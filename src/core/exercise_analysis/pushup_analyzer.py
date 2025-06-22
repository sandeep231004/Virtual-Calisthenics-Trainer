from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
from enum import Enum
from collections import deque

from .base_analyzer_robust import BaseExerciseAnalyzer, ExerciseState, UserLevel


class PushupPhase(Enum):
    """Push-up exercise phases."""
    REST = "rest"           # Starting position (arms extended)
    DESCENT = "descent"     # Lowering phase
    BOTTOM = "bottom"       # Bottom position (chest near floor)
    ASCENT = "ascent"       # Rising phase
    TOP = "top"            # Top position (arms extended)


class PushupAnalyzer(BaseExerciseAnalyzer):
    """Analyzer for push-up exercise form with improved state machine and form checking."""

    def __init__(self, user_level: UserLevel = UserLevel.BEGINNER):
        """Initialize the push-up analyzer with enhanced tracking."""
        super().__init__(user_level)
        self._rep_count = 0
        self._current_phase = PushupPhase.REST
        self._phase_start_time = None
        self._min_phase_duration = 0.2  # Reduced for more responsive detection
        self._min_confidence = 0.3
        self._min_landmark_visibility = 0.5
        
        # Rep quality tracking
        self._current_rep_quality = {"form_violations": [], "range_quality": 0.0}
        self._completed_reps = []
        
        # Phase transition thresholds (adjusted by user level)
        self._phase_thresholds = self._get_phase_thresholds()
        
        # Movement tracking with smoothing
        self._elbow_angle_history = {"left": deque(maxlen=5), "right": deque(maxlen=5)}
        self._movement_direction = "stationary"
        self._movement_velocity = 0.0  # degrees per second
        
        # Tempo tracking
        self._descent_start_time = None
        self._ascent_start_time = None
        self._rep_start_time = None
        
        # Camera angle detection with temporal smoothing
        self._camera_angle = "unknown"
        self._camera_angle_history = deque(maxlen=10)
        
        # Asymmetry tracking
        self._asymmetry_history = deque(maxlen=10)
        
        # Adaptive frame rate for performance
        self._last_analysis_time = 0
        self._analysis_interval = 0.1  # 10 FPS base rate
        self._last_significant_change = 0

    def _get_phase_thresholds(self) -> Dict[str, float]:
        """Get phase transition thresholds adjusted for user level."""
        base_thresholds = {
            "descent_start": 155,   # Start descent when elbow angle drops below this
            "bottom_reached": 90,   # Bottom reached when elbow angle reaches this
            "ascent_start": 85,     # Start ascent when elbow angle rises above this (with hysteresis)
            "top_reached": 150,     # Top reached when elbow angle reaches this
            "rest_threshold": 160,  # Return to rest when angle exceeds this
        }
        
        # Adjust for user level
        adjustments = {
            UserLevel.BEGINNER: {"bottom_reached": 100, "top_reached": 145},
            UserLevel.INTERMEDIATE: {"bottom_reached": 85, "top_reached": 160},
            UserLevel.ADVANCED: {"bottom_reached": 75, "top_reached": 165},
        }
        
        if self.user_level in adjustments:
            base_thresholds.update(adjustments[self.user_level])
            
        return base_thresholds

    def _update_movement_direction(self, angles: Dict[str, float]) -> None:
        """Update movement direction based on elbow angle changes."""
        current_angles = {
            "left": angles.get("left_elbow"),
            "right": angles.get("right_elbow")
        }
        
        if self._elbow_angle_history["left"] and self._elbow_angle_history["right"]:
            left_change = current_angles["left"] - self._elbow_angle_history["left"][-1][0] if current_angles["left"] is not None else 0
            right_change = current_angles["right"] - self._elbow_angle_history["right"][-1][0] if current_angles["right"] is not None else 0
            
            avg_change = (left_change + right_change) / 2
            
            if abs(avg_change) < 2:  # Less than 2 degrees change
                self._movement_direction = "stationary"
            elif avg_change > 0:
                self._movement_direction = "ascending"
            else:
                self._movement_direction = "descending"
        
        self._elbow_angle_history["left"].append((angles["left_elbow"], time.time()))
        self._elbow_angle_history["right"].append((angles["right_elbow"], time.time()))

    def _detect_camera_angle(self, landmarks: Dict[str, List[float]]) -> str:
        """Detect primary camera viewing angle."""
        if "left_shoulder" not in landmarks or "right_shoulder" not in landmarks:
            return "unknown"
            
        shoulder_depth = abs(landmarks["left_shoulder"][2] - landmarks["right_shoulder"][2])
        shoulder_width = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0])
        
        if shoulder_depth / shoulder_width > 0.3:
            return "angled_view"
        elif shoulder_width < 0.3:
            return "front_view"
        else:
            return "side_view"

    def _update_camera_angle(self, landmarks: Dict[str, List[float]]) -> None:
        """Update camera angle detection with temporal smoothing."""
        new_angle = self._detect_camera_angle(landmarks)
        self._camera_angle_history.append(new_angle)
        
        # Keep last 5 detections
        if len(self._camera_angle_history) > 5:
            self._camera_angle_history.pop(0)
            
        # Use most common angle
        from collections import Counter
        self._camera_angle = Counter(self._camera_angle_history).most_common(1)[0][0]

    def _update_phase_state_machine(self, angles: Dict[str, float], current_time: float) -> None:
        """Fixed state machine with proper phase transitions."""
        if not self._phase_start_time:
            self._phase_start_time = current_time
            
        phase_duration = current_time - self._phase_start_time
        avg_elbow = self._get_average_elbow_angle(angles)
        
        if avg_elbow is None:
            return
        
        new_phase = None
        
        # State machine transitions
        if self._current_phase == PushupPhase.REST:
            if avg_elbow < self._phase_thresholds["descent_start"]:
                new_phase = PushupPhase.DESCENT
                self._rep_start_time = current_time
                self._descent_start_time = current_time
                
        elif self._current_phase == PushupPhase.DESCENT:
            if (avg_elbow <= self._phase_thresholds["bottom_reached"] and 
                self._movement_direction in ["stationary", "descending"]):
                new_phase = PushupPhase.BOTTOM
                
        elif self._current_phase == PushupPhase.BOTTOM:
            if (avg_elbow > self._phase_thresholds["ascent_start"] and 
                self._movement_direction == "ascending"):
                new_phase = PushupPhase.ASCENT
                self._ascent_start_time = current_time
                
        elif self._current_phase == PushupPhase.ASCENT:
            if avg_elbow >= self._phase_thresholds["top_reached"]:
                new_phase = PushupPhase.TOP
                
        elif self._current_phase == PushupPhase.TOP:
            if (phase_duration >= self._min_phase_duration and 
                avg_elbow >= self._phase_thresholds["rest_threshold"]):
                # Complete repetition
                self._rep_count += 1
                rep_quality = self._calculate_rep_quality(angles, self._current_phase)
                self._completed_reps.append(rep_quality)
                new_phase = PushupPhase.REST
                self._rep_start_time = None
        
        # Apply valid transition
        if (new_phase and 
            self._is_valid_phase_transition(self._current_phase, new_phase, angles, phase_duration)):
            self._current_phase = new_phase
            self._phase_start_time = current_time

    def _calculate_rep_quality(self, angles: Dict[str, float], phase: PushupPhase) -> Dict[str, Any]:
        """Enhanced quality metrics for the current repetition."""
        quality = {
            "form_violations": [],
            "range_quality": 0.0,
            "symmetry_score": 0.0,
            "tempo_score": 0.0,
            "depth_achieved": 0.0
        }
        
        # Check form violations
        quality["form_violations"] = self.check_form_violations(angles)
        
        # Calculate range quality (0-1)
        avg_elbow = self._get_average_elbow_angle(angles)
        if avg_elbow is not None:
            if phase == PushupPhase.BOTTOM:
                # Score based on depth achieved
                min_depth = self._phase_thresholds["bottom_reached"]
                ideal_depth = 70  # Ideal depth for advanced users
                max_depth = 130   # Shallow end
                
                if avg_elbow <= ideal_depth:
                    quality["range_quality"] = 1.0
                elif avg_elbow <= min_depth:
                    quality["range_quality"] = 0.8
                else:
                    # Linear decrease for shallow reps
                    quality["range_quality"] = max(0, 1 - (avg_elbow - min_depth) / (max_depth - min_depth))
                
                quality["depth_achieved"] = max_depth - avg_elbow
        
        # Calculate symmetry score (0-1)
        asymmetry = self._calculate_asymmetry_score(angles)
        if asymmetry > 0:
            quality["symmetry_score"] = max(0, 1 - (asymmetry / 25))  # 25 degrees = 0 score
        
        # Calculate tempo score (0-1)
        if self._rep_start_time:
            rep_duration = time.time() - self._rep_start_time
            ideal_duration = 3.0  # 3 seconds for full rep (down + up)
            tempo_deviation = abs(rep_duration - ideal_duration) / ideal_duration
            quality["tempo_score"] = max(0, 1 - tempo_deviation)
        
        return quality

    def _calculate_asymmetry_score(self, angles: Dict[str, float]) -> float:
        """Calculate asymmetry between left and right sides."""
        if "left_elbow" not in angles or "right_elbow" not in angles:
            return 0.0
            
        if np.isnan(angles["left_elbow"]) or np.isnan(angles["right_elbow"]):
            return 0.0
            
        angle_diff = abs(angles["left_elbow"] - angles["right_elbow"])
        self._asymmetry_history.append(angle_diff)
        
        # Return average asymmetry over recent history
        return np.mean(list(self._asymmetry_history))

    def _is_rep_start_condition(self, angles: Dict[str, float], phase: str) -> bool:
        """Determine if the current frame marks the start of a repetition."""
        if phase != PushupPhase.REST.value:
            return False
            
        # Get average elbow angle
        elbow_angles = []
        if "left_elbow" in angles and not np.isnan(angles["left_elbow"]):
            elbow_angles.append(angles["left_elbow"])
        if "right_elbow" in angles and not np.isnan(angles["right_elbow"]):
            elbow_angles.append(angles["right_elbow"])
            
        if not elbow_angles:
            return False
            
        avg_elbow_angle = np.mean(elbow_angles)
        
        # Apply user level tolerance
        tolerance = self.level_config.rep_counting_tolerance * 10  # Convert to degrees
        return avg_elbow_angle < (self._phase_thresholds["descent_start"] + tolerance)

    def _is_rep_end_condition(self, angles: Dict[str, float], phase: str) -> bool:
        """Determine if the current frame marks the end of a repetition."""
        if phase != PushupPhase.DESCENT.value:
            return False
            
        # Get average elbow angle
        elbow_angles = []
        if "left_elbow" in angles and not np.isnan(angles["left_elbow"]):
            elbow_angles.append(angles["left_elbow"])
        if "right_elbow" in angles and not np.isnan(angles["right_elbow"]):
            elbow_angles.append(angles["right_elbow"])
            
        if not elbow_angles:
            return False
            
        avg_elbow_angle = np.mean(elbow_angles)
        
        # Apply user level tolerance
        tolerance = self.level_config.rep_counting_tolerance * 10  # Convert to degrees
        return avg_elbow_angle > (self._phase_thresholds["bottom_reached"] - tolerance)

    def _should_analyze_frame(self, current_time: float) -> bool:
        """Adaptive frame rate based on movement speed and activity."""
        time_since_last = current_time - self._last_analysis_time
        
        # High frequency during active movement
        if self._movement_direction != "stationary":
            return time_since_last >= 0.05  # 20 FPS during movement
        
        # Lower frequency during static holds
        return time_since_last >= self._analysis_interval

    def analyze_frame(
        self, 
        landmarks: Dict[str, List[float]], 
        angles: Dict[str, float],
        exercise_variant: Optional[str] = None
    ) -> ExerciseState:
        """Enhanced frame analysis with fixes for critical issues."""
        current_time = time.time()
        
        # Adaptive frame rate check
        if not self._should_analyze_frame(current_time):
            # Return previous state for skipped frames
            return ExerciseState(
                name=self.get_exercise_name(),
                phase=self._current_phase.value,
                rep_count=self._rep_count,
                is_correct_form=True,  # Assume no change
                violations=[],
                angles=angles,
                confidence=1.0,
                analysis_reliable=True,
                user_level=self.user_level
            )
        
        self._last_analysis_time = current_time
        
        # Update camera angle detection
        self._update_camera_angle(landmarks)
        
        # Calculate confidence
        confidence = self.calculate_confidence(landmarks)
        
        # Skip analysis if confidence is too low
        if confidence < self.level_config.min_confidence:
            return ExerciseState(
                name=self.get_exercise_name(),
                phase=self._current_phase.value,
                rep_count=self._rep_count,
                is_correct_form=False,
                violations=["Low confidence in pose detection"],
                angles=angles,
                confidence=confidence,
                analysis_reliable=False,
                user_level=self.user_level
            )
        
        # Validate camera position
        camera_valid, camera_error = self._validate_camera_position(landmarks)
        if not camera_valid:
            return ExerciseState(
                name=self.get_exercise_name(),
                phase=self._current_phase.value,
                rep_count=self._rep_count,
                is_correct_form=False,
                violations=[camera_error],
                angles=angles,
                confidence=confidence,
                analysis_reliable=False,
                error_message=camera_error,
                user_level=self.user_level
            )
        
        # Validate inputs
        is_valid, error_message = self.validate_inputs(landmarks, angles)
        if not is_valid:
            return ExerciseState(
                name=self.get_exercise_name(),
                phase=self._current_phase.value,
                rep_count=self._rep_count,
                is_correct_form=False,
                violations=[error_message],
                angles=angles,
                confidence=confidence,
                analysis_reliable=False,
                error_message=error_message,
                user_level=self.user_level
            )

        # Update movement tracking
        self._update_movement_tracking(angles, current_time)

        # Update phase state machine
        self._update_phase_state_machine(angles, current_time)

        # Calculate current rep quality
        self._current_rep_quality = self._calculate_rep_quality(angles, self._current_phase)

        # Check form violations
        violations = self.check_form_violations(angles, exercise_variant)

        return ExerciseState(
            name=self.get_exercise_name(),
            phase=self._current_phase.value,
            rep_count=self._rep_count,
            is_correct_form=len(violations) == 0,
            violations=violations,
            angles=angles,
            confidence=confidence,
            analysis_reliable=True,
            user_level=self.user_level,
            rep_quality=self._current_rep_quality
        )

    def get_exercise_name(self) -> str:
        """Get the name of the exercise."""
        return "pushup"

    def get_required_landmarks(self) -> List[str]:
        """Get the list of required landmarks."""
        return [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_ankle", "right_ankle",
            "left_ear", "right_ear",
            "nose"
        ]

    def get_required_angles(self) -> List[str]:
        """Get the list of required angles."""
        return [
            "left_elbow", "right_elbow",
            "body_floor_angle",
            "neck_angle",
            "hip_angle",
            "left_wrist_angle",
            "right_wrist_angle"
        ]

    def get_form_rules(self, exercise_variant: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Enhanced form rules with proper biomechanical thresholds."""
        base_rules = {
            "left_elbow": {
                "min": {
                    "threshold": 70,  # Full depth for standard push-up
                    "message": "Go deeper - bend elbows more for full range",
                    "level_adjustments": {
                        "beginner": {"threshold": 90, "message": "Try to bend your elbows more"},
                        "intermediate": {"threshold": 75, "message": "Achieve better depth"},
                        "advanced": {"threshold": 65, "message": "Achieve full depth"}
                    }
                },
                "max": {
                    "threshold": 170,  # Near full extension
                    "message": "Extend arms fully at the top",
                    "level_adjustments": {
                        "beginner": {"threshold": 160},
                        "intermediate": {"threshold": 175},
                        "advanced": {"threshold": 180}
                    }
                }
            },
            "right_elbow": {
                "min": {
                    "threshold": 70,
                    "message": "Go deeper - bend elbows more for full range",
                    "level_adjustments": {
                        "beginner": {"threshold": 90, "message": "Try to bend your elbows more"},
                        "intermediate": {"threshold": 75, "message": "Achieve better depth"},
                        "advanced": {"threshold": 65, "message": "Achieve full depth"}
                    }
                },
                "max": {
                    "threshold": 170,
                    "message": "Extend arms fully at the top",
                    "level_adjustments": {
                        "beginner": {"threshold": 160},
                        "intermediate": {"threshold": 175},
                        "advanced": {"threshold": 180}
                    }
                }
            },
            "body_floor_angle": {
                "max": {
                    "threshold": 10,  # Stricter body alignment
                    "message": "Keep your body straighter - don't let hips sag or pike",
                    "level_adjustments": {
                        "beginner": {"threshold": 15},
                        "intermediate": {"threshold": 8},
                        "advanced": {"threshold": 5}
                    }
                }
            },
            "hip_angle": {
                "min": {
                    "threshold": 165,  # Prevent hip sagging
                    "message": "Keep your body straight - don't let hips sag",
                    "level_adjustments": {
                        "beginner": {"threshold": 160},
                        "intermediate": {"threshold": 170},
                        "advanced": {"threshold": 175}
                    }
                },
                "max": {
                    "threshold": 185,  # Prevent piking
                    "message": "Don't pike your hips up - keep body straight",
                    "level_adjustments": {
                        "beginner": {"threshold": 190},
                        "intermediate": {"threshold": 180},
                        "advanced": {"threshold": 175}
                    }
                }
            },
            "neck_angle": {
                "min": {
                    "threshold": 150,
                    "message": "Keep your neck aligned with your spine",
                    "level_adjustments": {
                        "beginner": {"threshold": 145},
                        "intermediate": {"threshold": 155},
                        "advanced": {"threshold": 160}
                    }
                },
                "max": {
                    "threshold": 180,
                    "message": "Don't look too far up - maintain neutral neck",
                    "level_adjustments": {
                        "beginner": {"threshold": 185},
                        "intermediate": {"threshold": 175},
                        "advanced": {"threshold": 170}
                    }
                }
            },
            "left_wrist_angle": {
                "min": {
                    "threshold": 160,
                    "message": "Keep your left wrist straight to prevent injury",
                    "level_adjustments": {
                        "beginner": {"threshold": 155},
                        "intermediate": {"threshold": 165},
                        "advanced": {"threshold": 170}
                    }
                }
            },
            "right_wrist_angle": {
                "min": {
                    "threshold": 160,
                    "message": "Keep your right wrist straight to prevent injury",
                    "level_adjustments": {
                        "beginner": {"threshold": 155},
                        "intermediate": {"threshold": 165},
                        "advanced": {"threshold": 170}
                    }
                }
            },
            "elbow_asymmetry": {
                "max": {
                    "threshold": 12,  # Stricter asymmetry tolerance
                    "message": "Keep both arms even - maintain symmetrical form",
                    "level_adjustments": {
                        "beginner": {"threshold": 18},
                        "intermediate": {"threshold": 10},
                        "advanced": {"threshold": 6}
                    }
                }
            }
        }
        
        # Apply variant-specific modifications
        if exercise_variant == "diamond":
            # Diamond push-ups: hands close together, more tricep focus
            for side in ["left_elbow", "right_elbow"]:
                base_rules[side]["min"]["threshold"] += 10  # Less depth required
        elif exercise_variant == "wide":
            # Wide push-ups: hands wider apart, more chest focus
            for side in ["left_elbow", "right_elbow"]:
                base_rules[side]["min"]["threshold"] -= 5   # More depth possible
        elif exercise_variant == "incline":
            # Incline push-ups: easier variant, more forgiving
            for side in ["left_elbow", "right_elbow"]:
                base_rules[side]["min"]["threshold"] += 15
            base_rules["body_floor_angle"]["max"]["threshold"] += 5
        elif exercise_variant == "decline":
            # Decline push-ups: harder variant, stricter form
            for side in ["left_elbow", "right_elbow"]:
                base_rules[side]["min"]["threshold"] -= 5
            base_rules["body_floor_angle"]["max"]["threshold"] -= 3
        
        return base_rules

    def _get_average_elbow_angle(self, angles: Dict[str, float]) -> Optional[float]:
        """Get average elbow angle from both sides, handling missing data."""
        elbow_angles = []
        
        if "left_elbow" in angles and not np.isnan(angles["left_elbow"]):
            elbow_angles.append(angles["left_elbow"])
        if "right_elbow" in angles and not np.isnan(angles["right_elbow"]):
            elbow_angles.append(angles["right_elbow"])
            
        return np.mean(elbow_angles) if elbow_angles else None

    def _validate_camera_position(self, landmarks: Dict[str, List[float]]) -> tuple[bool, Optional[str]]:
        """
        Validate if the camera position is optimal for detection.

        Args:
            landmarks: Dictionary of landmark coordinates

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check visibility of critical landmarks
        critical_landmarks = {
            "shoulders": ["left_shoulder", "right_shoulder"],
            "elbows": ["left_elbow", "right_elbow"],
            "hips": ["left_hip", "right_hip"],
            "ankles": ["left_ankle", "right_ankle"]
        }

        visibility_issues = []
        
        for group, points in critical_landmarks.items():
            # Check if at least one point in each group is visible
            group_visible = False
            for point in points:
                if point in landmarks and landmarks[point][3] > self._min_landmark_visibility:
                    group_visible = True
                    break
            
            if not group_visible:
                visibility_issues.append(group)

        if visibility_issues:
            error_message = "Camera position needs adjustment: "
            if "shoulders" in visibility_issues:
                error_message += "Move camera to better see shoulders. "
            if "elbows" in visibility_issues:
                error_message += "Move camera to better see elbows. "
            if "hips" in visibility_issues:
                error_message += "Move camera to better see hips. "
            if "ankles" in visibility_issues:
                error_message += "Move camera to better see ankles. "
            return False, error_message

        # Check if body is too close to camera
        if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
            shoulder_distance = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0])
            if shoulder_distance > 0.8:  # If shoulders take up more than 80% of frame width
                return False, "Move camera back: body is too close to camera"

        # Check if body is too far from camera
        if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
            shoulder_distance = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0])
            if shoulder_distance < 0.2:  # If shoulders take up less than 20% of frame width
                return False, "Move camera closer: body is too far from camera"

        # Check if body is centered in frame
        if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
            center_x = (landmarks["left_shoulder"][0] + landmarks["right_shoulder"][0]) / 2
            if abs(center_x - 0.5) > 0.3:  # If body center is more than 30% off center
                return False, "Center your body in the frame"

        return True, None

    def check_form_violations(self, angles: Dict[str, float], exercise_variant: Optional[str] = None) -> List[str]:
        """
        Check for form violations based on joint angles.

        Args:
            angles: Dictionary of joint angles
            exercise_variant: Optional variant of the exercise

        Returns:
            List of violation messages
        """
        violations = []
        rules = self.get_form_rules(exercise_variant)

        # Check elbow angles
        for side in ["left_elbow", "right_elbow"]:
            if side in angles and not np.isnan(angles[side]):
                if angles[side] < rules[side]["min"]["threshold"]:
                    violations.append(f"{side}_too_deep")
                elif angles[side] > rules[side]["max"]["threshold"]:
                    violations.append(f"{side}_too_shallow")

        # Check body-floor angle
        if "body_floor_angle" in angles and not np.isnan(angles["body_floor_angle"]):
            if angles["body_floor_angle"] > rules["body_floor_angle"]["max"]["threshold"]:
                violations.append("body_not_parallel_to_floor")

        # Check neck angle
        if "neck_angle" in angles and not np.isnan(angles["neck_angle"]):
            if angles["neck_angle"] < rules["neck_angle"]["min"]["threshold"]:
                violations.append("neck_not_aligned")

        # Check shoulder angle
        if "hip_angle" in angles and not np.isnan(angles["hip_angle"]):
            if angles["hip_angle"] < rules["hip_angle"]["min"]["threshold"]:
                violations.append("hips_too_forward")
            elif angles["hip_angle"] > rules["hip_angle"]["max"]["threshold"]:
                violations.append("hips_too_backward")

        return violations

    def _update_movement_tracking(self, angles: Dict[str, float], current_time: float) -> None:
        """Enhanced movement tracking with velocity calculation."""
        # Update angle history
        for side in ["left", "right"]:
            angle_key = f"{side}_elbow"
            if angle_key in angles and not np.isnan(angles[angle_key]):
                self._elbow_angle_history[side].append((angles[angle_key], current_time))
        
        # Calculate movement direction and velocity
        velocities = []
        for side in ["left", "right"]:
            history = self._elbow_angle_history[side]
            if len(history) >= 2:
                # Calculate velocity from last two points
                angle_diff = history[-1][0] - history[-2][0]
                time_diff = history[-1][1] - history[-2][1]
                if time_diff > 0:
                    velocity = angle_diff / time_diff
                    velocities.append(velocity)
        
        if velocities:
            self._movement_velocity = np.mean(velocities)
            
            # Update movement direction with deadband
            if abs(self._movement_velocity) < 5:  # Less than 5 degrees/second
                self._movement_direction = "stationary"
            elif self._movement_velocity > 0:
                self._movement_direction = "ascending"
                self._last_significant_change = current_time
            else:
                self._movement_direction = "descending"
                self._last_significant_change = current_time

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for the session."""
        if not self._completed_reps:
            return {
                "total_reps": self._rep_count,
                "average_quality": 0.0,
                "best_rep_quality": 0.0,
                "consistency_score": 0.0,
                "common_violations": [],
                "symmetry_score": 0.0,
                "tempo_score": 0.0,
                "depth_achieved": 0.0
            }
        
        # Calculate aggregate metrics
        quality_scores = [rep.get("range_quality", 0) for rep in self._completed_reps]
        symmetry_scores = [rep.get("symmetry_score", 0) for rep in self._completed_reps]
        tempo_scores = [rep.get("tempo_score", 0) for rep in self._completed_reps]
        depth_scores = [rep.get("depth_achieved", 0) for rep in self._completed_reps]
        
        # Calculate consistency score (standard deviation of quality scores)
        consistency_score = 1.0 - min(1.0, np.std(quality_scores) if quality_scores else 1.0)
        
        # Collect all violations
        all_violations = []
        for rep in self._completed_reps:
            all_violations.extend(rep.get("form_violations", []))
        
        from collections import Counter
        common_violations = Counter(all_violations).most_common(3)
        
        return {
            "total_reps": self._rep_count,
            "average_quality": np.mean(quality_scores) if quality_scores else 0.0,
            "best_rep_quality": max(quality_scores) if quality_scores else 0.0,
            "consistency_score": consistency_score,
            "common_violations": [violation for violation, _ in common_violations],
            "symmetry_score": np.mean(symmetry_scores) if symmetry_scores else 0.0,
            "tempo_score": np.mean(tempo_scores) if tempo_scores else 0.0,
            "depth_achieved": np.mean(depth_scores) if depth_scores else 0.0
        }