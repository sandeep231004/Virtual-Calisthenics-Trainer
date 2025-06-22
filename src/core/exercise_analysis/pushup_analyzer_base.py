from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
from enum import Enum
from collections import deque, Counter
from .base_analyzer_robust import BaseExerciseAnalyzer, ExerciseState, UserLevel
import logging
import json
import tracemalloc


class PushupPhase(Enum):  # Define an enum class for the pushup exercise phases.
    """Push-up exercise phases.""" # Define set of named constants for each phase of the pushup exercise.
    REST = "rest"           # Starting position (arms extended)
    DESCENT = "descent"     # Lowering phase
    BOTTOM = "bottom"       # Bottom position (chest near floor)
    ASCENT = "ascent"       # Rising phase
    TOP = "top"            # Top position (arms extended)


# --- View Analyzer Registry ---
VIEW_ANALYZER_REGISTRY = {} # Dictionary to map view types to their corresponding analyzer classes.

def register_view_analyzer(view_type): # Decorator to automatically add new view specific analyzers classes to the registry when they are defined.
    def decorator(cls):
        VIEW_ANALYZER_REGISTRY[view_type] = cls
        return cls
    return decorator


# --- View-Specific Analyzers ---

class ViewSpecificAnalyzer: # Abstract base class for view-specific analyzers. It defines the interface for all view-specific analyzers.
    def __init__(self, view_type: str):
        self.view_type = view_type
        self.min_landmark_visibility = 0.5

    def get_required_angles(self) -> List[str]: # Abstract method to get the list of required angles for the analyzer. It must be implemented by the subclass.
        raise NotImplementedError # Ensures that the subclass can't be instantiated if this method is not defined.

    def get_phase_thresholds(self, user_level: UserLevel) -> Dict[str, float]: # Abstract method to get the phase thresholds for the analyzer. It must be implemented by the subclass.
        raise NotImplementedError

    def validate_camera_position(self, landmarks: Dict[str, List[float]]) -> Tuple[bool, Optional[str]]: # Abstract method to validate the camera position for the analyzer. It must be implemented by the subclass.
        raise NotImplementedError

    def get_form_rules(self, user_level: UserLevel, exercise_variant: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]: # Abstract method to get the form rules for the analyzer. It must be implemented by the subclass.
        raise NotImplementedError

@register_view_analyzer('side')
class SideViewAnalyzer(ViewSpecificAnalyzer):
    def __init__(self):
        super().__init__('side')

    def get_required_angles(self) -> List[str]:
        return ["left_elbow", "right_elbow", "body_floor_angle", "hip_angle", "neck_angle"]

    def get_phase_thresholds(self, user_level: UserLevel) -> Dict[str, float]:
        base = {
            "descent_start": 155,
            "bottom_reached": 90,
            "ascent_start": 95,
            "top_reached": 165,
            "rest_threshold": 160
        }
        level_adj = {
            UserLevel.BEGINNER: {"bottom_reached": 100, "top_reached": 160},
            UserLevel.INTERMEDIATE: {"bottom_reached": 95, "top_reached": 165},
            UserLevel.ADVANCED: {"bottom_reached": 85, "top_reached": 170}
        }
        if user_level in level_adj:
            base.update(level_adj[user_level])
        return base

    def validate_camera_position(self, landmarks):
        if "left_shoulder" not in landmarks or "right_shoulder" not in landmarks:
            return False, "Cannot detect shoulders — adjust camera"
        shoulder_width = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0])
        torso_length = self._calculate_torso_length(landmarks)
        if torso_length > 0 and (shoulder_width / torso_length) > 0.35:
            return False, "Shift camera to a clearer side profile"
        critical = ["left_shoulder","right_shoulder","left_elbow","right_elbow","left_hip","right_hip","left_ankle","right_ankle"]
        visible = sum(1 for l in critical if l in landmarks and landmarks[l][3] > self.min_landmark_visibility)
        if visible < len(critical) * 0.75:
            return False, "Move camera to fully frame your body"
        return True, None

    def get_form_rules(self, user_level, exercise_variant=None):
        rules = {
            "left_elbow": {
                "min": {
                    "threshold": 100,
                    "message": "Go deeper — bend elbows more",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 100},
                        UserLevel.INTERMEDIATE: {"threshold": 90},
                        UserLevel.ADVANCED: {"threshold": 85}
                    }
                },
                "max": {
                    "threshold": 175,
                    "message": "Extend arms fully at top",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 170},
                        UserLevel.INTERMEDIATE: {"threshold": 175},
                        UserLevel.ADVANCED: {"threshold": 180}
                    }
                }
            },
            "right_elbow": {
                "min": {
                    "threshold": 90,
                    "message": "Go deeper — bend elbows more",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 100},
                        UserLevel.INTERMEDIATE: {"threshold": 90},
                        UserLevel.ADVANCED: {"threshold": 85}
                    }
                },
                "max": {
                    "threshold": 175,
                    "message": "Extend arms fully at top",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 170},
                        UserLevel.INTERMEDIATE: {"threshold": 175},
                        UserLevel.ADVANCED: {"threshold": 180}
                    }
                }
            },
            "body_floor_angle": {
                "max": {
                    "threshold": 10,
                    "message": "Keep body straighter — avoid sag",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 15},
                        UserLevel.INTERMEDIATE: {"threshold": 10},
                        UserLevel.ADVANCED: {"threshold": 5}
                    }
                }
            },
            "hip_angle": {
                "min": {
                    "threshold": 165,
                    "message": "Don't let hips sag",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 160},
                        UserLevel.INTERMEDIATE: {"threshold": 165},
                        UserLevel.ADVANCED: {"threshold": 170}
                    }
                },
                "max": {
                    "threshold": 185,
                    "message": "Don't pike your hips up",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 190},
                        UserLevel.INTERMEDIATE: {"threshold": 185},
                        UserLevel.ADVANCED: {"threshold": 180}
                    }
                }
            },
            "neck_angle": {
                "min": {
                    "threshold": 150,
                    "message": "Keep neck aligned with spine",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 145},
                        UserLevel.INTERMEDIATE: {"threshold": 150},
                        UserLevel.ADVANCED: {"threshold": 155}
                    }
                },
                "max": {
                    "threshold": 180,
                    "message": "Don't look too far up",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 185},
                        UserLevel.INTERMEDIATE: {"threshold": 180},
                        UserLevel.ADVANCED: {"threshold": 175}
                    }
                }
            }
        }
        return self._apply_variant_adjustments(rules, exercise_variant)

    def _calculate_torso_length(self, landmarks):
        if "left_shoulder" in landmarks and "left_hip" in landmarks:
            return abs(landmarks["left_shoulder"][1] - landmarks["left_hip"][1])
        if "right_shoulder" in landmarks and "right_hip" in landmarks:
            return abs(landmarks["right_shoulder"][1] - landmarks["right_hip"][1])
        return 0.0

    def _apply_variant_adjustments(self, rules: Dict, exercise_variant: Optional[str]) -> Dict:
        # Apply variant tweaks
        if exercise_variant == "diamond":
            for side in ("left_elbow", "right_elbow"):
                rules[side]["min"]["threshold"] += 10
        elif exercise_variant == "wide":
            for side in ("left_elbow", "right_elbow"):
                rules[side]["min"]["threshold"] -= 5
        elif exercise_variant == "incline":
            for side in ("left_elbow", "right_elbow"):
                rules[side]["min"]["threshold"] += 15
            rules["body_floor_angle"]["max"]["threshold"] += 3
        elif exercise_variant == "decline":
            for side in ("left_elbow", "right_elbow"):
                rules[side]["min"]["threshold"] -= 5
            rules["body_floor_angle"]["max"]["threshold"] -= 3

        return rules

@register_view_analyzer('front')
class FrontViewAnalyzer(ViewSpecificAnalyzer):
    def __init__(self):
        super().__init__('front')

    def get_required_angles(self) -> List[str]:
        return ["left_elbow", "right_elbow", "left_wrist_angle", "right_wrist_angle", "elbow_torso_angle"]

    def get_phase_thresholds(self, user_level: UserLevel) -> Dict[str, float]:
        base = {
            "descent_start": 150,
            "bottom_reached": 95,
            "ascent_start": 90,
            "top_reached": 150,
            "rest_threshold": 155
        }
        adj = {
            UserLevel.BEGINNER: {"bottom_reached": 105, "top_reached": 140},
            UserLevel.INTERMEDIATE: {"bottom_reached": 95, "top_reached": 150},
            UserLevel.ADVANCED: {"bottom_reached": 85, "top_reached": 160}
        }
        if user_level in adj:
            base.update(adj[user_level])
        return base

    def validate_camera_position(self, landmarks: Dict[str, List[float]]) -> Tuple[bool, Optional[str]]:
        # Shoulder visibility check
        if "left_shoulder" not in landmarks or "right_shoulder" not in landmarks:
            return False, "Cannot detect shoulders — adjust camera"
        if "left_wrist" not in landmarks or "right_wrist" not in landmarks:
            return False, "Cannot detect wrists — adjust camera"

        shoulder_width = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0])
        torso_length = self._calculate_torso_length(landmarks)
        if torso_length > 0:
            ratio = shoulder_width / torso_length
            if ratio < 0.3:
                return False, "Move camera closer to frame your upper body"
            if ratio > 0.8:
                return False, "Move camera back to capture full body"

        # Ensure both arms visible
        for side in ["left", "right"]:
            for joint in [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]:
                if joint not in landmarks or landmarks[joint][3] < self.min_landmark_visibility:
                    return False, "Ensure both arms are fully visible in front view"

        return True, None

    def get_form_rules(self, user_level: UserLevel, exercise_variant: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        rules = {
            "left_elbow": {
                "min": {
                    "threshold": 75,
                    "message": "Bend left elbow more for full depth",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 95},
                        UserLevel.INTERMEDIATE: {"threshold": 80},
                        UserLevel.ADVANCED: {"threshold": 70}
                    }
                },
                "max": {
                    "threshold": 165,
                    "message": "Extend left arm fully at top",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 155},
                        UserLevel.INTERMEDIATE: {"threshold": 170},
                        UserLevel.ADVANCED: {"threshold": 175}
                    }
                }
            },
            "right_elbow": {
                "min": {
                    "threshold": 75,
                    "message": "Bend right elbow more for full depth",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 95},
                        UserLevel.INTERMEDIATE: {"threshold": 80},
                        UserLevel.ADVANCED: {"threshold": 70}
                    }
                },
                "max": {
                    "threshold": 165,
                    "message": "Extend right arm fully at top",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 155},
                        UserLevel.INTERMEDIATE: {"threshold": 170},
                        UserLevel.ADVANCED: {"threshold": 175}
                    }
                }
            },
            "left_wrist_angle": {
                "min": {
                    "threshold": 160,
                    "message": "Keep left wrist straight",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 155},
                        UserLevel.INTERMEDIATE: {"threshold": 165},
                        UserLevel.ADVANCED: {"threshold": 170}
                    }
                }
            },
            "right_wrist_angle": {
                "min": {
                    "threshold": 160,
                    "message": "Keep right wrist straight",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 155},
                        UserLevel.INTERMEDIATE: {"threshold": 165},
                        UserLevel.ADVANCED: {"threshold": 170}
                    }
                }
            },
            "elbow_torso_angle": {
                "max": {
                    "threshold": 60,
                    "message": "Keep elbows tucked (~45° from torso)",
                    "level_adjustments": {
                        UserLevel.BEGINNER: {"threshold": 70},
                        UserLevel.INTERMEDIATE: {"threshold": 60},
                        UserLevel.ADVANCED: {"threshold": 50}
                    }
                }
            }
        }

        return self._apply_variant_adjustments(rules, exercise_variant)

    def _calculate_torso_length(self, landmarks: Dict[str, List[float]]) -> float:
        if "left_shoulder" in landmarks and "left_hip" in landmarks:
            return abs(landmarks["left_shoulder"][1] - landmarks["left_hip"][1])
        if "right_shoulder" in landmarks and "right_hip" in landmarks:
            return abs(landmarks["right_shoulder"][1] - landmarks["right_hip"][1])
        return 0.0

    def _apply_variant_adjustments(self, rules: Dict, exercise_variant: Optional[str]) -> Dict:
        # Variant adaptations
        if exercise_variant == "diamond":
            for side in ("left_elbow", "right_elbow"):
                rules[side]["min"]["threshold"] += 10
        elif exercise_variant == "wide":
            for side in ("left_elbow", "right_elbow"):
                rules[side]["min"]["threshold"] -= 5
        elif exercise_variant == "incline":
            for side in ("left_elbow", "right_elbow"):
                rules[side]["min"]["threshold"] += 15
        elif exercise_variant == "decline":
            for side in ("left_elbow", "right_elbow"):
                rules[side]["min"]["threshold"] -= 5

        return rules

# --- Logger Setup ---
logger = logging.getLogger("PushupAnalyzer")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Configurable Threshold Loader (JSON/dict) ---
THRESHOLD_CONFIG = {
    "pushup": {
        "side": {
            "default": {
                "descent_start": 155,     # Begin lowering if elbow > 155°
                "bottom_reached": 90,     # Bottom when elbow ≤ 90°
                "ascent_start": 95,       # Start rising when elbow ≤ 95°
                "top_reached": 165,       # Top detected if elbow ≥ 165°
                "rest_threshold": 160     # Enter resting if elbow ≥ 160° and paused
            }
        },
        "front": {
            "default": {
                "descent_start": 150,     # Begin descent
                "bottom_reached": 90,     # Bottom detection
                "ascent_start": 95,       # Begin ascent
                "top_reached": 150,       # Top of push-up
                "rest_threshold": 155     # Ready state
            }
        }
    }
}


def load_thresholds_from_config(exercise_name, view_type, user_level):
    try:
        # Try to load from config dict (could be replaced with file/db)
        user_key = user_level.value if hasattr(user_level, 'value') else str(user_level)
        return THRESHOLD_CONFIG.get(exercise_name, {}).get(view_type, {}).get(user_key) or \
               THRESHOLD_CONFIG.get(exercise_name, {}).get(view_type, {}).get("default")
    except Exception as e:
        logger.error(f"Threshold config load error: {e}")
        return None


# --- Feedback Templates ---
class FeedbackGenerator:
    @staticmethod
    def missing_landmark(landmark):
        return f"Missing landmark: {landmark}"
    @staticmethod
    def missing_angle(angle):
        return f"Missing angle: {angle}"
    @staticmethod
    def partial_feedback(available, missing, confidence=None):
        msg = f"Partial analysis: available={available}, missing={missing}"
        if confidence is not None:
            msg += f" (confidence: {confidence:.2f})"
        return msg
    @staticmethod
    def ambiguous_view():
        return "Ambiguous camera angle. Please use a clear side or front view."
    @staticmethod
    def low_confidence():
        return "Low confidence in pose detection."
    @staticmethod
    def camera_error(msg):
        return f"Camera error: {msg}"
    @staticmethod
    def symmetry_not_available():
        return "Symmetry score not available due to missing data."
    @staticmethod
    def camera_reposition_side():
        return "⚠️ Please move camera to view your body from the side"
    @staticmethod
    def camera_reposition_front():
        return "⚠️ Please move camera to view your body from the front"

# --- Generic FormRule and ExercisePhase ---
class FormRule:
    def __init__(self, angle_name, min_val=None, max_val=None, message=None):
        self.angle_name = angle_name
        self.min_val = min_val
        self.max_val = max_val
        self.message = message
    def check(self, angles):
        val = angles.get(self.angle_name)
        if val is None or (isinstance(val, float) and (np.isnan(val))):
            return FeedbackGenerator.missing_angle(self.angle_name)
        if self.min_val is not None and val < self.min_val:
            return self.message or f"{self.angle_name} below minimum"
        if self.max_val is not None and val > self.max_val:
            return self.message or f"{self.angle_name} above maximum"
        return None

class ExercisePhase:
    def __init__(self, name, entry_condition, exit_condition):
        self.name = name
        self.entry_condition = entry_condition
        self.exit_condition = exit_condition
    def is_entry(self, angles):
        return self.entry_condition(angles)
    def is_exit(self, angles):
        return self.exit_condition(angles)

# --- Per-session calibration for view detection ---
class SessionCalibration:
    def __init__(self):
        self.torso_lengths = []
        self.shoulder_widths = []
        self.calibrated = False
        self.avg_torso = 1.0
        self.avg_shoulder = 1.0
    def update(self, landmarks):
        try:
            if "left_shoulder" in landmarks and "left_hip" in landmarks:
                torso = abs(landmarks["left_shoulder"][1] - landmarks["left_hip"][1])
                self.torso_lengths.append(torso)
            if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
                shoulder = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0])
                self.shoulder_widths.append(shoulder)
            if len(self.torso_lengths) > 10 and len(self.shoulder_widths) > 10:
                self.avg_torso = np.mean(self.torso_lengths)
                self.avg_shoulder = np.mean(self.shoulder_widths)
                self.calibrated = True
        except Exception as e:
            logger.error(f"Calibration error: {e}")
    def normalize(self, shoulder_width, torso_length):
        if self.calibrated and self.avg_torso > 0:
            return shoulder_width / self.avg_torso, torso_length / self.avg_torso
        return shoulder_width, torso_length
    def recalibrate(self):
        self.torso_lengths.clear()
        self.shoulder_widths.clear()
        self.calibrated = False
        self.avg_torso = 1.0
        self.avg_shoulder = 1.0

# --- Main Analyzer ---
class PushupAnalyzerBase(BaseExerciseAnalyzer):
    """Robust push-up analyzer for side, front, and ambiguous views."""
    _MIN_PHASE_DURATION = 0.2  # Seconds required to be in a phase to count it
    _VIEW_DETECTION_INTERVAL = 0.1  # Throttle view detection
    _VIEW_HISTORY_LEN = 8  # Rolling buffer for view confidence smoothing
    _ASYMMETRY_HISTORY_LEN = 10  # History for asymmetry detection
    _ELBOW_HISTORY_LEN = 5  # History for elbow angle smoothing
    _VELOCITY_SMOOTHING = 3  # Number of frames to average over
    _VELOCITY_THRESHOLDS = {'side': 8, 'front': 12}  # Velocity thresholds for phase detection
    _SHOULDER_WIDTH_RANGE = {'side': (0, 0.4), 'front': (0.3, 1.0)}  # Range for side view detection
    _HYSTERESIS_MARGIN = 0.05  # Hysteresis margin for view detection

    # Smooths noisy switching and enforces stable view confidence
    _AMBIGUOUS_VIEW_MIN_TIME = 0.5  # Minimum time to be considered ambiguous
    _AMBIGUOUS_VIEW_MIN_CONF = 0.7  # Minimum confidence for ambiguous view
    _STICKY_VIEW_MIN_TIME = 1.0  # Minimum time to be considered sticky
    _STICKY_VIEW_MIN_CONF = 0.7  # Minimum confidence for sticky view
    _DEBUG_MODE = False  # Debug flag to control profiling

    def __init__(self, user_level: UserLevel = UserLevel.BEGINNER):
        super().__init__(user_level)
        self._rep_count = 0
        self._current_phase = PushupPhase.REST
        self._phase_start_time = None
        self._current_view = "unknown"
        self._view_history = deque(maxlen=self._VIEW_HISTORY_LEN)
        self._last_view_detection_time = 0
        self._view_confidence = 0.0
        self._side_analyzer = SideViewAnalyzer()
        self._front_analyzer = FrontViewAnalyzer()
        self._current_analyzer = None
        self._elbow_angle_history = {"left": deque(maxlen=self._ELBOW_HISTORY_LEN), "right": deque(maxlen=self._ELBOW_HISTORY_LEN)}
        self._movement_direction = "stationary"
        self._movement_velocity = 0.0
        self._rep_start_time = None
        self._current_rep_quality = {"form_violations": [], "range_quality": 0.0}
        self._completed_reps = []
        self._asymmetry_history = deque(maxlen=self._ASYMMETRY_HISTORY_LEN)
        self._cached_phase_thresholds = {}
        self._cached_form_rules = {}
        self._velocity_history = deque(maxlen=self._VELOCITY_SMOOTHING)
        self._last_view = None
        self._last_view_confidence = 0.0
        self._view_analyzers = {k: v() for k, v in VIEW_ANALYZER_REGISTRY.items()}
        self._session_calibration = SessionCalibration()
        self._last_ambiguous_time = 0
        self._ambiguous_since = None
        self._sticky_view = None
        self._sticky_view_since = None
        self._sticky_view_conf = 0.0
        
        # Initialize tracemalloc if debug mode is enabled
        if self._DEBUG_MODE:
            tracemalloc.start()

    def __del__(self):
        """Cleanup profiling resources on object destruction"""
        if self._DEBUG_MODE:
            tracemalloc.stop()

    @classmethod
    def enable_debug(cls):
        """Enable debug mode"""
        cls._DEBUG_MODE = True
        tracemalloc.start()
        
    @classmethod
    def disable_debug(cls):
        """Disable debug mode"""
        if cls._DEBUG_MODE:
            tracemalloc.stop()
        cls._DEBUG_MODE = False

    def recalibrate(self):
        """Allow recalibration mid-session (e.g., if user moves/camera changes)."""
        self._session_calibration.recalibrate()
        logger.info("Session recalibrated.")

    def _detect_view(self, landmarks: Dict[str, List[float]]) -> Tuple[str, float, Optional[str]]:
        try:
            # Update calibration with new landmarks
            self._session_calibration.update(landmarks) # Helps normalize person size across frames and users.
            
            # Calculate body measurements
            torso_length = self._calculate_torso_length(landmarks)
            shoulder_width = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0]) if ("left_shoulder" in landmarks and "right_shoulder" in landmarks) else 0.0
            
            # Normalize measurements using calibration
            norm_shoulder, norm_torso = self._session_calibration.normalize(shoulder_width, torso_length)
            
            # Load view detection thresholds from config
            thresholds = load_thresholds_from_config("pushup", self._current_view, self.user_level) or {}
            side_range = thresholds.get("side_range", (0, 0.4))
            front_range = thresholds.get("front_range", (0.3, 1.0))
            
            reposition_warning = None
            if norm_shoulder >= front_range[0] and norm_shoulder <= front_range[1]:
                view, conf = "front", 1.0
            elif norm_shoulder >= side_range[0] and norm_shoulder <= side_range[1]:
                view, conf = "side", 1.0
            else:
                view, conf = "unknown", 0.0
                # Only add reposition warning when clearly outside both ranges
                if norm_shoulder < side_range[0]:
                    reposition_warning = FeedbackGenerator.camera_reposition_side()
                elif norm_shoulder > front_range[1]:
                    reposition_warning = FeedbackGenerator.camera_reposition_front()

            now = time.time()
            if self._sticky_view is None or view != self._sticky_view:
                # Initialize or change sticky view
                self._sticky_view = view
                self._sticky_view_since = now
                self._sticky_view_conf = conf
            elif view == self._sticky_view:
                # Update confidence if same view
                self._sticky_view_conf = max(self._sticky_view_conf, conf)

            if (now - self._sticky_view_since) < self._STICKY_VIEW_MIN_TIME or self._sticky_view_conf < self._STICKY_VIEW_MIN_CONF:
                return self._last_view or "unknown", self._last_view_confidence or 0.0, reposition_warning

            self._last_view = self._sticky_view
            self._last_view_confidence = self._sticky_view_conf
            return self._sticky_view, self._sticky_view_conf, reposition_warning

        except Exception as e:
            logger.error(f"View detection error: {e}")
            return "unknown", 0.0, None

    # "_update_view_detection" keeps a rolling buffer of recent detections and uses a voting mechanism to smooth out random fluctuations.
    # It also ensures that the view is not changed too frequently.
    def _update_view_detection(self, landmarks: Dict[str, List[float]]) -> None: # Updates the view detection.
        current_time = time.time()
        
        # Check if enough time has passed since last detection
        if current_time - self._last_view_detection_time < self._VIEW_DETECTION_INTERVAL:
            return None

        self._last_view_detection_time = current_time
        
        # Get new view detection
        new_view, confidence = self._detect_view(landmarks)
        
        # Add to history, keeps a rolling buffer of recent detections for voting.
        self._view_history.append((new_view, confidence))
        
        # Process history if enough samples (3 or more) to ensure a valid view.
        if len(self._view_history) >= 3:
            weighted_votes = {} # Dictionary to store the weighted votes for each view.
            total_weight = 0 # Total weight of all votes.
            
            # Calculate weighted votes for each view, weighted by confidence.
            # Using soft voting to stabilize detection.
            for view, conf in self._view_history:
                if view not in weighted_votes:
                    weighted_votes[view] = 0
                weighted_votes[view] += conf
                total_weight += conf
                
            # Determine best view based on weighted voting, if the total weight is greater than 0.
            if total_weight > 0:
                best_view = max(weighted_votes.items(), key=lambda x: x[1])
                if best_view[1] / total_weight > 0.6:  # 60% threshold, if the best view is greater than 60% of the total weight, set the current view to the best view.
                    self._current_view = best_view[0]
                    self._view_confidence = best_view[1] / total_weight
                else:
                    self._current_view = "unknown"
                    self._view_confidence = 0.0


        # Clear caches if view changed, helps prevent stale data from being used.
        if self._current_view != self._last_view:
            self._cached_phase_thresholds.clear()
            self._cached_form_rules.clear()

        self._last_view = self._current_view
        self._last_view_confidence = self._view_confidence
        # No reposition_warning available in this context; nothing to return

    def _get_current_analyzer(self) -> Optional[ViewSpecificAnalyzer]: # Returns the current analyzer based on the current view.
        return self._view_analyzers.get(self._current_view)

    def _calculate_torso_length(self, landmarks: Dict[str, List[float]]) -> float:

        # Calculate torso length from shoulder and hip landmarks.
        if all(key in landmarks for key in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
            left_length = abs(landmarks["left_shoulder"][1] - landmarks["left_hip"][1])
            right_length = abs(landmarks["right_shoulder"][1] - landmarks["right_hip"][1])
            return (left_length + right_length) / 2

        # If only one side is available, use that side's landmarks.
        elif "left_shoulder" in landmarks and "left_hip" in landmarks:
            return abs(landmarks["left_shoulder"][1] - landmarks["left_hip"][1])
        elif "right_shoulder" in landmarks and "right_hip" in landmarks:
            return abs(landmarks["right_shoulder"][1] - landmarks["right_hip"][1]) 

        return 0.0 # If no side is available, return 0.0.

    def _get_phase_thresholds(self) -> Dict[str, float]:
        cache_key = (self._current_view, self.user_level)
        if cache_key not in self._cached_phase_thresholds:
            # Get the current analyzer based on the current view.
            analyzer = self._get_current_analyzer()
            if analyzer:
                # Get the phase thresholds from the current analyzer.
                self._cached_phase_thresholds[cache_key] = analyzer.get_phase_thresholds(self.user_level)
            else:
                self._cached_phase_thresholds[cache_key] = {"descent_start": 155, "bottom_reached": 90, "ascent_start": 95, "top_reached": 165, "rest_threshold": 160}
        return self._cached_phase_thresholds[cache_key]

    def _get_required_angles(self) -> List[str]: # Returns the required angles for the current view.
        analyzer = self._get_current_analyzer()
        if analyzer:
            return analyzer.get_required_angles()
        return ["left_elbow", "right_elbow"]

    def _get_form_rules(self, exercise_variant: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]: # Returns the form rules for the current view.
        cache_key = (self._current_view, self.user_level, exercise_variant) # Composite key to prevent collisions accross variants and views.
        if cache_key not in self._cached_form_rules:
            analyzer = self._get_current_analyzer()
            if analyzer:
                self._cached_form_rules[cache_key] = analyzer.get_form_rules(self.user_level, exercise_variant)
            else:
                self._cached_form_rules[cache_key] = {}
        return self._cached_form_rules[cache_key]

    # Average elbow angle for phase traching, rep depth and form scoring.   
    def _get_average_elbow_angle(self, angles: Dict[str, float]) -> Optional[float]: # Returns the average elbow angle for the current view.

        elbow_angles = []
        if self._current_view == "side":
            left_valid = "left_elbow" in angles and not np.isnan(angles["left_elbow"])
            right_valid = "right_elbow" in angles and not np.isnan(angles["right_elbow"])
            if left_valid and right_valid:
                elbow_angles = [angles["left_elbow"], angles["right_elbow"]]
            elif left_valid:
                elbow_angles = [angles["left_elbow"]]
            elif right_valid:
                elbow_angles = [angles["right_elbow"]]
        else:
            if "left_elbow" in angles and not np.isnan(angles["left_elbow"]):
                elbow_angles.append(angles["left_elbow"])
            if "right_elbow" in angles and not np.isnan(angles["right_elbow"]):
                elbow_angles.append(angles["right_elbow"])
        return np.mean(elbow_angles) if elbow_angles else None

    def _update_movement_tracking(self, angles: Dict[str, float], current_time: float) -> None:
        for side in ["left", "right"]:
            angle_key = f"{side}_elbow"
            if angle_key in angles and not np.isnan(angles[angle_key]):
                self._elbow_angle_history[side].append((angles[angle_key], current_time)) # Appends latest elbow angle and time to history.
        velocities = [] # List to store the velocities.
        for side in ["left", "right"]:
            history = self._elbow_angle_history[side]
            if len(history) >= 2:
                angle_diff = history[-1][0] - history[-2][0]
                time_diff = history[-1][1] - history[-2][1]
                if time_diff > 0:
                    velocity = angle_diff / time_diff # per-side angular velocity using last two data points
                    velocities.append(velocity) # Append to list of velocities of left and right arms.
        if velocities: # If there are velocities, append the mean velocity to the history.
            mean_velocity = np.mean(velocities) # Average between left and right arms
            self._velocity_history.append(mean_velocity) # Appends the mean velocity to the history.
            
            # Second averaging: Smooth out the movement over time
            smoothed_velocity = np.mean(self._velocity_history) # Average over recent history
            threshold = self._VELOCITY_THRESHOLDS.get(self._current_view, 10) # Gets the threshold for the current view.
            if abs(smoothed_velocity) < threshold:
                self._movement_direction = "stationary"
            elif smoothed_velocity > 0:
                self._movement_direction = "ascending"
            else:
                self._movement_direction = "descending"
            self._movement_velocity = smoothed_velocity

    def _update_phase_state_machine(self, angles: Dict[str, float], current_time: float) -> None: # State machine for push-up phases:
        if not self._phase_start_time: # If the phase has not started, set the phase start time.
            self._phase_start_time = current_time
        phase_duration = current_time - self._phase_start_time # Calculate the duration of the current phase.
        avg_elbow = self._get_average_elbow_angle(angles) # Get the average elbow angle for phase traching, rep depth and form scoring.
        if avg_elbow is None: # If the average elbow angle is None, set the current phase to rest and return.
            self._current_phase = PushupPhase.REST
            self._phase_start_time = current_time
            return # Return to avoid further processing.
        
        thresholds = self._get_phase_thresholds()
        new_phase = None
        if self._current_phase == PushupPhase.REST:
            if avg_elbow < thresholds["descent_start"]:
                new_phase = PushupPhase.DESCENT
                self._rep_start_time = current_time
        elif self._current_phase == PushupPhase.DESCENT:
            if (avg_elbow <= thresholds["bottom_reached"] and self._movement_direction in ["stationary", "descending"]):
                new_phase = PushupPhase.BOTTOM
        elif self._current_phase == PushupPhase.BOTTOM:
            if (avg_elbow > thresholds["ascent_start"] and self._movement_direction == "ascending"):
                new_phase = PushupPhase.ASCENT
        elif self._current_phase == PushupPhase.ASCENT:
            if avg_elbow >= thresholds["top_reached"]:
                new_phase = PushupPhase.TOP
        elif self._current_phase == PushupPhase.TOP:
            if (phase_duration >= self._MIN_PHASE_DURATION and avg_elbow >= thresholds["rest_threshold"]):
                self._rep_count += 1
                self._current_rep_quality = self._calculate_rep_quality(angles, self._current_phase)
                self._completed_reps.append(self._current_rep_quality)
                new_phase = PushupPhase.REST
                self._rep_start_time = None
        if new_phase and self._is_valid_phase_transition(self._current_phase, new_phase, angles, phase_duration):
            self._current_phase = new_phase
            self._phase_start_time = current_time

    def _is_valid_phase_transition(self, current_phase, new_phase, angles, phase_duration): # Validates the phase transition.
        if phase_duration < self._MIN_PHASE_DURATION: # If the phase duration is less than the minimum phase duration, return False.
            return False
        if (current_phase == PushupPhase.DESCENT and new_phase == PushupPhase.REST): # If the current phase is descent then the next phase should be bottom, if the new phase is rest, return False.
            return False
        if (current_phase == PushupPhase.ASCENT and new_phase == PushupPhase.DESCENT): # If the current phase is ascent then the next phase should be top, if the new phase is descent, return False.
            return False
        return True # If the phase transition is valid, return True.

    def _calculate_rep_quality(self, angles: Dict[str, float], phase) -> Dict[str, Any]: # Calculates the quality of the current rep.
        if phase != PushupPhase.TOP: # If the current phase is not top, return the current rep quality.
            return self._current_rep_quality
        quality = {"form_violations": [], "range_quality": 0.0, "symmetry_score": None, "tempo_score": None, "depth_achieved": 0.0} # Initialize the quality dictionary.
        quality["form_violations"] = self.check_form_violations(angles) # Check for form violations.
        avg_elbow = self._get_average_elbow_angle(angles) # Get the average elbow angle for phase traching, rep depth and form scoring.
        thresholds = self._get_phase_thresholds() # Get the phase thresholds.
        if avg_elbow is not None: # If the average elbow angle is not None, calculate the range quality.
            min_depth = thresholds["bottom_reached"] # Get the minimum depth.
            ideal_depth = 70
            max_depth = 130
            if avg_elbow <= ideal_depth:
                quality["range_quality"] = 1.0
            elif avg_elbow <= min_depth:
                quality["range_quality"] = 0.8
            else:
                # Linear scaling for partial depth.
                quality["range_quality"] = max(0, 1 - (avg_elbow - min_depth) / (max_depth - min_depth))
            quality["depth_achieved"] = max_depth - avg_elbow # Calculate the depth achieved.
        symmetry = self._calculate_symmetry_score(angles) # Calculate the symmetry score.
        if symmetry is not None: # If the symmetry score is not None, calculate the symmetry score.
            quality["symmetry_score"] = max(0, 1 - symmetry) # Converts asymmetry measure to symmetry score.
        else:
            quality["symmetry_score"] = None
        # Adaptive tempo: use average of last 3 reps if available
        if self._rep_start_time: # If the rep start time is not None, calculate the tempo score.
            rep_duration = time.time() - self._rep_start_time # Calculate the duration of the current rep.
            valid_tempos = [rep.get("tempo_score") for rep in self._completed_reps[-3:] if rep.get("tempo_score") is not None] # Get the tempo scores of the last 3 reps.
            if valid_tempos:
                avg_tempo = np.mean(valid_tempos) # Calculate the average tempo.
                tempo_deviation = abs(rep_duration - avg_tempo) / avg_tempo  # Measures how much current rep differs from average.
                quality["tempo_score"] = max(0, 1 - tempo_deviation) # Convert deviation to score.
            else:
                quality["tempo_score"] = None  # Not enough data.
        return quality # Return the quality dictionary.

    def _calculate_symmetry_score(self, angles: Dict[str, float]) -> float: # Calculates the symmetry score.
        diffs = [] # List to store the differences between the angles.
        for pair in [("left_elbow", "right_elbow"), ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")]: # Pairs of angles to compare.
            a, b = pair # Unpack the pair.
            if a in angles and b in angles and not (np.isnan(angles[a]) or np.isnan(angles[b])): # If the angles are not None and not NaN.
                diffs.append(abs(angles[a] - angles[b]) / 180.0) # Append the difference to the list.
        if not diffs: # If there are no differences, return None.
            logger.warning("Symmetry score not available due to missing angles.")
            return None
        norm_asymmetry = sum(diffs) / len(diffs) # Calculate the average difference.
        self._asymmetry_history.append(norm_asymmetry) # Append the average difference to the history.
        return np.mean(list(self._asymmetry_history)) # Return the average difference.

    def get_required_landmarks(self) -> List[str]: # Returns the required landmarks for the current view.
        return ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_ankle", "right_ankle", "left_ear", "right_ear", "nose"]

    def get_required_angles(self) -> List[str]: # Returns the required angles for the current view.
        analyzer = self._get_current_analyzer()
        if analyzer:
            return analyzer.get_required_angles()
        return []

    def get_form_rules(self, exercise_variant: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return self._get_form_rules(exercise_variant)

    def check_form_violations(self, angles: Dict[str, float], exercise_variant: Optional[str] = None) -> List[str]:
        violations = []
        rules = self.get_form_rules(exercise_variant)
        # Convert rules dict to FormRule objects if needed
        form_rules = []
        for angle_name, thresholds in rules.items():
            min_val = thresholds.get("min", {}).get("threshold")
            max_val = thresholds.get("max", {}).get("threshold")
            message = thresholds.get("min", {}).get("message") or thresholds.get("max", {}).get("message")
            form_rules.append(FormRule(angle_name, min_val, max_val, message))
        for rule in form_rules:
            result = rule.check(angles)
            if result:
                violations.append(result)
        return violations

    def calculate_confidence(self, landmarks: Dict[str, List[float]]) -> float:
        required_landmarks = self.get_required_landmarks()
        visibilities = [landmarks[l][3] for l in required_landmarks if l in landmarks and len(landmarks[l]) > 3]
        if not visibilities:
            return 0.0
        return np.mean(visibilities)

    def validate_inputs(self, landmarks: Dict[str, List[float]], angles: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        missing_landmarks = set(self.get_required_landmarks()) - set(landmarks.keys())
        missing_angles = set(self.get_required_angles()) - set(angles.keys())
        if missing_landmarks:
            return False, f"Missing required landmarks: {', '.join(missing_landmarks)}"
        if missing_angles:
            return False, f"Missing required angles for {self._current_view} view: {', '.join(missing_angles)}"
        return True, None

    def analyze_frame(self, landmarks: Dict[str, List[float]], angles: Dict[str, float], exercise_variant: Optional[str] = None) -> ExerciseState:
        """
        Analyze a single frame of push-up performance. Provides partial feedback if only some data is available.
        Degraded mode: If only partial data, feedback is weighted by confidence.
        """
        # Only track time and memory if debugging is enabled
        start_time = time.time() if self._DEBUG_MODE else None
        try:
            current_time = time.time()
            
            # Detect and stabilize view.
            self._update_view_detection(landmarks)
            
            # Handle ambiguous view with min time/confidence
            if self._current_view not in ("side", "front"):
                if self._ambiguous_since is None:
                    self._ambiguous_since = current_time
                elif (current_time - self._ambiguous_since) < self._AMBIGUOUS_VIEW_MIN_TIME:
                    # Hold previous view for a bit longer
                    return self._last_state if hasattr(self, '_last_state') else super().analyze_frame(landmarks, angles, exercise_variant)
                
                # Degraded feedback for partial view
                available = [k for k in angles if not np.isnan(angles[k])]
                missing = [k for k in self.get_required_angles() if k not in angles or np.isnan(angles[k])]
                side_detected = "left" if any("left" in k for k in available) else ("right" if any("right" in k for k in available) else None)
                
                violations = []
                if side_detected:
                    violations.append(f"Only {side_detected} side detected, partial analysis.")
                violations.append(FeedbackGenerator.partial_feedback(available, missing))
                
                state = ExerciseState(
                    name=self.get_exercise_name(),
                    phase=self._current_phase.value,
                    rep_count=self._rep_count,
                    is_correct_form=False,
                    violations=violations,
                    angles=angles,
                    confidence=0.0,
                    analysis_reliable=False,
                    error_message=FeedbackGenerator.ambiguous_view(),
                    user_level=self.user_level
                )
                self._last_state = state
                return state
            else:
                self._ambiguous_since = None

            # Low confidence frame rejection
            confidence = self.calculate_confidence(landmarks)
            if confidence < self.level_config.min_confidence:
                state = ExerciseState(
                    name=self.get_exercise_name(),
                    phase=self._current_phase.value,
                    rep_count=self._rep_count,
                    is_correct_form=False,
                    violations=[FeedbackGenerator.low_confidence()],
                    angles=angles,
                    confidence=confidence,
                    analysis_reliable=False,
                    user_level=self.user_level
                )
                self._last_state = state
                return state

            # Validate camera positioning.
            analyzer = self._get_current_analyzer()
            if analyzer:
                camera_valid, camera_error = analyzer.validate_camera_position(landmarks)
                if not camera_valid:
                    state = ExerciseState(
                        name=self.get_exercise_name(),
                        phase=self._current_phase.value,
                        rep_count=self._rep_count,
                        is_correct_form=False,
                        violations=[FeedbackGenerator.camera_error(camera_error)],
                        angles=angles,
                        confidence=confidence,
                        analysis_reliable=False,
                        error_message=FeedbackGenerator.camera_error(camera_error),
                        user_level=self.user_level
                    )
                    self._last_state = state
                    return state

            # Validate landmarks and angles
            is_valid, error_message = self.validate_inputs(landmarks, angles)
            if not is_valid:
                available = [k for k in angles if not np.isnan(angles[k])]
                missing = [k for k in self.get_required_angles() if k not in angles or np.isnan(angles[k])]
                side_detected = "left" if any("left" in k for k in available) else ("right" if any("right" in k for k in available) else None)
                msg = error_message
                if side_detected:
                    msg += f" Only {side_detected} side detected, partial analysis."
                state = ExerciseState(
                    name=self.get_exercise_name(),
                    phase=self._current_phase.value,
                    rep_count=self._rep_count,
                    is_correct_form=False,
                    violations=[msg, FeedbackGenerator.partial_feedback(available, missing, confidence)],
                    angles=angles,
                    confidence=confidence,
                    analysis_reliable=False,
                    error_message=msg,
                    user_level=self.user_level
                )
                self._last_state = state
                return state

            # Track motion and detect rep phase.
            self._update_movement_tracking(angles, current_time)
            self._update_phase_state_machine(angles, current_time)

            # Analyze rep quality and violations.
            self._current_rep_quality = self._calculate_rep_quality(angles, self._current_phase)
            violations = self.check_form_violations(angles, exercise_variant)

            # Return success state.
            state = ExerciseState(
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
            self._last_state = state
            return state
    
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            if self._DEBUG_MODE:
                logger.exception("Detailed error in frame analysis:")
            return self._create_error_state(str(e))
            
        finally:
            # Only log timing info if debug mode is enabled
            if self._DEBUG_MODE and start_time is not None:
                elapsed = time.time() - start_time
                logger.debug(f"Frame analysis time: {elapsed:.3f}s")

    def get_exercise_name(self) -> str: # Returns the name of the exercise.
        return "pushup"

    def get_performance_metrics(self) -> Dict[str, Any]: # Returns the performance metrics.

        if not self._completed_reps:
            return {
                "total_reps": self._rep_count,
                "average_quality": 0.0,
                "best_rep_quality": 0.0,
                "consistency_score": 0.0,
                "common_violations": [],
                "symmetry_score": 0.0,
                "tempo_score": 0.0,
                "depth_achieved": 0.0,
                "current_view": self._current_view
            }

        # Calculate performance metrics.
        quality_scores = [rep.get("range_quality", 0) for rep in self._completed_reps]
        symmetry_scores = [rep.get("symmetry_score", 0) for rep in self._completed_reps]
        tempo_scores = [rep.get("tempo_score", 0) for rep in self._completed_reps]
        depth_scores = [rep.get("depth_achieved", 0) for rep in self._completed_reps]
        consistency_score = 1.0 - min(1.0, np.std(quality_scores) if quality_scores else 1.0) # Calculate the consistency score.
        
        all_violations = [] # List to store all violations.
        for rep in self._completed_reps:
            all_violations.extend(rep.get("form_violations", []))
        common_violations = Counter(all_violations).most_common(3)
        return {
            "total_reps": self._rep_count,
            "average_quality": np.mean(quality_scores) if quality_scores else 0.0,
            "best_rep_quality": max(quality_scores) if quality_scores else 0.0,
            "consistency_score": consistency_score,
            "common_violations": [violation for violation, _ in common_violations],
            "symmetry_score": np.mean(symmetry_scores) if symmetry_scores else 0.0,
            "tempo_score": np.mean(tempo_scores) if tempo_scores else 0.0,
            "depth_achieved": np.mean(depth_scores) if depth_scores else 0.0,
            "current_view": self._current_view
        }

# --- Performance profiling ---
def profile_performance(start_time):
    current, peak = tracemalloc.get_traced_memory()
    elapsed = time.time() - start_time
    logger.info(f"Frame time: {elapsed:.3f}s, Memory: {current/1024:.1f}KB (peak {peak/1024:.1f}KB)")
    tracemalloc.stop()
