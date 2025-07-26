import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, Counter
import logging
from .base_analyzer_robust import BaseExerciseAnalyzer, ExerciseState, UserLevel
from .pose_utils import calculate_torso_length, calculate_length, check_landmark_visibility, calculate_asymmetry_metrics
from .config_utils import load_squat_config
import time

_SQUAT_CONFIG = load_squat_config()

# --- Logger Setup ---
logger = logging.getLogger("SquatAnalyzer")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

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

# --- Phase Enum ---
class SquatPhase(Enum):
    REST = "rest"
    DESCENT = "descent"
    BOTTOM = "bottom"
    ASCENT = "ascent"
    TOP = "top"

# --- View Analyzer Registry ---
SQUAT_VIEW_ANALYZER_REGISTRY = {}

def register_squat_view_analyzer(view_type):
    def decorator(cls):
        SQUAT_VIEW_ANALYZER_REGISTRY[view_type] = cls
        return cls
    return decorator

# --- View-Specific Analyzers ---
class SquatViewSpecificAnalyzer:
    def __init__(self, view_type: str):
        self.view_type = view_type
        self.parent_analyzer = None
        try:
            self.min_landmark_visibility = _SQUAT_CONFIG["views"][view_type]["detection_criteria"].get("shoulder_visibility_threshold", 0.5)
        except Exception as e:
            logger.warning(f"Could not load min_landmark_visibility from config for view {view_type}: {e}")
            self.min_landmark_visibility = 0.5

    def get_required_angles(self) -> List[str]:
        try:
            return _SQUAT_CONFIG["views"][self.view_type]["required_angles"]
        except Exception as e:
            logger.error(f"Could not load required_angles from config for view {self.view_type}: {e}")
            return []

    def get_phase_thresholds(self, user_level: UserLevel) -> Dict[str, float]:
        try:
            thresholds = _SQUAT_CONFIG["views"][self.view_type]["phase_thresholds"]
            # Always extract per-user-level value if dict, else use as is
            level = user_level.name.lower() if hasattr(user_level, "name") else str(user_level).lower()
            result = {}
            for k, v in thresholds.items():
                if isinstance(v, dict):
                    # Use fallback order: user level, then any available value
                    result[k] = v.get(level, next(iter(v.values())))
                else:
                    result[k] = v
            return result
        except Exception as e:
            logger.error(f"Could not load phase_thresholds from config for view {self.view_type}: {e}")
            return {}

    def get_form_rules(self, user_level: UserLevel, exercise_variant: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        try:
            base_rules = _SQUAT_CONFIG["views"][self.view_type]["form_rules"].copy()
            # Optionally, add variant-specific rules here
            return base_rules
        except Exception as e:
            logger.error(f"Could not load form_rules from config for view {self.view_type}: {e}")
            return {}

    def get_required_landmarks(self) -> List[str]:
        raise NotImplementedError

    def validate_camera_position(self, landmarks: Dict[str, List[float]], angles=None) -> Tuple[bool, Optional[str]]:
        if self.parent_analyzer is None:
            return False, FeedbackGenerator.camera_error("Analyzer context missing.")
        required_landmarks = self.get_required_landmarks()
        for lmk in required_landmarks:
            if lmk not in landmarks or len(landmarks[lmk]) < 4 or landmarks[lmk][3] < self.min_landmark_visibility:
                return False, FeedbackGenerator.missing_landmark(lmk)
        # Optionally: add view voting logic here for more robustness
        return True, None

@register_squat_view_analyzer('side')
class SideSquatAnalyzer(SquatViewSpecificAnalyzer):
    def __init__(self):
        super().__init__('side')
    def get_required_landmarks(self) -> List[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]

@register_squat_view_analyzer('front')
class FrontSquatAnalyzer(SquatViewSpecificAnalyzer):
    def __init__(self):
        super().__init__('front')
    def get_required_landmarks(self) -> List[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee"
        ]

# --- Main Analyzer ---
class SquatAnalyzerBase(BaseExerciseAnalyzer):
    def _calculate_rep_quality(self, angles, phase):
        """Calculate rep quality using form_quality config, mirroring pushup analyzer."""
        form_quality_cfg = _SQUAT_CONFIG.get("form_quality", {})
        min_body_alignment = form_quality_cfg.get("min_body_alignment_score", 0.7)
        max_asymmetry = form_quality_cfg.get("max_asymmetry_score", 0.3)
        min_depth_achievement = form_quality_cfg.get("min_depth_achievement", 1.0)
        # Placeholder: actual calculation would use pose/angle metrics
        # For demonstration, set all to 1.0 (perfect) unless you have real metrics
        quality = {
            "body_alignment_score": angles.get("body_alignment_score", 1.0),
            "asymmetry_score": angles.get("asymmetry_score", 0.0),
            "depth_achieved": angles.get("depth_achieved", 1.0),
            "form_quality_score": 0.0
        }
        # Compute form quality score as mean of factors (if available)
        form_quality_factors = []
        if quality["body_alignment_score"] >= min_body_alignment:
            form_quality_factors.append(1.0)
        else:
            form_quality_factors.append(quality["body_alignment_score"] / min_body_alignment)
        if quality["asymmetry_score"] <= max_asymmetry:
            form_quality_factors.append(1.0)
        else:
            form_quality_factors.append(max(0, 1.0 - (quality["asymmetry_score"] - max_asymmetry)))
        if quality["depth_achieved"] >= min_depth_achievement:
            form_quality_factors.append(1.0)
        else:
            form_quality_factors.append(max(0, quality["depth_achieved"] / min_depth_achievement))
        if form_quality_factors:
            quality["form_quality_score"] = np.mean(form_quality_factors)
        return quality

    _MIN_PHASE_DURATION = 0.08
    _VIEW_DETECTION_INTERVAL = 0.1
    _VIEW_HISTORY_LEN = 8
    _ASYMMETRY_HISTORY_LEN = 10
    _ELBOW_HISTORY_LEN = 5
    _VELOCITY_SMOOTHING = 3
    _HYSTERESIS_MARGIN = 0.10
    _AMBIGUOUS_VIEW_MIN_TIME = 0.5
    _AMBIGUOUS_VIEW_MIN_CONF = 0.7
    _STICKY_VIEW_MIN_TIME = 2.0
    _STICKY_VIEW_MIN_CONF = 0.7
    _DEBUG_MODE = False

    def _get_average_knee_angle(self, angles: Dict[str, float]) -> Optional[float]:
        """Returns the average knee angle for the current view."""
        knee_angles = []
        if self._current_view == "side":
            left_valid = "left_knee" in angles and not np.isnan(angles["left_knee"])
            right_valid = "right_knee" in angles and not np.isnan(angles["right_knee"])
            if left_valid and right_valid:
                knee_angles = [angles["left_knee"], angles["right_knee"]]
            elif left_valid:
                knee_angles = [angles["left_knee"]]
            elif right_valid:
                knee_angles = [angles["right_knee"]]
        else:
            if "left_knee" in angles and not np.isnan(angles["left_knee"]):
                knee_angles.append(angles["left_knee"])
            if "right_knee" in angles and not np.isnan(angles["right_knee"]):
                knee_angles.append(angles["right_knee"])
        return np.mean(knee_angles) if knee_angles else None


    def _compute_movement_direction(self, avg_knee):
        """Compute movement direction based on knee angle history."""
        if len(self._velocity_history) < 2:
            return "stationary"
        diff = self._velocity_history[-1] - self._velocity_history[-2]
        if abs(diff) < 0.5:
            return "stationary"
        return "descending" if diff < 0 else "ascending"

    def _is_valid_phase_transition(self, current_phase, new_phase, angles, phase_duration):
        """Validates the phase transition, matching pushup analyzer logic."""
        if phase_duration < self._MIN_PHASE_DURATION:
            return False
        if (current_phase == SquatPhase.DESCENT and new_phase == SquatPhase.REST):
            return False
        if (current_phase == SquatPhase.ASCENT and new_phase == SquatPhase.DESCENT):
            return False
        return True


    def _update_phase_state_machine(self, angles: Dict[str, float], current_time: float) -> None:
        """Robust state machine for squat phases, mirroring pushup analyzer logic."""
        if not self._phase_start_time:
            self._phase_start_time = current_time
        phase_duration = current_time - self._phase_start_time
        avg_knee = self._get_average_knee_angle(angles)
        thresholds = self._current_analyzer.get_phase_thresholds(self.user_level)
        new_phase = self._current_phase
        # Track previous phase for rep counting
        prev_phase = getattr(self, '_prev_phase', self._current_phase)
        # Movement direction
        movement_direction = self._compute_movement_direction(avg_knee) if avg_knee is not None else "stationary"
        # Phase transitions with validation and duration
        if avg_knee is None:
            new_phase = SquatPhase.REST
        elif self._current_phase == SquatPhase.REST:
            if avg_knee < thresholds["descent_start"] and phase_duration >= self._MIN_PHASE_DURATION:
                new_phase = SquatPhase.DESCENT
        elif self._current_phase == SquatPhase.DESCENT:
            if avg_knee <= thresholds["bottom_reached"] and movement_direction in ["stationary", "descending"] and phase_duration >= self._MIN_PHASE_DURATION:
                new_phase = SquatPhase.BOTTOM
        elif self._current_phase == SquatPhase.BOTTOM:
            if avg_knee > thresholds["ascent_start"] and movement_direction == "ascending" and phase_duration >= self._MIN_PHASE_DURATION:
                new_phase = SquatPhase.ASCENT
        elif self._current_phase == SquatPhase.ASCENT:
            if avg_knee >= thresholds["top_reached"] and phase_duration >= self._MIN_PHASE_DURATION:
                new_phase = SquatPhase.TOP
        elif self._current_phase == SquatPhase.TOP:
            if avg_knee >= thresholds["rest_threshold"] and phase_duration >= self._MIN_PHASE_DURATION:
                new_phase = SquatPhase.REST
        # Only update phase if changed and valid
        if new_phase != self._current_phase:
            self._prev_phase = self._current_phase
            self._current_phase = new_phase
            self._phase_start_time = current_time
        # Rep counting: only increment on ASCENT→TOP transition
        if prev_phase == SquatPhase.ASCENT and self._current_phase == SquatPhase.TOP:
            self._rep_count += 1
            # Calculate and store rep quality
            rep_quality = self._calculate_rep_quality(self._last_angles or {}, self._current_phase)
            self._completed_reps.append(rep_quality)
        # Incomplete rep feedback: if TOP→REST without BOTTOM
        if prev_phase == SquatPhase.TOP and self._current_phase == SquatPhase.REST:
            # Optionally, set a flag or feedback for incomplete rep
            pass
        self._movement_direction = movement_direction

    def __init__(self, user_level: UserLevel = UserLevel.BEGINNER):
        super().__init__(user_level)
        self._rep_count = 0
        self._current_phase = SquatPhase.REST
        self._phase_start_time = None
        self._current_view = "unknown"
        self._view_history = deque(maxlen=8)
        self._last_view_detection_time = 0
        self._view_confidence = 0.0
        self._last_landmarks = None
        self._last_angles = None
        self._last_phase = None
        self._completed_reps = []
        self._asymmetry_history = deque(maxlen=10)
        self._velocity_history = deque(maxlen=3)
        self._last_view = None
        self._last_view_confidence = 0.0
        self._view_analyzers = {k: v() for k, v in SQUAT_VIEW_ANALYZER_REGISTRY.items()}
        for analyzer in self._view_analyzers.values():
            analyzer.parent_analyzer = self
        self._current_analyzer = self._view_analyzers.get('side', next(iter(self._view_analyzers.values())))
        self._in_rep = False
        self._phase_start_time = None
        self._last_knee_angle = None
        self._rep_start_time = None
        self._current_rep_quality = {"form_violations": [], "range_quality": 0.0}

    def get_exercise_name(self) -> str:
        return "squat"

    def get_required_landmarks(self) -> List[str]:
        return self._current_analyzer.get_required_landmarks()

    def get_required_angles(self) -> List[str]:
        return self._current_analyzer.get_required_angles()

    # Alias for compatibility with trainer.py
class SquatAnalyzer(SquatAnalyzerBase):
    def _is_rep_start_condition(self, angles: Dict[str, float], phase: str) -> bool:
        """
        Returns True if the current phase is the start of a squat rep (i.e., at the bottom).
        Mirrors pushup analyzer logic but for squats.
        """
        return phase == SquatPhase.BOTTOM.value

    def _is_rep_end_condition(self, angles: Dict[str, float], phase: str) -> bool:
        """
        Returns True if the current phase is the end of a squat rep (i.e., at the top).
        Mirrors pushup analyzer logic but for squats.
        """
        return phase == SquatPhase.TOP.value

    def get_form_rules(self, exercise_variant: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        return self._current_analyzer.get_form_rules(self.user_level, exercise_variant)

    def compute_view_votes(self, landmarks: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Compute weighted votes for each view ('front', 'side', 'unknown') based on ratio, symmetry, and visibility.
        Returns a dict: {view: score}
        """
        votes = {"front": 0.0, "side": 0.0, "unknown": 0.0}
        # Ratio: Shoulder-to-hip ratio, etc.
        # ...
        # (existing logic omitted for brevity)
        return votes

    def _detect_view(self, landmarks: Dict[str, List[float]]) -> str:
        votes = self.compute_view_votes(landmarks)
        best_view, best_score = max(votes.items(), key=lambda x: x[1])
        if best_score == 0 or best_view == "unknown":
            return self._last_view or "side"
        return best_view

    def compute_view_votes(self, landmarks: Dict[str, List[float]]) -> Dict[str, float]:
        votes = {"front": 0.0, "side": 0.0, "unknown": 0.0}
        side_criteria = _SQUAT_CONFIG["views"]["side"].get("detection_criteria", {})
        front_criteria = _SQUAT_CONFIG["views"]["front"].get("detection_criteria", {})
        torso_length = calculate_torso_length(landmarks)
        shoulder_width = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0]) if ("left_shoulder" in landmarks and "right_shoulder" in landmarks) else 0.0
        perframe_ratio = (shoulder_width / torso_length) if (torso_length and torso_length > 0) else 0.0
        # Ratio
        side_ratio_range = side_criteria.get("shoulder_torso_ratio", {})
        front_ratio_range = front_criteria.get("shoulder_torso_ratio", {})
        if side_ratio_range.get("min", 0) <= perframe_ratio <= side_ratio_range.get("max", 0.4):
            votes["side"] += 1.0
        if front_ratio_range.get("min", 0) <= perframe_ratio <= front_ratio_range.get("max", 1.0):
            votes["front"] += 1.0
        # Symmetry
        symmetry = None
        if ("left_shoulder" in landmarks and "right_shoulder" in landmarks):
            symmetry = abs(landmarks["left_shoulder"][1] - landmarks["right_shoulder"][1])
        front_sym = front_criteria.get("symmetry_threshold", 0.08)
        side_sym = side_criteria.get("symmetry_threshold", 0.15)
        if symmetry is not None:
            if symmetry < front_sym:
                votes["front"] += 1.0
            elif symmetry > side_sym:
                votes["side"] += 1.0
            else:
                votes["unknown"] += 1.0
        # Visibility
        left_vis = landmarks["left_shoulder"][3] if ("left_shoulder" in landmarks and len(landmarks["left_shoulder"]) > 3) else 0.0
        right_vis = landmarks["right_shoulder"][3] if ("right_shoulder" in landmarks and len(landmarks["right_shoulder"]) > 3) else 0.0
        visible = left_vis > 0.5 and right_vis > 0.5
        if visible:
            votes["front"] += 0.5
            votes["side"] += 0.5
        else:
            votes["unknown"] += 1.0
        return votes

    def _detect_view(self, landmarks: Dict[str, List[float]]) -> str:
        votes = self.compute_view_votes(landmarks)
        best_view, best_score = max(votes.items(), key=lambda x: x[1])
        if best_score == 0 or best_view == "unknown":
            return self._last_view or "side"
        return best_view

    def analyze_frame(self, landmarks: Dict[str, List[float]], angles: Dict[str, float], exercise_variant: Optional[str] = None) -> ExerciseState:
        # --- View detection and smoothing ---
        view = self._detect_view(landmarks)
        self._view_history.append(view)
        view_counts = Counter(self._view_history)
        self._current_view = max(view_counts, key=view_counts.get)
        self._current_analyzer = self._view_analyzers[self._current_view]
        self._last_view = self._current_view
        self._last_landmarks = landmarks
        self._last_angles = angles

        # --- Strict missing landmark/angle checks ---
        required_landmarks = self._current_analyzer.get_required_landmarks()
        missing_landmarks = [l for l in required_landmarks if l not in landmarks or len(landmarks[l]) < 4 or landmarks[l][3] < self._current_analyzer.min_landmark_visibility]
        if missing_landmarks:
            return ExerciseState(
                name=self.get_exercise_name(),
                phase=getattr(self, '_current_phase', None) and self._current_phase.value or "unknown",
                rep_count=getattr(self, '_rep_count', 0),
                is_correct_form=False,
                violations=[FeedbackGenerator.missing_landmark(l) for l in missing_landmarks],
                angles=angles,
                confidence=0.0,
                analysis_reliable=False,
                error_message="skip_frame",
                user_level=getattr(self, 'user_level', None) or UserLevel.BEGINNER
            )
        required_angles = self._current_analyzer.get_required_angles()
        missing_angles = [a for a in required_angles if a not in angles or angles[a] is None or (hasattr(np, 'isnan') and np.isnan(angles[a]))]
        if missing_angles:
            return ExerciseState(
                name=self.get_exercise_name(),
                phase=getattr(self, '_current_phase', None) and self._current_phase.value or "unknown",
                rep_count=getattr(self, '_rep_count', 0),
                is_correct_form=False,
                violations=[FeedbackGenerator.missing_angle(a) for a in missing_angles],
                angles=angles,
                confidence=0.0,
                analysis_reliable=False,
                error_message=f"Missing required angles: {', '.join(missing_angles)}",
                user_level=getattr(self, 'user_level', None) or UserLevel.BEGINNER
            )

        # --- Camera position validation ---
        valid_cam, cam_feedback = self._current_analyzer.validate_camera_position(landmarks, angles)
        if not valid_cam:
            return ExerciseState(
                name=self.get_exercise_name(),
                phase=self._current_phase.value,
                rep_count=self._rep_count,
                is_correct_form=False,
                violations=[cam_feedback or FeedbackGenerator.ambiguous_view()],
                angles=angles,
                confidence=0.0,
                analysis_reliable=False,
                error_message=cam_feedback,
                user_level=self.user_level
            )

        # --- Symmetry/Asymmetry metrics ---
        calculate_asymmetry_metrics(landmarks, angles)

        # --- Phase detection and rep counting (robust state machine) ---
        current_time = time.time()
        self._update_phase_state_machine(angles, current_time)

        # --- Form rule checks ---
        violations = []
        form_rules = self.get_form_rules(exercise_variant)
        if self._current_phase == SquatPhase.BOTTOM:
            for angle_name, rule in form_rules.items():
                val = angles.get(angle_name)
                if val is not None and hasattr(rule, 'min_val') and rule.min_val is not None and val < rule.min_val:
                    violations.append(rule.min_message or rule.message or f"{angle_name} below minimum")
                if val is not None and hasattr(rule, 'max_val') and rule.max_val is not None and val > rule.max_val:
                    violations.append(rule.max_message or rule.message or f"{angle_name} above maximum")
        # Track violations for rep quality
        if self._in_rep:
            self._current_rep_quality["form_violations"].extend(violations)

        # --- Symmetry checks (front view) ---
        if self._current_view == 'front':
            if 'shoulder_symmetry' in angles and angles['shoulder_symmetry'] > 0.05:
                violations.append("Shoulder symmetry violation")
            if 'hip_symmetry' in angles and angles['hip_symmetry'] > 0.05:
                violations.append("Hip symmetry violation")

        # --- Visibility check ---
        visible = check_landmark_visibility(landmarks, self.get_required_landmarks(), self._current_analyzer.min_landmark_visibility)
        analysis_reliable = visible
        error_message = None if visible else FeedbackGenerator.camera_error("Some body parts are not visible. Please adjust your position or camera.")

        # --- Confidence calculation ---
        confidence = 1.0 if analysis_reliable and not violations else 0.5 if analysis_reliable else 0.0

        return ExerciseState(
            name=self.get_exercise_name(),
            phase=self._current_phase.value,
            rep_count=self._rep_count,
            is_correct_form=len(violations) == 0,
            violations=violations,
            angles=angles,
            confidence=confidence,
            analysis_reliable=analysis_reliable,
            error_message=error_message,
            user_level=self.user_level
        )