import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, Counter
import logging
from .base_analyzer_robust import BaseExerciseAnalyzer, ExerciseState, UserLevel
from .pose_utils import calculate_torso_length, calculate_length, check_landmark_visibility, calculate_asymmetry_metrics
from .config_utils import load_squat_config

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
            if isinstance(thresholds.get("bottom_reached"), dict):
                level = user_level.name.lower() if hasattr(user_level, "name") else str(user_level).lower()
                return {k: (v[level] if isinstance(v, dict) and level in v else v)
                        for k, v in thresholds.items()}
            return thresholds
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
    _VIEW_HISTORY_LEN = 8
    _ASYMMETRY_HISTORY_LEN = 10

    def __init__(self, user_level: UserLevel = UserLevel.BEGINNER):
        super().__init__(user_level)
        self._rep_count = 0
        self._current_phase = SquatPhase.REST
        self._last_landmarks = None
        self._last_angles = None
        self._last_phase = None
        self._completed_reps = []
        self._asymmetry_history = deque(maxlen=self._ASYMMETRY_HISTORY_LEN)
        self._view_history = deque(maxlen=self._VIEW_HISTORY_LEN)
        self._last_view = None
        self._current_view = 'side'
        # View analyzers
        self._view_analyzers = {k: v() for k, v in SQUAT_VIEW_ANALYZER_REGISTRY.items()}
        for analyzer in self._view_analyzers.values():
            analyzer.parent_analyzer = self
        self._current_analyzer = self._view_analyzers['side']  # Default

    def get_exercise_name(self) -> str:
        return "squat"

    def get_required_landmarks(self) -> List[str]:
        return self._current_analyzer.get_required_landmarks()

    def get_required_angles(self) -> List[str]:
        return self._current_analyzer.get_required_angles()

    def get_form_rules(self, exercise_variant: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        return self._current_analyzer.get_form_rules(self.user_level, exercise_variant)

    def compute_view_votes(self, landmarks: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Compute weighted votes for each view ('front', 'side', 'unknown') based on ratio, symmetry, and visibility.
        Returns a dict: {view: score}
        """
        votes = {"front": 0.0, "side": 0.0, "unknown": 0.0}
        # Ratio-based heuristic (shoulder width / torso length)
        torso_length = calculate_torso_length(landmarks)
        shoulder_width = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0]) if ("left_shoulder" in landmarks and "right_shoulder" in landmarks) else 0.0
        perframe_ratio = (shoulder_width / torso_length) if (torso_length and torso_length > 0) else 0.0
        # Use config thresholds if available
        side_criteria = _SQUAT_CONFIG["views"]["side"].get("detection_criteria", {})
        front_criteria = _SQUAT_CONFIG["views"]["front"].get("detection_criteria", {})
        # Side view: ratio typically lower, both ankles visible
        if side_criteria.get("shoulder_visibility_threshold", 0.5) <= min(
            landmarks.get("left_shoulder", [0, 0, 0, 0])[3],
            landmarks.get("right_shoulder", [0, 0, 0, 0])[3]
        ):
            if perframe_ratio > 0 and perframe_ratio < 0.35:
                votes["side"] += 1.0
        # Front view: ratio higher, both shoulders visible
        if front_criteria.get("shoulder_visibility_threshold", 0.5) <= min(
            landmarks.get("left_shoulder", [0, 0, 0, 0])[3],
            landmarks.get("right_shoulder", [0, 0, 0, 0])[3]
        ):
            if perframe_ratio > 0.3:
                votes["front"] += 1.0
        # Visibility of ankles/knees
        if check_landmark_visibility(landmarks, ["left_ankle", "right_ankle"], 0.5):
            votes["side"] += 1.0
        if check_landmark_visibility(landmarks, ["left_knee", "right_knee"], 0.5):
            votes["front"] += 0.5
            votes["side"] += 0.5
        # If not enough info, mark as unknown
        if sum(votes.values()) == 0:
            votes["unknown"] = 1.0
        return votes

    def _detect_view(self, landmarks: Dict[str, List[float]]) -> str:
        # Use voting logic for robust view detection
        votes = self.compute_view_votes(landmarks)
        best_view, best_score = max(votes.items(), key=lambda x: x[1])
        if best_score == 0 or best_view == "unknown":
            return self._last_view or "side"
        return best_view

    def analyze_frame(self, landmarks: Dict[str, List[float]], angles: Dict[str, float], exercise_variant: Optional[str] = None) -> ExerciseState:
        # --- View detection and smoothing ---
        view = self._detect_view(landmarks)
        self._view_history.append(view)
        # Use most common view in history for stability
        view_counts = Counter(self._view_history)
        self._current_view = max(view_counts, key=view_counts.get)
        self._current_analyzer = self._view_analyzers[self._current_view]
        self._last_view = self._current_view
        self._last_landmarks = landmarks
        self._last_angles = angles

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

        # --- Phase detection (robust state machine) ---
        left_knee = angles.get("left_knee", 180)
        right_knee = angles.get("right_knee", 180)
        avg_knee = (left_knee + right_knee) / 2
        thresholds = self._current_analyzer.get_phase_thresholds(self.user_level)
        prev_phase = self._current_phase
        if avg_knee <= thresholds["bottom_reached"]:
            self._current_phase = SquatPhase.BOTTOM
        elif avg_knee >= thresholds["top_reached"]:
            if self._current_phase == SquatPhase.BOTTOM:
                self._rep_count += 1
            self._current_phase = SquatPhase.TOP
        elif avg_knee < thresholds["top_reached"]:
            if self._current_phase in [SquatPhase.TOP, SquatPhase.REST]:
                self._current_phase = SquatPhase.DESCENT
            elif self._current_phase == SquatPhase.BOTTOM:
                self._current_phase = SquatPhase.ASCENT

        # --- Form rule checks ---
        violations = []
        form_rules = self.get_form_rules(exercise_variant)
        for angle_name, rule in form_rules.items():
            val = angles.get(angle_name)
            if val is not None:
                if "min" in rule and val < rule["min"]:
                    violations.append(f"{angle_name} too small: {val:.1f} < {rule['min']}")
                if "max" in rule and val > rule["max"]:
                    violations.append(f"{angle_name} too large: {val:.1f} > {rule['max']}")

        # --- Symmetry checks (front view) ---
        if self._current_view == 'front':
            if 'shoulder_symmetry' in angles and angles['shoulder_symmetry'] > 0.05:
                violations.append(f"Shoulder symmetry off: {angles['shoulder_symmetry']:.3f}")
            if 'hip_symmetry' in angles and angles['hip_symmetry'] > 0.05:
                violations.append(f"Hip symmetry off: {angles['hip_symmetry']:.3f}")
            # Optionally: knee distance checks for valgus/varus

        # --- Visibility check ---
        visible = check_landmark_visibility(landmarks, self.get_required_landmarks(), self._current_analyzer.min_landmark_visibility)
        analysis_reliable = visible
        error_message = None if visible else FeedbackGenerator.camera_error("Some body parts are not visible. Please adjust your position or camera.")

        return ExerciseState(
            name=self.get_exercise_name(),
            phase=self._current_phase.value,
            rep_count=self._rep_count,
            is_correct_form=len(violations) == 0,
            violations=violations,
            angles=angles,
            confidence=1.0 if analysis_reliable else 0.0,
            analysis_reliable=analysis_reliable,
            error_message=error_message,
            user_level=self.user_level
        )

class SquatAnalyzer(SquatAnalyzerBase):
    pass