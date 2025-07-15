from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
from enum import Enum
from collections import deque, Counter
from .base_analyzer_robust import BaseExerciseAnalyzer, ExerciseState, UserLevel
from .pose_utils import calculate_torso_length, calculate_length, check_landmark_visibility, calculate_pushup_specific_angles
from .pose_utils import EMAFilter, should_skip_frame
from .config_utils import load_pushup_config
import logging
import tracemalloc
import traceback

_PUSHUP_CONFIG = load_pushup_config()


class PushupPhase(Enum):  # Define an enum class for the pushup exercise phases.
    """Push-up exercise phases.""" # Define set of named constants for each phase of the pushup exercise.
    REST = "rest"           # Starting position (arms extended)
    DESCENT = "descent"     # Lowering phase
    BOTTOM = "bottom"       # Bottom position (chest near floor)
    ASCENT = "ascent"       # Rising phase
    TOP = "top"             # Top position (arms extended)


# --- View Analyzer Registry ---
VIEW_ANALYZER_REGISTRY = {} # Dictionary to map view types to their corresponding analyzer classes.

def register_view_analyzer(view_type): # Decorator to automatically add new view specific analyzers classes to the registry when they are defined.
    def decorator(cls):
        VIEW_ANALYZER_REGISTRY[view_type] = cls
        return cls
    return decorator


# --- View-Specific Analyzers ---

class ViewSpecificAnalyzer:
    def __init__(self, view_type: str):
        self.view_type = view_type
        self.parent_analyzer = None  # Always set, will be overwritten by PushupAnalyzerBase
        # Set min_landmark_visibility from config if available
        try:
            self.min_landmark_visibility = _PUSHUP_CONFIG["views"][view_type]["detection_criteria"].get("shoulder_visibility_threshold", 0.5)
        except Exception as e:
            logger.warning(f"Could not load min_landmark_visibility from config for view {view_type}: {e}")
            self.min_landmark_visibility = 0.5

    def get_required_angles(self) -> List[str]:
        try:
            return _PUSHUP_CONFIG["views"][self.view_type]["required_angles"]
        except Exception as e:
            logger.error(f"Could not load required_angles from config for view {self.view_type}: {e}")
            return []

    def get_phase_thresholds(self, user_level: UserLevel) -> Dict[str, float]:
        try:
            thresholds = _PUSHUP_CONFIG["views"][self.view_type]["phase_thresholds"]
            # Handle per-level dicts
            if isinstance(thresholds.get("bottom_reached"), dict):
                level = user_level.name.lower() if hasattr(user_level, "name") else str(user_level).lower()
                return {k: (v[level] if isinstance(v, dict) and level in v else v)
                        for k, v in thresholds.items()}
            return thresholds
        except Exception as e:
            logger.error(f"Could not load phase_thresholds from config for view {self.view_type}: {e}")
            return {}

    def validate_camera_position(self, landmarks: Dict[str, List[float]], angles=None) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError

    def get_form_rules(self, user_level: UserLevel, exercise_variant: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        try:
            base_rules = _PUSHUP_CONFIG["views"][self.view_type]["form_rules"].copy()
            
            # Expand elbow_range to left/right_elbow for convenience
            if "elbow_range" in base_rules:
                er = base_rules.pop("elbow_range")
                base_rules["left_elbow"] = er.copy()
                base_rules["right_elbow"] = er.copy()
               
            # Merge variant rules if present
            if exercise_variant and exercise_variant in _PUSHUP_CONFIG.get("variants", {}):
                variant_rules = _PUSHUP_CONFIG["variants"][exercise_variant].get("form_rules", {})
                for angle, limits in variant_rules.items():
                    if angle in base_rules:
                        base_rules[angle].update(limits)
                    else:
                        base_rules[angle] = limits
            return base_rules
        except Exception as e:
            logger.error(f"Could not load form_rules from config for view {self.view_type}: {e}")
            return {}
# Inherit get_required_angles, get_phase_thresholds, and get_form_rules from ViewSpecificAnalyzer

@register_view_analyzer('side')
class SideViewAnalyzer(ViewSpecificAnalyzer):
    def __init__(self):
        super().__init__('side')

    def get_required_landmarks(self) -> List[str]:
        # Landmarks required for side view push-up analysis (for all required angles)
        return [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_ankle", "right_ankle",
            "left_knee", "right_knee",
            "nose"
        ]

    def validate_camera_position(self, landmarks, angles=None):
        if self.parent_analyzer is None:
            return False, "Analyzer context missing."
        # 1. Check required landmarks for this view
        required_landmarks = self.get_required_landmarks()
        for lmk in required_landmarks:
            if lmk not in landmarks or len(landmarks[lmk]) < 4 or landmarks[lmk][3] < self.min_landmark_visibility:
                logger.warning(FeedbackGenerator.missing_landmark(lmk))
                return False, FeedbackGenerator.missing_landmark(lmk)
        # 2. Check required angles for this view
        if angles is not None:
            required_angles = self.get_required_angles()
            for ang in required_angles:
                if ang not in angles or angles[ang] is None or (hasattr(np, 'isnan') and np.isnan(angles[ang])):
                    logger.warning(FeedbackGenerator.missing_angle(ang))
                    return False, FeedbackGenerator.missing_angle(ang)
        # 3. Use centralized view voting logic
        parent = getattr(self, 'parent_analyzer', None)
        if parent is None:
            return False, "Analyzer context missing."
        votes = parent.compute_view_votes(landmarks)
        total = sum(votes.values())
        if total == 0:
            return False, "View could not be determined."
        best_view, best_score = max(votes.items(), key=lambda x: x[1])
        confidence = best_score / total if total > 0 else 0.0
        # Store for debug output
        shoulder_width = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0]) if ("left_shoulder" in landmarks and "right_shoulder" in landmarks) else 0.0
        torso_length = calculate_torso_length(landmarks)
        perframe_ratio = (shoulder_width / torso_length) if (torso_length and torso_length > 0) else 0.0
        if hasattr(parent, '_last_body_metrics'):
            parent._last_body_metrics = {
                "shoulder_width": shoulder_width,
                "torso_length": torso_length,
                "shoulder_torso_ratio": perframe_ratio
            }
        logger.debug(f"[CAM POS] View votes: {votes}, Best: {best_view}, Confidence: {confidence:.2f}")
        # Accept if best view matches this analyzer and confidence is high
        if best_view == self.view_type and confidence >= 0.4:
            return True, None
        feedback = f"Camera not positioned for {self.view_type} view. Current: {best_view} (conf {confidence:.2f}). Votes: {votes}"
        return False, feedback


@register_view_analyzer('front')
class FrontViewAnalyzer(ViewSpecificAnalyzer):
    def __init__(self):
        super().__init__('front')

    def get_required_landmarks(self) -> List[str]:
        # Landmarks required for front view push-up analysis (for all required angles)
        return [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip"
        ]

    def validate_camera_position(self, landmarks: Dict[str, List[float]], angles=None) -> Tuple[bool, Optional[str]]:
        if self.parent_analyzer is None:
            return False, "Analyzer context missing."
        # 1. Check required landmarks for this view
        required_landmarks = self.get_required_landmarks()
        for lmk in required_landmarks:
            if lmk not in landmarks or len(landmarks[lmk]) < 4 or landmarks[lmk][3] < self.min_landmark_visibility:
                logger.warning(FeedbackGenerator.missing_landmark(lmk))
                return False, FeedbackGenerator.missing_landmark(lmk)
        # 2. Check required angles for this view
        if angles is not None:
            required_angles = self.get_required_angles()
            for ang in required_angles:
                if ang not in angles or angles[ang] is None or (hasattr(np, 'isnan') and np.isnan(angles[ang])):
                    logger.warning(FeedbackGenerator.missing_angle(ang))
                    return False, FeedbackGenerator.missing_angle(ang)
        # 3. Use centralized view voting logic
        parent = getattr(self, 'parent_analyzer', None)
        if parent is None:
            return False, "Analyzer context missing."
        votes = parent.compute_view_votes(landmarks)
        total = sum(votes.values())
        if total == 0:
            return False, "View could not be determined."
        best_view, best_score = max(votes.items(), key=lambda x: x[1])
        confidence = best_score / total if total > 0 else 0.0
        # Store for debug output
        shoulder_width = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0]) if ("left_shoulder" in landmarks and "right_shoulder" in landmarks) else 0.0
        torso_length = calculate_torso_length(landmarks)
        perframe_ratio = (shoulder_width / torso_length) if (torso_length and torso_length > 0) else 0.0
        if hasattr(parent, '_last_body_metrics'):
            parent._last_body_metrics = {
                "shoulder_width": shoulder_width,
                "torso_length": torso_length,
                "shoulder_torso_ratio": perframe_ratio
            }
        logger.debug(f"[CAM POS] View votes: {votes}, Best: {best_view}, Confidence: {confidence:.2f}")
        # Accept if best view matches this analyzer and confidence is high
        if best_view == self.view_type and confidence >= 0.4:
            return True, None
        feedback = f"Camera not positioned for {self.view_type} view. Current: {best_view} (conf {confidence:.2f}). Votes: {votes}"
        return False, feedback


# --- Logger Setup ---
logger = logging.getLogger("PushupAnalyzer")
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

# --- Generic FormRule and ExercisePhase ---
class FormRule:
    def __init__(self, angle_name, min_val=None, max_val=None, message=None, min_message=None, max_message=None):
        self.angle_name = angle_name
        self.min_val = min_val
        self.max_val = max_val
        self.message = message
        self.min_message = min_message
        self.max_message = max_message
    def check(self, angles):
        val = angles.get(self.angle_name)
        if val is None or (isinstance(val, float) and (np.isnan(val))):
            return FeedbackGenerator.missing_angle(self.angle_name)
        if self.min_val is not None and val < self.min_val:
            return self.min_message or self.message or f"{self.angle_name} below minimum"
        if self.max_val is not None and val > self.max_val:
            return self.max_message or self.message or f"{self.angle_name} above maximum"
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
            # Use consistent torso length calculation
            torso = calculate_torso_length(landmarks)
            if torso is not None:
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
        """
        Normalize the given shoulder width and torso length by the average torso length from calibration.

        Args:
            shoulder_width (float): The measured shoulder width for the current frame.
            torso_length (float): The measured torso length for the current frame.

        Returns:
            Tuple[float, float]: Normalized (shoulder_width, torso_length) if calibrated, else raw values.
        """
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
    def compute_view_votes(self, landmarks: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Compute weighted votes for each view ('front', 'side', 'unknown') based on ratio, symmetry, and visibility.
        Returns a dict: {view: score}
        """
        votes = {"front": 0.0, "side": 0.0, "unknown": 0.0}
        # Config thresholds
        side_criteria = _PUSHUP_CONFIG["views"]["side"].get("detection_criteria", {})
        front_criteria = _PUSHUP_CONFIG["views"]["front"].get("detection_criteria", {})
        # Ratio
        torso_length = calculate_torso_length(landmarks)
        shoulder_width = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0]) if ("left_shoulder" in landmarks and "right_shoulder" in landmarks) else 0.0
        perframe_ratio = (shoulder_width / torso_length) if (torso_length and torso_length > 0) else 0.0
        
        # Symmetry - FIXED: Use consistent coordinate-based calculation
        symmetry = None
        if ("left_shoulder" in landmarks and "right_shoulder" in landmarks):
            symmetry = abs(landmarks["left_shoulder"][1] - landmarks["right_shoulder"][1])
        
        # Visibility
        left_vis = landmarks["left_shoulder"][3] if ("left_shoulder" in landmarks and len(landmarks["left_shoulder"]) > 3) else 0.0
        right_vis = landmarks["right_shoulder"][3] if ("right_shoulder" in landmarks and len(landmarks["right_shoulder"]) > 3) else 0.0
        visible = left_vis > 0.5 and right_vis > 0.5
        
        # --- Voting weights ---
        # Ratio
        side_ratio_range = side_criteria.get("shoulder_torso_ratio", {})
        front_ratio_range = front_criteria.get("shoulder_torso_ratio", {})
        if side_ratio_range.get("min", 0) <= perframe_ratio <= side_ratio_range.get("max", 0.4):
            votes["side"] += 1.0
        if front_ratio_range.get("min", 0) <= perframe_ratio <= front_ratio_range.get("max", 1.0):
            votes["front"] += 1.0
        
        # Symmetry - FIXED: Use config thresholds consistently
        front_sym = front_criteria.get("symmetry_threshold", 0.08)
        side_sym = side_criteria.get("symmetry_threshold", 0.15)
        if symmetry is not None:
            if symmetry < front_sym:
                votes["front"] += 1.0
            elif symmetry > side_sym:
                votes["side"] += 1.0
            else:
                votes["unknown"] += 0.5
        
        # Visibility
        if visible:
            votes["front"] += 0.5
            votes["side"] += 0.5
        else:
            votes["unknown"] += 1.0
        return votes

    """Robust push-up analyzer for side, front, and ambiguous views."""
    _MIN_PHASE_DURATION = 0.08  # Lowered for fast push-up detection
    _VIEW_DETECTION_INTERVAL = 0.1  # Throttle view detection
    _VIEW_HISTORY_LEN = 8  # Rolling buffer for view confidence smoothing
    _ASYMMETRY_HISTORY_LEN = 10  # History for asymmetry detection
    _ELBOW_HISTORY_LEN = 5  # History for elbow angle smoothing
    _VELOCITY_SMOOTHING = 3  # Number of frames to average over
    _VELOCITY_THRESHOLDS = {'side': 8, 'front': 12}  # Velocity thresholds for phase detection
    _HYSTERESIS_MARGIN = 0.10  # Increased hysteresis margin for view detection

    # Smooths noisy switching and enforces stable view confidence
    _AMBIGUOUS_VIEW_MIN_TIME = 0.5  # Minimum time to be considered ambiguous
    _AMBIGUOUS_VIEW_MIN_CONF = 0.7  # Minimum confidence for ambiguous view
    _STICKY_VIEW_MIN_TIME = 2.0  # Increased minimum time to be considered sticky
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
        self._side_analyzer.parent_analyzer = self
        self._front_analyzer = FrontViewAnalyzer()
        self._front_analyzer.parent_analyzer = self
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
        for analyzer in self._view_analyzers.values():
            analyzer.parent_analyzer = self
        self._session_calibration = SessionCalibration()
        self._last_ambiguous_time = 0
        self._ambiguous_since = None
        self._sticky_view = None
        self._sticky_view_since = None
        self._sticky_view_conf = 0.0
        # Store last per-frame body metrics for debug printing
        self._last_body_metrics = {"shoulder_width": None, "torso_length": None, "shoulder_torso_ratio": None}
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
            self._session_calibration.update(landmarks)
            # Store for debug output in trainer
            torso_length = calculate_torso_length(landmarks)
            shoulder_width = abs(landmarks["left_shoulder"][0] - landmarks["right_shoulder"][0]) if ("left_shoulder" in landmarks and "right_shoulder" in landmarks) else 0.0
            perframe_ratio = (shoulder_width / torso_length) if (torso_length and torso_length > 0) else 0.0
            self._last_body_metrics = {
                "shoulder_width": shoulder_width,
                "torso_length": torso_length,
                "shoulder_torso_ratio": perframe_ratio
            }
            # --- Unified voting logic ---
            votes = self.compute_view_votes(landmarks)
            total = sum(votes.values())
            if total == 0:
                return "unknown", 0.0, "View could not be determined."
            best_view = max(votes.items(), key=lambda x: x[1])
            confidence = best_view[1] / total if total > 0 else 0.0
            reposition_warning = None
            if confidence < 0.6:
                reposition_warning = f"Camera angle ambiguous. Votes: {votes}. Try moving to side or front."
            return best_view[0], confidence, reposition_warning
        except Exception as e:
            logger.error(f"Error in _detect_view: {e}")
            return "unknown", 0.0, None
    
    # "_update_view_detection" keeps a rolling buffer of recent detections and uses a voting mechanism to smooth out random fluctuations.
    # It also ensures that the view is not changed too frequently.
    def _update_view_detection(self, landmarks: Dict[str, List[float]]):
        # Unified logic: always unpack three values from _detect_view
        new_view, confidence, reposition_warning = self._detect_view(landmarks)
        self._reposition_warning = reposition_warning
        current_time = time.time()

        # Check if enough time has passed since last detection
        if current_time - self._last_view_detection_time < self._VIEW_DETECTION_INTERVAL:
            return

        self._last_view_detection_time = current_time
        # Add to history, keeps a rolling buffer of recent detections for voting.
        self._view_history.append((new_view, confidence))

        # Process history if enough samples (3 or more) to ensure a valid view.
        if len(self._view_history) >= 3:
            weighted_votes = {}  # Dictionary to store the weighted votes for each view.
            total_weight = 0  # Total weight of all votes.
            # Calculate weighted votes for each view, weighted by confidence.
            for view, conf in self._view_history:
                if view not in weighted_votes:
                    weighted_votes[view] = 0
                weighted_votes[view] += conf
                total_weight += conf
            # Determine best view based on weighted voting, if the total weight is greater than 0.
            if total_weight > 0:
                best_view = max(weighted_votes.items(), key=lambda x: x[1])
                if best_view[1] / total_weight > 0.6:  # 60% threshold
                    self._current_view = best_view[0]
                    self._view_confidence = best_view[1] / total_weight
        else:
            self._current_view = new_view
            self._view_confidence = confidence

        # Clear caches if view changed, helps prevent stale data from being used.
        if self._current_view != self._last_view:
            logger.info(f"[VIEW] Changed to {self._current_view} (confidence={self._view_confidence:.2f})")
            self._cached_phase_thresholds.clear()
            self._cached_form_rules.clear()

        self._last_view = self._current_view
        self._last_view_confidence = self._view_confidence
        # No reposition_warning available in this context; nothing to return

    def _get_current_analyzer(self) -> Optional[ViewSpecificAnalyzer]: # Returns the current analyzer based on the current view.
        return self._view_analyzers.get(self._current_view)

    def _get_phase_thresholds(self) -> Dict[str, float]:
        cache_key = (self._current_view, self.user_level)
        if cache_key not in self._cached_phase_thresholds:
            analyzer = self._get_current_analyzer()
            if analyzer:
                self._cached_phase_thresholds[cache_key] = analyzer.get_phase_thresholds(self.user_level)
            else:
                # Use config default or raise error if not found
                view_cfg = _PUSHUP_CONFIG["views"].get(self._current_view, {})
                thresholds = view_cfg.get("phase_thresholds")
                if thresholds:
                    # Handle user level for bottom_reached/top_reached if dict
                    level = self.user_level.name.lower() if hasattr(self.user_level, "name") else str(self.user_level).lower()
                    result = {}
                    for k, v in thresholds.items():
                        if isinstance(v, dict):
                            result[k] = v.get(level, v.get("beginner"))
                        else:
                            result[k] = v
                    self._cached_phase_thresholds[cache_key] = result
                else:
                    raise ValueError(f"No phase thresholds for view {self._current_view} in config")
        return self._cached_phase_thresholds[cache_key]

    def _get_required_angles(self) -> List[str]:
        analyzer = self._get_current_analyzer()
        if analyzer:
            return analyzer.get_required_angles()
        return _PUSHUP_CONFIG["views"].get(self._current_view, {}).get("required_angles", [])

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
                # Rep is counted only when transitioning from ASCENT to TOP
                # Check robust vertical depth before counting rep
                vertical_depth = self._calculate_vertical_depth(self._last_landmarks) if hasattr(self, '_last_landmarks') else None
                form_quality_cfg = _PUSHUP_CONFIG.get("form_quality", {})
                min_depth = form_quality_cfg.get("min_depth_achievement", 1.0)
                if vertical_depth is not None and vertical_depth < min_depth:
                    print(f"[DEBUG][WARNING] Rep counted but depth insufficient: {vertical_depth:.3f} < {min_depth}")
                    self._last_violation = "Go a little bit lower."
                else:
                    self._last_violation = None
                    self._rep_count += 1
                    self._current_rep_quality = self._calculate_rep_quality(angles, PushupPhase.TOP)
                    self._completed_reps.append(self._current_rep_quality)
                    new_phase = PushupPhase.TOP
        elif self._current_phase == PushupPhase.TOP:
            # Only transition to REST, do not increment rep count here
            if avg_elbow >= thresholds["rest_threshold"]:
                # Detect incomplete rep: up to up without reaching bottom
                if hasattr(self, '_last_phase') and self._last_phase == PushupPhase.TOP:
                    # If the last phase was also TOP, and no BOTTOM was reached, give feedback
                    print("[DEBUG][WARNING] Incomplete rep: Did not reach bottom. Provide feedback.")
                    self._last_violation = "Incomplete rep: go all the way down."
                new_phase = PushupPhase.REST
                self._rep_start_time = None
        # Track last phase for incomplete rep detection
        self._last_phase = self._current_phase
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

    def _calculate_vertical_depth(self, landmarks: Dict[str, List[float]]) -> Optional[float]:
        """
        Calculate push-up depth as the vertical (Y) distance between shoulder and wrist,
        normalized by torso length. Handles both front and side views robustly.
        """
        if self._current_view == "side":
            # Use the side with better visibility
            def side_visibility(side):
                keys = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", f"{side}_hip", f"{side}_ankle"]
                return np.mean([landmarks[k][3] for k in keys if k in landmarks and len(landmarks[k]) > 3])
            left_vis = side_visibility("left")
            right_vis = side_visibility("right")
            primary = "left" if left_vis >= right_vis else "right"
            if all(f"{primary}_shoulder" in landmarks and f"{primary}_wrist" in landmarks and f"{primary}_hip" in landmarks):
                y_shoulder = landmarks[f"{primary}_shoulder"][1]
                y_wrist = landmarks[f"{primary}_wrist"][1]
                raw_depth = y_wrist - y_shoulder
                torso_length = calculate_length(landmarks[f"{primary}_shoulder"], landmarks[f"{primary}_hip"])
                if torso_length and torso_length > 0:
                    norm_depth = raw_depth / torso_length
                else:
                    norm_depth = raw_depth
                return max(0.0, norm_depth)
            else:
                return None
        else:
            # Front view: use average of both sides
            if not ("left_shoulder" in landmarks and "right_shoulder" in landmarks and "left_wrist" in landmarks and "right_wrist" in landmarks):
                return None
            y_shoulder = (landmarks["left_shoulder"][1] + landmarks["right_shoulder"][1]) / 2
            y_wrist = (landmarks["left_wrist"][1] + landmarks["right_wrist"][1]) / 2
            raw_depth = y_wrist - y_shoulder
            torso_length = None
            if "left_shoulder" in landmarks and "left_hip" in landmarks:
                torso_length = calculate_length(landmarks["left_shoulder"], landmarks["left_hip"])
            elif "right_shoulder" in landmarks and "right_hip" in landmarks:
                torso_length = calculate_length(landmarks["right_shoulder"], landmarks["right_hip"])
            if torso_length and torso_length > 0:
                norm_depth = raw_depth / torso_length
            else:
                norm_depth = raw_depth
            return max(0.0, norm_depth)

    def _calculate_rep_quality(self, angles: Dict[str, float], phase) -> Dict[str, Any]:
        if phase != PushupPhase.TOP:
            return self._current_rep_quality
        
        # Get form quality thresholds from config
        form_quality_cfg = _PUSHUP_CONFIG.get("form_quality", {})
        min_body_alignment = form_quality_cfg.get("min_body_alignment_score", 0.7)
        max_asymmetry = form_quality_cfg.get("max_asymmetry_score", 0.3)
        min_depth_achievement = form_quality_cfg.get("min_depth_achievement", 0.8)
        
        quality = {
            "form_violations": [], 
            "range_quality": 0.0, 
            "symmetry_score": None, 
            "tempo_score": None, 
            "depth_achieved": 0.0,
            "body_alignment_score": None,
            "asymmetry_score": None,
            "form_quality_score": 0.0
        }
        
        quality["form_violations"] = self.check_form_violations(angles)
        
        # Calculate robust depth achievement
        vertical_depth = self._calculate_vertical_depth(self._last_landmarks) if hasattr(self, '_last_landmarks') else None
        if vertical_depth is not None:
            quality["depth_achieved"] = vertical_depth
            print(f"[DEBUG] Robust vertical depth achieved: {vertical_depth:.3f}")
        else:
            # Fallback to elbow angle method if vertical depth not available
            avg_elbow = self._get_average_elbow_angle(angles)
            thresholds = self._get_phase_thresholds()
            view_cfg = _PUSHUP_CONFIG["views"].get(self._current_view, {})
            ideal_depth = view_cfg.get("ideal_depth", 70)
            max_depth = view_cfg.get("max_depth", 130)
            if avg_elbow is not None:
                min_depth = thresholds["bottom_reached"]
                if avg_elbow <= ideal_depth:
                    quality["range_quality"] = 1.0
                elif avg_elbow <= min_depth:
                    quality["range_quality"] = 0.8
                else:
                    quality["range_quality"] = max(0, 1 - (avg_elbow - min_depth) / (max_depth - min_depth))
                quality["depth_achieved"] = max(0.0, max_depth - avg_elbow)
        
        # Calculate symmetry score
        symmetry = self._calculate_symmetry_score(angles)
        if symmetry is not None:
            quality["symmetry_score"] = max(0, 1 - symmetry)
            quality["asymmetry_score"] = symmetry
        else:
            quality["symmetry_score"] = None
            quality["asymmetry_score"] = None
        
        # Calculate body alignment score
        if "body_alignment_score" in angles and not np.isnan(angles["body_alignment_score"]):
            quality["body_alignment_score"] = angles["body_alignment_score"]
        
        # Calculate overall form quality score
        form_quality_factors = []
        
        # Range quality factor
        form_quality_factors.append(quality["range_quality"])
        
        # Symmetry factor
        if quality["symmetry_score"] is not None:
            symmetry_factor = 1.0 if quality["asymmetry_score"] <= max_asymmetry else max(0, 1 - (quality["asymmetry_score"] - max_asymmetry) / (1 - max_asymmetry))
            form_quality_factors.append(symmetry_factor)
        
        # Body alignment factor
        if quality["body_alignment_score"] is not None:
            alignment_factor = 1.0 if quality["body_alignment_score"] >= min_body_alignment else max(0, quality["body_alignment_score"] / min_body_alignment)
            form_quality_factors.append(alignment_factor)
        
        # Depth achievement factor
        if quality["depth_achieved"] > 0:
            depth_factor = 1.0 if quality["depth_achieved"] >= min_depth_achievement else max(0, quality["depth_achieved"] / min_depth_achievement)
            form_quality_factors.append(depth_factor)
        
        # Calculate overall form quality as average of available factors
        if form_quality_factors:
            quality["form_quality_score"] = np.mean(form_quality_factors)
        
        # Adaptive tempo: use average of last 3 reps if available
        if self._rep_start_time:
            rep_duration = time.time() - self._rep_start_time
            valid_tempos = [rep.get("tempo_score") for rep in self._completed_reps[-3:] if rep.get("tempo_score") is not None]
            if valid_tempos:
                avg_tempo = np.mean(valid_tempos)
                tempo_deviation = abs(rep_duration - avg_tempo) / avg_tempo
                quality["tempo_score"] = max(0, 1 - tempo_deviation)
            else:
                quality["tempo_score"] = None
        
        return quality
    
    def _calculate_symmetry_score(self, angles: Dict[str, float]) -> Optional[float]:
        """
        Calculate symmetry score using both angle-based and coordinate-based asymmetry metrics.
        Returns a normalized score where 0 = perfect symmetry, 1 = poor symmetry.
        """
        try:
            symmetry_factors = []
            
            # 1. Angle-based asymmetry (normalized by typical angle ranges)
            angle_pairs = [
                ("left_elbow", "right_elbow", 180.0),  # Elbow angles typically 0-180°
                ("left_shoulder", "right_shoulder", 180.0),  # Shoulder angles typically 0-180°
                ("left_hip", "right_hip", 180.0)  # Hip angles typically 0-180°
            ]
            
            for left_angle, right_angle, max_range in angle_pairs:
                if left_angle in angles and right_angle in angles:
                    if not (np.isnan(angles[left_angle]) or np.isnan(angles[right_angle])):
                        asymmetry = abs(angles[left_angle] - angles[right_angle])
                        normalized_asymmetry = asymmetry / max_range
                        normalized_asymmetry = min(1.0, max(0.0, normalized_asymmetry))  # Clamp to [0, 1]
                        symmetry_factors.append(normalized_asymmetry)
            
            # 2. Coordinate-based asymmetry (from pose_utils)
            coordinate_metrics = [
                ("shoulder_symmetry", 0.1),  # Threshold for poor symmetry
                ("hip_symmetry", 0.08),  # Threshold for poor symmetry
                ("wrist_distance", 0.5)  # Threshold for poor wrist positioning
            ]
            
            for metric, threshold in coordinate_metrics:
                if metric in angles and not np.isnan(angles[metric]):
                    normalized_asymmetry = angles[metric] / threshold
                    normalized_asymmetry = min(1.0, max(0.0, normalized_asymmetry))  # Clamp to [0, 1]
                    symmetry_factors.append(normalized_asymmetry)
            
            if not symmetry_factors:
                logger.warning("Symmetry score not available due to missing data.")
                return None
            
            # Calculate average asymmetry and add to history
            avg_asymmetry = np.mean(symmetry_factors)
            self._asymmetry_history.append(avg_asymmetry)
            
            # Return smoothed average from history
            return np.mean(list(self._asymmetry_history))
            
        except Exception as e:
            logger.error(f"Error calculating symmetry score: {str(e)}")
            return None

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
            if isinstance(thresholds, dict):
                min_val = thresholds.get("min")
                max_val = thresholds.get("max")
                message = thresholds.get("message")
                min_message = thresholds.get("min_message")
                max_message = thresholds.get("max_message")
                form_rules.append(FormRule(angle_name, min_val, max_val, message, min_message, max_message))
        
        # Check all form rules
        for rule in form_rules:
            result = rule.check(angles)
            if result:
                violations.append(result)
        
        # Additional asymmetry and alignment checks
        try:
            # Check shoulder symmetry
            if "shoulder_symmetry" in angles and not np.isnan(angles["shoulder_symmetry"]):
                max_threshold = rules.get("shoulder_symmetry", {}).get("max", 0.1)
                if angles["shoulder_symmetry"] > max_threshold:
                    violations.append(f"Shoulder asymmetry too high: {angles['shoulder_symmetry']:.3f} > {max_threshold}")
            
            # Check hip symmetry
            if "hip_symmetry" in angles and not np.isnan(angles["hip_symmetry"]):
                max_threshold = rules.get("hip_symmetry", {}).get("max", 0.08)
                if angles["hip_symmetry"] > max_threshold:
                    violations.append(f"Hip asymmetry too high: {angles['hip_symmetry']:.3f} > {max_threshold}")
            
            # Check wrist distance
            if "wrist_distance" in angles and not np.isnan(angles["wrist_distance"]):
                wrist_rules = rules.get("wrist_distance", {})
                min_threshold = wrist_rules.get("min", 0.15)
                max_threshold = wrist_rules.get("max", 0.35)
                if angles["wrist_distance"] < min_threshold:
                    violations.append(f"Wrists too close: {angles['wrist_distance']:.3f} < {min_threshold}")
                elif angles["wrist_distance"] > max_threshold:
                    violations.append(f"Wrists too far apart: {angles['wrist_distance']:.3f} > {max_threshold}")
            
            # Check body alignment score using form_quality config
            if "body_alignment_score" in angles and not np.isnan(angles["body_alignment_score"]):
                form_quality_cfg = _PUSHUP_CONFIG.get("form_quality", {})
                min_threshold = form_quality_cfg.get("min_body_alignment_score", 0.7)
                if angles["body_alignment_score"] < min_threshold:
                    violations.append(f"Poor body alignment: {angles['body_alignment_score']:.3f} < {min_threshold}")
            
            # Check overall asymmetry score using form_quality config
            asymmetry_score = self._calculate_symmetry_score(angles)
            if asymmetry_score is not None:
                form_quality_cfg = _PUSHUP_CONFIG.get("form_quality", {})
                max_threshold = form_quality_cfg.get("max_asymmetry_score", 0.3)
                if asymmetry_score > max_threshold:
                    violations.append(f"High asymmetry: {asymmetry_score:.3f} > {max_threshold}")
            
            # Check depth achievement using robust vertical depth
            vertical_depth = self._calculate_vertical_depth(self._last_landmarks) if hasattr(self, '_last_landmarks') else None
            form_quality_cfg = _PUSHUP_CONFIG.get("form_quality", {})
            min_threshold = form_quality_cfg.get("min_depth_achievement", 1.0)  # Suggest 1.0 for normalized robust depth
            if vertical_depth is not None:
                print(f"[DEBUG] Robust vertical depth (for violation): {vertical_depth:.3f} vs threshold {min_threshold}")
                if vertical_depth < min_threshold:
                    violations.append("Go a little bit lower.")
            else:
                print("[DEBUG] Robust vertical depth not available for violation check.")
            
        except Exception as e:
            logger.error(f"Error in additional form checks: {str(e)}")
        
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

    def _create_error_state(self, error_message: str) -> ExerciseState:
        """Create an ExerciseState indicating an error or unreliable analysis."""
        return ExerciseState(
            name=self.get_exercise_name() if hasattr(self, 'get_exercise_name') else "pushup",
            phase=getattr(self, '_current_phase', None) and self._current_phase.value or "unknown",
            rep_count=getattr(self, '_rep_count', 0),
            is_correct_form=False,
            violations=[error_message],
            angles={},
            confidence=0.0,
            analysis_reliable=False,
            error_message=error_message,
            user_level=getattr(self, 'user_level', None) or UserLevel.BEGINNER
        )

# --- Concrete PushupAnalyzer implementation ---
class PushupAnalyzer(PushupAnalyzerBase):
    def _is_rep_start_condition(self, angles, phase):
        return super()._is_rep_start_condition(angles, phase)

    def _is_rep_end_condition(self, angles, phase):
        return super()._is_rep_end_condition(angles, phase)

    def analyze_frame(self, landmarks, angles, exercise_variant=None):
        # 1. Update view detection and select analyzer
        self._update_view_detection(landmarks)
        analyzer = self._get_current_analyzer()
        current_view = self._current_view
        print(f"[DEBUG] Current view: {current_view}, Analyzer: {analyzer}")
        if analyzer is None:
            print("[DEBUG] Analyzer context missing.")
            return self._create_error_state("Analyzer context missing.")

        # 2. Strict missing landmark/angle check
        required_landmarks = analyzer.get_required_landmarks()
        missing_landmarks = [l for l in required_landmarks if l not in landmarks or len(landmarks[l]) < 4 or landmarks[l][3] < analyzer.min_landmark_visibility]
        if missing_landmarks:
            print(f"[DEBUG] Missing landmarks: {missing_landmarks}")
            # Return a special error state to indicate skip-frame
            return ExerciseState(
                name=self.get_exercise_name(),
                phase=getattr(self, '_current_phase', None) and self._current_phase.value or "unknown",
                rep_count=getattr(self, '_rep_count', 0),
                is_correct_form=False,
                violations=[],
                angles=angles,
                confidence=0.0,
                analysis_reliable=False,
                error_message="skip_frame",
                user_level=getattr(self, 'user_level', None) or UserLevel.BEGINNER
            )
        required_angles = analyzer.get_required_angles()
        missing_angles = [a for a in required_angles if a not in angles or angles[a] is None or (hasattr(np, 'isnan') and np.isnan(angles[a]))]
        if missing_angles:
            print(f"[DEBUG] Missing angles: {missing_angles}")
            return self._create_error_state(f"Missing required angles: {', '.join(missing_angles)}")

        # 3. Camera position validation
        camera_valid, camera_error = analyzer.validate_camera_position(landmarks, angles)
        print(f"[DEBUG] Camera valid: {camera_valid}, Error: {camera_error}")
        if not camera_valid:
            return self._create_error_state(camera_error or "Camera position invalid.")

        # 4. Form rule & violation checks
        violations = self.check_form_violations(angles, exercise_variant)
        # Add last phase-based violation if present
        if hasattr(self, '_last_violation') and self._last_violation:
            violations.append(self._last_violation)
            self._last_violation = None
        is_correct_form = not violations

        # 5. Movement & phase tracking
        current_time = time.time()
        self._last_landmarks = landmarks  # Store for robust depth calculation
        self._update_movement_tracking(angles, current_time)
        self._update_phase_state_machine(angles, current_time)

        # 6. Rep quality & feedback
        quality = self._calculate_rep_quality(angles, getattr(self, '_current_phase', None))
        # Clamp and normalize depth_achieved
        if "depth_achieved" in quality:
            quality["depth_achieved"] = max(0.0, quality["depth_achieved"])
        confidence = self.calculate_confidence(landmarks)

        # 7. Return ExerciseState
        # Remove 'form_violations' from quality and use as 'violations'
        violations_from_quality = quality.pop("form_violations", [])
        all_violations = violations or violations_from_quality
        # Only pass valid fields to ExerciseState
        return ExerciseState(
            name=self.get_exercise_name(),
            phase=getattr(self, '_current_phase', None) and self._current_phase.value or "unknown",
            rep_count=getattr(self, '_rep_count', 0),
            is_correct_form=is_correct_form,
            violations=all_violations,
            angles=angles,
            confidence=confidence,
            analysis_reliable=True,
            error_message=all_violations[0] if all_violations else None,
            user_level=getattr(self, 'user_level', None) or UserLevel.BEGINNER
        )
    def get_exercise_name(self):
        return super().get_exercise_name()
