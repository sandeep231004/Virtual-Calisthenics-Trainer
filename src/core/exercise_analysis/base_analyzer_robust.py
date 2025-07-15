from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np


class UserLevel(Enum):
    """Enum representing different user experience levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class LevelConfig:
    """Configuration for a specific user level."""
    min_confidence: float  # Minimum confidence required for analysis
    strictness_factor: float  # How strict the form checking should be (0.0 to 1.0)
    rep_counting_tolerance: float  # Tolerance for rep counting (higher = more forgiving)
    description: str  # Description of the level's characteristics
    min_landmark_visibility: float = 0.5  # Minimum visibility threshold for landmarks


@dataclass
class ExerciseState:
    """Represents the current state of an exercise."""
    name: str
    phase: str  # e.g., "up", "down", "rest"
    rep_count: int
    is_correct_form: bool
    violations: List[str]
    angles: Dict[str, float]
    confidence: float  # Confidence score for the pose detection in the current frame
    analysis_reliable: bool = True  # Flag indicating if the analysis is reliable
    error_message: Optional[str] = None  # Optional error message if analysis fails
    user_level: UserLevel = UserLevel.BEGINNER  # Current user level


class BaseExerciseAnalyzer(ABC):
    """Base class for exercise analysis implementations."""

    # Default configurations for different user levels
    DEFAULT_LEVEL_CONFIGS = {
        UserLevel.BEGINNER: LevelConfig(
            min_confidence=0.5,
            strictness_factor=0.7,  # More forgiving
            rep_counting_tolerance=0.3,  # More forgiving rep counting
            description="Suitable for those new to the exercise. Focus on basic form and safety.",
            min_landmark_visibility=0.4  # More forgiving visibility threshold
        ),
        UserLevel.INTERMEDIATE: LevelConfig(
            min_confidence=0.6,
            strictness_factor=0.85,  # Moderately strict
            rep_counting_tolerance=0.2,  # Moderate rep counting accuracy
            description="For those with basic proficiency. Focus on proper form and technique.",
            min_landmark_visibility=0.5  # Standard visibility threshold
        ),
        UserLevel.ADVANCED: LevelConfig(
            min_confidence=0.7,
            strictness_factor=1.0,  # Strictest form checking
            rep_counting_tolerance=0.1,  # Most accurate rep counting
            description="For experienced users. Focus on perfect form and advanced techniques.",
            min_landmark_visibility=0.6  # Stricter visibility threshold
        )
    }

    def __init__(self, user_level: UserLevel = UserLevel.BEGINNER):
        """
        Initialize the analyzer with configuration parameters.
        
        Args:
            user_level: User's experience level
        """
        self.user_level = user_level
        self.level_config = self.DEFAULT_LEVEL_CONFIGS[user_level]

    def set_user_level(self, user_level: UserLevel) -> None:
        """
        Update the user level and corresponding configuration.

        Args:
            user_level: New user level to set
        """
        self.user_level = user_level
        self.level_config = self.DEFAULT_LEVEL_CONFIGS[user_level]

    def _validate_camera_position(self, landmarks: Dict[str, List[float]]) -> Tuple[bool, Optional[str]]:
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
                if point in landmarks and landmarks[point][3] > self.level_config.min_landmark_visibility:
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

    @abstractmethod
    def analyze_frame(
        self, 
        landmarks: Dict[str, List[float]], 
        angles: Dict[str, float],
        exercise_variant: Optional[str] = None
    ) -> ExerciseState:
        """
        Analyze a single frame of exercise performance.

        Args:
            landmarks: Dictionary of landmark coordinates
            angles: Dictionary of joint angles (angles calculated from landmarks)
            exercise_variant: Optional variant of the exercise (e.g., "wide", "narrow", "diamond")

        Returns:
            ExerciseState object containing analysis results
        """
        pass

    @abstractmethod
    def get_exercise_name(self) -> str:
        """Get the name of the exercise being analyzed."""
        pass

    @abstractmethod
    def get_required_landmarks(self) -> List[str]:
        """Get the list of landmarks required for this exercise."""
        pass

    @abstractmethod
    def get_required_angles(self) -> List[str]:
        """Get the list of angle names required for this exercise."""
        pass

    @abstractmethod
    def get_form_rules(
        self,
        exercise_variant: Optional[str] = None
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get the form rules for this exercise, adjusted for the current user level.

        Args:
            exercise_variant: Optional variant of the exercise

        Returns:
            Dictionary of rules with their thresholds and messages, adjusted for user level
            Example:
            {
                "left_knee": {
                    "min": {
                        "threshold": 90,
                        "message": "Left knee not deep enough",
                        "level_adjustments": {
                            "beginner": {"threshold": 80, "message": "Try to bend your knee more"},
                            "advanced": {"threshold": 95, "message": "Maintain proper depth"}
                        }
                    }
                }
            }
        """
        pass

    def check_form_violations(
        self, 
        angles: Dict[str, float],
        exercise_variant: Optional[str] = None
    ) -> List[str]:
        """
        Check for form violations based on joint angles, considering user level.

        Args:
            angles: Dictionary of joint angles
            exercise_variant: Optional variant of the exercise

        Returns:
            List of violation messages
        """
        violations = []
        rules = self.get_form_rules(exercise_variant)

        for angle_name, thresholds in rules.items():
            if angle_name not in angles:
                continue

            angle_value = angles[angle_name]
            
            # Check minimum threshold
            if "min" in thresholds:
                min_config = thresholds["min"]
                threshold = min_config["threshold"]
                message = min_config["message"]
                
                # Apply level-specific adjustments if available
                if "level_adjustments" in min_config:
                    level_adj = min_config["level_adjustments"].get(self.user_level.value, {})
                    threshold = level_adj.get("threshold", threshold)
                    message = level_adj.get("message", message)
                
                if angle_value < threshold:
                    violations.append(message)

            # Check maximum threshold
            if "max" in thresholds:
                max_config = thresholds["max"]
                threshold = max_config["threshold"]
                message = max_config["message"]
                
                # Apply level-specific adjustments if available
                if "level_adjustments" in max_config:
                    level_adj = max_config["level_adjustments"].get(self.user_level.value, {})
                    threshold = level_adj.get("threshold", threshold)
                    message = level_adj.get("message", message)
                
                if angle_value > threshold:
                    violations.append(message)

        return violations

    def calculate_confidence(self, landmarks: Dict[str, List[float]]) -> float:
        """
        Calculate confidence score for the detection by checking the visibility of the required landmarks.

        Args:
            landmarks: Dictionary of landmark coordinates (x, y, z, visibility)

        Returns:
            Confidence score between 0 and 1 (0: not visible, 1: fully visible)
        """
        required_landmarks = self.get_required_landmarks()
        visibilities = []

        for landmark in required_landmarks:
            if landmark in landmarks:
                visibilities.append(landmarks[landmark][3])  # visibility is the 4th value

        if not visibilities:
            return 0.0

        return np.mean(visibilities)

    def validate_inputs(
        self,
        landmarks: Dict[str, List[float]],
        angles: Dict[str, float]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that all required landmarks and angles are present.

        Args:
            landmarks: Dictionary of landmark coordinates
            angles: Dictionary of joint angles

        Returns:
            Tuple of (is_valid, error_message)
        """
        missing_landmarks = set(self.get_required_landmarks()) - set(landmarks.keys())
        missing_angles = set(self.get_required_angles()) - set(angles.keys())

        if missing_landmarks:
            return False, f"Missing required landmarks: {', '.join(missing_landmarks)}"
        if missing_angles:
            return False, f"Missing required angles: {', '.join(missing_angles)}"

        return True, None

    @abstractmethod
    def _is_rep_start_condition(self, angles: Dict[str, float], phase: str) -> bool:
        """
        Determine if the current frame marks the start of a repetition.
        Should consider the user level's rep_counting_tolerance.

        Args:
            angles: Dictionary of joint angles
            phase: Current exercise phase

        Returns:
            True if the frame marks the start of a repetition
        """
        pass

    @abstractmethod
    def _is_rep_end_condition(self, angles: Dict[str, float], phase: str) -> bool:
        """
        Determine if the current frame marks the end of a repetition.
        Should consider the user level's rep_counting_tolerance.

        Args:
            angles: Dictionary of joint angles
            phase: Current exercise phase

        Returns:
            True if the frame marks the end of a repetition
        """
        pass