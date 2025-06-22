from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np


@dataclass
class ExerciseState: # Data class to store the current state of an exercise.
    """Represents the current state of an exercise."""
    name: str
    phase: str  # e.g., "up", "down", "rest"
    rep_count: int
    is_correct_form: bool
    violations: List[str]
    angles: Dict[str, float]
    confidence: float # Confidence score for the pose detection in the current frame.
    analysis_reliable: bool = True  # Flag indicating if the analysis is reliable
    error_message: Optional[str] = None  # Optional error message if analysis fails


class BaseExerciseAnalyzer(ABC): # Abstract base class for exercise analysis implementations.
    """Base class for exercise analysis implementations."""
    # Provides a common interface and some shared functionality for analyzing different exercises.

    def __init__(self, min_analysis_confidence: float = 0.5):
        """
        Initialize the analyzer with configuration parameters.
        
        Args:
            min_analysis_confidence: Minimum confidence threshold for reliable analysis
        """
        self.min_analysis_confidence = min_analysis_confidence
        self._min_landmark_visibility = 0.5  # Minimum visibility threshold for landmarks

    def _validate_camera_position(self, landmarks: Dict[str, List[float]]) -> Tuple[bool, Optional[str]]:
        """
        Validate if the camera position is optimal for detection.

        Args:
            landmarks: Dictionary of landmark coordinates

        Returns:
            Tuple of (is_valid, error_message)
        """
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
        user_level: str = "beginner",
        exercise_variant: Optional[str] = None
    ) -> ExerciseState:
        """
        Analyze a single frame of exercise performance.

        Args:
            landmarks: Dictionary of landmark coordinates
            angles: Dictionary of joint angles (angles calculated from landmarks)
            user_level: User's experience level (e.g., "beginner", "intermediate", "advanced")
            exercise_variant: Optional variant of the exercise (e.g., "wide", "narrow", "diamond")

        Returns:
            ExerciseState object containing analysis results
        """
        pass

    @abstractmethod
    def get_exercise_name(self) -> str:
        """
        Get the name of the exercise being analyzed.

        Returns:
            Exercise name
        """
        pass

    @abstractmethod
    def get_required_landmarks(self) -> List[str]:
        """
        Get the list of landmarks required for this exercise.

        Returns:
            List of required landmark names
        """
        pass

    @abstractmethod
    def get_required_angles(self) -> List[str]:
        """
        Get the list of angle names required for this exercise.

        Returns:
            List of required angle names
        """
        pass

    @abstractmethod
    def get_form_rules(
        self,
        user_level: str = "beginner",
        exercise_variant: Optional[str] = None
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get the form rules for this exercise.

        Args:
            user_level: User's experience level
            exercise_variant: Optional variant of the exercise

        Returns:
            Dictionary of rules with their thresholds and messages
            Example:
            {
                "left_knee": {
                    "min": {"threshold": 90, "message": "Left knee not deep enough"},
                    "max": {"threshold": 180, "message": "Left knee hyperextended"}
                }
            }
        """
        pass

    def check_form_violations(
        self, 
        angles: Dict[str, float],
        user_level: str = "beginner",
        exercise_variant: Optional[str] = None
    ) -> List[str]:
        """
        Check for form violations based on joint angles.

        Args:
            angles: Dictionary of joint angles
            user_level: User's experience level
            exercise_variant: Optional variant of the exercise

        Returns:
            List of violation messages
        """
        violations = []
        rules = self.get_form_rules(user_level, exercise_variant)

        for angle_name, thresholds in rules.items():
            if angle_name not in angles:
                continue

            angle_value = angles[angle_name]
            if "min" in thresholds and angle_value < thresholds["min"]["threshold"]:
                violations.append(thresholds["min"]["message"])
            if "max" in thresholds and angle_value > thresholds["max"]["threshold"]:
                violations.append(thresholds["max"]["message"])

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

        Args:
            angles: Dictionary of joint angles
            phase: Current exercise phase

        Returns:
            True if the frame marks the end of a repetition
        """
        pass 