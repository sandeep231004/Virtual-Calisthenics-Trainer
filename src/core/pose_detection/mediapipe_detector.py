from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np
import cv2

from .base_detector import BasePoseDetector # Abstract base class


class MediaPipePoseDetector(BasePoseDetector): # MediapiePoseDetector inherits from BasePoseDetector. Concrete implementation of the abstract "BasePoseDetector" class.
    """MediaPipe implementation of pose detection."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initialize the MediaPipe pose detector.

        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose # Assigning the "pose" solution from mediapipe.solutions
        self.pose = self.mp_pose.Pose(   # Initializing the pose detection model, "pose" instance holding the pose detection model
            static_image_mode=False,
            model_complexity=1,          # Pose Complexity for pose landmark model.
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self._landmark_names = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer", "left_ear",
            "right_ear", "mouth_left", "mouth_right", "left_shoulder",
            "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
            "right_wrist", "left_pinky", "right_pinky", "left_index",
            "right_index", "left_thumb", "right_thumb", "left_hip",
            "right_hip", "left_knee", "right_knee", "left_ankle",
            "right_ankle", "left_heel", "right_heel", "left_foot_index",
            "right_foot_index"
        ]

    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[Dict[str, List[float]]]]:
        """
        Detect pose landmarks using MediaPipe.

        Args:
            frame: Input frame as numpy array

        Returns:
            Tuple containing:
            - Boolean indicating if detection was successful
            - Dictionary of landmark coordinates (if successful) or None
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return False, None
            
        # Convert landmarks to dictionary
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks[self._landmark_names[idx]] = [
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ]
            
        return True, landmarks

    def get_landmark_names(self) -> List[str]:
        """Get the list of landmark names provided by MediaPipe."""
        return self._landmark_names

    def calculate_angles(self, landmarks: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate joint angles from MediaPipe landmarks.

        Args:
            landmarks: Dictionary of landmark coordinates [x, y, z, visibility]

        Returns:
            Dictionary of angle names and their values in degrees
        """
        angles = {}
        
        # Define angle calculations with their required landmarks and whether they need full range
        angle_definitions = {
            # Basic angles (0-180)
            "left_elbow": ("left_shoulder", "left_elbow", "left_wrist", False),
            "right_elbow": ("right_shoulder", "right_elbow", "right_wrist", False),
            "left_knee": ("left_hip", "left_knee", "left_ankle", False),
            "right_knee": ("right_hip", "right_knee", "right_ankle", False),
            
            # Full range angles (0-360)
            "left_shoulder": ("left_hip", "left_shoulder", "left_elbow", True),
            "right_shoulder": ("right_hip", "right_shoulder", "right_elbow", True),
            "left_hip": ("left_shoulder", "left_hip", "left_knee", True),
            "right_hip": ("right_shoulder", "right_hip", "right_knee", True),
        }

        # Minimum visibility threshold for landmarks
        VISIBILITY_THRESHOLD = 0.5

        for angle_name, (p1_name, p2_name, p3_name, full_range) in angle_definitions.items():
            try:
                # Check if all required landmarks exist
                if not all(lm_name in landmarks for lm_name in [p1_name, p2_name, p3_name]):
                    angles[angle_name] = np.nan
                    continue

                # Check visibility of all points
                p1_visible = landmarks[p1_name][3] > VISIBILITY_THRESHOLD
                p2_visible = landmarks[p2_name][3] > VISIBILITY_THRESHOLD
                p3_visible = landmarks[p3_name][3] > VISIBILITY_THRESHOLD

                if not all([p1_visible, p2_visible, p3_visible]):
                    angles[angle_name] = np.nan
                    continue

                # Calculate angle
                angle = self._calculate_angle(
                    landmarks[p1_name],
                    landmarks[p2_name],
                    landmarks[p3_name],
                    full_range
                )

                # Validate angle is within reasonable range
                max_angle = 360 if full_range else 180
                if 0 <= angle <= max_angle:
                    angles[angle_name] = angle
                else:
                    angles[angle_name] = np.nan

            except Exception as e:
                # Log error and set angle to NaN
                print(f"Error calculating {angle_name}: {str(e)}")
                angles[angle_name] = np.nan

        return angles

    def _calculate_angle(self, a: List[float], b: List[float], c: List[float], full_range: bool = False) -> float:
        """
        Calculate the angle between three points with robust error handling.
        Supports both 0-180 degree range (default) and 0-360 degree range.

        Args:
            a: First point coordinates [x, y, z, visibility]
            b: Middle point coordinates [x, y, z, visibility]
            c: Last point coordinates [x, y, z, visibility]
            full_range: If True, returns angle in 0-360 range, otherwise 0-180 range

        Returns:
            Angle in degrees, or 0.0 if calculation is not possible
        """
        try:
            # Convert to numpy arrays (ignore visibility value)
            a = np.array([a[0], a[1], a[2]])
            b = np.array([b[0], b[1], b[2]])
            c = np.array([c[0], c[1], c[2]])
            
            # Calculate vectors from middle point to other points
            ba = a - b  # Vector from b to a
            bc = c - b  # Vector from b to c
            
            # Calculate vector norms (lengths)
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            
            # Check for degenerate vectors (points too close together)
            if norm_ba < 1e-6 or norm_bc < 1e-6:
                return 0.0
                
            # Calculate cosine of angle using dot product
            cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
            
            # Ensure cosine is within valid range [-1, 1]
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            # Calculate basic angle in degrees (0-180)
            angle = np.degrees(np.arccos(cosine_angle))
            
            if full_range:
                # For full range (0-360), we need to determine the direction
                # Calculate cross product to determine the orientation
                cross_product = np.cross(ba, bc)
                
                # The sign of the z-component tells us if the angle is > 180
                if cross_product[2] < 0:
                    angle = 360 - angle
                    
                # Additional validation for full range angles
                if not (0 <= angle <= 360):
                    return 0.0
            else:
                # Additional validation for limited range angles
                if not (0 <= angle <= 180):
                    return 0.0
            
            return angle
            
        except Exception as e:
            print(f"Error in angle calculation: {str(e)}")
            return 0.0 