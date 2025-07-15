from typing import Dict, List, Optional, Tuple
from collections import deque

import mediapipe as mp
import numpy as np
import cv2

from .base_detector import BasePoseDetector
from ..exercise_analysis.pose_utils import calculate_angle, calculate_pushup_specific_angles, calculate_asymmetry_metrics, calculate_body_alignment_score

class MediaPipePoseDetector(BasePoseDetector):
    """MediaPipe implementation of pose detection with improved angle calculations."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5, model_complexity: int = 1):
        """
        Initialize the MediaPipe pose detector.

        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            model_complexity: Complexity of the pose landmark model (0, 1, or 2)
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
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
        self._min_landmark_visibility = 0.3  # Minimum visibility threshold for landmarks
        
        # Initialize temporal smoothing
        self._angle_history = {}
        self._window_size = 5  # Number of frames to average over
        
        # Camera angle detection
        self._camera_angle = "unknown"
        self._camera_angle_history = deque(maxlen=10)

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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return False, None
            
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            # Only include landmarks that meet the minimum visibility threshold
            if landmark.visibility >= self._min_landmark_visibility:
                landmarks[self._landmark_names[idx]] = [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility
                ]
        
        # Update camera angle detection
        self._update_camera_angle(landmarks)
            
        return True, landmarks

    def _update_camera_angle(self, landmarks: Dict[str, List[float]]) -> None:
        """Update camera angle detection based on landmark positions."""
        if "left_shoulder" not in landmarks or "right_shoulder" not in landmarks:
            return
            
        left_shoulder = np.array(landmarks["left_shoulder"][:3])
        right_shoulder = np.array(landmarks["right_shoulder"][:3])
        
        # Calculate shoulder separation in different dimensions
        width_separation = abs(left_shoulder[0] - right_shoulder[0])  # X-axis
        depth_separation = abs(left_shoulder[2] - right_shoulder[2])  # Z-axis
        
        # FIXED: Improved camera angle detection with more robust thresholds
        if width_separation < 0.08:  # Shoulders very close = front view
            angle = "front_view"
        elif width_separation > 0.25:  # Shoulders far apart = side view
            angle = "side_view"
        else:  # In between = angled view
            angle = "angled_view"
            
        self._camera_angle_history.append(angle)
        
        # Use most common angle from recent history
        if len(self._camera_angle_history) >= 3:
            from collections import Counter
            self._camera_angle = Counter(self._camera_angle_history).most_common(1)[0][0]

    def get_camera_angle(self) -> str:
        """Get the detected camera viewing angle."""
        return self._camera_angle

    def get_landmark_names(self) -> List[str]:
        """Get the list of landmark names provided by MediaPipe."""
        return self._landmark_names

    def calculate_angles(self, landmarks: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate joint angles from MediaPipe landmarks with temporal smoothing and camera angle compensation.

        Args:
            landmarks: Dictionary of landmark coordinates [x, y, z, visibility]

        Returns:
            Dictionary of angle names and their values in degrees
        """
        angles = {}
        # Angle definitions with consistent point ordering: (point1, middle_point, point3, full_range)
        # The angle is calculated at the middle_point between vectors to point1 and point3
        angle_definitions = {
            "left_elbow": ("left_shoulder", "left_elbow", "left_wrist", False),
            "right_elbow": ("right_shoulder", "right_elbow", "right_wrist", False),
            "left_knee": ("left_hip", "left_knee", "left_ankle", False),
            "right_knee": ("right_hip", "right_knee", "right_ankle", False),
            "left_shoulder": ("left_hip", "left_shoulder", "left_elbow", False),
            "right_shoulder": ("right_hip", "right_shoulder", "right_elbow", False),
            "left_hip": ("left_shoulder", "left_hip", "left_knee", False),
            "right_hip": ("right_shoulder", "right_hip", "right_knee", False),
        }
        for angle_name, (p1_name, p2_name, p3_name, full_range) in angle_definitions.items():
            try:
                if not all(lm_name in landmarks for lm_name in [p1_name, p2_name, p3_name]):
                    angles[angle_name] = np.nan
                    continue
                p1_visible = landmarks[p1_name][3] >= self._min_landmark_visibility
                p2_visible = landmarks[p2_name][3] >= self._min_landmark_visibility
                p3_visible = landmarks[p3_name][3] >= self._min_landmark_visibility
                if not all([p1_visible, p2_visible, p3_visible]):
                    angles[angle_name] = np.nan
                    continue
                angle = calculate_angle(
                    landmarks[p1_name],
                    landmarks[p2_name],
                    landmarks[p3_name],
                    full_range
                )
                max_angle = 360 if full_range else 180
                if 0 <= angle <= max_angle:
                    if angle_name not in self._angle_history:
                        self._angle_history[angle_name] = deque(maxlen=self._window_size)
                    self._angle_history[angle_name].append(angle)
                    angles[angle_name] = np.mean(self._angle_history[angle_name])
                else:
                    angles[angle_name] = np.nan
            except Exception as e:
                angles[angle_name] = np.nan
        def temporal_smoother(name, value):
            if name not in self._angle_history:
                self._angle_history[name] = deque(maxlen=self._window_size)
            self._angle_history[name].append(value)
            return np.mean(self._angle_history[name])
        calculate_pushup_specific_angles(landmarks, angles, self._min_landmark_visibility, temporal_smoother)
        calculate_asymmetry_metrics(landmarks, angles)
        
        # Calculate overall body alignment score
        alignment_score = calculate_body_alignment_score(landmarks, angles)
        if alignment_score is not None:
            angles["body_alignment_score"] = alignment_score
        
        return angles

    def _apply_temporal_smoothing(self, angle_name: str, angle: float, angles: Dict[str, float]) -> None:
        """Apply temporal smoothing to an angle and store in angles dict."""
        if angle_name not in self._angle_history:
            self._angle_history[angle_name] = deque(maxlen=self._window_size)
        self._angle_history[angle_name].append(angle)
        angles[angle_name] = np.mean(self._angle_history[angle_name])