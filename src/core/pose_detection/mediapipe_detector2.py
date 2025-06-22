from typing import Dict, List, Optional, Tuple
from collections import deque

import mediapipe as mp
import numpy as np
import cv2

from .base_detector import BasePoseDetector


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
        self._min_landmark_visibility = 0.5  # Minimum visibility threshold for landmarks
        
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
        
        # Determine camera angle based on shoulder visibility
        if width_separation < 0.15:  # Shoulders appear close together
            angle = "front_view"
        elif depth_separation / width_separation > 0.4:  # Significant depth difference
            angle = "angled_view"
        else:
            angle = "side_view"
            
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
        
        # Define angle calculations with their required landmarks
        angle_definitions = {
            # Basic joint angles (0-180)
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

        # Calculate basic angles
        for angle_name, (p1_name, p2_name, p3_name, full_range) in angle_definitions.items():
            try:
                # Check if all required landmarks exist and are visible
                if not all(lm_name in landmarks for lm_name in [p1_name, p2_name, p3_name]):
                    angles[angle_name] = np.nan
                    continue

                # Check visibility of all points
                p1_visible = landmarks[p1_name][3] >= self._min_landmark_visibility
                p2_visible = landmarks[p2_name][3] >= self._min_landmark_visibility
                p3_visible = landmarks[p3_name][3] >= self._min_landmark_visibility

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
                    # Apply temporal smoothing
                    if angle_name not in self._angle_history:
                        self._angle_history[angle_name] = deque(maxlen=self._window_size)
                    self._angle_history[angle_name].append(angle)
                    angles[angle_name] = np.mean(self._angle_history[angle_name])
                else:
                    angles[angle_name] = np.nan

            except Exception as e:
                print(f"Error calculating {angle_name}: {str(e)}")
                angles[angle_name] = np.nan

        # Calculate push-up specific angles
        self._calculate_pushup_specific_angles(landmarks, angles)

        # Calculate asymmetry metrics
        self._calculate_asymmetry_metrics(angles)

        return angles

    def _calculate_pushup_specific_angles(self, landmarks: Dict[str, List[float]], angles: Dict[str, float]) -> None:
        """Calculate push-up specific angles with improved accuracy."""
        try:
            # Determine which side has better visibility
            left_visibility = self._get_side_visibility(landmarks, "left")
            right_visibility = self._get_side_visibility(landmarks, "right")
            primary_side = "left" if left_visibility > right_visibility else "right"
            
            # 1. FIXED: Body-floor angle (body alignment)
            if self._can_calculate_body_alignment(landmarks, primary_side):
                ankle = np.array(landmarks[f"{primary_side}_ankle"][:3])
                hip = np.array(landmarks[f"{primary_side}_hip"][:3])
                shoulder = np.array(landmarks[f"{primary_side}_shoulder"][:3])
                
                # Calculate body line vector
                body_vector = shoulder - ankle
                
                # Calculate angle based on camera position
                if self._camera_angle == "side_view":
                    # For side view, use Y-component for vertical deviation
                    horizontal_distance = np.sqrt(body_vector[0]**2 + body_vector[2]**2)
                    vertical_distance = abs(body_vector[1])
                    angle = np.degrees(np.arctan2(vertical_distance, horizontal_distance))
                else:
                    # For angled/front view, project onto best plane
                    # Use the plane that shows most variation
                    if abs(body_vector[1]) > abs(body_vector[2]):
                        # Use XY plane
                        horizontal_distance = abs(body_vector[0])
                        vertical_distance = abs(body_vector[1])
                    else:
                        # Use XZ plane
                        horizontal_distance = np.sqrt(body_vector[0]**2 + body_vector[2]**2)
                        vertical_distance = abs(body_vector[1])
                    
                    angle = np.degrees(np.arctan2(vertical_distance, horizontal_distance))
                
                # Apply temporal smoothing
                self._apply_temporal_smoothing("body_floor_angle", angle, angles)

            # 2. FIXED: Hip angle (critical for push-up form)
            if self._can_calculate_hip_angle(landmarks, primary_side):
                shoulder = np.array(landmarks[f"{primary_side}_shoulder"][:3])
                hip = np.array(landmarks[f"{primary_side}_hip"][:3])
                ankle = np.array(landmarks[f"{primary_side}_ankle"][:3])
                
                # Calculate hip angle (should be ~180° for straight body)
                angle = self._calculate_angle(
                    landmarks[f"{primary_side}_shoulder"],
                    landmarks[f"{primary_side}_hip"],
                    landmarks[f"{primary_side}_ankle"],
                    False
                )
                
                if 0 <= angle <= 180:
                    self._apply_temporal_smoothing("hip_angle", angle, angles)

            # 3. IMPROVED: Neck angle (neck alignment with spine)
            if self._can_calculate_neck_angle(landmarks, primary_side):
                shoulder = np.array(landmarks[f"{primary_side}_shoulder"][:3])
                hip = np.array(landmarks[f"{primary_side}_hip"][:3])
                ear = np.array(landmarks[f"{primary_side}_ear"][:3])
                
                # Calculate torso vector
                torso_vector = shoulder - hip
                # Calculate neck vector
                neck_vector = ear - shoulder
                
                # Calculate angle between torso and neck
                if np.linalg.norm(torso_vector) > 1e-6 and np.linalg.norm(neck_vector) > 1e-6:
                    cos_angle = np.dot(torso_vector, neck_vector) / (
                        np.linalg.norm(torso_vector) * np.linalg.norm(neck_vector)
                    )
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))
                    
                    # Adjust angle based on camera position
                    if self._camera_angle == "side_view":
                        # For side view, we can directly use the angle
                        pass
                    else:
                        # For other views, we need to project onto the best plane
                        if abs(neck_vector[1]) > abs(neck_vector[2]):
                            # Use XY plane
                            angle = np.degrees(np.arctan2(neck_vector[1], neck_vector[0]))
                        else:
                            # Use XZ plane
                            angle = np.degrees(np.arctan2(neck_vector[2], neck_vector[0]))
                    
                    self._apply_temporal_smoothing("neck_angle", angle, angles)

            # 4. NEW: Wrist angle (injury prevention)
            for side in ["left", "right"]:
                if self._can_calculate_wrist_angle(landmarks, side):
                    elbow = np.array(landmarks[f"{side}_elbow"][:3])
                    wrist = np.array(landmarks[f"{side}_wrist"][:3])
                    # Use index finger as approximation for hand direction
                    if f"{side}_index" in landmarks:
                        index_finger = np.array(landmarks[f"{side}_index"][:3])
                        
                        angle = self._calculate_angle(
                            landmarks[f"{side}_elbow"],
                            landmarks[f"{side}_wrist"],
                            landmarks[f"{side}_index"],
                            False
                        )
                        
                        if 120 <= angle <= 180:  # Valid wrist angle range
                            self._apply_temporal_smoothing(f"{side}_wrist_angle", angle, angles)

        except Exception as e:
            print(f"Error calculating push-up specific angles: {str(e)}")

    def _calculate_asymmetry_metrics(self, angles: Dict[str, float]) -> None:
        """Calculate asymmetry between left and right sides."""
        try:
            # Elbow asymmetry
            if "left_elbow" in angles and "right_elbow" in angles:
                if not (np.isnan(angles["left_elbow"]) or np.isnan(angles["right_elbow"])):
                    asymmetry = abs(angles["left_elbow"] - angles["right_elbow"])
                    angles["elbow_asymmetry"] = asymmetry

            # Shoulder asymmetry
            if "left_shoulder" in angles and "right_shoulder" in angles:
                if not (np.isnan(angles["left_shoulder"]) or np.isnan(angles["right_shoulder"])):
                    asymmetry = abs(angles["left_shoulder"] - angles["right_shoulder"])
                    angles["shoulder_asymmetry"] = asymmetry

        except Exception as e:
            print(f"Error calculating asymmetry metrics: {str(e)}")

    def _get_side_visibility(self, landmarks: Dict[str, List[float]], side: str) -> float:
        """Get average visibility for one side of the body."""
        side_landmarks = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", 
                         f"{side}_hip", f"{side}_ankle"]
        visibilities = []
        
        for landmark in side_landmarks:
            if landmark in landmarks:
                visibilities.append(landmarks[landmark][3])
        
        return np.mean(visibilities) if visibilities else 0.0

    def _can_calculate_body_alignment(self, landmarks: Dict[str, List[float]], side: str) -> bool:
        """Check if body alignment angle can be calculated."""
        required = [f"{side}_ankle", f"{side}_hip", f"{side}_shoulder"]
        return all(
            landmark in landmarks and landmarks[landmark][3] >= self._min_landmark_visibility
            for landmark in required
        )

    def _can_calculate_hip_angle(self, landmarks: Dict[str, List[float]], side: str) -> bool:
        """Check if hip angle can be calculated."""
        required = [f"{side}_shoulder", f"{side}_hip", f"{side}_ankle"]
        return all(
            landmark in landmarks and landmarks[landmark][3] >= self._min_landmark_visibility
            for landmark in required
        )

    def _can_calculate_neck_angle(self, landmarks: Dict[str, List[float]], side: str) -> bool:
        """Check if neck angle can be calculated."""
        required = [f"{side}_shoulder", f"{side}_hip", f"{side}_ear"]
        return all(
            landmark in landmarks and landmarks[landmark][3] >= self._min_landmark_visibility
            for landmark in required
        )

    def _can_calculate_wrist_angle(self, landmarks: Dict[str, List[float]], side: str) -> bool:
        """Check if wrist angle can be calculated."""
        required = [f"{side}_elbow", f"{side}_wrist", f"{side}_index"]
        return all(
            landmark in landmarks and landmarks[landmark][3] >= self._min_landmark_visibility
            for landmark in required
        )

    def _apply_temporal_smoothing(self, angle_name: str, angle: float, angles: Dict[str, float]) -> None:
        """Apply temporal smoothing to an angle and store in angles dict."""
        if angle_name not in self._angle_history:
            self._angle_history[angle_name] = deque(maxlen=self._window_size)
        self._angle_history[angle_name].append(angle)
        angles[angle_name] = np.mean(self._angle_history[angle_name])

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
                
                # The sign of the appropriate component tells us if the angle is > 180
                # Choose component based on camera angle for better accuracy
                if self._camera_angle == "side_view":
                    # For side view, use Z-component
                    if len(cross_product.shape) == 0:  # scalar case
                        direction_indicator = cross_product
                    else:
                        direction_indicator = cross_product[2] if len(cross_product) > 2 else cross_product[0]
                else:
                    # For front/angled view, use Y-component
                    if len(cross_product.shape) == 0:  # scalar case
                        direction_indicator = cross_product
                    else:
                        direction_indicator = cross_product[1] if len(cross_product) > 1 else cross_product[0]
                
                if direction_indicator < 0:
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