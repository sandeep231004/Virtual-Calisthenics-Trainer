"""
pose_utils.py - Shared utilities for pose estimation, smoothing, and geometry.
"""
import numpy as np
from typing import Dict, List, Optional
from collections import deque

# --- Math & Geometry Utilities ---
def calculate_angle(a: List[float], b: List[float], c: List[float], full_range: bool = False) -> float:
    """
    Calculate the angle between three points with robust error handling.
    Supports both 0-180 degree range (default) and 0-360 degree range.
    
    Point ordering convention:
    - a: First point (e.g., shoulder for elbow angle)
    - b: Middle point (e.g., elbow for elbow angle) 
    - c: Last point (e.g., wrist for elbow angle)
    - The angle is calculated at point 'b' between vectors 'ba' and 'bc'
    
    Args:
        a: First point coordinates [x, y, z, visibility]
        b: Middle point coordinates [x, y, z, visibility] - angle is calculated here
        c: Last point coordinates [x, y, z, visibility]
        full_range: If True, returns angle in 0-360 range, otherwise 0-180 range
    Returns:
        Angle in degrees, or NaN if calculation is not possible
    """
    try:
        a = np.array([a[0], a[1], a[2]])
        b = np.array([b[0], b[1], b[2]])
        c = np.array([c[0], c[1], c[2]])
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return np.nan
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        if full_range:
            cross_product = np.cross(ba, bc)
            direction_indicator = cross_product[2] if cross_product.shape[-1] > 2 else cross_product[0]
            if direction_indicator < 0:
                angle = 360 - angle
            if not (0 <= angle <= 360):
                return np.nan
        else:
            if not (0 <= angle <= 180):
                return np.nan
        return angle
    except Exception as e:
        return np.nan

def calculate_length(a: List[float], b: List[float]) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))
    
def calculate_pushup_specific_angles(landmarks: Dict[str, List[float]], angles: Dict[str, float], min_visibility: float = 0.3, temporal_smoother=None) -> None:
    # temporal_smoother: Optional[Callable[[str, float], float]]
    left_visibility = get_side_visibility(landmarks, "left")
    right_visibility = get_side_visibility(landmarks, "right")
    primary_side = "left" if left_visibility > right_visibility else "right"
    
    # 1. Body-floor angle - FIXED: Use consistent three-point calculation
    if can_calculate_body_alignment(landmarks, primary_side, min_visibility):
        try:
            # Use three points: ankle, hip, shoulder to calculate body alignment angle
            # This is more consistent with other angle calculations
            ankle = landmarks[f"{primary_side}_ankle"]
            hip = landmarks[f"{primary_side}_hip"]
            shoulder = landmarks[f"{primary_side}_shoulder"]
            
            # Calculate angle between vertical line and body line
            # Use a virtual point above the ankle to create a vertical reference
            vertical_point = [ankle[0], ankle[1] - 1.0, ankle[2]]  # Point 1 unit above ankle
            angle = calculate_angle(vertical_point, ankle, shoulder, False)
            
            # Body-floor angle should be 0-90 degrees for proper pushup form
            if 0 <= angle <= 90:
                if temporal_smoother:
                    angle = temporal_smoother("body_floor_angle", angle)
                angles["body_floor_angle"] = angle
        except Exception as e:
            print(f"Error calculating body-floor angle: {str(e)}")
    
    # 2. Hip angle - FIXED: Improved error handling
    if can_calculate_hip_angle(landmarks, primary_side, min_visibility):
        try:
            angle = calculate_angle(
                landmarks[f"{primary_side}_shoulder"],
                landmarks[f"{primary_side}_hip"],
                landmarks[f"{primary_side}_ankle"],
                False
            )
            # Hip angle should be 120-200 degrees for proper form
            if 120 <= angle <= 200:
                if temporal_smoother:
                    angle = temporal_smoother("hip_angle", angle)
                angles["hip_angle"] = angle
        except Exception as e:
            print(f"Error calculating hip angle: {str(e)}")
    
    # 3. Neck angle - FIXED: Improved error handling
    if can_calculate_neck_angle(landmarks, primary_side, min_visibility, use_nose=True):
        try:
            shoulder = landmarks[f"{primary_side}_shoulder"]
            hip = landmarks[f"{primary_side}_hip"]
            nose = landmarks["nose"]
            angle = calculate_angle(hip, shoulder, nose, False)
            # Neck angle should be 100-180 degrees for proper form
            if 100 <= angle <= 180:
                if temporal_smoother:
                    angle = temporal_smoother("neck_angle", angle)
                angles["neck_angle"] = angle
        except Exception as e:
            print(f"Error calculating neck angle: {str(e)}")
    
    # 4. Wrist alignment - FIXED: Improved error handling
    for side in ["left", "right"]:
        if can_calculate_wrist_angle(landmarks, side, min_visibility):
            try:
                if f"{side}_index" in landmarks:
                    angle = calculate_angle(
                        landmarks[f"{side}_elbow"],
                        landmarks[f"{side}_wrist"],
                        landmarks[f"{side}_index"],
                        False
                    )
                    # Wrist angle should be 140-180 degrees for proper alignment
                    if 140 <= angle <= 180:
                        if temporal_smoother:
                            angle = temporal_smoother(f"{side}_wrist_angle", angle)
                        angles[f"{side}_wrist_angle"] = angle
            except Exception as e:
                print(f"Error calculating {side} wrist angle: {str(e)}")
    
    # 5. Elbow flare - FIXED: Improved error handling
    if can_calculate_elbow_flare(landmarks, min_visibility):
        for side in ["left", "right"]:
            try:
                if all(f"{side}_{part}" in landmarks for part in ["shoulder", "elbow", "hip"]):
                    angle = calculate_angle(
                        landmarks[f"{side}_hip"],
                        landmarks[f"{side}_shoulder"],
                        landmarks[f"{side}_elbow"],
                        False
                    )
                    # Elbow flare should be 30-120 degrees for proper form
                    if 30 <= angle <= 120:
                        if temporal_smoother:
                            angle = temporal_smoother(f"{side}_elbow_flare", angle)
                        angles[f"{side}_elbow_flare"] = angle
            except Exception as e:
                print(f"Error calculating {side} elbow flare: {str(e)}")

def calculate_asymmetry_metrics(landmarks: Dict[str, List[float]], angles: Dict[str, float]) -> None:
    """
    Calculate comprehensive asymmetry metrics for form analysis.
    Includes both coordinate-based and angle-based asymmetry measures.
    """
    # 1. Angle-based asymmetry (existing logic, improved)
    # Elbow asymmetry
    if "left_elbow" in angles and "right_elbow" in angles:
        if not (np.isnan(angles["left_elbow"]) or np.isnan(angles["right_elbow"])):
            asymmetry = abs(angles["left_elbow"] - angles["right_elbow"])
            angles["elbow_asymmetry"] = asymmetry
    # Shoulder asymmetry (angle-based)
    if "left_shoulder" in angles and "right_shoulder" in angles:
        if not (np.isnan(angles["left_shoulder"]) or np.isnan(angles["right_shoulder"])):
            asymmetry = abs(angles["left_shoulder"] - angles["right_shoulder"])
            angles["shoulder_asymmetry"] = asymmetry
    # Wrist asymmetry
    if "left_wrist_angle" in angles and "right_wrist_angle" in angles:
        if not (np.isnan(angles["left_wrist_angle"]) or np.isnan(angles["right_wrist_angle"])):
            asymmetry = abs(angles["left_wrist_angle"] - angles["right_wrist_angle"])
            angles["wrist_asymmetry"] = asymmetry
    
    # 2. Coordinate-based asymmetry (for view detection and form validation)
    try:
        # Shoulder symmetry (Y-coordinate difference)
        if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
            shoulder_symmetry = abs(landmarks["left_shoulder"][1] - landmarks["right_shoulder"][1])
            angles["shoulder_symmetry"] = shoulder_symmetry
        
        # Hip symmetry (Y-coordinate difference)
        if "left_hip" in landmarks and "right_hip" in landmarks:
            hip_symmetry = abs(landmarks["left_hip"][1] - landmarks["right_hip"][1])
            angles["hip_symmetry"] = hip_symmetry
        
        # Wrist distance (X-coordinate difference)
        if "left_wrist" in landmarks and "right_wrist" in landmarks:
            wrist_distance = abs(landmarks["left_wrist"][0] - landmarks["right_wrist"][0])
            angles["wrist_distance"] = wrist_distance
            
    except Exception as e:
        print(f"Error calculating coordinate-based asymmetry: {str(e)}")

def get_side_visibility(landmarks: Dict[str, List[float]], side: str) -> float:
    side_landmarks = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", f"{side}_hip", f"{side}_ankle"]
    visibilities = [landmarks[l][3] for l in side_landmarks if l in landmarks]
    return np.mean(visibilities) if visibilities else 0.0

def can_calculate_body_alignment(landmarks: Dict[str, List[float]], side: str, min_visibility: float = 0.3) -> bool:
    required = [f"{side}_ankle", f"{side}_hip", f"{side}_shoulder"]
    return all(l in landmarks and landmarks[l][3] >= min_visibility for l in required)

def can_calculate_hip_angle(landmarks: Dict[str, List[float]], side: str, min_visibility: float = 0.3) -> bool:
    required = [f"{side}_shoulder", f"{side}_hip", f"{side}_ankle"]
    return all(l in landmarks and landmarks[l][3] >= min_visibility for l in required)

def can_calculate_neck_angle(landmarks: Dict[str, List[float]], side: str, min_visibility: float = 0.3, use_nose=True) -> bool:
    required = [f"{side}_shoulder", f"{side}_hip", "nose" if use_nose else f"{side}_ear"]
    return all(l in landmarks and landmarks[l][3] >= min_visibility for l in required)

def can_calculate_wrist_angle(landmarks: Dict[str, List[float]], side: str, min_visibility: float = 0.3) -> bool:
    required = [f"{side}_elbow", f"{side}_wrist", f"{side}_index"]
    return all(l in landmarks and landmarks[l][3] >= min_visibility for l in required)

def can_calculate_elbow_flare(landmarks: Dict[str, List[float]], min_visibility: float = 0.3) -> bool:
    for side in ["left", "right"]:
        required = [f"{side}_hip", f"{side}_shoulder", f"{side}_elbow"]
        if all(l in landmarks and landmarks[l][3] >= min_visibility for l in required):
            return True
    return False

def check_landmark_visibility(landmarks: Dict[str, List[float]], names: List[str], min_visibility: float = 0.5) -> bool:
    """Check if all named landmarks are present and visible above threshold."""
    return all(
        name in landmarks and len(landmarks[name]) > 3 and landmarks[name][3] > min_visibility
        for name in names
    )

def calculate_torso_length(landmarks: Dict[str, List[float]]) -> Optional[float]:
    """
    Calculate torso length robustly:
    - If all four (left/right shoulder & hip) are present, return average of both sides.
    - If only left or only right side is present, return that side's length.
    - If neither side is available, return None.
    """
    left_ready = "left_shoulder" in landmarks and "left_hip" in landmarks
    right_ready = "right_shoulder" in landmarks and "right_hip" in landmarks
    left_length = calculate_length(landmarks["left_shoulder"], landmarks["left_hip"]) if left_ready else None
    right_length = calculate_length(landmarks["right_shoulder"], landmarks["right_hip"]) if right_ready else None
    if left_length is not None and right_length is not None:
        return (left_length + right_length) / 2
    elif left_length is not None:
        return left_length
    elif right_length is not None:
        return right_length
    else:
        return None


def calculate_body_alignment_score(landmarks: Dict[str, List[float]], angles: Dict[str, float]) -> Optional[float]:
    """
    Calculate overall body alignment score based on multiple factors.
    Returns a score between 0 (poor alignment) and 1 (perfect alignment).
    """
    try:
        alignment_factors = []
        
        # 1. Body-floor angle alignment (0-90 degrees is good)
        if "body_floor_angle" in angles and not np.isnan(angles["body_floor_angle"]):
            angle = angles["body_floor_angle"]
            # Score is highest when angle is close to 0 (perfect vertical)
            if 0 <= angle <= 90:
                alignment_factors.append(1.0 - (angle / 90.0))
        
        # 2. Hip angle alignment (160-180 degrees is good)
        if "hip_angle" in angles and not np.isnan(angles["hip_angle"]):
            angle = angles["hip_angle"]
            if 160 <= angle <= 180:
                alignment_factors.append((angle - 160) / 20.0)
            elif 120 <= angle < 160:
                alignment_factors.append(0.5)  # Moderate alignment
            else:
                alignment_factors.append(0.0)  # Poor alignment
        
        # 3. Neck angle alignment (150-170 degrees is good)
        if "neck_angle" in angles and not np.isnan(angles["neck_angle"]):
            angle = angles["neck_angle"]
            if 150 <= angle <= 170:
                alignment_factors.append(1.0 - abs(angle - 160) / 10.0)
            elif 100 <= angle < 150:
                alignment_factors.append(0.5)  # Moderate alignment
            else:
                alignment_factors.append(0.0)  # Poor alignment
        
        # 4. Shoulder symmetry (lower values are better)
        if "shoulder_symmetry" in angles and not np.isnan(angles["shoulder_symmetry"]):
            symmetry = angles["shoulder_symmetry"]
            # Normalize: 0.0 = perfect symmetry, 0.1+ = poor symmetry
            if symmetry <= 0.05:
                alignment_factors.append(1.0)
            elif symmetry <= 0.1:
                alignment_factors.append(1.0 - (symmetry - 0.05) / 0.05)
            else:
                alignment_factors.append(0.0)
        
        # 5. Hip symmetry (lower values are better)
        if "hip_symmetry" in angles and not np.isnan(angles["hip_symmetry"]):
            symmetry = angles["hip_symmetry"]
            # Normalize: 0.0 = perfect symmetry, 0.08+ = poor symmetry
            if symmetry <= 0.04:
                alignment_factors.append(1.0)
            elif symmetry <= 0.08:
                alignment_factors.append(1.0 - (symmetry - 0.04) / 0.04)
            else:
                alignment_factors.append(0.0)
        
        if alignment_factors:
            return np.mean(alignment_factors)
        else:
            return None
            
    except Exception as e:
        print(f"Error calculating body alignment score: {str(e)}")
        return None
