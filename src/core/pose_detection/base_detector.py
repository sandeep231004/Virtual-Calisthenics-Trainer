from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class BasePoseDetector(ABC):
    """Base class for pose detection implementations."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[Dict[str, List[float]]]]:
        """
        Detect pose landmarks in the given frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            Tuple containing:
            - Boolean indicating if detection was successful
            - Dictionary of landmark coordinates (if successful) or None
        """
        pass

    @abstractmethod
    def get_landmark_names(self) -> List[str]:
        """
        Get the list of landmark names that this detector provides.

        Returns:
            List of landmark names
        """
        pass

    @abstractmethod
    def calculate_angles(self, landmarks: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate joint angles from landmarks.

        Args:
            landmarks: Dictionary of landmark coordinates

        Returns:
            Dictionary of angle names and their values in degrees
        """
        pass 