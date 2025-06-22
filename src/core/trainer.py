from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .exercise_analysis.base_analyzer_robust import UserLevel
from .exercise_analysis.pushup_analyzer import PushupAnalyzer
from .feedback.voice_feedback import VoiceFeedback
from .pose_detection.mediapipe_detector2 import MediaPipePoseDetector


class VirtualCalisthenicsTrainer:
    """Main class for the Virtual Calisthenics Trainer."""

    def __init__(self, exercise_type: str = "pushup", user_level: UserLevel = UserLevel.BEGINNER):
        """
        Initialize the trainer.

        Args:
            exercise_type: Type of exercise to analyze
            user_level: User's experience level
        """
        # Initialize components
        self.pose_detector = MediaPipePoseDetector()
        self.voice_feedback = VoiceFeedback()
        
        # Initialize exercise analyzer
        if exercise_type == "pushup":
            self.exercise_analyzer = PushupAnalyzer(user_level)
        else:
            raise ValueError(f"Unsupported exercise type: {exercise_type}")
        
        # Initialize frame buffer for temporal analysis
        self.frame_buffer = deque(maxlen=30)  # 1-second buffer at 30fps
        
        # Initialize video capture
        self.cap = None
        self.is_running = False

    def set_user_level(self, user_level: UserLevel) -> None:
        """
        Update the user level for exercise analysis.

        Args:
            user_level: New user level to set
        """
        self.exercise_analyzer.set_user_level(user_level)

    def start(self, camera_id: int = 0) -> None:
        """
        Start the trainer with the specified camera.

        Args:
            camera_id: Camera device ID
        """
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Process frame
                result = self.process_frame(frame)
                
                # Display results
                self._display_results(frame, result)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the trainer and release resources."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame.

        Args:
            frame: Input frame

        Returns:
            Dictionary containing processing results
        """
        # Detect pose
        success, landmarks = self.pose_detector.detect(frame)
        if not success:
            return {"error": "No pose detected"}
            
        # Calculate angles
        angles = self.pose_detector.calculate_angles(landmarks)
        
        # Analyze exercise form
        exercise_state = self.exercise_analyzer.analyze_frame(landmarks, angles)
        
        # Generate feedback
        feedback = self.voice_feedback.generate_feedback(exercise_state.__dict__)
        if feedback:
            self.voice_feedback.speak_async(feedback)
            
        # Update frame buffer
        self.frame_buffer.append(exercise_state)
        
        return {
            "landmarks": landmarks,
            "angles": angles,
            "exercise_state": exercise_state,
            "feedback": feedback
        }

    def _display_results(self, frame: np.ndarray, result: Dict) -> None:
        """
        Display results on the frame.

        Args:
            frame: Input frame
            result: Processing results
        """
        if "error" in result:
            cv2.putText(
                frame,
                result["error"],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            cv2.imshow("Virtual Calisthenics Trainer", frame)
            return
            
        # Draw landmarks
        landmarks = result["landmarks"]
        for name, coords in landmarks.items():
            x, y = int(coords[0] * frame.shape[1]), int(coords[1] * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
        # Draw exercise state
        state = result["exercise_state"]
        cv2.putText(
            frame,
            f"Exercise: {state.name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Phase: {state.phase}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Reps: {state.rep_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Draw violations
        if state.violations:
            cv2.putText(
                frame,
                f"Form: Incorrect - {state.violations[0]}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        else:
            cv2.putText(
                frame,
                "Form: Correct",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
        # Display frame
        cv2.imshow("Virtual Calisthenics Trainer", frame) 