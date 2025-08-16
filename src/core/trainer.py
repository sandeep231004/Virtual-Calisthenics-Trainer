from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .exercise_analysis.base_analyzer_robust import UserLevel
from .exercise_analysis.pushup_analyzer_base import PushupAnalyzer
from .exercise_analysis.squat_analyzer_base import SquatAnalyzer
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
            self.exercise_analyzer = PushupAnalyzer(user_level=user_level)
        elif exercise_type == "squat":
            self.exercise_analyzer = SquatAnalyzer(user_level=user_level)
        else:
            raise ValueError(f"Unsupported exercise type: {exercise_type}")
        
        # Initialize frame buffer for temporal analysis
        self.frame_buffer = deque(maxlen=30)  # 1-second buffer at 30fps
        
        # Initialize video capture
        self.cap = None
        self.is_running = False
        self.missing_landmarks_counter = 0
        self.missing_landmarks_threshold = 30  # ~1 second at 30fps

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
                try:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    result = self.process_frame(frame)
                    exercise_state = result.get("exercise_state")
                    if exercise_state is not None and not getattr(exercise_state, "analysis_reliable", True) and getattr(exercise_state, "error_message", None) == "skip_frame":
                        self.missing_landmarks_counter += 1
                        if self.missing_landmarks_counter >= self.missing_landmarks_threshold:
                            # Show user-friendly warning
                            warning_msg = "We can't see your full body. Please adjust your position or camera."
                            print(f"[UI WARNING] {warning_msg}")
                            cv2.putText(frame, warning_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.imshow('Virtual Calisthenics Trainer', frame)
                            cv2.waitKey(1)
                        continue
                    else:
                        self.missing_landmarks_counter = 0
                    self._display_results(frame, result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except KeyboardInterrupt:
                    print("\n[INFO] KeyboardInterrupt received. Exiting gracefully...")
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
            # Show blank/no-pose window for user feedback
            debug_frame = frame.copy()
            cv2.putText(debug_frame, 'No pose detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Debug - Landmarks', debug_frame)
            cv2.waitKey(1)
            return {"error": "No pose detected"}
        else:
            # Show detected landmarks in a dedicated window
            debug_frame = frame.copy()
            for name, coords in landmarks.items():
                x, y = int(coords[0] * frame.shape[1]), int(coords[1] * frame.shape[0])
                cv2.circle(debug_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Debug - Landmarks', debug_frame)
            cv2.waitKey(1)
        
        # Calculate angles
        angles = self.pose_detector.calculate_angles(landmarks)

        # --- Timed debug output ---
        import time
        if not hasattr(self, '_last_debug_print'):
            self._last_debug_print = 0
        now = time.time()
        if now - self._last_debug_print >= 2.0:
            print(f"[DEBUG] Landmarks detected: {len(landmarks)}")
            print('[DEBUG] Calculated angles:', angles)
            # Print missing angles
            expected_angles = getattr(self.pose_detector, 'EXPECTED_ANGLES', None)
            if expected_angles:
                missing = [a for a in expected_angles if a not in angles or angles[a] is None]
                if missing:
                    print('[DEBUG] Missing angles:', missing)
            # Print per-frame shoulder/torso metrics
            if hasattr(self.exercise_analyzer, 'get_last_body_metrics'):
                metrics = self.exercise_analyzer.get_last_body_metrics()
                if metrics and all(v is not None for v in metrics.values()):
                    print(f"[DEBUG] Body metrics: Shoulder width: {metrics['shoulder_width']:.3f}, Torso length: {metrics['torso_length']:.3f}, Ratio: {metrics['shoulder_torso_ratio']:.3f}")
            self._last_debug_print = now

        # --- Inject normalized shoulder/torso ratio ---
        # Try to get the analyzer's last computed ratio (if available)
        ratio = None
        if hasattr(self.exercise_analyzer, '_session_calibration'):
            torso_length = self.exercise_analyzer._session_calibration.avg_torso
            shoulder_width = self.exercise_analyzer._session_calibration.avg_shoulder
            if torso_length and torso_length > 0:
                ratio = shoulder_width / torso_length
        # If not available, fallback to None
        if ratio is not None:
            angles['shoulder_torso_ratio'] = ratio

        # Analyze exercise form
        exercise_state = self.exercise_analyzer.analyze_frame(landmarks, angles)

        # Generate feedback only if analysis is reliable
        feedback = None
        if exercise_state.analysis_reliable:
            feedback = self.voice_feedback.generate_feedback(exercise_state)
            if feedback:
                self.voice_feedback.speak_async(feedback)
        else:
            # Optionally, speak error/ambiguous feedback for user, but prevent spam
            if exercise_state.error_message:
                feedback = self.voice_feedback.generate_feedback(exercise_state)
                if feedback:
                    self.voice_feedback.speak_async(feedback)

        # Update frame buffer
        self.frame_buffer.append(exercise_state)
        
        return {
            "landmarks": landmarks,
            "angles": angles,
            "exercise_state": exercise_state,
            "feedback": feedback,
            "error_message": exercise_state.error_message if not exercise_state.analysis_reliable else None
        }

    def _display_results(self, frame: np.ndarray, result: Dict) -> None:
        """
        Display results on the frame.

        Args:
            frame: Input frame
            result: Processing results
        """
        # Show pose detection or analysis errors prominently
        error_msg = result.get("error") or result.get("error_message")
        if error_msg:
            print(f"[UI ERROR] {error_msg}")  # Print to terminal
            cv2.putText(
                frame,
                error_msg,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
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
            0.6,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Phase: {state.phase}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Reps: {state.rep_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        
        # Draw violations
        if state.violations:
            for idx, violation in enumerate(state.violations):
                print(f"[UI VIOLATION] {violation}")  # Print each violation to terminal
                if idx == 0:
                    # Show the first violation on the UI
                    cv2.putText(
                        frame,
                        f"Form: Incorrect - {violation}",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )
        else:
            # Only show encouragement if form is correct, analysis is reliable, and no missing data
            if state.analysis_reliable and not state.violations and not state.error_message:
                cv2.putText(
                    frame,
                    "Form: Correct",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    "Go on! You can proceed.",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "Form: Correct",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
        # Display shoulder-torso ratio if available
        if 'angles' in result and 'shoulder_torso_ratio' in result['angles']:
            ratio = result['angles']['shoulder_torso_ratio']
            cv2.putText(
                frame,
                f"Shoulder/Torso Ratio: {ratio:.2f}",
                (10, 180),  # Adjust position as needed
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
        # Display frame
        cv2.imshow("Virtual Calisthenics Trainer", frame) 