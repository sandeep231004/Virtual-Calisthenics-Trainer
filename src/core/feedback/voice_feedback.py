import time
from typing import Dict, List, Optional
import threading
import queue
import pyttsx3


import time

class VoiceFeedback:
    """Voice feedback system for exercise form correction."""

    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Initialize the voice feedback system.

        Args:
            rate: Speech rate (words per minute)
            volume: Speech volume (0.0 to 1.0)
        """
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

        self._tts_queue = queue.Queue()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()
        
        self.last_feedback_time = 0
        self.feedback_cooldown = 7.0  # seconds (increased to reduce repetition)
        self.feedback_queue = []
        self._last_feedback_message = None
        self._last_violation = None
        self._violation_persist_count = 0
        self._violation_debounce_threshold = 2  # frames
        
        # Feedback messages for different violations
        self.feedback_messages = {
            # Push-up violations
            "pushup_elbow_too_small": "Keep your elbows closer to your body",
            "pushup_elbow_too_large": "Go lower - chest closer to the ground",
            "pushup_body_not_straight": "Keep your core tight and body straight",
            
            # Squat violations
            "squat_knee_too_small": "Push your knees out over your toes",
            "squat_knee_too_large": "Don't go too deep",
            "squat_hip_too_small": "Keep your chest up and torso more upright",
            "squat_hip_too_large": "Don't lean forward too much",
            
            # General feedback
            "good_form": "Good form! Keep it up!",
            "rep_complete": "Rep complete!",
            "rest_needed": "Take a short rest",
            "exercise_complete": "Great work! Exercise complete!",
            "ambiguous_view": "Camera angle is ambiguous. Please adjust your position or camera.",
            "partial_view": "Partial view detected. Make sure your whole body is visible to the camera.",
            "unknown_view": "Unable to determine view. Please adjust your position or camera angle.",
        }
        
        self.last_message = None
        self.last_time = 0
        self.cooldown = 4  # seconds

    def generate_feedback(self, exercise_state) -> Optional[str]:
        """
        Generate voice feedback based on exercise state.

        Args:
            exercise_state: Current state of the exercise (ExerciseState object)

        Returns:
            Feedback message if any, None otherwise
        """
        current_time = time.time()

        # Avoid feedback spam
        if current_time - self.last_feedback_time < self.feedback_cooldown:
            return None

        # Handle unreliable analysis (ambiguous/partial/unknown view)
        if not exercise_state.analysis_reliable:
            err = exercise_state.error_message
            if err:
                ambiguous_map = {
                    "Ambiguous view": "Camera angle is ambiguous. Please adjust your position or camera.",
                    "partial view": "Partial view detected. Make sure your whole body is visible to the camera.",
                    "Unable to determine torso/shoulder ratio": "Camera cannot see enough of your body. Adjust your position or camera angle.",
                    "Some body parts are not visible": "Some body parts are not visible. Please adjust your position or camera.",
                }
                for key, msg in ambiguous_map.items():
                    if key.lower() in err.lower():
                        feedback = msg
                        break
                else:
                    feedback = err
                if feedback == self._last_feedback_message:
                    return None
                self._last_feedback_message = feedback
                self.last_feedback_time = current_time
                return feedback
            return None

        # Get violations
        violations = exercise_state.violations
        violation = violations[0] if violations else None
        # Debounce logic: only speak if violation persists for threshold frames
        if violation:
            if violation == self._last_violation:
                self._violation_persist_count += 1
            else:
                self._violation_persist_count = 1
                self._last_violation = violation
            if self._violation_persist_count < self._violation_debounce_threshold:
                return None
            # Always use the violation string as feedback if not in feedback_messages
            feedback = self.feedback_messages.get(violation, violation)
        else:
            self._violation_persist_count = 0
            self._last_violation = None
            # Positive feedback based on phase
            phase = exercise_state.phase
            if phase == "up":
                feedback = self.feedback_messages["good_form"]
            elif phase == "rest":
                feedback = self.feedback_messages["rest_needed"]
            else:
                return None

        # Only speak if feedback message changes
        if feedback == self._last_feedback_message:
            return None
        self._last_feedback_message = feedback
        self.last_feedback_time = current_time
        return feedback

    def speak(self, message: str) -> None:
        """
        Queue the given message to be spoken by the background TTS thread.

        Args:
            message: Message to speak
        """
        self._tts_queue.put(message)

    def _tts_worker(self):
        while True:
            msg = self._tts_queue.get()
            if msg is None:
                break  # Allow for clean shutdown if needed
            self.engine.say(msg)
            self.engine.runAndWait()

    def speak_async(self, message: str) -> None:
        """
        Queue the given message to be spoken asynchronously by the background TTS thread.

        Args:
            message: Message to speak
        """
        self.speak(message) 