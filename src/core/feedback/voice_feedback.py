import time
from typing import Dict, List, Optional

import pyttsx3


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
        
        self.last_feedback_time = 0
        self.feedback_cooldown = 3.0  # seconds
        self.feedback_queue = []
        
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
            "exercise_complete": "Great work! Exercise complete!"
        }

    def generate_feedback(self, exercise_state: Dict) -> Optional[str]:
        """
        Generate voice feedback based on exercise state.

        Args:
            exercise_state: Current state of the exercise

        Returns:
            Feedback message if any, None otherwise
        """
        current_time = time.time()
        
        # Avoid feedback spam
        if current_time - self.last_feedback_time < self.feedback_cooldown:
            return None
            
        # Get violations
        violations = exercise_state.get("violations", [])
        
        # Generate appropriate feedback
        if violations:
            # Prioritize the first violation
            violation = violations[0]
            feedback = self.feedback_messages.get(violation, "Check your form")
        else:
            # Positive feedback based on phase
            phase = exercise_state.get("phase", "")
            if phase == "up":
                feedback = self.feedback_messages["good_form"]
            elif phase == "rest":
                feedback = self.feedback_messages["rest_needed"]
            else:
                return None
        
        # Update last feedback time
        self.last_feedback_time = current_time
        
        return feedback

    def speak(self, message: str) -> None:
        """
        Speak the given message.

        Args:
            message: Message to speak
        """
        self.engine.say(message)
        self.engine.runAndWait()

    def speak_async(self, message: str) -> None:
        """
        Speak the given message asynchronously.

        Args:
            message: Message to speak
        """
        self.engine.say(message)
        self.engine.runAndWait()

    def add_feedback_message(self, key: str, message: str) -> None:
        """
        Add a new feedback message.

        Args:
            key: Message key
            message: Message text
        """
        self.feedback_messages[key] = message

    def set_feedback_cooldown(self, seconds: float) -> None:
        """
        Set the feedback cooldown period.

        Args:
            seconds: Cooldown period in seconds
        """
        self.feedback_cooldown = seconds 