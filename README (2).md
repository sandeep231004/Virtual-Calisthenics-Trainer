# Virtual Calisthenics Trainer

**Virtual Calisthenics Trainer** is a Python-based application that provides real-time feedback and evaluation of exercise form during calisthenics workouts. The system leverages computer vision and voice feedback technologies to help users maintain proper technique and reduce injury risk.

## Features

- **Real-Time Pose Estimation:** Uses computer vision models to analyze live video input and compute joint angles.
- **Form Violation Detection:** Detects multiple exercise form violations (such as hip drop and knee valgus) based on pose analysis.
- **Immediate Voice Feedback:** Integrates text-to-speech to provide corrective cues during workouts.
- **Modular Architecture:** Clear separation for pose detection, feedback delivery, and evaluation logic.
- **Evaluation Protocols:** Includes mechanisms for assessing detection accuracy and minimizing false positives.

## Technologies Used

- Python
- MediaPipe (for pose estimation)
- OpenCV (computer vision processing)
- pyttsx3 (text-to-speech feedback)

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/sandeep231004/Virtual-Calisthenics-Trainer.git
    cd Virtual-Calisthenics-Trainer
    ```
2. Install dependencies (Python >= 3.8 recommended):
    ```bash
    pip install -r requirements.txt
    ```
3. (Optional) Ensure your webcam is connected and working for real-time video input.

## Usage

1. Run the main application:
    ```bash
    python main.py
    ```
2. Follow on-screen instructions to start your workout.
3. Listen for immediate voice feedback and adjust your form in real time.

## Repository Structure

- `main.py` – Entry point for the application
- `pose_estimation/` – Pose detection logic and models
- `feedback/` – Voice feedback and form violation alerts
- `evaluation/` – Accuracy assessment and protocol scripts

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.

## License

This project currently has no license specified.

## Author

[Sandeep231004](https://github.com/sandeep231004)

---
