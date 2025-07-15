import argparse
import sys
import traceback
import os
import cv2
from core.trainer import VirtualCalisthenicsTrainer
# Now uses PushupAnalyzerBase from pushup_analyzer_base.py for all pushup logic


def main():
    """Main entry point for the Virtual Calisthenics Trainer."""
    parser = argparse.ArgumentParser(description="Virtual Calisthenics Trainer")
    parser.add_argument(
        "--exercise",
        type=str,
        default="pushup",
        choices=["pushup", "squat"],
        help="Type of exercise to analyze"
    )
    parser.add_argument(
        "--user_level",
        type=str,
        default="beginner",
        choices=["beginner", "intermediate", "advanced"],
        help="User level (beginner/intermediate/advanced)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test component initialization without running the trainer"
    )
    
    args = parser.parse_args()
    from core.exercise_analysis.base_analyzer_robust import UserLevel
    user_level = UserLevel[args.user_level.upper()]

    try:
        print("Initializing Virtual Calisthenics Trainer...")
        # Create and start trainer
        trainer = VirtualCalisthenicsTrainer(exercise_type=args.exercise, user_level=user_level)
        print("Trainer created successfully, starting...")
        trainer.start(camera_id=args.camera)
    except Exception as e:
        print(f"Error starting trainer: {e}")
        traceback.print_exc()
        sys.exit(1)


def main_video():
    parser = argparse.ArgumentParser(description="Virtual Calisthenics Trainer - Video Analysis Mode")
    parser.add_argument('--video', type=str, required=True, help='Path to the video file to analyze')
    parser.add_argument('--exercise', type=str, default='pushup', choices=["pushup", "squat"], help='Exercise type (default: pushup)')
    parser.add_argument('--user_level', type=str, default='beginner', choices=["beginner", "intermediate", "advanced"], help='User level (beginner/intermediate/advanced)')
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"Video file not found: {video_path}")
        return

    print(f"Initializing Virtual Calisthenics Trainer (Video Mode)...")
    from core.trainer import VirtualCalisthenicsTrainer
    from core.exercise_analysis.base_analyzer_robust import UserLevel
    user_level = UserLevel[args.user_level.upper()]
    trainer = VirtualCalisthenicsTrainer(exercise_type=args.exercise, user_level=user_level)
    print("Trainer created successfully, starting video analysis...")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        result = trainer.process_frame(frame)
        # Optionally, display the frame and overlay feedback
        cv2.imshow('Virtual Calisthenics Trainer - Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video analysis complete. Processed {frame_count} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual Calisthenics Trainer")
    parser.add_argument('--mode', type=str, choices=['camera', 'video'], default='camera', help='Run mode: camera (default) or video')
    parser.add_argument('--video', type=str, help='Path to the video file to analyze (required if mode=video)')
    parser.add_argument('--exercise', type=str, default='pushup', choices=["pushup", "squat"], help='Exercise type (default: pushup)')
    parser.add_argument('--user_level', type=str, default='beginner', choices=["beginner", "intermediate", "advanced"], help='User level (beginner/intermediate/advanced)')
    args = parser.parse_args()

    if args.mode == 'video':
        if not args.video:
            print("Error: --video argument is required when mode is 'video'.")
            exit(1)
        main_video_func = globals().get('main_video')
        if main_video_func is None:
            print("Error: main_video function not found.")
            exit(1)
        # Pass args to main_video
        import sys
        sys.argv = [sys.argv[0], '--video', args.video, '--exercise', args.exercise, '--user_level', args.user_level]
        main_video_func()
    else:
        # Pass user_level to main
        import sys
        sys.argv = [sys.argv[0], '--exercise', args.exercise, '--user_level', args.user_level, '--camera', str(args.camera) if hasattr(args, 'camera') else '0']
        main() 