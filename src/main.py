import argparse
from core.trainer import VirtualCalisthenicsTrainer


def main():
    """Main entry point for the Virtual Calisthenics Trainer."""
    parser = argparse.ArgumentParser(description="Virtual Calisthenics Trainer")
    parser.add_argument(
        "--exercise",
        type=str,
        default="pushup",
        choices=["pushup"],
        help="Type of exercise to analyze"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID"
    )
    
    args = parser.parse_args()
    
    # Create and start trainer
    trainer = VirtualCalisthenicsTrainer(exercise_type=args.exercise)
    trainer.start(camera_id=args.camera)


if __name__ == "__main__":
    main() 