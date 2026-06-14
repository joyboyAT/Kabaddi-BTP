from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_DATA = Path("data_pose.yaml")
DEFAULT_WEIGHTS = "yolov8n-pose.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8-Pose model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Path to the pose data YAML.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Pretrained pose weights to start from.")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default=None,
                        help="Device to train on, for example 0 or cpu.")
    parser.add_argument("--project", default="runs/pose",
                        help="Output project directory.")
    parser.add_argument("--name", default="train",
                        help="Run name inside the project directory.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Pose data config not found: {args.data}")

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
