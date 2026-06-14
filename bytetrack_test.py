from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ByteTrack on extracted frames and save an annotated video."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/train-4/weights/best.pt"),
        help="Path to the trained YOLO weights.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("frames"),
        help="Folder containing extracted frame images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/bytetrack_test"),
        help="Directory where the annotated video will be saved.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a progress message every N frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5000,
        help="Maximum number of frames to process from the frames folder.",
    )
    return parser.parse_args()


def frame_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def load_frame_paths(source: Path, max_frames: int) -> list[Path]:
    if not source.is_dir():
        raise ValueError(
            f"Expected a frames folder for --source, got: {source}")

    frame_paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        frame_paths.extend(source.glob(pattern))

    frame_paths = sorted(frame_paths, key=frame_sort_key)
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]

    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in folder: {source}")

    return frame_paths


def color_from_id(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return 0, 255, 0

    palette = (
        (255, 87, 51),
        (51, 153, 255),
        (255, 204, 51),
        (102, 255, 102),
        (255, 102, 204),
        (0, 204, 204),
    )
    return palette[track_id % len(palette)]


def annotate_frame(frame, result, class_names) -> "cv2.Mat":
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return frame

    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    track_ids = None
    if boxes.id is not None:
        track_ids = boxes.id.cpu().numpy().astype(int)

    for index, box in enumerate(xyxy):
        x1, y1, x2, y2 = box.astype(int)
        cls_id = int(classes[index])
        track_id = int(track_ids[index]) if track_ids is not None else None
        label = class_names.get(cls_id, str(cls_id))
        text = f"ID {track_id} {label}" if track_id is not None else label
        color = color_from_id(track_id if track_id is not None else cls_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )
        text_y = max(y1 - 10, text_height + 10)
        cv2.rectangle(
            frame,
            (x1, text_y - text_height - baseline),
            (x1 + text_width + 8, text_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return frame


def main() -> None:
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    frame_paths = load_frame_paths(args.source, args.max_frames)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.source.name}_bytetrack.mp4"

    model = YOLO(str(args.weights))
    total_frames = len(frame_paths)
    fps = 25.0
    writer = None
    frame_index = 0

    print(f"Tracking first {total_frames} frames from {args.source}")

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read frame: {frame_path}")

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                verbose=False,
            )[0]

            frame_index += 1
            annotated = annotate_frame(frame, result, model.names)

            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

            writer.write(annotated)
            cv2.imshow("ByteTrack Preview", annotated)

            if args.progress_every > 0 and frame_index % args.progress_every == 0:
                percent = (frame_index / total_frames) * 100
                print(
                    f"Processed {frame_index}/{total_frames} frames ({percent:.1f}%)"
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Preview closed by user.")
                break
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Annotated tracking video saved to: {output_path}")


if __name__ == "__main__":
    main()
