from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ultralytics import YOLO


DEFAULT_SOURCE = Path("videos/kabaddi.mp4")
DEFAULT_WEIGHTS = "yolov8n-pose.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8-Pose inference and export detections.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Image, video, folder, or camera source.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Pose model weights or a trained checkpoint.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size.")
    parser.add_argument("--device", default=None,
                        help="Device to run on, for example 0 or cpu.")
    parser.add_argument("--project", default="runs/pose_infer",
                        help="Output project directory.")
    parser.add_argument("--name", default="predict",
                        help="Run name inside the project directory.")
    parser.add_argument("--save-vis", action="store_true",
                        help="Save Ultralytics annotated outputs.")
    parser.add_argument("--track", action="store_true",
                        help="Run tracking so track IDs are included when available.")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        help="Tracker config to use when --track is set.")
    parser.add_argument("--output-json", type=Path, default=Path(
        "runs/pose_infer/predictions.json"), help="Path to write JSON results.")
    return parser.parse_args()


def _to_float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def _frame_sort_key(path: Path) -> tuple[int, Any]:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match is not None:
        return (0, int(match.group(1)))
    return (1, path.name.lower())


def _collect_source_paths(source: Path) -> list[str]:
    if source.is_file():
        return [str(source)]

    if source.is_dir():
        image_paths = [
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        image_paths.sort(key=_frame_sort_key)
        return [str(path) for path in image_paths]

    return [str(source)]


def _iter_results(
    model: YOLO,
    source_paths: list[str],
    *,
    track: bool,
    tracker: str,
    predict_kwargs: dict[str, Any],
):
    for source_path in source_paths:
        if track:
            results = model.track(
                source=source_path,
                persist=True,
                tracker=tracker,
                stream=False,
                **predict_kwargs,
            )
        else:
            results = model.predict(
                source=source_path, stream=False, **predict_kwargs)

        for result in results:
            yield result


def _frame_number_from_path(path_text: str) -> int | None:
    match = re.search(r"(\d+)(?!.*\d)", Path(path_text).stem)
    return int(match.group(1)) if match is not None else None


def main() -> None:
    args = parse_args()

    if not args.source.exists() and not str(args.source).isdigit():
        raise FileNotFoundError(f"Source not found: {args.source}")

    model = YOLO(args.weights)
    source_paths = _collect_source_paths(args.source)
    predict_kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "save": args.save_vis,
        "exist_ok": True,
        "verbose": False,
    }

    results = _iter_results(
        model,
        source_paths,
        track=args.track,
        tracker=args.tracker,
        predict_kwargs=predict_kwargs,
    )

    output_path = args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    for frame_index, result in enumerate(results):
        result_path = str(getattr(result, "path", args.source))
        frame_data: dict[str, Any] = {
            "frame_index": frame_index,
            "frame_path": result_path,
            "frame_number": _frame_number_from_path(result_path),
            "source": str(args.source),
            "detections": [],
        }

        boxes = result.boxes
        keypoints = getattr(result, "keypoints", None)

        if boxes is None or len(boxes) == 0:
            frames.append(frame_data)
            continue

        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else [
            None] * len(xyxy)
        classes = boxes.cls.cpu().tolist() if boxes.cls is not None else [
            None] * len(xyxy)
        track_ids = boxes.id.cpu().tolist() if getattr(
            boxes, "id", None) is not None else [None] * len(xyxy)
        keypoint_values = keypoints.data.cpu().tolist() if keypoints is not None and getattr(
            keypoints, "data", None) is not None else [None] * len(xyxy)

        for index, bbox in enumerate(xyxy):
            detection = {
                "bbox": _to_float_list(bbox),
                "confidence": None if confidences[index] is None else float(confidences[index]),
                "class_id": None if classes[index] is None else int(classes[index]),
                "track_id": None if track_ids[index] is None else int(track_ids[index]),
                "keypoints": keypoint_values[index],
            }
            frame_data["detections"].append(detection)

        frames.append(frame_data)

    payload = {
        "weights": args.weights,
        "source": str(args.source),
        "source_paths": source_paths if len(source_paths) > 1 else None,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "track": args.track,
        "frames": frames,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved pose predictions to {output_path}")


if __name__ == "__main__":
    main()
