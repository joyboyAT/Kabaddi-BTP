from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker


DEFAULT_FRAMES_DIR = Path("frames")
DEFAULT_ANNOTATIONS_DIR = Path("annotations")
DEFAULT_WEIGHTS = "yolov8n-pose.pt"
DEFAULT_OUTPUT_JSON = Path("runs/pose_infer/annotation_guided_predictions.json")
DEFAULT_VIS_DIR = Path("runs/pose_infer/annotation_guided_frames_full_vis")
DEFAULT_POSE_YAML = Path("data_pose.yaml")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = [
    "Dubki",
    "Back Kick",
    "Hand Touch",
    "Lion Jump",
    "Side Kick",
    "Toe Touch",
    "idle",
    "Ankle Hold",
    "Thigh Hold",
    "Waist Hold",
    "Block",
    "Dash",
    "Chain Tackle",
    "bonus",
    "w",
    "d",
    "idlew",
    "tackle",
    "toe touch",
]
PLAYER_ACTION_CLASS_IDS = set(range(0, 14))


@dataclass
class AnnotationBox:
    class_id: int | None
    class_name: str | None
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0


class TrackerDetections:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> "TrackerDetections":
        return TrackerDetections(self.xyxy[index], self.conf[index], self.cls[index])

    @property
    def xywh(self) -> np.ndarray:
        xywh = self.xyxy.copy()
        xywh[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        xywh[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
        xywh[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
        xywh[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
        return xywh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pose inference inside annotation boxes, ByteTrack the annotated players, and render visuals."
    )
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--drive", choices=("annotations", "frames"), default="annotations",
                        help="Use annotations as the source list, or scan every frame.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--pose-yaml", type=Path, default=DEFAULT_POSE_YAML)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_VIS_DIR)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.15)
    parser.add_argument("--keypoint-conf", type=float, default=0.15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--roi-pad", type=float, default=0.08,
                        help="Fractional padding added around each annotation box before pose inference.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all frames.")
    parser.add_argument("--include-classes", nargs="*", default=None)
    parser.add_argument("--exclude-classes", nargs="*", default=None)
    parser.add_argument("--players-only", action="store_true",
                        help="Keep only player action annotation classes.")
    parser.add_argument("--hide-boxes", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    return parser.parse_args()


def frame_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match is not None:
        return (0, int(match.group(1)))
    return (1, path.name.lower())


def frame_number_from_path(path: Path) -> int | None:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    return int(match.group(1)) if match is not None else None


def resolve_class_token(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    try:
        return int(token)
    except ValueError:
        lower_map = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}
        return lower_map.get(token.lower())


def class_name(class_id: int | None) -> str | None:
    if class_id is None:
        return None
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return str(class_id)


def parse_filter_values(values: list[str] | None) -> set[int] | None:
    if not values:
        return None
    resolved = {class_id for value in values if (class_id := resolve_class_token(value)) is not None}
    return resolved or None


def class_allowed(class_id: int | None, include: set[int] | None, exclude: set[int] | None) -> bool:
    if class_id is None:
        return include is None
    if include is not None and class_id not in include:
        return False
    if exclude is not None and class_id in exclude:
        return False
    return True


def yolo_to_pixel(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    xc, yc, bw, bh = box
    return (
        (xc - bw / 2.0) * width,
        (yc - bh / 2.0) * height,
        (xc + bw / 2.0) * width,
        (yc + bh / 2.0) * height,
    )


def clamp_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(width - 1)))
    y1 = max(0.0, min(float(y1), float(height - 1)))
    x2 = max(0.0, min(float(x2), float(width - 1)))
    y2 = max(0.0, min(float(y2), float(height - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def read_yolo_txt(path: Path, width: int, height: int) -> list[AnnotationBox]:
    boxes: list[AnnotationBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = resolve_class_token(parts[0])
        if class_id is None:
            continue
        try:
            raw_box = tuple(map(float, parts[1:5]))
        except ValueError:
            continue
        boxes.append(AnnotationBox(class_id, class_name(class_id), clamp_box(yolo_to_pixel(raw_box, width, height), width, height)))
    return boxes


def read_voc_xml(path: Path, width: int, height: int) -> list[AnnotationBox]:
    boxes: list[AnnotationBox] = []
    root = ET.parse(path).getroot()
    for obj in root.findall("object"):
        label = obj.findtext("name", default="").strip()
        class_id = resolve_class_token(label)
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            box = (
                float(bndbox.findtext("xmin", default="0")),
                float(bndbox.findtext("ymin", default="0")),
                float(bndbox.findtext("xmax", default="0")),
                float(bndbox.findtext("ymax", default="0")),
            )
        except ValueError:
            continue
        boxes.append(AnnotationBox(class_id, class_name(class_id) or label or None, clamp_box(box, width, height)))
    return boxes


def iter_json_entries(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("annotations"), list):
                yield from (entry for entry in item["annotations"] if isinstance(entry, dict))
            elif isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        entries = payload.get("annotations") or payload.get("objects") or []
        if isinstance(entries, list):
            yield from (entry for entry in entries if isinstance(entry, dict))


def read_json_annotations(path: Path, width: int, height: int) -> list[AnnotationBox]:
    boxes: list[AnnotationBox] = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in iter_json_entries(payload):
        label = entry.get("label") or entry.get("name") or entry.get("class")
        class_id = resolve_class_token(str(label)) if label is not None else None
        coords = entry.get("coordinates") or entry.get("bbox") or entry.get("bndbox")
        if coords is None:
            continue

        try:
            if isinstance(coords, dict) and {"x", "y", "width", "height"}.issubset(coords):
                x = float(coords["x"])
                y = float(coords["y"])
                box = (x, y, x + float(coords["width"]), y + float(coords["height"]))
            elif isinstance(coords, dict) and {"xmin", "ymin", "xmax", "ymax"}.issubset(coords):
                box = (
                    float(coords["xmin"]),
                    float(coords["ymin"]),
                    float(coords["xmax"]),
                    float(coords["ymax"]),
                )
            elif isinstance(coords, (list, tuple)) and len(coords) >= 4:
                box = tuple(map(float, coords[:4]))
            else:
                continue
        except (TypeError, ValueError):
            continue

        boxes.append(AnnotationBox(class_id, class_name(class_id) or (str(label) if label else None), clamp_box(box, width, height)))
    return boxes


def annotation_path_for_frame(annotations_dir: Path, image_path: Path) -> Path | None:
    for suffix in (".txt", ".xml", ".json"):
        candidate = annotations_dir / f"{image_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def read_annotations(path: Path | None, width: int, height: int) -> list[AnnotationBox]:
    if path is None:
        return []
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_yolo_txt(path, width, height)
    if suffix == ".xml":
        return read_voc_xml(path, width, height)
    if suffix == ".json":
        return read_json_annotations(path, width, height)
    return []


def load_skeleton(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [(int(a), int(b)) for a, b in data.get("skeleton", [])]


def make_tracker(track_buffer: int, match_thresh: float) -> BYTETracker:
    args = SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        fuse_score=True,
    )
    return BYTETracker(args)


def track_annotations(tracker: BYTETracker, annotations: list[AnnotationBox], image: np.ndarray) -> dict[int, int]:
    if annotations:
        xyxy = np.asarray([ann.bbox for ann in annotations], dtype=np.float32)
        conf = np.asarray([ann.confidence for ann in annotations], dtype=np.float32)
        cls = np.asarray([ann.class_id if ann.class_id is not None else -1 for ann in annotations], dtype=np.float32)
    else:
        xyxy = np.empty((0, 4), dtype=np.float32)
        conf = np.empty((0,), dtype=np.float32)
        cls = np.empty((0,), dtype=np.float32)

    tracks = tracker.update(TrackerDetections(xyxy, conf, cls), img=image)
    id_by_annotation_index: dict[int, int] = {}
    for row in tracks:
        if len(row) < 8:
            continue
        annotation_index = int(round(float(row[7])))
        if 0 <= annotation_index < len(annotations):
            id_by_annotation_index[annotation_index] = int(round(float(row[4])))
    return id_by_annotation_index


def padded_roi(box: tuple[float, float, float, float], width: int, height: int, pad_fraction: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * pad_fraction
    pad_y = bh * pad_fraction
    rx1, ry1, rx2, ry2 = clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)
    return int(round(rx1)), int(round(ry1)), int(round(rx2)), int(round(ry2))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def infer_pose_in_roi(
    model: YOLO,
    image: np.ndarray,
    annotation_box: tuple[float, float, float, float],
    *,
    roi_pad: float,
    conf: float,
    imgsz: int,
    device: str | None,
) -> tuple[list[list[float]] | None, list[float] | None, float | None, list[float]]:
    height, width = image.shape[:2]
    rx1, ry1, rx2, ry2 = padded_roi(annotation_box, width, height, roi_pad)
    if rx2 <= rx1 or ry2 <= ry1:
        return None, None, None, [float(rx1), float(ry1), float(rx2), float(ry2)]

    crop = image[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None, None, None, [float(rx1), float(ry1), float(rx2), float(ry2)]

    result = model.predict(
        source=crop,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
        stream=False,
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None, None, None, [float(rx1), float(ry1), float(rx2), float(ry2)]

    crop_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    target = np.asarray([[annotation_box[0] - rx1, annotation_box[1] - ry1, annotation_box[2] - rx1, annotation_box[3] - ry1]], dtype=np.float32)
    best_index = int(np.argmax(box_iou(target, crop_boxes)[0]))

    pose_bbox = crop_boxes[best_index].copy()
    pose_bbox[[0, 2]] += rx1
    pose_bbox[[1, 3]] += ry1
    pose_conf = float(result.boxes.conf.cpu().numpy()[best_index]) if result.boxes.conf is not None else None

    keypoints_obj = getattr(result, "keypoints", None)
    if keypoints_obj is None or getattr(keypoints_obj, "data", None) is None:
        return None, pose_bbox.tolist(), pose_conf, [float(rx1), float(ry1), float(rx2), float(ry2)]

    keypoints = keypoints_obj.data.cpu().numpy()[best_index].astype(np.float32)
    keypoints[:, 0] += rx1
    keypoints[:, 1] += ry1
    return keypoints.tolist(), pose_bbox.tolist(), pose_conf, [float(rx1), float(ry1), float(rx2), float(ry2)]


def color_for_track(track_id: int | None) -> tuple[int, int, int]:
    palette = (
        (255, 87, 51),
        (51, 153, 255),
        (255, 204, 51),
        (102, 255, 102),
        (255, 102, 204),
        (0, 204, 204),
        (180, 120, 255),
        (80, 220, 180),
    )
    if track_id is None:
        return (0, 165, 255)
    return palette[track_id % len(palette)]


def draw_keypoints(image: np.ndarray, keypoints: list[list[float]] | None, skeleton: list[tuple[int, int]], conf_thresh: float) -> None:
    if not keypoints:
        return
    for a, b in skeleton:
        if a >= len(keypoints) or b >= len(keypoints):
            continue
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca >= conf_thresh and cb >= conf_thresh:
            cv2.line(image, (int(round(xa)), int(round(ya))), (int(round(xb)), int(round(yb))), (0, 128, 255), 2)
    for x, y, score in keypoints:
        if score >= conf_thresh:
            cv2.circle(image, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)


def draw_box_label(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    track_id: int | None,
    class_id: int | None,
    label_name: str | None,
    color: tuple[int, int, int],
    draw_box: bool,
    draw_label: bool,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    if draw_box:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if not draw_label:
        return

    parts = []
    if track_id is not None:
        parts.append(f"id:{track_id}")
    if label_name is not None:
        parts.append(label_name)
    elif class_id is not None:
        parts.append(f"cls:{class_id}")
    label = " ".join(parts) or "annotation"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(th + baseline, y1 - 4)
    cv2.rectangle(image, (x1, ty - th - baseline), (x1 + tw + 6, ty + baseline), color, -1)
    cv2.putText(image, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def collect_frame_paths(frames_dir: Path, start: int, limit: int) -> list[Path]:
    image_paths = [path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    image_paths.sort(key=frame_sort_key)
    if not image_paths:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    start = max(0, min(start, len(image_paths) - 1))
    end = len(image_paths) if limit <= 0 else min(len(image_paths), start + limit)
    return image_paths[start:end]


def collect_annotation_paths(annotations_dir: Path, start: int, limit: int) -> list[Path]:
    annotation_paths = [
        path for path in annotations_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".txt", ".xml", ".json"}
        and path.stem.startswith("frame_")
    ]
    annotation_paths.sort(key=frame_sort_key)
    if not annotation_paths:
        raise FileNotFoundError(f"No frame annotation files found in {annotations_dir}")
    start = max(0, min(start, len(annotation_paths) - 1))
    end = len(annotation_paths) if limit <= 0 else min(len(annotation_paths), start + limit)
    return annotation_paths[start:end]


def frame_path_for_annotation(frames_dir: Path, annotation_path: Path) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = frames_dir / f"{annotation_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def collect_work_items(args: argparse.Namespace) -> list[tuple[Path, Path | None]]:
    if args.drive == "frames":
        return [
            (frame_path, annotation_path_for_frame(args.annotations_dir, frame_path))
            for frame_path in collect_frame_paths(args.frames_dir, args.start, args.limit)
        ]

    items: list[tuple[Path, Path | None]] = []
    missing_frames = 0
    for annotation_path in collect_annotation_paths(args.annotations_dir, args.start, args.limit):
        frame_path = frame_path_for_annotation(args.frames_dir, annotation_path)
        if frame_path is None:
            missing_frames += 1
            continue
        items.append((frame_path, annotation_path))

    if not items:
        raise FileNotFoundError(
            f"No matching frame images found for annotations in {args.annotations_dir}"
        )
    if missing_frames:
        print(f"warning: skipped {missing_frames} annotation files with no matching frame image")
    return items


def main() -> None:
    args = parse_args()
    if not args.frames_dir.exists():
        raise FileNotFoundError(f"Frames folder not found: {args.frames_dir}")
    if not args.annotations_dir.exists():
        raise FileNotFoundError(f"Annotations folder not found: {args.annotations_dir}")

    include = parse_filter_values(args.include_classes)
    exclude = parse_filter_values(args.exclude_classes)
    if args.players_only:
        include = PLAYER_ACTION_CLASS_IDS if include is None else include & PLAYER_ACTION_CLASS_IDS

    work_items = collect_work_items(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    tracker = make_tracker(args.track_buffer, args.match_thresh)
    skeleton = load_skeleton(args.pose_yaml)

    frames_payload: list[dict[str, Any]] = []
    total_detections = 0
    total_with_pose = 0

    for frame_index, (image_path, annotation_path) in enumerate(work_items):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"warning: skipping unreadable frame {image_path}")
            continue

        height, width = image.shape[:2]
        annotations = [
            ann for ann in read_annotations(annotation_path, width, height)
            if class_allowed(ann.class_id, include, exclude)
        ]
        track_ids = track_annotations(tracker, annotations, image)
        visual = image.copy()

        frame_data: dict[str, Any] = {
            "frame_index": frame_index,
            "frame_path": str(image_path),
            "frame_number": frame_number_from_path(image_path),
            "source": str(args.frames_dir),
            "annotation_path": None if annotation_path is None else str(annotation_path),
            "detections": [],
        }

        for ann_index, ann in enumerate(annotations):
            track_id = track_ids.get(ann_index)
            keypoints, pose_bbox, pose_conf, roi_box = infer_pose_in_roi(
                model,
                image,
                ann.bbox,
                roi_pad=args.roi_pad,
                conf=args.pose_conf,
                imgsz=args.imgsz,
                device=args.device,
            )

            detection = {
                "bbox": [float(value) for value in ann.bbox],
                "confidence": float(ann.confidence),
                "class_id": None if ann.class_id is None else int(ann.class_id),
                "class_name": ann.class_name,
                "track_id": track_id,
                "keypoints": keypoints,
                "pose_bbox": pose_bbox,
                "pose_confidence": pose_conf,
                "roi_bbox": roi_box,
            }
            frame_data["detections"].append(detection)
            total_detections += 1
            if keypoints is not None:
                total_with_pose += 1

            color = color_for_track(track_id)
            draw_keypoints(visual, keypoints, skeleton, args.keypoint_conf)
            draw_box_label(
                visual,
                ann.bbox,
                track_id=track_id,
                class_id=ann.class_id,
                label_name=ann.class_name,
                color=color,
                draw_box=not args.hide_boxes,
                draw_label=not args.hide_labels,
            )

        out_path = args.out_dir / image_path.name
        cv2.imwrite(str(out_path), visual)
        frames_payload.append(frame_data)

        if args.progress_every > 0 and (frame_index == 0 or (frame_index + 1) % args.progress_every == 0):
            print(f"processed {frame_index + 1}/{len(work_items)} annotated frames -> {out_path}")

    payload = {
        "weights": args.weights,
        "source": str(args.frames_dir),
        "annotations_dir": str(args.annotations_dir),
        "drive": args.drive,
        "output_mode": "annotation_guided_pose",
        "imgsz": args.imgsz,
        "pose_conf": args.pose_conf,
        "keypoint_conf": args.keypoint_conf,
        "roi_pad": args.roi_pad,
        "track": True,
        "tracker": "bytetrack",
        "visual_output_dir": str(args.out_dir),
        "frames": frames_payload,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved annotation-guided pose JSON to {args.output_json}")
    print(f"Saved visual frames to {args.out_dir}")
    print(f"Pose found for {total_with_pose}/{total_detections} annotation boxes")


if __name__ == "__main__":
    main()
