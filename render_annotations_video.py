from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import cv2


DEFAULT_FRAMES_DIR = Path("frames")
DEFAULT_ANNOTATIONS_DIR = Path("annotations")
DEFAULT_OUTPUT = Path("kabaddi.avi")
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
]

PLAYER_ACTION_CLASS_IDS = set(range(0, 13))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a kabaddi video from frames and annotation files.")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR,
                        help="Folder containing the original frame images.")
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR,
                        help="Folder containing per-frame annotation files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output AVI path.")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Video frame rate.")
    parser.add_argument("--start", type=int, default=0,
                        help="Start at this zero-based frame index.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Number of frames to render. 0 means all frames.")
    parser.add_argument("--include-classes", nargs="*", default=None,
                        help="Optional class names or ids to keep. If omitted, renders all annotation boxes.")
    parser.add_argument("--exclude-classes", nargs="*", default=None,
                        help="Optional class names or ids to skip.")
    parser.add_argument("--players-only", action="store_true",
                        help="Render only the player-action classes from the annotation set.")
    parser.add_argument("--thickness", type=int, default=2,
                        help="Bounding-box thickness.")
    parser.add_argument("--label-scale", type=float, default=0.7,
                        help="Font scale for labels.")
    parser.add_argument("--label-thickness", type=int, default=2,
                        help="Thickness for label text.")
    parser.add_argument("--fourcc", default="XVID",
                        help="Video codec fourcc. Default: XVID for AVI.")
    return parser.parse_args()


def _frame_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match is not None:
        return (0, int(match.group(1)))
    return (1, path.name.lower())


def _resolve_class_token(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    try:
        return int(token)
    except ValueError:
        if token in CLASS_NAMES:
            return CLASS_NAMES.index(token)
        lower_map = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}
        return lower_map.get(token.lower())


def _parse_filter_values(values: list[str] | None) -> set[int] | None:
    if not values:
        return None
    resolved: set[int] = set()
    for value in values:
        class_id = _resolve_class_token(value)
        if class_id is not None:
            resolved.add(class_id)
    return resolved if resolved else None


def _yolo_annotation_to_box(line: str) -> tuple[int | None, tuple[float, float, float, float] | None]:
    parts = line.split()
    if len(parts) < 5:
        return None, None
    class_id = _resolve_class_token(parts[0])
    if class_id is None:
        return None, None
    try:
        xc, yc, w, h = map(float, parts[1:5])
    except ValueError:
        return None, None
    return class_id, (xc, yc, w, h)


def _read_yolo_txt(path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        class_id, box = _yolo_annotation_to_box(line)
        if class_id is None or box is None:
            continue
        boxes.append((class_id, box))
    return boxes


def _read_voc_xml(path: Path) -> list[tuple[int | None, tuple[float, float, float, float]]]:
    boxes: list[tuple[int | None, tuple[float, float, float, float]]] = []
    tree = ET.parse(path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        class_id = _resolve_class_token(name)
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.findtext("xmin", default="0"))
            ymin = float(bndbox.findtext("ymin", default="0"))
            xmax = float(bndbox.findtext("xmax", default="0"))
            ymax = float(bndbox.findtext("ymax", default="0"))
        except ValueError:
            continue
        boxes.append((class_id, (xmin, ymin, xmax, ymax)))
    return boxes


def _read_json_annotations(path: Path) -> list[tuple[int | None, tuple[float, float, float, float]]]:
    boxes: list[tuple[int | None, tuple[float, float, float, float]]] = []
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        entries: Iterable[dict] = payload
    elif isinstance(payload, dict):
        entries = payload.get("annotations", [])
        if not entries and "objects" in payload:
            entries = payload["objects"]
    else:
        return boxes

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label") or entry.get("name") or entry.get("class")
        class_id = _resolve_class_token(
            str(label)) if label is not None else None

        coords = entry.get("coordinates") or entry.get(
            "bbox") or entry.get("bndbox")
        if coords is None:
            continue

        if isinstance(coords, dict):
            if {"x", "y", "width", "height"}.issubset(coords):
                x = float(coords["x"])
                y = float(coords["y"])
                width = float(coords["width"])
                height = float(coords["height"])
                boxes.append((class_id, (x, y, x + width, y + height)))
                continue
            if {"xmin", "ymin", "xmax", "ymax"}.issubset(coords):
                boxes.append((
                    class_id,
                    (
                        float(coords["xmin"]),
                        float(coords["ymin"]),
                        float(coords["xmax"]),
                        float(coords["ymax"]),
                    ),
                ))
                continue

        if isinstance(coords, (list, tuple)) and len(coords) >= 4:
            x1, y1, x2, y2 = map(float, coords[:4])
            boxes.append((class_id, (x1, y1, x2, y2)))

    return boxes


def _frame_annotation_path(annotations_dir: Path, image_path: Path) -> Path | None:
    candidates = [
        annotations_dir / f"{image_path.stem}.txt",
        annotations_dir / f"{image_path.stem}.xml",
        annotations_dir / f"{image_path.stem}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _to_pixel_box(box: tuple[float, float, float, float], width: int, height: int, *, is_yolo: bool) -> tuple[int, int, int, int]:
    if is_yolo:
        xc, yc, bw, bh = box
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
    else:
        x1, y1, x2, y2 = box

    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))
    return x1, y1, x2, y2


def _class_allowed(class_id: int | None, include: set[int] | None, exclude: set[int] | None) -> bool:
    if class_id is None:
        return include is None
    if include is not None and class_id not in include:
        return False
    if exclude is not None and class_id in exclude:
        return False
    return True


def _draw_boxes(
    image,
    boxes: list[tuple[int | None, tuple[float, float, float, float]]],
    *,
    width: int,
    height: int,
    is_yolo: bool,
    include: set[int] | None,
    exclude: set[int] | None,
    thickness: int,
    label_scale: float,
    label_thickness: int,
):
    for class_id, box in boxes:
        if not _class_allowed(class_id, include, exclude):
            continue

        x1, y1, x2, y2 = _to_pixel_box(box, width, height, is_yolo=is_yolo)
        color = (0, 165, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        if class_id is None:
            label = "object"
        elif 0 <= class_id < len(CLASS_NAMES):
            label = CLASS_NAMES[class_id]
        else:
            label = f"id:{class_id}"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, label_thickness)
        top = max(th + baseline, y1)
        left = x1
        cv2.rectangle(
            image,
            (left, top - th - baseline),
            (left + tw + 6, top + baseline),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (left + 3, top),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_scale,
            (255, 255, 255),
            label_thickness,
            cv2.LINE_AA,
        )


def _load_boxes(annotation_path: Path) -> tuple[list[tuple[int | None, tuple[float, float, float, float]]], bool]:
    if annotation_path.suffix.lower() == ".txt":
        return _read_yolo_txt(annotation_path), True
    if annotation_path.suffix.lower() == ".xml":
        return _read_voc_xml(annotation_path), False
    if annotation_path.suffix.lower() == ".json":
        return _read_json_annotations(annotation_path), False
    return [], False


def main() -> None:
    args = parse_args()
    if not args.frames_dir.exists():
        raise FileNotFoundError(f"Frames folder not found: {args.frames_dir}")
    if not args.annotations_dir.exists():
        raise FileNotFoundError(
            f"Annotations folder not found: {args.annotations_dir}")

    include = _parse_filter_values(args.include_classes)
    exclude = _parse_filter_values(args.exclude_classes)
    if args.players_only:
        include = PLAYER_ACTION_CLASS_IDS if include is None else include & PLAYER_ACTION_CLASS_IDS

    image_paths = [
        path for path in args.frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort(key=_frame_sort_key)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.frames_dir}")

    start = max(0, min(args.start, len(image_paths) - 1))
    end = len(image_paths) if args.limit <= 0 else min(
        len(image_paths), start + args.limit)
    selected_paths = image_paths[start:end]

    first_image = cv2.imread(str(selected_paths[0]))
    if first_image is None:
        raise FileNotFoundError(
            f"Could not read first frame: {selected_paths[0]}")

    height, width = first_image.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = cv2.VideoWriter(str(args.output), fourcc,
                             args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.output}")

    rendered = 0
    for index, image_path in enumerate(selected_paths, start=start):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"warning: skipping unreadable frame {image_path}")
            continue

        annotation_path = _frame_annotation_path(
            args.annotations_dir, image_path)
        if annotation_path is not None:
            boxes, is_yolo = _load_boxes(annotation_path)
            if boxes:
                _draw_boxes(
                    image,
                    boxes,
                    width=width,
                    height=height,
                    is_yolo=is_yolo,
                    include=include,
                    exclude=exclude,
                    thickness=args.thickness,
                    label_scale=args.label_scale,
                    label_thickness=args.label_thickness,
                )

        label = f"{index + 1}/{len(image_paths)}  {image_path.name}"
        cv2.putText(image, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2, cv2.LINE_AA)
        writer.write(image)
        rendered += 1
        if rendered % 100 == 0 or rendered == 1:
            print(
                f"rendered {rendered}/{len(selected_paths)} -> {image_path.name}")

    writer.release()
    print(f"Saved {rendered} frames to {args.output}")


if __name__ == "__main__":
    main()
