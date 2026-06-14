from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _frame_sort_key(path: Path) -> tuple[int, int | str]:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match is not None:
        return (0, int(match.group(1)))
    return (1, path.name.lower())


def _latest_run_folder(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None

    folders = [path for path in base_dir.iterdir() if path.is_dir()]
    if not folders:
        return None

    folders.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return folders[0]


def _resolve_directory(directory: Path | None) -> Path:
    if directory is not None:
        return directory

    preferred = Path("runs/pose/runs/pose_infer")
    latest = _latest_run_folder(preferred)
    if latest is not None:
        return latest

    fallback = Path("runs/pose_infer")
    latest = _latest_run_folder(fallback)
    if latest is not None:
        return latest

    raise FileNotFoundError(
        "No pose result directory found. Pass --dir with the folder containing the annotated images.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step through YOLOv8-Pose result images one by one.")
    parser.add_argument("--dir", type=Path, default=None,
                        help="Folder containing annotated pose result images.")
    parser.add_argument("--start", type=int, default=0,
                        help="Zero-based index of the first image to show.")
    parser.add_argument("--fps", type=float, default=0.0,
                        help="Playback speed. Use >0 for continuous playback; 0 keeps manual stepping mode.")
    parser.add_argument("--no-loop", action="store_true",
                        help="Stop at last frame instead of looping to first frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = _resolve_directory(args.dir)

    image_paths = [
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort(key=_frame_sort_key)

    if not image_paths:
        raise FileNotFoundError(f"No image files found in {directory}")

    index = max(0, min(args.start, len(image_paths) - 1))
    window_name = "Pose Results Viewer"
    autoplay = args.fps > 0
    paused = not autoplay
    frame_delay_ms = max(1, int(round(1000.0 / args.fps))) if autoplay else 0

    print(f"Viewing {len(image_paths)} images from: {directory}")
    if autoplay:
        print(
            "Controls: space = pause/resume, right arrow = next, left arrow = previous, q or esc = quit")
    else:
        print(
            "Controls: right arrow / space = next, left arrow = previous, q or esc = quit")

    while True:
        image_path = image_paths[index]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable file: {image_path}")
            index = (index + 1) % len(image_paths)
            continue

        display = image.copy()
        label = f"{index + 1}/{len(image_paths)}  {image_path.name}"
        cv2.putText(display, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2, cv2.LINE_AA)
        if autoplay:
            status = "PAUSED" if paused else f"PLAY {args.fps:.2f} fps"
            cv2.putText(display, status, (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, display)

        if autoplay and not paused:
            key = cv2.waitKey(frame_delay_ms) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == 255:
            if autoplay and not paused:
                if index == len(image_paths) - 1:
                    if args.no_loop:
                        break
                    index = 0
                else:
                    index += 1
            continue

        if key in (81, 2424832):
            index = (index - 1) % len(image_paths)
            continue
        if key in (83, 2555904, 32):
            if autoplay and key == 32:
                paused = not paused
                continue
            index = (index + 1) % len(image_paths)
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
