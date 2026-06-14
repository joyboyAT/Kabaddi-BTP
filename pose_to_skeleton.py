from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = Path("runs/pose_infer/predictions.json")
DEFAULT_OUTPUT = Path("runs/skeleton_sequences")


@dataclass
class TrackFrame:
    frame_index: int
    keypoints: np.ndarray
    confidence: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert tracked YOLOv8-Pose JSON results into skeleton windows.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT,
                        help="Pose prediction JSON from pose_infer.py.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory to write .npz windows and manifest.")
    parser.add_argument("--window-size", type=int, default=64,
                        help="Number of frames per skeleton window.")
    parser.add_argument("--stride", type=int, default=32,
                        help="Step size between windows for each track.")
    parser.add_argument("--min-confidence", type=float, default=0.25,
                        help="Minimum keypoint confidence to keep a joint.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Pose JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _keypoint_array(raw_keypoints: Any) -> np.ndarray:
    array = np.asarray(raw_keypoints, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"Expected keypoints with shape (V, 3), got {array.shape}")
    return array


def _center_and_scale(keypoints: np.ndarray) -> tuple[np.ndarray, float]:
    left_hip = keypoints[11, :2]
    right_hip = keypoints[12, :2]
    center = (left_hip + right_hip) / 2.0

    left_shoulder = keypoints[5, :2]
    right_shoulder = keypoints[6, :2]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    torso = np.linalg.norm(shoulder_center - center)

    if torso < 1e-6:
        visible = keypoints[:, 2] > 0
        if np.any(visible):
            pts = keypoints[visible, :2]
            span = np.max(pts, axis=0) - np.min(pts, axis=0)
            scale = float(np.linalg.norm(span))
        else:
            scale = 1.0
    else:
        scale = float(torso)

    return center, max(scale, 1.0)


def _normalize_frame(keypoints: np.ndarray, min_confidence: float) -> tuple[np.ndarray, np.ndarray]:
    coords = keypoints[:, :2].copy()
    confidence = keypoints[:, 2].copy()

    center, scale = _center_and_scale(keypoints)
    coords = (coords - center) / scale

    low_confidence = confidence < min_confidence
    coords[low_confidence] = 0.0
    confidence[low_confidence] = 0.0

    return coords, confidence


def _build_track_frames(payload: dict[str, Any]) -> dict[int, list[TrackFrame]]:
    tracks: dict[int, list[TrackFrame]] = {}

    for frame in payload.get("frames", []):
        frame_index = int(frame.get("frame_index", 0))
        for detection in frame.get("detections", []):
            track_id = detection.get("track_id")
            keypoints = detection.get("keypoints")
            if track_id is None or keypoints is None:
                continue

            raw = _keypoint_array(keypoints)
            tracks.setdefault(int(track_id), []).append(
                TrackFrame(
                    frame_index=frame_index,
                    keypoints=raw,
                    confidence=raw[:, 2].copy(),
                )
            )

    for track_id in tracks:
        tracks[track_id].sort(key=lambda item: item.frame_index)

    return tracks


def _pad_window(frames: list[TrackFrame], window_size: int) -> list[TrackFrame]:
    if not frames:
        return []

    padded = list(frames)
    while len(padded) < window_size:
        padded.append(padded[-1])
    return padded[:window_size]


def main() -> None:
    args = parse_args()
    payload = _load_json(args.input_json)
    tracks = _build_track_frames(payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for track_id, frames in tracks.items():
        if not frames:
            continue

        window_index = 0
        for start in range(0, len(frames), args.stride):
            window = frames[start:start + args.window_size]
            if not window:
                continue

            window = _pad_window(window, args.window_size)

            keypoints = []
            confidence = []
            frame_indices = []
            for frame in window:
                coords, conf = _normalize_frame(
                    frame.keypoints, args.min_confidence)
                keypoints.append(coords)
                confidence.append(conf)
                frame_indices.append(frame.frame_index)

            keypoints_array = np.asarray(
                keypoints, dtype=np.float32)  # (T, V, 2)
            confidence_array = np.asarray(
                confidence, dtype=np.float32)  # (T, V)

            output_file = args.output_dir / \
                f"track_{track_id:04d}_window_{window_index:04d}.npz"
            np.savez_compressed(
                output_file,
                keypoints=keypoints_array,
                confidence=confidence_array,
                track_id=np.int32(track_id),
                frame_indices=np.asarray(frame_indices, dtype=np.int32),
                source_json=str(args.input_json),
            )

            manifest.append({
                "file": output_file.name,
                "track_id": int(track_id),
                "window_index": window_index,
                "frame_indices": frame_indices,
                "shape": list(keypoints_array.shape),
            })
            window_index += 1

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest)} skeleton windows to {args.output_dir}")


if __name__ == "__main__":
    main()
