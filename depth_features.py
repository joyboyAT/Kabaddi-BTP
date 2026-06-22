"""
depth_features.py
─────────────────
Standalone script that generates per-player depth features using MiDaS Small.

Inputs
------
1. runs/detect/0.9matchallYellow/track_records.json
2. runs/testvedio/raw_frames/frame_<N>.jpg

Outputs
-------
- depth_features.json   (one record per player-frame)
- depth36.png … depth39.png  (normalised depth maps for debugging)
"""

import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

# ──────────────────────────────────────────────
# Paths (relative to the script's own directory)
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TRACK_JSON = os.path.join(
    SCRIPT_DIR,
    "runs", "detect", "0.9matchallYellow", "track_records.json",
)
FRAMES_DIR = os.path.join(
    SCRIPT_DIR,
    "runs", "testvedio", "raw_frames",
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "runs", "depth")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_JSON = os.path.join(OUTPUT_DIR, "depth_features.json")

# Frames for which we save debug depth PNGs
DEBUG_FRAMES = {36, 37, 38, 39}


# ──────────────────────────────────────────────
# 1. Load MiDaS Small  (loaded once)
# ──────────────────────────────────────────────
def load_midas():
    """Load MiDaS Small and its transforms exactly once."""
    print("[INFO] Loading MiDaS Small …")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform

    print(f"[INFO] MiDaS Small loaded on {device}")
    return model, transform, device


# ──────────────────────────────────────────────
# 2. Compute full-frame depth map
# ──────────────────────────────────────────────
@torch.no_grad()
def compute_depth_map(frame_bgr, model, transform, device):
    """Return a float32 depth map (H, W) for a BGR frame."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(frame_rgb).to(device)

    prediction = model(input_batch)

    # Resize back to original resolution
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=frame_bgr.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    return prediction.cpu().numpy()


# ──────────────────────────────────────────────
# 3. Save a normalised depth map as PNG
# ──────────────────────────────────────────────
def save_depth_png(depth_map, path):
    """Normalise depth to 0-255 and save as a grayscale PNG."""
    d = depth_map.copy()
    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 1e-6:
        d = (d - d_min) / (d_max - d_min)
    else:
        d = np.zeros_like(d)
    d = (d * 255).astype(np.uint8)
    cv2.imwrite(path, d)
    print(f"[DEBUG] Saved {path}")


# ──────────────────────────────────────────────
# 4. Extract median depth for a bbox crop
# ──────────────────────────────────────────────
def median_depth_in_bbox(depth_map, bbox):
    """
    Extract the median depth value inside a bounding box.

    Parameters
    ----------
    depth_map : np.ndarray  (H, W)
    bbox      : list [x1, y1, x2, y2]

    Returns
    -------
    float – median depth value
    """
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w, int(round(x2)))
    y2 = min(h, int(round(y2)))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = depth_map[y1:y2, x1:x2]
    return float(np.median(crop))


# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────
def main():
    # ── Load track records ──────────────────
    print(f"[INFO] Loading tracks from {TRACK_JSON}")
    with open(TRACK_JSON, "r") as f:
        records = json.load(f)
    print(f"[INFO] {len(records)} track records loaded")

    # ── Group records by frame ──────────────
    # NOTE: keys in the JSON have trailing colons  ("frame:", "track_id:")
    frame_to_tracks = defaultdict(list)
    for rec in records:
        fno = int(rec["frame:"])
        frame_to_tracks[fno].append(rec)

    sorted_frames = sorted(frame_to_tracks.keys())
    print(f"[INFO] {len(sorted_frames)} unique frames to process")

    # ── Load MiDaS (once) ───────────────────
    model, transform, device = load_midas()

    # ── Process frame by frame ──────────────
    results = []
    processed = 0

    for fno in sorted_frames:
        frame_path = os.path.join(FRAMES_DIR, f"frame_{fno}.jpg")

        if not os.path.isfile(frame_path):
            # Skip missing frames safely
            continue

        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            continue

        # Compute depth map once per frame
        depth_map = compute_depth_map(frame_bgr, model, transform, device)

        # Save debug PNGs for frames 36-39
        if fno in DEBUG_FRAMES:
            png_path = os.path.join(OUTPUT_DIR, f"depth{fno}.png")
            save_depth_png(depth_map, png_path)

        # Process every track in this frame
        for rec in frame_to_tracks[fno]:
            track_id = int(rec["track_id:"])
            bbox = rec["Bounding Box:"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            med_depth = median_depth_in_bbox(depth_map, bbox)

            results.append({
                "frame": fno,
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "center": [round(cx, 2), round(cy, 2)],
                "depth": round(med_depth, 4),
            })

        processed += 1
        if processed % 100 == 0:
            print(f"[PROGRESS] {processed}/{len(sorted_frames)} frames done")

    # ── Save output JSON ────────────────────
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] {len(results)} records written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
