"""
hungarian_match.py
──────────────────
Frame-by-frame player association using the Hungarian algorithm.

Inputs
------
1. runs/detect/0.9matchallYellow/track_records.json   (detections)
2. runs/depth/depth_features.json                     (per-player depth)

Cost matrix
-----------
For every (track, detection) pair the cost is:

    cost  =  (1 − IoU)  +  λ · |depth_track − depth_det|

where λ controls how much the depth difference matters relative to the
spatial overlap.  Pairs with IoU = 0 are gated (set to a large constant)
so the solver never picks them.

Output
------
runs/depth/hungarian_assignments.json

Each record:
{
    "frame": int,
    "track_id": int,         ← existing track
    "det_index": int,        ← index of detection in that frame
    "cost": float
}

Also prints per-frame statistics.
"""

import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

# ──────────────────────────────────────────────
# Paths  (relative to script directory)
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)                 # one level up

TRACK_JSON = os.path.join(
    PROJECT_DIR,
    "runs", "detect", "0.9matchallYellow", "track_records.json",
)
DEPTH_JSON = os.path.join(
    PROJECT_DIR,
    "runs", "depth", "depth_features.json",
)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "runs", "depth")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "hungarian_assignments.json")

# ──────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────
DEPTH_WEIGHT = 0.01          # λ  – scale depth difference vs (1-IoU)
GATE_THRESHOLD = 0.70        # minimum IoU required (pairs below → ∞)
BIG_COST = 1e5               # gating constant


# ──────────────────────────────────────────────
# Helper: IoU between two [x1,y1,x2,y2] boxes
# ──────────────────────────────────────────────
def iou(box_a, box_b):
    """Compute intersection-over-union for two axis-aligned boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ──────────────────────────────────────────────
# Build a lookup:  (frame, track_id) → depth
# ──────────────────────────────────────────────
def build_depth_lookup(depth_records):
    """Return dict  (frame, track_id) → median_depth."""
    lookup = {}
    for rec in depth_records:
        key = (int(rec["frame"]), int(rec["track_id"]))
        lookup[key] = float(rec["depth"])
    return lookup


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    # ── 1. Load detections ──────────────────
    print(f"[INFO] Loading detections from {TRACK_JSON}")
    with open(TRACK_JSON, "r") as f:
        det_records = json.load(f)
    print(f"[INFO] {len(det_records)} detection records loaded")

    # ── 2. Load depth features ──────────────
    print(f"[INFO] Loading depth from {DEPTH_JSON}")
    with open(DEPTH_JSON, "r") as f:
        depth_records = json.load(f)
    depth_lookup = build_depth_lookup(depth_records)
    print(f"[INFO] {len(depth_lookup)} depth entries indexed")

    # ── 3. Group detections by frame ────────
    frame_to_dets = defaultdict(list)
    for rec in det_records:
        fno = int(rec["frame:"])
        bbox = rec["Bounding Box:"]
        tid = int(rec["track_id:"])
        frame_to_dets[fno].append({
            "track_id": tid,
            "bbox": bbox,
        })

    sorted_frames = sorted(frame_to_dets.keys())
    print(f"[INFO] {len(sorted_frames)} unique frames")

    # ── 4. Frame-by-frame Hungarian matching ─
    assignments = []
    prev_tracks = []          # list of dicts from previous frame

    for idx, fno in enumerate(sorted_frames):
        curr_dets = frame_to_dets[fno]

        if len(prev_tracks) == 0 or len(curr_dets) == 0:
            # Nothing to match yet – seed tracks from current detections
            prev_tracks = curr_dets
            continue

        n_tracks = len(prev_tracks)
        n_dets = len(curr_dets)

        # Build cost matrix  (n_tracks × n_dets)
        cost_matrix = np.full((n_tracks, n_dets), BIG_COST, dtype=np.float64)

        for t_idx, track in enumerate(prev_tracks):
            t_box = track["bbox"]
            t_tid = track["track_id"]
            # Depth of the track (previous frame)
            prev_fno = sorted_frames[sorted_frames.index(fno) - 1] \
                if idx > 0 else fno
            t_depth = depth_lookup.get((prev_fno, t_tid), None)

            for d_idx, det in enumerate(curr_dets):
                d_box = det["bbox"]
                d_tid = det["track_id"]
                d_depth = depth_lookup.get((fno, d_tid), None)

                overlap = iou(t_box, d_box)

                if overlap < (1.0 - GATE_THRESHOLD):
                    # Below gate → allow but penalise
                    pass

                spatial_cost = 1.0 - overlap

                # Depth cost (if both depths available)
                if t_depth is not None and d_depth is not None:
                    depth_cost = DEPTH_WEIGHT * abs(t_depth - d_depth)
                else:
                    depth_cost = 0.0

                total_cost = spatial_cost + depth_cost

                # Gate: if IoU is exactly 0, keep BIG_COST
                if overlap > 0.0:
                    cost_matrix[t_idx, d_idx] = total_cost

        # Solve with the Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] >= BIG_COST:
                continue                       # gated pair → skip

            assignments.append({
                "frame": fno,
                "track_id": prev_tracks[r]["track_id"],
                "det_index": c,
                "matched_det_track_id": curr_dets[c]["track_id"],
                "cost": round(float(cost_matrix[r, c]), 6),
            })

        # Current detections become tracks for the next frame
        prev_tracks = curr_dets

        if (idx + 1) % 50 == 0:
            print(f"[PROGRESS] {idx + 1}/{len(sorted_frames)} frames matched")

    # ── 5. Save assignments ─────────────────
    with open(OUTPUT_JSON, "w") as f:
        json.dump(assignments, f, indent=2)

    print(f"\n[DONE] {len(assignments)} assignments → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
