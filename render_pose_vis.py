from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import yaml


def parse_args():
    p = argparse.ArgumentParser(
        description="Render pose keypoints from predictions JSON")
    p.add_argument("--pred-json", type=Path,
                   default=Path("runs/pose_infer/frames_predictions.json"))
    p.add_argument("--pose-yaml", type=Path, default=Path("data_pose.yaml"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("runs/pose_infer/frames_full_vis"))
    p.add_argument("--conf-thresh", type=float, default=0.15)
    p.add_argument("--box-thickness", type=int, default=2)
    p.add_argument("--hide-boxes", action="store_true",
                   help="Do not draw bounding boxes")
    p.add_argument("--hide-labels", action="store_true",
                   help="Do not draw labels (track id / class / confidence)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="0=all")
    return p.parse_args()


def load_skeleton(pose_yaml: Path) -> List[Tuple[int, int]]:
    if not pose_yaml.exists():
        return []
    data = yaml.safe_load(pose_yaml.read_text(encoding="utf-8"))
    sk = data.get("skeleton", [])
    pairs = [(int(a), int(b)) for a, b in sk]
    return pairs


def draw_detection(img, keypoints, skeleton, conf_thres=0.15):
    # keypoints: list of [x,y,conf] for each kp
    h, w = img.shape[:2]
    # colors
    kp_color = (0, 255, 0)
    sk_color = (0, 128, 255)
    for i, kp in enumerate(keypoints):
        try:
            x, y, c = kp
        except Exception:
            continue
        if c is None:
            continue
        if c >= conf_thres and x is not None and y is not None:
            ix, iy = int(round(x)), int(round(y))
            cv2.circle(img, (ix, iy), 3, kp_color, -1)
    # draw skeleton lines
    for a, b in skeleton:
        if a < 0 or b < 0 or a >= len(keypoints) or b >= len(keypoints):
            continue
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca is None or cb is None:
            continue
        if ca >= conf_thres and cb >= conf_thres:
            va = (int(round(xa)), int(round(ya)))
            vb = (int(round(xb)), int(round(yb)))
            cv2.line(img, va, vb, sk_color, 2)


def draw_bbox_and_label(
    img,
    bbox,
    track_id,
    class_id,
    confidence,
    *,
    draw_box=True,
    draw_label=True,
    thickness=2,
):
    if bbox is None or len(bbox) < 4:
        return

    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
    box_color = (255, 0, 0)
    text_color = (255, 255, 255)
    bg_color = (255, 0, 0)

    if draw_box:
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)

    if not draw_label:
        return

    parts = []
    if track_id is not None:
        parts.append(f"id:{track_id}")
    if class_id is not None:
        parts.append(f"cls:{class_id}")
    if confidence is not None:
        parts.append(f"{float(confidence):.2f}")
    if not parts:
        return

    label = " ".join(parts)
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx = max(0, x1)
    ty = max(th + baseline, y1 - 4)

    cv2.rectangle(
        img,
        (tx, ty - th - baseline),
        (tx + tw + 4, ty + baseline),
        bg_color,
        -1,
    )
    cv2.putText(
        img,
        label,
        (tx + 2, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    skeleton = load_skeleton(args.pose_yaml)

    with open(args.pred_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", []) if isinstance(data, dict) else data
    total = len(frames)
    start = args.start
    limit = args.limit or total

    end = min(total, start + limit)
    print(f"Rendering frames {start}..{end} (of {total}) -> {out}")

    for i in range(start, end):
        fr = frames[i]
        img_path = Path(fr.get("frame_path"))
        if not img_path.exists():
            # try relative to workspace
            img_path = Path.cwd() / fr.get("frame_path")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"warning: cannot open image {img_path}")
            continue

        for det in fr.get("detections", []):
            kps = det.get("keypoints") or []
            # kps is list of 17 [x,y,conf]
            draw_detection(img, kps, skeleton, conf_thres=args.conf_thresh)
            draw_bbox_and_label(
                img,
                det.get("bbox"),
                det.get("track_id"),
                det.get("class_id"),
                det.get("confidence"),
                draw_box=not args.hide_boxes,
                draw_label=not args.hide_labels,
                thickness=args.box_thickness,
            )

        out_path = out / Path(img_path.name)
        cv2.imwrite(str(out_path), img)
        if (i - start) % 100 == 0:
            print(f"saved {i - start + 1}/{end - start} -> {out_path}")

    print("Done")


if __name__ == "__main__":
    main()
