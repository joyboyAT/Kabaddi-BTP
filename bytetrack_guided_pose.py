import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor
import torch
# -------------------------
# Paths
# -------------------------
TRACK_JSON = r"runs/detect/0.9matchallYellow/track_records.json"
FRAME_DIR = r"runs/detect/0.9matchallYellow/tracked_frames"

POSE_MODEL = "yolov8n-pose.pt"

OUTPUT_JSON = r"runs/pose_infer/bytetrack_pose_predictions.json"


# -------------------------
# Load model
# -------------------------
pose_model = YOLO(POSE_MODEL)

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
sam.to(device)
sam.eval()
predictor = SamPredictor(sam)

# -------------------------
# Load track records
# -------------------------
with open(TRACK_JSON, "r") as f:
    records = json.load(f)

pose_records = []

current_frame_no = None
current_frame = None

for rec in records:

    frame_no = rec["frame:"]
    track_id = rec["track_id:"]

    x1, y1, x2, y2 = rec["Bounding Box:"]

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(max(x1 + 1, x2))
    y2 = int(max(y1 + 1, y2))

    # -------------------------
    # Load frame only once
    # -------------------------
    if frame_no != current_frame_no:

        frame_path = Path("runs/testvedio/raw_frames") / f"frame_{frame_no}.jpg"

        if not frame_path.exists():
            continue

        current_frame = cv2.imread(str(frame_path))
        current_frame_no = frame_no

    frame = current_frame

    if frame_no == 36 and track_id == 6:
        cv2.imwrite("raw_frame_debug.jpg", frame)

    h, w = frame.shape[:2]

    x1 = min(x1, w - 1)
    x2 = min(x2, w)

    y1 = min(y1, h - 1)
    y2 = min(y2, h)

    crop = frame[y1:y2, x1:x2]

    if frame_no == 36 and track_id == 6:
        cv2.imwrite("crop_id6_raw.jpg", crop)
        print("saved raw crop")
        exit()

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    predictor.set_image(crop_rgb)
    h, w = crop.shape[:2]
    # input_box = np.array([0, 0, w-1, h-1])
    # above was givig error for some crops, so using smaller boxes around torso and head
    input_box = np.array([int(w*0.2), int(h*0.1), int(w*0.8), int(h*0.9)])
    masks, scores, logits = predictor.predict(
        box=input_box, multimask_output=False)
    mask = masks[0]
    print(mask.shape)
    print(mask.sum())
    print(mask.size)
    print("foreground %= ", 100*mask.sum()/mask.size)
    player_crop = crop.copy()
    player_crop[~mask] = 0
    if frame_no == 36 and track_id == 6:
        cv2.imwrite("sam_playerid6.jpg", player_crop)
        print("saved SAM output")
        exit()
    if crop.size == 0:
        continue

    # -------------------------
    # Pose inference
    # -------------------------
    results = pose_model(crop, verbose=False)

    if len(results) == 0:
        continue

    result = results[0]

    if result.keypoints is None:
        continue

    if len(result.keypoints.xy) == 0:
        continue

    kp = result.keypoints.xy[0].cpu().numpy()

    # convert crop coords back to full image coords
    full_kp = []

    for x, y in kp:
        full_kp.append([
            float(x + x1),
            float(y + y1)
        ])

    pose_records.append(
        {
            "frame": frame_no,
            "track_id": track_id,
            "bbox": [x1, y1, x2, y2],
            "keypoints": full_kp
        }
    )

# -------------------------
# Save JSON
# -------------------------
Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_JSON, "w") as f:
    json.dump(pose_records, f)

print(f"Saved {len(pose_records)} pose records")
print(f"Output -> {OUTPUT_JSON}")
