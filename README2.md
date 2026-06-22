# README2.md
# Running the Custom ByteTrack (4D Kalman + Depth-aware Hungarian)

This document explains how to run the modified tracker on any video which consists of using custom Kalman Filter 4D and MiDas depth assisted Hungarian matching.

---

## Features

This implementation contains:

- YOLOv8 Person Detection
- Custom 4D Kalman Filter (State = x, y, vx, vy)
- ByteTrack Tracker
- Hungarian Matching
- IoU + MiDaS Depth based association
- Depth visualization generation
- Tracking logs

---

## Folder Structure

```
runs/
├── detect/
├── depth/
├── pose_infer/
└── testvideo/
```

---

## Required Models

Place the following models in the project.

```
yolov8n.pt
yolov8n-pose.pt
mobile_sam.pt
models/dpt_swin2_tiny_256.pt
```

---

## Important Modified Files

The tracker uses modified Ultralytics files.

```
.venv/
└── Lib/
    └── site-packages/
        └── ultralytics/
            └── trackers/
                ├── byte_tracker.py
                ├── track.py
                ├── kalman4d.py
                └── utils/
                    └── kalman_filter.py
```

Do NOT replace these with the original Ultralytics versions.

---

## Step 1 : Generate Depth Features

Run

```
python depth_generator.py
```

This creates

```
runs/depth/
    depth_features.json
```

and depth visualization images.

---

## Step 2 : Run Tracking

Run

```
python testbytetrackRED.py
```

(or the tracking script used in the repository.)

This performs

1. YOLO Detection
2. 4D Kalman Prediction
3. IoU + Depth Hungarian Matching
4. Track Update

---

## Output Files

Tracking results are saved in

```
runs/detect/customKF/
```

including

```
tracked_frames/
output_video.mp4
track_records.json
```

Depth outputs are saved in

```
runs/depth/
```

including

```
depth_features.json
depthforcustomKF/
cost_log.txt
```

---

## Cost Function

Association cost is computed as

```
Cost = α × (1 − IoU)
     + β × NormalizedDepthDifference
```

Current weights

```
α = 0.8
β = 0.2
```

---

## Notes

Depth is relative (MiDaS monocular depth), not metric distance.

The tracker first predicts using the custom 4D Kalman Filter and then performs Hungarian assignment using IoU and depth.

---

## Testing

Tested on Kabaddi raid videos with severe player overlap and temporary occlusion.

The implementation is intended to reduce ID switches during player interactions.