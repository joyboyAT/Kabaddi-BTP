# YOLOv8-Pose setup

This workspace already contains a detection-style dataset split in `dataset/`, plus a sample video at `videos/kabaddi.mp4`.

Important: the current annotations in `annotations/` are bounding boxes, not keypoints. That means `data_pose.yaml` is ready for YOLOv8-Pose, but pose training will only work after you create keypoint labels in YOLO pose format or convert from a pose-annotated source.

## Files added

- `data_pose.yaml`: pose training config for a single `person` class with COCO-17 keypoints.
- `train_pose.py`: a small wrapper around Ultralytics training.
- `pose_infer.py`: runs pose inference on a video, image, folder, or camera source and writes JSON output.
- `annotation_guided_pose.py`: tracks annotated player boxes with ByteTrack and runs pose inference inside each annotated ROI.
- `pose_to_skeleton.py`: converts tracked pose JSON into normalized skeleton windows for action recognition.

## Quick run

Train:

```powershell
python train_pose.py --weights yolov8n-pose.pt --data data_pose.yaml --epochs 80 --imgsz 640 --batch 8
```

Infer on the sample video:

```powershell
python pose_infer.py --source videos/kabaddi.mp4 --weights yolov8n-pose.pt --track --save-vis
```

The inference script writes a JSON file at `runs/pose_infer/predictions.json` by default.

Separate-pass run on your extracted frames folder:

```powershell
python pose_infer.py --source frames --weights yolov8n-pose.pt --track --save-vis --output-json runs/pose_infer/frames_predictions.json
```

The runner sorts image files by their numeric frame number before inference, so `frame_2.jpg` stays before `frame_10.jpg`.

Convert the tracked pose JSON into skeleton windows for action recognition:

```powershell
python pose_to_skeleton.py --input-json runs/pose_infer/predictions.json --output-dir runs/skeleton_sequences --window-size 64 --stride 32
```

The exporter saves one compressed `.npz` per window plus a `manifest.json` with the exported track segments.

## Annotation-guided pose + ByteTrack

For the current annotation set, the safer pipeline is to use the labeled
player/action boxes as the region of interest, track those boxes with ByteTrack,
and run YOLOv8-Pose only inside each annotated ROI:

```powershell
python annotation_guided_pose.py --frames-dir frames --annotations-dir annotations --weights yolov8n-pose.pt
```

Default outputs:

- JSON predictions: `runs/pose_infer/annotation_guided_predictions.json`
- Visual frames: `runs/pose_infer/annotation_guided_frames_full_vis`

The output JSON is compatible with the skeleton exporter:

```powershell
python pose_to_skeleton.py --input-json runs/pose_infer/annotation_guided_predictions.json --output-dir runs/skeleton_sequences_annotation_guided --window-size 64 --stride 32
```

If you specifically want the rendered frames to land in the same style/location
as the earlier full-frame pose visuals, pass:

```powershell
python annotation_guided_pose.py --out-dir runs/pose_infer/frames_full_vis
```

Open the annotated pose results one by one:

```powershell
python view_pose_results.py --dir runs/pose/runs/pose_infer/predict-2
```

Use the right arrow or space to move forward, left arrow to move back, and `q` or `Esc` to quit.
