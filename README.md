🏃 Kabaddi Raider Tracking & Pose Analysis System

Overview

This project focuses on player detection, tracking, and pose estimation in Kabaddi videos using YOLOv8, ByteTrack, and YOLOv8-Pose.

The primary objective is to maintain consistent player identities during occlusions and overlaps while generating pose and skeleton representations for future action recognition and tactical analysis.

---

✨ Features

Player Detection

- YOLOv8-based player detection
- Bounding box generation for every frame
- Custom dataset training support

Multi-Object Tracking

- ByteTrack integration
- Persistent player IDs across frames
- Kalman Filter-based motion prediction
- Lost-track recovery experiments

Pose Estimation

- YOLOv8-Pose model
- 17 body keypoints extraction
- Frame-wise pose prediction

Annotation-Guided Pose Recovery

- Uses tracked/annotated bounding boxes
- Generates pose predictions even when detector confidence is low
- Improves continuity during severe occlusions

Skeleton Generation

- Converts keypoints into skeleton representations
- Generates skeleton image sequences
- Supports future action recognition pipelines

Visualization

- Bounding box rendering
- Tracking ID visualization
- Pose skeleton visualization
- Debug video generation

---

📁 Project Structure

BTP/
│
├── dataset/
├── annotations/
├── annotations_backup/
├── frames/
├── frames_part2/
├── tracked_frames/
│
├── runs/
│ ├── detect/
│ ├── pose/
│ ├── bytetrack_test/
│ └── pose_infer/
│ ├── frames_full_vis/
│ ├── annotation_guided_frames_full_vis/
│ ├── skeleton_sequences_full/
│ ├── skeleton_sequences_annotation_guided/
│ └── skeleton_sequences_test/
│
├── video/
├── videos/
│
├── yolov8n.pt
├── yolov8n-pose.pt
│
└── Python Scripts

---

🚀 Main Scripts

Detection & Tracking

"testbytetrack.py"

Runs YOLO detection with ByteTrack tracking.

Outputs

- Bounding boxes
- Player IDs
- Tracking visualizations

---

"bytetrack_test.py"

Experimental ByteTrack implementation used for testing and debugging tracking performance.

---

"tstyolo.py"

Runs YOLO inference on images and videos.

---

"trainyolo.py"

Trains a custom YOLO detection model using the prepared dataset.

---

Pose Estimation

"pose_infer.py"

Performs pose estimation on tracked players.

Outputs

- "raid_pose.json"
- Frame-wise pose predictions

---

"annotation_guided_pose.py"

Runs pose estimation using annotation-guided bounding boxes.

Useful when:

- Players overlap
- Detections are missed
- Occlusions occur

---

"view_pose_results.py"

Visualizes generated pose predictions.

---

"render_pose_vis.py"

Creates pose visualization images and videos.

---

Skeleton Processing

"pose_to_skeleton.py"

Converts pose keypoints into skeleton representations.

---

"action_compilation.py"

Compiles skeleton sequences for future action recognition experiments.

---

Dataset & Annotation Utilities

"datasetcreate.py"

Creates and organizes the training dataset.

---

"extract_frames.py"

Extracts frames from input videos.

---

"render_annotations_video.py"

Renders annotations directly onto videos.

---

"mergeToetoe.py"

Utility script for annotation processing and dataset preparation.

---

"count.py"

Dataset statistics and counting utility.

---

"team_color.py"

Team color classification experiments.

---

"checkchest.py"

Experimental script for extracting torso/chest-based features for player re-identification.

---

Debug Utilities

"checkfps.py"

Checks video FPS and related metadata.

---

"debug_tracks.mp4"

Debug output video showing tracking results.

---

📊 Generated Outputs

Tracking Outputs

tracked_frames/
runs/bytetrack_test/

Contains:

- Bounding boxes
- Tracking IDs
- Tracking visualizations

---

Pose Outputs

raid_pose.json
frames_predictions.json
annotation_guided_predictions.json

Contains:

- Keypoints
- Confidence scores
- Frame-wise pose predictions

---

Skeleton Outputs

runs/pose_infer/skeleton_sequences_full/
runs/pose_infer/skeleton_sequences_annotation_guided/
runs/pose_infer/skeleton_sequences_test/

Contains:

- Skeleton images
- Sequence data for future action recognition

---

🔬 Current Research Focus

The current work focuses on improving identity preservation during severe player overlaps.

Existing Method

- IoU-based track reassignment
- Kalman Filter prediction

Ongoing Improvements

Combining:

- IoU similarity
- Head keypoint location
- Chest/Torso keypoint location
- Pose consistency

to reduce ID switching during player overlap and occlusion scenarios.

---

📈 Recent Updates

Tracking

- Added lost-track recovery experiments.
- Improved ID consistency during occlusions.
- Added tracking debug logs.

Pose Estimation

- Integrated YOLOv8-Pose pipeline.
- Added annotation-guided pose estimation.
- Generated pose JSON outputs for analysis.

Visualization

- Added skeleton rendering pipeline.
- Generated pose visualization outputs.
- Created debugging videos.

Dataset Pipeline

- Automated frame extraction.
- Dataset organization utilities.
- Annotation rendering support.

---

🎯 Future Work

- Action recognition using skeleton sequences
- Raider movement analysis
- Automatic raid detection
- Team strategy analytics
- Pose-based player re-identification
- Deep feature matching for long-term tracking
- Multi-camera player tracking

---

🛠️ Technologies Used

- Python
- OpenCV
- YOLOv8
- YOLOv8-Pose
- ByteTrack
- NumPy
- Kalman Filter Tracking

---
