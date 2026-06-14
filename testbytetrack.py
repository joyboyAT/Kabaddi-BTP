from ultralytics import YOLO

model = YOLO("runs/detect/train-4/weights/best.pt")

model.track(
    source="test_raid.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    save=True
)
