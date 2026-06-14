from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train-4/weights/best.pt")

# Run detection only
model.predict(
    source="test3.mp4",
    save=True,
    conf=0.25,
    show=False
)
