# from ultralytics import YOLO

# model = YOLO(r"runs/detect/train-4/weights/best.pt")
# print(model.task)
# print(model.names)
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

print(model.task)
