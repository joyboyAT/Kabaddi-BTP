import cv2
import os

frames_folder = "frames"
labels_folder = "annotations"

output_video = "annotated_video.mp4"

class_names = {
    0: "Dubki",
    1: "Back Kick",
    2: "Hand Touch",
    3: "Lion Jump",
    4: "Side Kick",
    5: "Toe Touch",
    6: "Idle",
    7: "Ankle Hold",
    8: "Thigh Hold",
    9: "Waist Hold",
    10: "Block",
    11: "Dash",
    12: "Chain Tackle"
}

# Get sorted image files
image_files = sorted(
    [f for f in os.listdir(frames_folder) if f.endswith(".jpg")],
    key=lambda x: int(x.split("_")[1].split(".")[0])
)

# Read first image
first_img = cv2.imread(os.path.join(frames_folder, image_files[0]))
height, width = first_img.shape[:2]

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(
    output_video,
    fourcc,
    10,
    (width, height)
)

for img_file in image_files:

    frame_num = int(img_file.split("_")[1].split(".")[0])

    if frame_num > 3800:
        break

    img_path = os.path.join(frames_folder, img_file)
    frame = cv2.imread(img_path)

    label_file = img_file.replace(".jpg", ".txt")
    label_path = os.path.join(labels_folder, label_file)

    if os.path.exists(label_path):

        with open(label_path, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                box_width = float(parts[3]) * width
                box_height = float(parts[4]) * height

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)

                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                label = class_names.get(class_id, str(class_id))

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    video.write(frame)

video.release()

print("Video saved as:", output_video)
