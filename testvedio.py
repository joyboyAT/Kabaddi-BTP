import cv2
import os

FRAMES_FOLDER = "dataset/images/val"

START_FRAME = 801
END_FRAME = 1000

OUTPUT_VIDEO = "test_raid.mp4"
FPS = 20

# Find first existing frame
first_frame = None

for i in range(START_FRAME, END_FRAME + 1):
    path = os.path.join(FRAMES_FOLDER, f"frame_{i}.jpg")

    if os.path.exists(path):
        first_frame = cv2.imread(path)
        break

if first_frame is None:
    print("No frames found!")
    exit()

height, width = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    FPS,
    (width, height)
)

written = 0

for frame_num in range(
    START_FRAME,
    END_FRAME + 1
):

    frame_path = os.path.join(
        FRAMES_FOLDER,
        f"frame_{frame_num}.jpg"
    )

    if not os.path.exists(frame_path):
        continue

    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    video.write(frame)
    written += 1

video.release()

print("Video saved:", OUTPUT_VIDEO)
print("Frames written:", written)
