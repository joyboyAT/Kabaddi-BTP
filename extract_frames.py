import cv2
import os

# video path
video_path = "videos/kabaddi.mp4"

# time window to extract from
start_time_sec = 61
end_time_sec = 13 * 60 + 50

# number of frames to save
target_frames = 5000

# output folder
output_folder = "frames"

# create folder if not exists
os.makedirs(output_folder, exist_ok=True)

# open video
cap = cv2.VideoCapture(video_path)

# check video
if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

# total frames in video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Total Frames in Video:", total_frames)

if total_frames == 0:
    print("ERROR: Video has no frames")
    cap.release()
    exit()

if fps <= 0:
    print("ERROR: Cannot determine video FPS")
    cap.release()
    exit()

start_frame = round(start_time_sec * fps)
end_frame = round(end_time_sec * fps)

start_frame = max(0, min(start_frame, total_frames - 1))
end_frame = max(0, min(end_frame, total_frames - 1))

if end_frame <= start_frame:
    print("ERROR: Invalid time range")
    cap.release()
    exit()

segment_frames = end_frame - start_frame + 1
target_frames = min(target_frames, segment_frames)

for filename in os.listdir(output_folder):
    if filename.startswith("frame_") and filename.endswith(".jpg"):
        os.remove(os.path.join(output_folder, filename))

if target_frames == 1:
    target_positions = [start_frame]
else:
    target_positions = [round(start_frame + i * (segment_frames - 1) / (target_frames - 1))
                        for i in range(target_frames)]

saved_count = 0

for frame_index in target_positions:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()

    if not ret:
        continue

    frame_name = f"{output_folder}/frame_{saved_count}.jpg"

    cv2.imwrite(frame_name, frame)

    saved_count += 1

    print(f"Saved: {frame_name}")

cap.release()

print("DONE")
print("Total Saved Frames:", saved_count)
