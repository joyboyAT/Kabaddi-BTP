from pathlib import Path
import shutil

# CHANGE THESE PATHS
images_src = Path("frames")
labels_src = Path("annotations")

train_count = 800
val_count = 200

# Create folders
Path("dataset/images/train").mkdir(parents=True, exist_ok=True)
Path("dataset/images/val").mkdir(parents=True, exist_ok=True)
Path("dataset/labels/train").mkdir(parents=True, exist_ok=True)
Path("dataset/labels/val").mkdir(parents=True, exist_ok=True)

# Get frame numbers
files = []

for file_path in labels_src.glob("frame_*.txt"):
    try:
        num = int(file_path.stem.split("_")[1])
    except (IndexError, ValueError):
        continue

    files.append(num)

files.sort()

# First 800 -> train
for num in files[:train_count]:

    shutil.copy(
        images_src / f"frame_{num}.jpg",
        Path("dataset/images/train") / f"frame_{num}.jpg"
    )

    shutil.copy(
        labels_src / f"frame_{num}.txt",
        Path("dataset/labels/train") / f"frame_{num}.txt"
    )

# Next 200 -> val
for num in files[train_count:train_count+val_count]:

    shutil.copy(
        images_src / f"frame_{num}.jpg",
        Path("dataset/images/val") / f"frame_{num}.jpg"
    )

    shutil.copy(
        labels_src / f"frame_{num}.txt",
        Path("dataset/labels/val") / f"frame_{num}.txt"
    )

print("Dataset created successfully!")
