import os
from collections import defaultdict

ANNOTATIONS_FOLDER = "annotations"
START_FRAME = 1
END_FRAME = 3800

with open(
    os.path.join(ANNOTATIONS_FOLDER, "classes.txt"),
    "r",
    encoding="utf-8"
) as f:
    class_names = [line.strip() for line in f if line.strip()]

action_counts = defaultdict(int)

# For each class track whether previous frame had it
active_sequences = {}

for class_name in class_names:
    active_sequences[class_name] = False

for frame_number in range(START_FRAME, END_FRAME + 1):

    txt_file = os.path.join(
        ANNOTATIONS_FOLDER,
        f"frame_{frame_number}.txt"
    )

    current_actions = set()

    if os.path.exists(txt_file):

        with open(txt_file, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:
                    class_id = int(parts[0])
                except:
                    continue

                if class_id >= len(class_names):
                    continue

                label = class_names[class_id]

                if label.lower() == "idle":
                    continue

                current_actions.add(label)

    for action in class_names:

        if action.lower() == "idle":
            continue

        # New sequence starts
        if action in current_actions and not active_sequences[action]:
            action_counts[action] += 1
            active_sequences[action] = True

        # Sequence ends
        elif action not in current_actions:
            active_sequences[action] = False

print("\nINDIVIDUAL ACTION SEQUENCE COUNTS")
print("=" * 40)

total = 0

for action in sorted(action_counts):

    print(f"{action:<20} {action_counts[action]}")
    total += action_counts[action]

print("=" * 40)
print("TOTAL:", total)
