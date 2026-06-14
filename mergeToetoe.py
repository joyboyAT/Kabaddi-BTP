import os

ANNOTATIONS_FOLDER = "annotations"

OLD_CLASS_ID = 18   # toe touch
NEW_CLASS_ID = 5    # Toe Touch

updated_files = 0
updated_labels = 0

for filename in os.listdir(ANNOTATIONS_FOLDER):

    if not filename.endswith(".txt"):
        continue

    if filename == "classes.txt":
        continue

    path = os.path.join(ANNOTATIONS_FOLDER, filename)

    changed = False
    new_lines = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) < 5:
                new_lines.append(line)
                continue

            if parts[0] == str(OLD_CLASS_ID):
                parts[0] = str(NEW_CLASS_ID)
                changed = True
                updated_labels += 1

            new_lines.append(" ".join(parts) + "\n")

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        updated_files += 1

print("Updated files:", updated_files)
print("Updated labels:", updated_labels)
