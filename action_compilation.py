import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np


FRAMES_FOLDER = "frames"
ANNOTATIONS_FOLDER = "annotations"
CLASSES_FILE = os.path.join(ANNOTATIONS_FOLDER, "classes.txt")
OUTPUT_VIDEO = "action_compilation_annotated.mp4"

START_FRAME = 1
END_FRAME = 3750
FPS = 10
PAUSE_SECONDS = 2
CONTEXT_SECONDS_BEFORE = 2
CONTEXT_SECONDS_AFTER = 2

CONTEXT_FRAMES_BEFORE = FPS * CONTEXT_SECONDS_BEFORE
CONTEXT_FRAMES_AFTER = FPS * CONTEXT_SECONDS_AFTER
ACTION_SECTION_GAP_FRAMES = 75
IDLE_LABEL = "idle"


def load_class_names(classes_file):
    if not os.path.exists(classes_file):
        return []

    with open(classes_file, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def clamp(value, low, high):
    return max(low, min(high, value))


def yolo_to_box(parts, image_width, image_height):
    x_center, y_center, box_width, box_height = map(float, parts)

    xmin = int((x_center - box_width / 2) * image_width)
    ymin = int((y_center - box_height / 2) * image_height)
    xmax = int((x_center + box_width / 2) * image_width)
    ymax = int((y_center + box_height / 2) * image_height)

    return (
        clamp(xmin, 0, image_width - 1),
        clamp(ymin, 0, image_height - 1),
        clamp(xmax, 0, image_width - 1),
        clamp(ymax, 0, image_height - 1),
    )


def parse_yolo_annotation(annotation_path, class_names, image_width, image_height):
    boxes = []

    with open(annotation_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id_text, *bbox_parts = parts
            try:
                class_id = int(class_id_text)
                box = yolo_to_box(bbox_parts, image_width, image_height)
            except ValueError:
                continue

            label = (
                class_names[class_id]
                if 0 <= class_id < len(class_names)
                else f"class_{class_id}"
            )
            boxes.append((label, box))

    return boxes


def parse_xml_annotation(annotation_path):
    boxes = []
    root = ET.parse(annotation_path).getroot()

    for object_node in root.findall("object"):
        label = object_node.findtext("name", default="unknown").strip()
        box_node = object_node.find("bndbox")
        if box_node is None:
            continue

        try:
            xmin = int(float(box_node.findtext("xmin")))
            ymin = int(float(box_node.findtext("ymin")))
            xmax = int(float(box_node.findtext("xmax")))
            ymax = int(float(box_node.findtext("ymax")))
        except (TypeError, ValueError):
            continue

        boxes.append((label, (xmin, ymin, xmax, ymax)))

    return boxes


def load_annotations(frame_number, image_width, image_height, class_names):
    base_name = f"frame_{frame_number}"
    txt_path = os.path.join(ANNOTATIONS_FOLDER, f"{base_name}.txt")
    xml_path = os.path.join(ANNOTATIONS_FOLDER, f"{base_name}.xml")

    if os.path.exists(txt_path):
        return parse_yolo_annotation(txt_path, class_names, image_width, image_height)

    if os.path.exists(xml_path):
        return parse_xml_annotation(xml_path)

    return []


def is_idle_label(label):
    return label.strip().lower() == IDLE_LABEL


def get_action_boxes(boxes):
    return [(label, box) for label, box in boxes if not is_idle_label(label)]


def get_action_signature(boxes):
    signature = []
    seen = set()

    for label, _ in boxes:
        normalized = label.strip().lower()
        if normalized not in seen:
            signature.append(normalized)
            seen.add(normalized)

    return tuple(signature)


def get_action_name(boxes):
    labels = []
    seen = set()

    for label, _ in boxes:
        normalized = label.strip().lower()
        if normalized in seen:
            continue
        labels.append(label.strip())
        seen.add(normalized)

    return " + ".join(labels).upper()


def draw_labelled_boxes(frame, boxes):
    height, width = frame.shape[:2]

    for label, (xmin, ymin, xmax, ymax) in boxes:
        xmin = clamp(int(xmin), 0, width - 1)
        ymin = clamp(int(ymin), 0, height - 1)
        xmax = clamp(int(xmax), 0, width - 1)
        ymax = clamp(int(ymax), 0, height - 1)

        color = (80, 220, 80) if label.strip(
        ).lower() != IDLE_LABEL else (80, 80, 220)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2,
        )
        text_width, text_height = text_size
        label_ymin = max(ymin - text_height - baseline - 6, 0)
        label_xmax = min(xmin + text_width + 8, width - 1)

        cv2.rectangle(
            frame,
            (xmin, label_ymin),
            (label_xmax, label_ymin + text_height + baseline + 6),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (xmin + 4, label_ymin + text_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def make_sequence_card(width, height, sequence_number, action_name, start_frame, end_frame):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        (f"ACTION SEQUENCE {sequence_number}", 2.0, 4),
        (action_name, 2.2, 4),
        (f"Frames: {start_frame} - {end_frame}", 1.2, 3),
    ]
    max_text_width = int(width * 0.88)
    rendered_lines = []

    for text, scale, thickness in lines:
        while scale > 0.6:
            text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
            if text_size[0] <= max_text_width:
                break
            scale -= 0.1

        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        rendered_lines.append((text, scale, thickness, text_size, baseline))

    line_gap = 45
    total_height = sum(text_size[1] + baseline for _,
                       _, _, text_size, baseline in rendered_lines)
    total_height += line_gap * (len(rendered_lines) - 1)
    y = (height - total_height) // 2

    for text, scale, thickness, text_size, baseline in rendered_lines:
        text_width, text_height = text_size
        x = (width - text_width) // 2
        y += text_height
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += baseline + line_gap

    return frame


def build_action_sequences(class_names):
    """
    Build action sequences using YOLO non-idle boxes as the signal.

    Each sequence:
      - groups consecutive frames with the same "action signature"
        (and breaks when the gap is too large)
      - expands start/end by CONTEXT_FRAMES_BEFORE/AFTER
      - stores an 'action_frames' set for frames that actually have non-idle boxes
        (used only for deciding whether to show label/bboxes)
    """
    sequences = []
    current_sequence = None

    skipped_missing_frames = 0
    skipped_without_action = 0

    # Track last annotated frame to help with gap logic
    last_seen_frame = None

    for frame_number in range(START_FRAME, END_FRAME + 1):
        frame_path = os.path.join(FRAMES_FOLDER, f"frame_{frame_number}.jpg")
        if not os.path.exists(frame_path):
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        height, width = frame.shape[:2]
        boxes = get_action_boxes(load_annotations(
            frame_number, width, height, class_names))

        # If no action boxes (or only Idle), treat as no-action at this frame.
        if not boxes:
            continue

        action_signature = get_action_signature(boxes)
        action_name = get_action_name(boxes)

        if current_sequence is None:
            # Start first sequence
            start_frame = max(START_FRAME, frame_number -
                              CONTEXT_FRAMES_BEFORE)
            end_frame = min(END_FRAME, frame_number + CONTEXT_FRAMES_AFTER)

            current_sequence = {
                "sequence_number": 1,  # updated below when appended
                "action_name": action_name,
                "action_signature": action_signature,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "action_frames": {frame_number},
            }
            last_seen_frame = frame_number
            continue

        # Decide whether to start a new sequence
        gap_frames = (
            last_seen_frame is not None and frame_number - last_seen_frame
        )
        starts_new_sequence = (
            action_signature != current_sequence["action_signature"]
            or (gap_frames is not None and gap_frames > ACTION_SECTION_GAP_FRAMES)
        )

        if starts_new_sequence:
            # finalize previous sequence
            current_sequence["end_frame"] = min(
                END_FRAME, current_sequence["end_frame"]
            )

            # append and start new
            current_sequence["sequence_number"] = len(sequences) + 1
            sequences.append(current_sequence)

            start_frame = max(START_FRAME, frame_number -
                              CONTEXT_FRAMES_BEFORE)
            end_frame = min(END_FRAME, frame_number + CONTEXT_FRAMES_AFTER)

            current_sequence = {
                "sequence_number": 0,  # filled on append
                "action_name": action_name,
                "action_signature": action_signature,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "action_frames": {frame_number},
            }
            last_seen_frame = frame_number
        else:
            # extend current sequence context window
            current_sequence["action_name"] = action_name
            current_sequence["action_signature"] = action_signature
            current_sequence["action_frames"].add(frame_number)

            current_sequence["start_frame"] = max(
                START_FRAME,
                min(current_sequence["start_frame"],
                    frame_number - CONTEXT_FRAMES_BEFORE),
            )
            current_sequence["end_frame"] = min(
                END_FRAME,
                max(current_sequence["end_frame"],
                    frame_number + CONTEXT_FRAMES_AFTER),
            )

            last_seen_frame = frame_number

    if current_sequence is not None:
        current_sequence["sequence_number"] = len(sequences) + 1
        sequences.append(current_sequence)

    stats = {
        "skipped_missing_frames": skipped_missing_frames,
        "skipped_without_action": skipped_without_action,
    }

    return sequences, stats


def write_sequence_video(sequences):
    if not sequences:
        return 0, 0

    class_names = load_class_names(CLASSES_FILE)

    # Determine output size from the first available frame (context ranges may exist)
    def load_first_frame_for_size():
        for seq in sequences:
            for fn in range(seq["start_frame"], seq["end_frame"] + 1):
                p = os.path.join(FRAMES_FOLDER, f"frame_{fn}.jpg")
                if not os.path.exists(p):
                    continue
                img = cv2.imread(p)
                if img is not None:
                    return img.shape[1], img.shape[0]
        # fallback (shouldn't happen if dataset exists)
        return 640, 480

    first_width, first_height = load_first_frame_for_size()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS,
                            (first_width, first_height))

    title_card_frames = int(FPS * PAUSE_SECONDS)
    written_action_frames = 0
    written_title_frames = 0

    # Render all sequences in order, with title cards between them.
    # To avoid missing frames due to overlaps, we write strictly per-sequence range
    # as requested by the task.
    for i, sequence in enumerate(sequences):
        print(
            f"Showing ACTION SEQUENCE {sequence['sequence_number']} "
            f"| {sequence['action_name']} "
            f"| frames {sequence['start_frame']} - {sequence['end_frame']}"
        )

        # Title card before the action clip (including first sequence)
        title_card = make_sequence_card(
            first_width,
            first_height,
            sequence["sequence_number"],
            sequence["action_name"],
            sequence["start_frame"],
            sequence["end_frame"],
        )
        for _ in range(title_card_frames):
            video.write(title_card)
            written_title_frames += 1

        # Write every frame in expanded context range
        for frame_number in range(sequence["start_frame"], sequence["end_frame"] + 1):
            # Clamp defensively (requirements)
            frame_number = clamp(frame_number, 0, END_FRAME)

            frame_path = os.path.join(
                FRAMES_FOLDER, f"frame_{frame_number}.jpg")
            if not os.path.exists(frame_path):
                # Requirement says: still write frame if possible.
                # If the file truly doesn't exist, we cannot draw; write a black frame.
                frame = np.zeros(
                    (first_height, first_width, 3), dtype=np.uint8)
                video.write(frame)
                continue

            frame = cv2.imread(frame_path)
            if frame is None:
                frame = np.zeros(
                    (first_height, first_width, 3), dtype=np.uint8)
                video.write(frame)
                continue

            # Resize to output size if needed
            if (frame.shape[1], frame.shape[0]) != (first_width, first_height):
                frame = cv2.resize(frame, (first_width, first_height))

            # During action frames: draw boxes/labels
            # During context frames: show original frames (no need to draw boxes)
            if frame_number in sequence.get("action_frames", set()):
                height, width = frame.shape[:2]
                boxes = get_action_boxes(
                    load_annotations(frame_number, width, height, class_names)
                )
                if boxes:
                    draw_labelled_boxes(frame, boxes)

            video.write(frame)
            written_action_frames += 1

    video.release()
    return written_action_frames, written_title_frames


def main():
    class_names = load_class_names(CLASSES_FILE)
    sequences, stats = build_action_sequences(class_names)

    if not sequences:
        print("No action sequences found.")
        return

    total_action_frames = sum(
        len(sequence.get("action_frames", set())) for sequence in sequences
    )

    print("Action frames found:", total_action_frames)
    print("Action sequences:", len(sequences))
    print("Output FPS:", FPS)
    print("Title card length:", f"{PAUSE_SECONDS} seconds")

    written_action_frames, written_title_frames = write_sequence_video(
        sequences)
    total_output_frames = written_action_frames + written_title_frames
    duration_seconds = total_output_frames / FPS

    print("Video saved:", OUTPUT_VIDEO)
    print("Action frames written:", written_action_frames)
    print("Title card frames written:", written_title_frames)
    print("Total output frames:", total_output_frames)
    print("Final duration:", f"{duration_seconds:.2f} seconds")
    print("Missing/unreadable frames skipped:",
          stats["skipped_missing_frames"])
    print("Idle/no-annotation frames skipped:",
          stats["skipped_without_action"])


if __name__ == "__main__":
    main()
