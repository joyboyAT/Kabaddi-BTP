import cv2
import numpy as np


def get_team_color(frame, tlbr):
    x1, y1, x2, y2 = map(int, tlbr)

    h, w = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return "unknown"

    # use upper half only
    crop = crop[: crop.shape[0] // 2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    h_mean = np.mean(hsv[:, :, 0])

    # adjust ranges after observing jersey colors
    if h_mean < 20:
        return "red"

    elif h_mean < 50:
        return "yellow"

    elif h_mean < 90:
        return "green"

    else:
        return "blue"
