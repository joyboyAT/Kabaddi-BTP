import cv2
from ultralytics import YOLO
log = open("iou_log.txt", "w")
model = YOLO("runs/detect/train-4/weights/best.pt")

cap = cv2.VideoCapture("test_raid.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "debug_tracks.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)
frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    annotated = results[0].plot()
    tracker = model.predictor.trackers[0]
    log.write(f"Frame {frame_no}   |  "
              f"Lost tracks: {len(tracker.lost_stracks)}   |  "
              f"Tracked: {len(tracker.tracked_stracks)}\n"
              )
    for track in tracker.lost_stracks:
        log.write(f"  Lost ID {track.track_id}\n")
        x, y, w, h = track.tlwh
        tx1 = x
        ty1 = y
        tx2 = x+w
        ty2 = y+h
        best_iou = 0
        log.write(f"\nFrame {frame_no}\n")
        log.write(f"Lost ID {track.track_id}\n")
        for det_idx, det in enumerate(results[0].boxes):
            dx1, dy1, dx2, dy2 = det.xyxy[0].cpu().numpy()
            ix1 = max(tx1, dx1)
            iy1 = max(ty1, dy1)
            ix2 = min(tx2, dx2)
            iy2 = min(ty2, dy2)

            iw = max(0, ix2-ix1)
            ih = max(0, iy2-iy1)
            inter = iw*ih
            area_track = (tx2-tx1)*(ty2-ty1)
            area_det = (dx2-dx1)*(dy2-dy1)
            union = area_track + area_det - inter
            iou = inter/union if union > 0 else 0
            log.write(f"   Detection{det_idx}:IOU = {iou:.3f}\n")
            best_iou = max(best_iou, iou)

        cv2.rectangle(
            annotated,
            (int(x), int(y)),
            (int(x+w), int(y+h)),
            (0, 255, 255),  # Yellow for lost tracks
            3
        )

        cv2.putText(
            annotated,
            f"LOST {track.track_id} IOU: {best_iou:.2f}",
            (int(x), int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3
        )
    log.write(f"Active IDs:  ")
    for t in tracker.tracked_stracks:
        log.write(f" {t.track_id}  ")
    log.write("\n______________________________________\n")
    out.write(annotated)
    cv2.imshow("debug", annotated)

    if cv2.waitKey(1) == 27:
        break

log.close()
out.release()
cap.release()
cv2.destroyAllWindows()
