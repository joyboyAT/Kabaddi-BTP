from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


TEAM_INDIA = 0
TEAM_IRAN = 1

SOURCE_VIDEO = "bytetrack_sample_video.mp4"
POSE_MODEL = "yolov8n-pose.pt"
PLAYER_FILTER_MODEL = "best.pt"
OUTPUT_VIDEO = "kabaddi_pose_team_output.mp4"
FRAME_EXPORT_DIR = "pant_crops"

WHITE_THRESHOLD = 0.05
BLUE_THRESHOLD = 0.03
WHITE_CONFIDENCE_MARGIN = 0.02

USE_PLAYER_FILTER = True
PLAYER_FILTER_CONFIDENCE = 0.25
REQUIRE_PLAYER_FILTER_MATCH = True
PLAYER_FILTER_IOU_THRESHOLD = 0.20
PLAYER_FILTER_CENTER_MARGIN = 0.08

ALLOW_AMBIGUOUS_TEAM_CONTINUATION = False

MAX_MISSING_FRAMES = 45
MIN_CENTER_GATE = 70.0
CENTER_GATE_FACTOR = 0.8
MOTION_GATE_PER_FRAME = 8.0
RAW_ID_MATCH_BONUS = 0.35

TEAM_LABELS = {
    TEAM_INDIA: "INDIA",
    TEAM_IRAN: "IRAN",
}

TEAM_ID_PREFIX = {
    TEAM_INDIA: "B",
    TEAM_IRAN: "W",
}

TEAM_COLORS = {
    TEAM_INDIA: (255, 0, 0),
    TEAM_IRAN: (0, 0, 255),
}


def other_team(team: int) -> int:
    return TEAM_IRAN if team == TEAM_INDIA else TEAM_INDIA


def box_xyxy_array(box) -> np.ndarray:
    return box.xyxy[0].cpu().numpy().astype(float)


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.array(
        [
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
        ],
        dtype=float,
    )


def bbox_diagonal(bbox: np.ndarray) -> float:
    width = max(1.0, float(bbox[2] - bbox[0]))
    height = max(1.0, float(bbox[3] - bbox[1]))
    return float(np.hypot(width, height))


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def expand_bbox(bbox: np.ndarray, margin_ratio: float) -> np.ndarray:
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    x_margin = width * margin_ratio
    y_margin = height * margin_ratio

    return np.array(
        [
            bbox[0] - x_margin,
            bbox[1] - y_margin,
            bbox[2] + x_margin,
            bbox[3] + y_margin,
        ],
        dtype=float,
    )


def point_inside_bbox(point: np.ndarray, bbox: np.ndarray) -> bool:
    return bool(
        bbox[0] <= point[0] <= bbox[2]
        and bbox[1] <= point[1] <= bbox[3]
    )


def bbox_matches_player_filter(
    pose_bbox: np.ndarray,
    player_filter_bboxes: list[np.ndarray],
) -> bool:
    if not player_filter_bboxes:
        return not REQUIRE_PLAYER_FILTER_MATCH

    pose_center = bbox_center(pose_bbox)
    expanded_pose_bbox = expand_bbox(pose_bbox, PLAYER_FILTER_CENTER_MARGIN)

    for player_bbox in player_filter_bboxes:
        if bbox_iou(pose_bbox, player_bbox) >= PLAYER_FILTER_IOU_THRESHOLD:
            return True

        expanded_player_bbox = expand_bbox(player_bbox, PLAYER_FILTER_CENTER_MARGIN)
        if point_inside_bbox(pose_center, expanded_player_bbox):
            return True

        if point_inside_bbox(bbox_center(player_bbox), expanded_pose_bbox):
            return True

    return False


def detect_player_filter_bboxes(model: YOLO | None, frame: np.ndarray) -> list[np.ndarray]:
    if model is None:
        return []

    results = model.predict(
        source=frame,
        conf=PLAYER_FILTER_CONFIDENCE,
        verbose=False,
    )

    if not results or results[0].boxes is None:
        return []

    return [box_xyxy_array(box) for box in results[0].boxes]


@dataclass
class StableTrack:
    team: int
    number: int
    bbox: np.ndarray
    center: np.ndarray
    velocity: np.ndarray
    last_seen: int
    hits: int = 1
    raw_track_ids: set[int] = field(default_factory=set)

    @property
    def display_id(self) -> str:
        return f"{TEAM_ID_PREFIX[self.team]}{self.number:02d}"

    def predicted_center(self, frame_number: int) -> np.ndarray:
        frames_missing = max(0, frame_number - self.last_seen)
        return self.center + self.velocity * frames_missing

    def update(self, bbox: np.ndarray, frame_number: int, raw_track_id: int) -> None:
        frames_elapsed = max(1, frame_number - self.last_seen)
        new_center = bbox_center(bbox)
        observed_velocity = (new_center - self.center) / frames_elapsed

        self.velocity = (self.velocity * 0.65) + (observed_velocity * 0.35)
        self.bbox = bbox
        self.center = new_center
        self.last_seen = frame_number
        self.hits += 1

        if raw_track_id >= 0:
            self.raw_track_ids.add(raw_track_id)


class TeamAwareIdentityTracker:
    """Keeps display IDs stable and locked to one team color.

    ByteTrack IDs are useful hints, but they are not allowed to carry a blue
    display ID onto a white player or the reverse. Matching happens inside the
    team bucket first, then uses predicted spatial position to survive raw-ID
    changes.
    """

    def __init__(self) -> None:
        self.tracks: dict[int, dict[int, StableTrack]] = {
            TEAM_INDIA: {},
            TEAM_IRAN: {},
        }
        self.next_number = {
            TEAM_INDIA: 1,
            TEAM_IRAN: 1,
        }
        self.raw_id_to_track: dict[int, dict[int, int]] = {}
        self.assignment_frame = -1
        self.assigned_this_frame: set[tuple[int, int]] = set()

    def assign(
        self,
        raw_track_id: int,
        observed_team: int | None,
        white_ratio: float | None,
        bbox: np.ndarray,
        frame_number: int,
        allow_new_track: bool = True,
    ) -> tuple[str | None, int | None]:
        if frame_number != self.assignment_frame:
            self.assignment_frame = frame_number
            self.assigned_this_frame.clear()
            self._purge_stale_tracks(frame_number)

        candidate_teams = self._candidate_teams(observed_team, white_ratio)
        track = self._match_by_raw_id(
            raw_track_id,
            candidate_teams,
            bbox,
            frame_number,
        )

        if track is None:
            track = self._match_by_position(candidate_teams, bbox, frame_number)

        if track is None and not allow_new_track:
            return None, None

        if track is None:
            track = self._create_track(
                observed_team if observed_team is not None else TEAM_INDIA,
                bbox,
                frame_number,
                raw_track_id,
            )
        else:
            track.update(bbox, frame_number, raw_track_id)

        self.assigned_this_frame.add((track.team, track.number))

        if raw_track_id >= 0:
            self.raw_id_to_track.setdefault(raw_track_id, {})[track.team] = track.number

        return track.display_id, track.team

    def _candidate_teams(
        self,
        observed_team: int | None,
        white_ratio: float | None,
    ) -> list[int]:
        if observed_team is None:
            return [TEAM_INDIA, TEAM_IRAN]

        if self._has_strong_team_evidence(observed_team, white_ratio):
            return [observed_team]

        return [observed_team, other_team(observed_team)]

    def _has_strong_team_evidence(
        self,
        observed_team: int,
        white_ratio: float | None,
    ) -> bool:
        if white_ratio is None:
            return False

        if observed_team == TEAM_IRAN:
            return white_ratio >= WHITE_THRESHOLD + WHITE_CONFIDENCE_MARGIN

        return white_ratio <= max(0.0, WHITE_THRESHOLD - WHITE_CONFIDENCE_MARGIN)

    def _match_by_raw_id(
        self,
        raw_track_id: int,
        candidate_teams: list[int],
        bbox: np.ndarray,
        frame_number: int,
    ) -> StableTrack | None:
        if raw_track_id < 0:
            return None

        team_mapping = self.raw_id_to_track.get(raw_track_id)
        if not team_mapping:
            return None

        best_track: StableTrack | None = None
        best_score = float("inf")

        for team in candidate_teams:
            number = team_mapping.get(team)
            if number is None:
                continue

            track = self.tracks[team].get(number)
            score = self._match_score(track, bbox, frame_number, raw_id_bonus=True)
            if score is not None and score < best_score:
                best_track = track
                best_score = score

        return best_track

    def _match_by_position(
        self,
        candidate_teams: list[int],
        bbox: np.ndarray,
        frame_number: int,
    ) -> StableTrack | None:
        best_track: StableTrack | None = None
        best_score = float("inf")

        for team in candidate_teams:
            for track in self.tracks[team].values():
                score = self._match_score(track, bbox, frame_number, raw_id_bonus=False)
                if score is not None and score < best_score:
                    best_track = track
                    best_score = score

        return best_track

    def _match_score(
        self,
        track: StableTrack | None,
        bbox: np.ndarray,
        frame_number: int,
        raw_id_bonus: bool,
    ) -> float | None:
        if track is None:
            return None

        if (track.team, track.number) in self.assigned_this_frame:
            return None

        frames_missing = frame_number - track.last_seen
        if frames_missing < 0 or frames_missing > MAX_MISSING_FRAMES:
            return None

        center = bbox_center(bbox)
        distance = float(np.linalg.norm(center - track.predicted_center(frame_number)))
        size_gate = CENTER_GATE_FACTOR * max(bbox_diagonal(bbox), bbox_diagonal(track.bbox))
        motion_gate = MOTION_GATE_PER_FRAME * max(0, frames_missing)
        gate = max(MIN_CENTER_GATE, size_gate) + motion_gate
        overlap = bbox_iou(bbox, track.bbox)

        if distance > gate and overlap < 0.05:
            return None

        score = (distance / max(1.0, gate)) - overlap
        if raw_id_bonus:
            score -= RAW_ID_MATCH_BONUS

        return score

    def _create_track(
        self,
        team: int,
        bbox: np.ndarray,
        frame_number: int,
        raw_track_id: int,
    ) -> StableTrack:
        number = self.next_number[team]
        self.next_number[team] += 1

        track = StableTrack(
            team=team,
            number=number,
            bbox=bbox,
            center=bbox_center(bbox),
            velocity=np.zeros(2, dtype=float),
            last_seen=frame_number,
        )

        if raw_track_id >= 0:
            track.raw_track_ids.add(raw_track_id)

        self.tracks[team][number] = track
        return track

    def _purge_stale_tracks(self, frame_number: int) -> None:
        for team, team_tracks in self.tracks.items():
            stale_numbers = [
                number
                for number, track in team_tracks.items()
                if frame_number - track.last_seen > MAX_MISSING_FRAMES
            ]

            for number in stale_numbers:
                del team_tracks[number]

        for raw_track_id, team_mapping in list(self.raw_id_to_track.items()):
            for team, number in list(team_mapping.items()):
                if number not in self.tracks[team]:
                    del team_mapping[team]

            if not team_mapping:
                del self.raw_id_to_track[raw_track_id]


def get_pant_crop(frame: np.ndarray, keypoints) -> np.ndarray | None:
    """Crop the pant area using COCO hip keypoints 11 and 12."""
    if keypoints is None or len(keypoints) <= 12:
        return None

    left_hip = keypoints[11].cpu().numpy()
    right_hip = keypoints[12].cpu().numpy()

    if np.all(left_hip == 0) or np.all(right_hip == 0):
        return None

    mid_x = int((left_hip[0] + right_hip[0]) / 2)
    mid_y = int((left_hip[1] + right_hip[1]) / 2)
    hip_width = int(abs(left_hip[0] - right_hip[0]))

    if hip_width < 2:
        return None

    x1 = max(0, mid_x - int(hip_width * 0.7))
    x2 = min(frame.shape[1], mid_x + int(hip_width * 0.7))
    y1 = max(0, mid_y)
    y2 = min(frame.shape[0], mid_y + int(hip_width * 1.8))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def get_lower_body_crop(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    """Fallback crop when pose hip keypoints are missing or unreliable."""
    x1, y1, x2, y2 = map(int, bbox)
    width = x2 - x1
    height = y2 - y1

    if width < 2 or height < 2:
        return None

    lower_y1 = y1 + int(height * 0.45)
    x1 = max(0, x1)
    y1 = max(0, lower_y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def team_color_ratios(pant_crop: np.ndarray | None) -> tuple[float, float] | None:
    if pant_crop is None or pant_crop.size == 0:
        return None

    hsv = cv2.cvtColor(pant_crop, cv2.COLOR_BGR2HSV)
    white_mask = (hsv[:, :, 1] <= 50) & (hsv[:, :, 2] >= 200)
    blue_mask = (
        (hsv[:, :, 0] >= 90)
        & (hsv[:, :, 0] <= 135)
        & (hsv[:, :, 1] >= 60)
        & (hsv[:, :, 2] >= 50)
    )
    pixel_count = max(1, pant_crop.shape[0] * pant_crop.shape[1])
    white_ratio = np.count_nonzero(white_mask) / pixel_count
    blue_ratio = np.count_nonzero(blue_mask) / pixel_count
    return white_ratio, blue_ratio


def white_pixel_ratio(pant_crop: np.ndarray | None) -> float | None:
    ratios = team_color_ratios(pant_crop)
    return None if ratios is None else ratios[0]


def classify_team(pant_crop: np.ndarray | None) -> tuple[int | None, float | None]:
    """Return the observed team and white-pixel ratio.

    None means the crop was unavailable, so the identity tracker should use
    spatial continuity from previous frames instead of forcing a team switch.
    """
    ratios = team_color_ratios(pant_crop)

    if ratios is None:
        return None, None

    white_ratio, blue_ratio = ratios

    if white_ratio >= WHITE_THRESHOLD and white_ratio >= blue_ratio:
        return TEAM_IRAN, white_ratio

    if blue_ratio >= BLUE_THRESHOLD:
        return TEAM_INDIA, white_ratio

    return None, white_ratio


def is_team_iran(pant_crop: np.ndarray | None) -> bool:
    """Classify Iran by checking the white-pixel ratio in the pant crop."""
    team, _ = classify_team(pant_crop)
    return team == TEAM_IRAN


def draw_player_label(
    frame: np.ndarray,
    bbox: np.ndarray,
    display_id: str,
    team: int,
) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    color = TEAM_COLORS[team]
    label = TEAM_LABELS[team]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{label} ID:{display_id}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def open_video_writer(source: str) -> cv2.VideoWriter:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def main() -> None:
    if not Path(SOURCE_VIDEO).exists():
        raise FileNotFoundError(f"Missing source video: {SOURCE_VIDEO}")

    if not Path(POSE_MODEL).exists():
        raise FileNotFoundError(f"Missing pose model: {POSE_MODEL}")

    if USE_PLAYER_FILTER and not Path(PLAYER_FILTER_MODEL).exists():
        raise FileNotFoundError(f"Missing player filter model: {PLAYER_FILTER_MODEL}")

    pose_model = YOLO(POSE_MODEL)
    player_filter_model = YOLO(PLAYER_FILTER_MODEL) if USE_PLAYER_FILTER else None
    video_writer = open_video_writer(SOURCE_VIDEO)
    frame_export_dir = Path(FRAME_EXPORT_DIR)
    frame_export_dir.mkdir(exist_ok=True)
    identity_tracker = TeamAwareIdentityTracker()
    frame_number = 0

    try:
        for result in pose_model.track(source=SOURCE_VIDEO, tracker="bytetrack.yaml", stream=True):
            frame_number += 1
            frame = result.orig_img
            if frame is None:
                continue

            player_filter_bboxes = detect_player_filter_bboxes(player_filter_model, frame)

            has_detections = result.boxes is not None and len(result.boxes) > 0

            if has_detections:
                has_keypoints = (
                    result.keypoints is not None
                    and result.keypoints.xy is not None
                    and len(result.keypoints.xy) > 0
                )
                player_count = len(result.boxes)

                for i in range(player_count):
                    box = result.boxes[i]
                    raw_track_id = int(box.id[0]) if box.id is not None else -1
                    bbox = box_xyxy_array(box)

                    if not bbox_matches_player_filter(bbox, player_filter_bboxes):
                        continue

                    pant_crop = None

                    if has_keypoints and i < len(result.keypoints.xy):
                        keypoints = result.keypoints.xy[i]
                        pant_crop = get_pant_crop(frame, keypoints)

                    if pant_crop is None or pant_crop.size == 0:
                        pant_crop = get_lower_body_crop(frame, bbox)

                    observed_team, white_ratio = classify_team(pant_crop)

                    if (
                        observed_team is None
                        and not ALLOW_AMBIGUOUS_TEAM_CONTINUATION
                    ):
                        continue

                    display_id, team = identity_tracker.assign(
                        raw_track_id=raw_track_id,
                        observed_team=observed_team,
                        white_ratio=white_ratio,
                        bbox=bbox,
                        frame_number=frame_number,
                        allow_new_track=observed_team is not None,
                    )

                    if display_id is None or team is None:
                        continue

                    draw_player_label(frame, bbox, display_id, team)

            video_writer.write(frame)
            frame_path = frame_export_dir / f"frame_{frame_number:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            cv2.imshow("Kabaddi Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_writer.release()
        cv2.destroyAllWindows()

    print(f"Run complete. Output saved to {OUTPUT_VIDEO}")
    print(f"Annotated full frames saved to {FRAME_EXPORT_DIR}")


if __name__ == "__main__":
    main()
