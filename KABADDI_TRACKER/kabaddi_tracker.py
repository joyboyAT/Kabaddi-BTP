import cv2
import numpy as np
import os
from collections import deque
from ultralytics import YOLO

# Stores persistent history for each player, keyed by persisted_id.
global_player_memory = {}
# Maps current tracker IDs to persisted IDs for the current session.
track_id_to_persisted_id = {}

TEAM_BLUE = 0
TEAM_WHITE_GREEN = 1

TEAM_COLORS = {
    TEAM_BLUE: (255, 0, 0),
    TEAM_WHITE_GREEN: (255, 255, 255),
}

def detect_team_from_crop(crop: np.ndarray) -> tuple[float, float]:
    if crop.size == 0: return 0.0, 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total_pixels = max(1, crop.shape[0] * crop.shape[1])
    
    # Tighten masks to ignore the blue mat
    blue_ratio = np.count_nonzero(((hsv[:,:,0] >= 90) & (hsv[:,:,0] <= 135)) & (hsv[:,:,1] >= 100) & (hsv[:,:,2] > 50)) / total_pixels
    green_ratio = np.count_nonzero(((hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 90)) & (hsv[:,:,1] >= 60)) / total_pixels
    white_ratio = np.count_nonzero((hsv[:,:,1] <= 50) & (hsv[:,:,2] >= 190)) / total_pixels

    india_score = (blue_ratio * 3.0) 
    iran_score = (green_ratio * 5.0) + (white_ratio * 1.5) 
    
    return india_score, iran_score

def crop_center_region(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    x1, x2 = sorted((max(0, x1), min(frame.shape[1], x2)))
    y1, y2 = sorted((max(0, y1), min(frame.shape[0], y2)))
    
    width = x2 - x1
    height = y2 - y1
    torso_width = int(width * 0.30)
    upper_height = int(height * 0.40)
    
    cx1 = x1 + max(0, (width - torso_width) // 2)
    cy1 = y1 + int(height * 0.12)
    return frame[cy1:cy1+upper_height, cx1:cx1+torso_width]



def draw_tracking(frame, box, final_id, team):
    x1, y1, x2, y2 = map(int, box)
    color = TEAM_COLORS.get(team, (0, 255, 0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"ID: {final_id}", (x1, max(0, y1 - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0) if team == TEAM_WHITE_GREEN else (255,255,255), 2)

def main() -> None:
    source_video = "bytetrack_sample_video.mp4"
    model = YOLO("best.pt")
    
    cap = cv2.VideoCapture(source_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    video_writer = cv2.VideoWriter('kabaddi_final_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    os.makedirs("saved_frames", exist_ok=True)
    frame_count = 0 

    for result in model.track(source=source_video, tracker="bytetrack.yaml", stream=True):
        frame = result.orig_img
        if frame is None: continue

        for entry in global_player_memory.values():
            entry['seen_this_frame'] = False

        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes:
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # 1. PRE-DETECT TEAM
                crop = crop_center_region(frame, box.xyxy[0].cpu().numpy())
                ind_raw, irn_raw = detect_team_from_crop(crop)
                current_team_raw = TEAM_BLUE if ind_raw >= irn_raw else TEAM_WHITE_GREEN

                # 2. MATCHING WITH TEAM VERIFICATION
                if track_id not in track_id_to_persisted_id:
                    best_match, best_dist = None, float('inf')
                    for old_id, old_entry in global_player_memory.items():
                        if not old_entry['seen_this_frame'] and (frame_count - old_entry['last_frame'] < 30):
                            # ONLY match if it's the same team AND close in distance
                            if old_entry['team_lock'] == current_team_raw:
                                dist = np.hypot(center_x - old_entry['last_known_center'][0], center_y - old_entry['last_known_center'][1])
                                if dist < 60 and dist < best_dist: 
                                    best_dist, best_match = dist, old_id
                    
                    persisted_id = best_match if best_match is not None else track_id
                    track_id_to_persisted_id[track_id] = persisted_id
                
                persisted_id = track_id_to_persisted_id[track_id]
                
                # 3. INITIALIZE MEMORY
                if persisted_id not in global_player_memory:
                    global_player_memory[persisted_id] = {
                        'ema_india': ind_raw, 
                        'ema_iran': irn_raw, 
                        'team_lock': current_team_raw, 
                        'last_known_center': (center_x, center_y), 
                        'last_frame': frame_count, 
                        'initial_team': current_team_raw, 
                        'seen_this_frame': True
                    }

                entry = global_player_memory[persisted_id]
                entry.update({'seen_this_frame': True, 'last_known_center': (center_x, center_y), 'last_frame': frame_count})

                # 4. EMA & HYSTERESIS
                alpha = 0.15
                margin = 0.10
                entry['ema_india'] = (1 - alpha) * entry['ema_india'] + alpha * ind_raw
                entry['ema_iran'] = (1 - alpha) * entry['ema_iran'] + alpha * irn_raw

                if entry['team_lock'] == TEAM_BLUE and entry['ema_iran'] > entry['ema_india'] + margin:
                    entry['team_lock'] = TEAM_WHITE_GREEN
                elif entry['team_lock'] == TEAM_WHITE_GREEN and entry['ema_india'] > entry['ema_iran'] + margin:
                    entry['team_lock'] = TEAM_BLUE

                # 5. DRAWING
                current_team = entry['team_lock']
                final_id = track_id if entry['initial_team'] == current_team else track_id + 1000
                draw_tracking(frame, box.xyxy[0].cpu().numpy(), final_id, current_team)
                
                print(f"[DIAGNOSTIC] ID:{track_id} | Persisted:{persisted_id} | Pos:{center_x},{center_y} | India:{ind_raw:.2f} | Iran:{irn_raw:.2f} | Team:{current_team}")

        stale_ids = [tid for tid, entry in global_player_memory.items() if frame_count - entry['last_frame'] > 30]
        for stale_id in stale_ids:
            del global_player_memory[stale_id]
            for tid, pid in list(track_id_to_persisted_id.items()):
                if pid == stale_id:
                    del track_id_to_persisted_id[tid]

        video_writer.write(frame)
        cv2.imwrite(os.path.join("saved_frames", f"frame_{frame_count:04d}.jpg"), frame)
        frame_count += 1
        cv2.imshow("Kabaddi Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    video_writer.release()
    cv2.destroyAllWindows()
    print(f"✅ Run complete. Processed {frame_count} frames.")

if __name__ == "__main__":
    main()