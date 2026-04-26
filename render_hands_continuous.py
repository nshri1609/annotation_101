import json
import cv2
import sys
from pathlib import Path
from tqdm import tqdm
import math

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Base
]

def get_centroid(points):
    if not points: return (0, 0)
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return (x, y)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def render_video_with_smoothing(video_path, json_path, output_path, max_hold_frames=15):
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    frame_annotations = {}
    for ann in data.get('annotations', []):
        image_id = ann.get('image_id', '')
        if "_frame_" in image_id:
            frame_num = int(image_id.split("_frame_")[1])
        else:
            continue
            
        if frame_num not in frame_annotations:
            frame_annotations[frame_num] = []
            
        keypoints = ann.get('keypoints', [])
        if len(keypoints) >= 63:
            points = []
            for i in range(21):
                x = int(keypoints[i*3])
                y = int(keypoints[i*3 + 1])
                points.append((x, y))
            frame_annotations[frame_num].append(points)
        
    print(f"Loaded annotations for {len(frame_annotations)} frames.")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Rendering video with temporal smoothing to {output_path}...")
    frame_idx = 1
    
    # Active tracks: dict of track_id -> {"points": last_points, "missing_frames": 0, "centroid": (x,y)}
    active_tracks = {}
    next_track_id = 0
    
    pbar = tqdm(total=total_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_hands = frame_annotations.get(frame_idx, [])
        
        # Match current detections to active tracks
        matched_tracks = set()
        unmatched_hands = list(current_hands)
        
        # Match by centroid distance
        for track_id, track_info in list(active_tracks.items()):
            if not unmatched_hands:
                break
                
            best_match_idx = -1
            best_dist = float('inf')
            
            for i, hand_pts in enumerate(unmatched_hands):
                dist = distance(track_info["centroid"], get_centroid(hand_pts))
                if dist < 150: # max distance threshold
                    if dist < best_dist:
                        best_dist = dist
                        best_match_idx = i
                        
            if best_match_idx != -1:
                # Update track
                matched_pts = unmatched_hands.pop(best_match_idx)
                active_tracks[track_id] = {
                    "points": matched_pts,
                    "missing_frames": 0,
                    "centroid": get_centroid(matched_pts)
                }
                matched_tracks.add(track_id)
                
        # Create new tracks for remaining unmatched hands
        for hand_pts in unmatched_hands:
            active_tracks[next_track_id] = {
                "points": hand_pts,
                "missing_frames": 0,
                "centroid": get_centroid(hand_pts)
            }
            matched_tracks.add(next_track_id)
            next_track_id += 1
            
        # Increment missing counter for unmatched tracks and remove old ones
        for track_id in list(active_tracks.keys()):
            if track_id not in matched_tracks:
                active_tracks[track_id]["missing_frames"] += 1
                if active_tracks[track_id]["missing_frames"] > max_hold_frames:
                    del active_tracks[track_id]
                    
        # Draw all active tracks
        for track_id, track_info in active_tracks.items():
            points = track_info["points"]
            
            # Draw points
            for p in points:
                cv2.circle(frame, p, 5, (0, 0, 255), -1)
                
            # Draw connections
            for connection in HAND_CONNECTIONS:
                p1 = points[connection[0]]
                p2 = points[connection[1]]
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
                
        out.write(frame)
        frame_idx += 1
        pbar.update(1)
        
    pbar.close()
    cap.release()
    out.release()
    print("Done!")

if __name__ == "__main__":
    job_id = "8e293e83-fa15-401b-864b-6e3fba786d74"
    video_path = f"storage/jobs/{job_id}/hands.mp4"
    json_path = f"storage/jobs/{job_id}/hands_output/hands_hand_tracking.json"
    output_path = "annotated_hands_continuous.mp4"
    
    if not Path(video_path).exists() or not Path(json_path).exists():
        print("Could not find the job files. Make sure the job ID is correct.")
        sys.exit(1)
        
    render_video_with_smoothing(video_path, json_path, output_path, max_hold_frames=15)
