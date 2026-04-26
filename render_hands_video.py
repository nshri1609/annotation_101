import json
import cv2
import sys
from pathlib import Path
from tqdm import tqdm

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Base
]

def render_video(video_path, json_path, output_path):
    print(f"Loading annotations from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Build dictionary mapping frame number to list of hands
    # Each hand has keypoints: [x1, y1, v1, x2, y2, v2, ...]
    frame_annotations = {}
    for ann in data.get('annotations', []):
        image_id = ann.get('image_id', '')
        # image_id is usually formatted as "{video_id}_frame_{frame_number}"
        if "_frame_" in image_id:
            frame_num = int(image_id.split("_frame_")[1])
        else:
            continue
            
        if frame_num not in frame_annotations:
            frame_annotations[frame_num] = []
        frame_annotations[frame_num].append(ann.get('keypoints', []))
        
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
    
    print(f"Rendering video to {output_path}...")
    frame_idx = 1
    
    # We use tqdm to show progress
    pbar = tqdm(total=total_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        hands = frame_annotations.get(frame_idx, [])
        for keypoints in hands:
            if not keypoints or len(keypoints) < 63: # 21 points * 3 values
                continue
                
            points = []
            for i in range(21):
                x = int(keypoints[i*3])
                y = int(keypoints[i*3 + 1])
                points.append((x, y))
                # Draw point
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                
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
    output_path = "annotated_hands.mp4"
    
    if not Path(video_path).exists() or not Path(json_path).exists():
        print("Could not find the job files. Make sure the job ID is correct.")
        sys.exit(1)
        
    render_video(video_path, json_path, output_path)
