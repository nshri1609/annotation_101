import cv2
import sys
from pathlib import Path
from tqdm import tqdm

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("MediaPipe not installed.")
    sys.exit(1)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Base
]

def process_video_realtime(video_path, output_path):
    print("Initializing MediaPipe Hands Task at FULL FPS...")
    model_path = "models/mediapipe/hand_landmarker.task"
    if not Path(model_path).exists():
        print(f"Model missing at {model_path}. Please download it.")
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
    # Use lower confidence for more continuous tracking
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    hands = vision.HandLandmarker.create_from_options(options)
    
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
    
    print(f"Processing every frame continuously. Video has {total_frames} frames at {fps} FPS.")
    print(f"Saving to {output_path}...")
    
    pbar = tqdm(total=total_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect hands for THIS frame
        results = hands.detect(mp_image)
        
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                points = []
                for lm in hand_landmarks:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    points.append((x, y))
                    # Draw points
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    
                # Draw connections
                for connection in HAND_CONNECTIONS:
                    p1 = points[connection[0]]
                    p2 = points[connection[1]]
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)
                    
        out.write(frame)
        pbar.update(1)
        
    pbar.close()
    cap.release()
    out.release()
    hands.close()
    print("Done!")

if __name__ == "__main__":
    job_id = "8e293e83-fa15-401b-864b-6e3fba786d74"
    video_path = f"storage/jobs/{job_id}/hands.mp4"
    output_path = "annotated_hands_perfect.mp4"
    
    if not Path(video_path).exists():
        print(f"Could not find video at {video_path}")
        sys.exit(1)
        
    process_video_realtime(video_path, output_path)
