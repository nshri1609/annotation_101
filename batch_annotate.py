import os
import cv2
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Base
]

def init_hand_model():
    if not MEDIAPIPE_AVAILABLE:
        print("MediaPipe not available. Cannot run hand tracking.")
        sys.exit(1)
        
    model_path = "models/mediapipe/hand_landmarker.task"
    if not Path(model_path).exists():
        print(f"Downloading MediaPipe Hand Landmarker model...")
        os.makedirs("models/mediapipe", exist_ok=True)
        os.system(f"curl -sSL https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o {model_path}")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    return vision.HandLandmarker.create_from_options(options)

def init_object_model():
    if not ULTRALYTICS_AVAILABLE:
        print("Ultralytics not available. Cannot run object detection.")
        sys.exit(1)
    
    return YOLO("yolo11n.pt")

def process_video(video_path, output_path, run_hands, run_objects):
    hands_model = init_hand_model() if run_hands else None
    object_model = init_object_model() if run_objects else None
    
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
    
    print(f"\nProcessing {Path(video_path).name} ({total_frames} frames at {fps:.2f} FPS)")
    
    pbar = tqdm(total=total_frames, desc=Path(video_path).name)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # --- Object Detection ---
        if object_model:
            results = object_model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    cls_name = object_model.names.get(cls_id, f"obj_{cls_id}")
                    
                    if box.id is not None:
                        track_id = int(box.id[0].cpu().numpy())
                        label = f"{cls_name} {track_id} ({conf:.2f})"
                    else:
                        label = f"{cls_name} ({conf:.2f})"
                        
                    # Draw Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                    # Draw Label
                    cv2.putText(frame, label, (x1, max(15, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                
        # --- Hand Tracking ---
        if hands_model:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = hands_model.detect(mp_image)
            
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    points = []
                    for lm in hand_landmarks:
                        x = int(lm.x * width)
                        y = int(lm.y * height)
                        points.append((x, y))
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                        
                    for connection in HAND_CONNECTIONS:
                        p1 = points[connection[0]]
                        p2 = points[connection[1]]
                        cv2.line(frame, p1, p2, (0, 255, 0), 2)
                        
        out.write(frame)
        pbar.update(1)
        
    pbar.close()
    cap.release()
    out.release()
    if hands_model:
        hands_model.close()
    print(f"Saved annotated video to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch process videos and render annotations.")
    parser.add_argument("--input-dir", type=str, default="input_videos", help="Directory containing raw videos.")
    parser.add_argument("--output-dir", type=str, default="annotated_videos", help="Directory to save annotated videos.")
    parser.add_argument("--hands", action="store_true", help="Enable hand tracking pipeline.")
    parser.add_argument("--objects", action="store_true", help="Enable object detection pipeline.")
    args = parser.parse_args()
    
    if not args.hands and not args.objects:
        print("Please specify at least one pipeline to run: --hands or --objects")
        sys.exit(1)
        
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.mov")) + list(input_dir.glob("*.avi"))
    
    if not video_files:
        print(f"No video files found in {input_dir}. Please place some videos there and try again.")
        sys.exit(0)
        
    print(f"Found {len(video_files)} videos in {input_dir}. Beginning batch processing...")
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n--- Video {i}/{len(video_files)} ---")
        output_file = output_dir / f"{video_file.stem}_annotated{video_file.suffix}"
        process_video(str(video_file), str(output_file), args.hands, args.objects)
        
    print("\nBatch processing complete! All annotated videos are in the annotated_videos/ folder.")

if __name__ == "__main__":
    main()
