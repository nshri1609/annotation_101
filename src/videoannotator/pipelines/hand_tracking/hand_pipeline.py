import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from videoannotator.exporters.native_formats import (
    create_coco_annotation,
    create_coco_image_entry,
    create_coco_keypoints_annotation,
    export_coco_json,
    validate_coco_json,
)
from videoannotator.pipelines.base_pipeline import BasePipeline

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class HandTrackingPipeline(BasePipeline):
    """Hand tracking pipeline using MediaPipe for 21-point DOF tracing.

    This pipeline produces COCO-style annotations for tracked hands.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        default_config = {
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "max_num_hands": 2,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        
        self.logger = logging.getLogger(__name__)
        self.hands: Any = None
        
    def initialize(self) -> None:
        if self.is_initialized:
            return
            
        self.logger.info("Initializing HandTrackingPipeline with Tasks API...")
        
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe package required for hand tracking")
            
        model_path = "models/mediapipe/hand_landmarker.task"
        if not Path(model_path).exists():
            raise FileNotFoundError(f"MediaPipe model missing: {model_path}. Run: mkdir -p models/mediapipe && curl -sSL https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.config["max_num_hands"],
            min_hand_detection_confidence=self.config["min_detection_confidence"],
            min_hand_presence_confidence=self.config["min_tracking_confidence"],
            min_tracking_confidence=self.config["min_tracking_confidence"]
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self.set_model_info("mediapipe_hands_tasks")
        self.is_initialized = True
        self.logger.info("[OK] MediaPipe Hands Task model ready")

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float | None = None,
        pps: float = 5,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        video_metadata = self.get_video_info(video_path)
        video_metadata["video_id"] = Path(video_path).stem
        
        if not self.is_initialized:
            self.initialize()
            
        annotations: list[dict[str, Any]] = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = video_metadata["fps"]
        total_frames = video_metadata["frame_count"]
        
        if end_time is None:
            end_time = total_frames / fps if fps > 0 else 0
            
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_step = max(1, int(fps / pps)) if pps > 0 else 1
        
        annotation_id = 1
        
        try:
            for frame_num in range(start_frame, min(end_frame, total_frames), frame_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                    
                timestamp = frame_num / fps
                frame_annotations = self._process_frame(
                    frame, timestamp, video_metadata["video_id"], frame_num
                )
                
                for ann in frame_annotations:
                    ann["id"] = annotation_id
                    annotation_id += 1
                    
                annotations.extend(frame_annotations)
        finally:
            cap.release()
            
        if output_dir and annotations:
            self._save_coco_annotations(annotations, output_dir, video_metadata)
            
        self.logger.info(f"Hand tracking complete: {len(annotations)} detections")
        return annotations

    def _process_frame(
        self, frame: np.ndarray, timestamp: float, video_id: str, frame_number: int
    ) -> list[dict[str, Any]]:
        if self.hands is None:
            raise RuntimeError("MediaPipe Hands model is not initialized")
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        results = self.hands.detect(mp_image)
        
        annotations = []
        
        if results.hand_landmarks:
            h, w, _ = frame.shape
            for i, hand_landmarks in enumerate(results.hand_landmarks):
                # Calculate bounding box
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                coco_keypoints = []
                visible_keypoints = 0
                
                for lm in hand_landmarks:
                    x, y = lm.x * w, lm.y * h
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                    
                    # MediaPipe doesn't give confidence per point easily in this API, assume 1.0 (visible)
                    visibility = 2
                    coco_keypoints.extend([float(x), float(y), visibility])
                    visible_keypoints += 1
                
                # Add margin to bounding box
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                
                bbox_coco = [
                    float(x_min),
                    float(y_min),
                    float(x_max - x_min),
                    float(y_max - y_min),
                ]
                
                # Classification (Left/Right)
                label = ""
                score = 1.0
                if results.handedness and i < len(results.handedness):
                    classification = results.handedness[i][0]
                    label = classification.category_name
                    score = float(classification.score)
                    
                annotation = create_coco_keypoints_annotation(
                    annotation_id=0,
                    image_id=f"{video_id}_frame_{frame_number}",
                    category_id=2,  # 2 for Hand
                    keypoints=coco_keypoints,
                    bbox=bbox_coco,
                    num_keypoints=visible_keypoints,
                    score=score,
                    # VideoAnnotator extensions
                    track_id=i,  # Basic track ID based on order
                    timestamp=timestamp,
                    frame_number=frame_number,
                    person_label=label, # Reuse person_label for hand label
                )
                annotations.append(annotation)
                
        return annotations

    def _save_coco_annotations(
        self,
        annotations: list[dict[str, Any]],
        output_dir: str,
        video_metadata: dict[str, Any],
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images = []
        image_ids = set()
        
        for ann in annotations:
            image_id = ann["image_id"]
            if image_id not in image_ids:
                image_ids.add(image_id)
                frame_number = ann.get("frame_number", 0)
                timestamp = ann.get("timestamp", 0.0)
                
                image = create_coco_image_entry(
                    image_id=image_id,
                    width=video_metadata["width"],
                    height=video_metadata["height"],
                    file_name=f"frame_{frame_number:06d}.jpg",
                    video_id=video_metadata["video_id"],
                    frame_number=frame_number,
                    timestamp=timestamp,
                )
                images.append(image)
                
        categories = [
            {
                "id": 2,
                "name": "hand",
                "supercategory": "hand",
                "keypoints": [
                    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
                    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
                    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
                    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
                    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
                ],
                "skeleton": [
                    [1, 2], [2, 3], [3, 4], [4, 5],
                    [1, 6], [6, 7], [7, 8], [8, 9],
                    [1, 10], [10, 11], [11, 12], [12, 13],
                    [1, 14], [14, 15], [15, 16], [16, 17],
                    [1, 18], [18, 19], [19, 20], [20, 21],
                    [6, 10], [10, 14], [14, 18]
                ]
            }
        ]
        
        coco_path = output_path / f"{video_metadata['video_id']}_hand_tracking.json"
        export_coco_json(annotations, images, str(coco_path), categories)
        
    def cleanup(self):
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        self.is_initialized = False
        
    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "array",
            "description": "Hand tracking results in COCO format",
        }
