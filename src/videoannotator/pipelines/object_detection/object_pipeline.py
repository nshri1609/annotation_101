import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from videoannotator.exporters.native_formats import (
    create_coco_image_entry,
    create_coco_annotation,
    export_coco_json,
)
from videoannotator.pipelines.base_pipeline import BasePipeline

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class ObjectDetectionPipeline(BasePipeline):
    """Generic object detection pipeline using YOLOv11.

    This pipeline detects standard objects (e.g., cars, laptops, cups)
    and produces COCO-style bounding box annotations.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        default_config = {
            "model_name": "yolo11n.pt",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "classes": None,  # None means detect all classes
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        
        self.logger = logging.getLogger(__name__)
        self.model: Any = None
        self._class_names: dict[int, str] = {}
        
    def initialize(self) -> None:
        if self.is_initialized:
            return
            
        self.logger.info(f"Initializing ObjectDetectionPipeline with model {self.config['model_name']}")
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package required for object detection")
            
        # Initialize YOLO model
        # Note: ultralytics automatically downloads standard models like 'yolo11n.pt'
        self.model = YOLO(self.config["model_name"])
        self._class_names = self.model.names
        
        self.set_model_info(f"ultralytics_{self.config['model_name']}")
        self.is_initialized = True
        self.logger.info("[OK] YOLOv11 model ready")

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
            
        self.logger.info(f"Object detection complete: {len(annotations)} detections")
        return annotations

    def _process_frame(
        self, frame: np.ndarray, timestamp: float, video_id: str, frame_number: int
    ) -> list[dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized")
            
        # Run tracking/detection
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.config["confidence_threshold"],
            iou=self.config["iou_threshold"],
            classes=self.config["classes"],
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        annotations = []
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self._class_names.get(class_id, f"object_{class_id}")
                
                # Get track ID if available
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1
                
                # Format to COCO [x, y, width, height]
                w = x2 - x1
                h = y2 - y1
                bbox_coco = [float(x1), float(y1), float(w), float(h)]
                
                # Base COCO annotation
                annotation = create_coco_annotation(
                    annotation_id=0,
                    image_id=f"{video_id}_frame_{frame_number}",
                    category_id=class_id,
                    bbox=bbox_coco,
                    area=float(w * h),
                    score=score,
                    # Extended metadata
                    track_id=track_id,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    person_label=class_name, # Storing the object name here for convenience
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
                
        # Build categories from detected classes
        used_class_ids = {ann["category_id"] for ann in annotations}
        categories = []
        for class_id in used_class_ids:
            class_name = self._class_names.get(class_id, f"object_{class_id}")
            categories.append({
                "id": class_id,
                "name": class_name,
                "supercategory": "object"
            })
            
        coco_path = output_path / f"{video_metadata['video_id']}_object_detection.json"
        export_coco_json(annotations, images, str(coco_path), categories)
        
    def cleanup(self):
        self.model = None
        self._class_names = {}
        self.is_initialized = False
        
    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "array",
            "description": "Generic object tracking results in COCO format",
        }
