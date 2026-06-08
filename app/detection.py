from ultralytics import YOLO
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class ObjectDetector:
    def __init__(self):
        print(f"Loading YOLO model ({config.MODEL_NAME})...")
        self.model = YOLO(config.MODEL_NAME)
        self.class_names = self.model.names
        print("Model loaded successfully.")
        
    def detect(self, frame):
        """
        Runs object detection on a single frame using YOLO tracking
        for stable, consistent detections across frames.
        Returns a list of detections: [{"box": [x1, y1, x2, y2], "class_name": name, "confidence": conf}]
        """
        # Use track() instead of detect() for temporal consistency across frames
        # imgsz=640 gives full accuracy (default input size for YOLO training)
        # iou threshold: lower = more strict NMS = fewer duplicate boxes
        results = self.model.track(
            frame,
            persist=True,       # Track objects across frames for stability
            verbose=False,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=640           # Full resolution input for max accuracy
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class_name": class_name,
                    "confidence": conf
                })
                
        return detections

