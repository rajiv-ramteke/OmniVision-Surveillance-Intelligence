from ultralytics import YOLO
import os
import sys
import threading
import queue

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class ObjectDetector:
    def __init__(self):
        print(f"Loading YOLO model ({config.MODEL_NAME})...")
        self.model = YOLO(config.MODEL_NAME)

        # If using YOLO-World, set the custom vocabulary
        if 'world' in config.MODEL_NAME.lower() and hasattr(config, 'CUSTOM_CLASSES'):
            print(f"Setting custom YOLO-World classes: {config.CUSTOM_CLASSES}")
            self.model.set_classes(config.CUSTOM_CLASSES)

        self.class_names = self.model.names

        # Warm up the model once so the first real frame isn't slow
        import numpy as np
        dummy = np.zeros((480, 640, 3), dtype='uint8')
        self.model.predict(dummy, verbose=False, imgsz=getattr(config, 'IMAGE_SIZE', 640))
        print("Model loaded and warmed up successfully.")

        # ── Async detection pipeline ──────────────────────────────────────────
        # The detector runs on its own thread so the display loop is never
        # blocked waiting for inference.  We use a 1-slot "mailbox" pattern:
        # the main loop always reads the latest result, not a queued backlog.
        self._input_frame  = None
        self._output       = []
        self._lock         = threading.Lock()
        self._frame_event  = threading.Event()
        self._result_ready = threading.Event()
        self._stopped      = False

        self._thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._thread.start()

    # ── Background inference thread ──────────────────────────────────────────
    def _inference_worker(self):
        """Runs YOLO inference on the latest frame in a tight loop."""
        while not self._stopped:
            # Block until a new frame is posted
            self._frame_event.wait()
            self._frame_event.clear()

            with self._lock:
                frame = self._input_frame

            if frame is None:
                continue

            results = self.model.track(
                frame,
                persist=True,          # Track across frames for stability
                verbose=False,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                imgsz=getattr(config, 'IMAGE_SIZE', 640),  # Higher resolution for better small object detection
                half=False,            # half=True needs CUDA; keep False for CPU
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
                    track_id = int(box.id[0]) if box.id is not None else None
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "class_name": class_name,
                        "confidence": conf,
                        "track_id": track_id
                    })

            with self._lock:
                self._output = detections
            self._result_ready.set()

    # ── Public API ───────────────────────────────────────────────────────────
    def detect(self, frame):
        """
        Posts a frame for async inference and returns the latest results.
        Calling this every main-loop iteration keeps the display fluid because
        inference never blocks the display thread.
        """
        with self._lock:
            self._input_frame = frame
        self._frame_event.set()   # wake the inference thread

        # Return whatever the last completed inference produced
        with self._lock:
            return list(self._output)

    def stop(self):
        self._stopped = True
        self._frame_event.set()  # unblock the worker so it can exit
