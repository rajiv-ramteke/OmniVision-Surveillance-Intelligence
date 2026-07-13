import gradio as gr
import cv2
import os
import sys
import time
import requests
import datetime
import threading
from ultralytics import YOLO

# Ensure the root directory is in the sys path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from app.utils import draw_bounding_box

# Initialize YOLO Model
print(f"Loading YOLO model ({config.MODEL_NAME}) for Gradio...")
model = YOLO(config.MODEL_NAME)
if 'world' in config.MODEL_NAME.lower() and hasattr(config, 'CUSTOM_CLASSES'):
    model.set_classes(config.CUSTOM_CLASSES)
class_names = model.names

last_alert_time = {}

def send_ntfy_notification(class_name, confidence):
    """Sends background mobile notification via ntfy"""
    if not config.NTFY_TOPIC:
        return
    
    url = f"https://ntfy.envs.net/{config.NTFY_TOPIC}"
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    conf_str = f" — {int(confidence * 100)}% confident" if confidence else ""
    message = f"[ALERT] {class_name.capitalize()} detected at {timestamp}{conf_str}."
    
    headers = {
        "Title": f"{class_name.capitalize()} Detected!",
        "Tags": "warning,eyes",
        "Priority": "high",
    }
    
    try:
        requests.post(url, data=message.encode('utf-8'), headers=headers, timeout=5)
    except Exception as e:
        print(f"Ntfy error: {e}")

def process_frame(frame):
    """Process a single frame from the Gradio webcam stream"""
    if frame is None:
        return None
        
    # Run YOLO inference
    results = model.track(
        frame,
        persist=True,
        verbose=False,
        conf=config.CONFIDENCE_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        imgsz=getattr(config, 'IMAGE_SIZE', 640),
        half=False
    )
    
    # Convert RGB to BGR for drawing with cv2
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    current_time = time.time()
    
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            track_id = int(box.id[0]) if box.id is not None else None
            
            # Draw bounding box on the frame
            bgr_frame = draw_bounding_box(bgr_frame, [x1, y1, x2, y2], class_name, conf, track_id)
            
            # Trigger mobile alert based on cooldown
            if not config.ALERT_CLASSES or class_name in config.ALERT_CLASSES:
                if current_time - last_alert_time.get(class_name, 0) >= config.COOLDOWN_TIME:
                    last_alert_time[class_name] = current_time
                    threading.Thread(target=send_ntfy_notification, args=(class_name, conf), daemon=True).start()
                
    # Convert back to RGB for Gradio output
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return rgb_frame

# Build Gradio Web Interface
demo = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(sources=["webcam"], streaming=True),
    outputs="image",
    title="Real-Time Object Detection",
    description="YOLOv8 Object Detection running on Hugging Face Spaces. Make sure to allow webcam access.",
    live=True
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
