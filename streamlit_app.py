import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import av
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

st.set_page_config(page_title="Real-Time Object Detection", layout="wide")

# Initialize YOLO Model
@st.cache_resource
def load_model():
    print(f"Loading YOLO model ({config.MODEL_NAME}) for Streamlit...")
    m = YOLO(config.MODEL_NAME)
    if 'world' in config.MODEL_NAME.lower() and hasattr(config, 'CUSTOM_CLASSES'):
        m.set_classes(config.CUSTOM_CLASSES)
    return m

model = load_model()
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


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert WebRTC frame to OpenCV format
    img = frame.to_ndarray(format="bgr24")
    
    # Run YOLO inference
    results = model.track(
        img,
        persist=True,
        verbose=False,
        conf=config.CONFIDENCE_THRESHOLD,
        iou=config.IOU_THRESHOLD,
        imgsz=getattr(config, 'IMAGE_SIZE', 640),
        half=False
    )
    
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
            
            # Draw bounding box
            img = draw_bounding_box(img, [x1, y1, x2, y2], class_name, conf, track_id)
            
            # Trigger mobile alert based on cooldown
            if not config.ALERT_CLASSES or class_name in config.ALERT_CLASSES:
                if current_time - last_alert_time.get(class_name, 0) >= config.COOLDOWN_TIME:
                    last_alert_time[class_name] = current_time
                    threading.Thread(target=send_ntfy_notification, args=(class_name, conf), daemon=True).start()
                
    # Return processed frame back to the browser
    return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("Real-Time Object Detection System 📸")
st.markdown("This Streamlit app uses **WebRTC** to stream your webcam directly to the server, runs **YOLOv8-World**, and pushes notifications via **ntfy**.")

# WebRTC Streamer Setup
webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
