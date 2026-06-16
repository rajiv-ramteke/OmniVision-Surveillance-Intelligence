# Real-Time Object Detection System

This repository presents a real-time object detection system developed primarily for security purposes. The project addresses the need for automated monitoring by detecting and classifying objects from live video streams without human intervention. 

## Features
- **Real-Time Detection:** Uses YOLOv8 for high-speed, accurate detection from webcam feeds.
- **Visual Feedback:** Draws bounding boxes with class labels and confidence scores.
- **Alert System:** 
  - Voice alerts via `pyttsx3`.
  - Mobile notifications via `Pushbullet` (requires API key in config).

## Technologies
- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)
- Pyttsx3
- Requests

## Installation
1. Clone the repository or open the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Update configuration:
   Open `config/config.py` and optionally add your Pushbullet API Key if you want mobile notifications.

## Usage
Run the main script:
```bash
python run.py
```
Press `q` to quit the camera stream.

## Structure

```text
.
├── app/
│   ├── alerts.py
│   ├── camera.py
│   ├── detection.py
│   ├── main.py
│   ├── real time live object detection.code-workspace
│   └── utils.py
├── config/
│   └── config.py
├── models/
│   └── classes.txt
├── weights/
│   └── clip/
│       └── ViT-B-32.pt
├── .gitignore
├── README.md
├── index.html
├── requirements.txt
├── run.bat
├── run.py
├── stop.bat
├── test_tts.py
├── yolov8m.pt
├── yolov8n.pt
├── yolov8s-world.pt
└── yolov8s.pt
```

- `app/`: Core application logic (detection, camera, alerts, utils).
- `config/`: Configuration files for thresholds and API keys.
- `models/`: Contains YOLO label references.
- `run.py`: Entry point script.
