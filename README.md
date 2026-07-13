# Real-Time Object Detection & Alert System

A real-time object detection and security alert system built with **YOLOv8-World** and **OpenCV**. Detects and classifies objects from a live webcam feed without human intervention, with voice alerts, mobile push notifications, CSV logging, snapshot saving, and video clip recording.

## Features
- **Real-Time Detection:** YOLOv8-World open-vocabulary model — 94 customisable object classes
- **Object Tracking:** ByteTrack cross-frame tracking with unique Track IDs
- **Visual Feedback:** Bounding boxes with class labels, confidence scores, and a live FPS/count HUD
- **Voice Alerts:** Speaks detected object names aloud via `pyttsx3` (Windows TTS)
- **Mobile Notifications:** Push notifications via [ntfy](https://ntfy.sh) — danger classes (knife, fire, etc.) get urgent priority
- **Logging:** Timestamped CSV log of all detections
- **Snapshots:** Auto-saves a JPEG of every new detection event
- **Video Recording:** Clip-on-detection with pre-buffer (captures footage *before* the trigger), plus optional continuous session recording
- **Async Architecture:** Camera capture, YOLO inference, voice, and recording all run on separate threads — display loop never blocks

## Technologies
- Python 3.8+
- OpenCV (`opencv-python`)
- Ultralytics YOLOv8-World
- pyttsx3 (Windows TTS)
- requests (ntfy push notifications)
- Gradio (HuggingFace Spaces web UI)

## Installation

1. Clone the repository or open the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Set your ntfy topic as an environment variable to keep it private:
   ```bat
   setx NTFY_TOPIC "your-topic-name"
   ```
   Or edit `NTFY_TOPIC` directly in `config/config.py`.

## Usage

Run the main script:
```bash
python run.py
```

Press **`q`** to quit the camera window.

## Configuration

All settings are in [`config/config.py`](config/config.py):

| Setting | Default | Description |
|---|---|---|
| `MODEL_NAME` | `yolov8s-world.pt` | YOLO model to use |
| `CONFIDENCE_THRESHOLD` | `0.40` | Minimum detection confidence |
| `ALERT_CLASSES` | `[]` (all) | Classes to alert on — empty = all |
| `COOLDOWN_TIME` | `60s` | Seconds between ntfy notifications per class |
| `RECORD_CLIPS` | `True` | Save clip on each detection |
| `CLIP_DURATION_SEC` | `15` | Seconds of video after detection |
| `PRE_BUFFER_SEC` | `3` | Seconds of pre-detection footage in clip |

Custom detection classes are loaded from [`models/classes.txt`](models/classes.txt) — one class per line.

## Project Structure

```text
.
├── app/
│   ├── alerts.py       # Voice (pyttsx3) + ntfy mobile notifications
│   ├── camera.py       # Threaded camera capture (DirectShow)
│   ├── detection.py    # Async YOLOv8-World inference + ByteTrack
│   ├── main.py         # Main orchestration loop
│   ├── recorder.py     # ClipRecorder (trigger-based) + ContinuousRecorder
│   └── utils.py        # Bounding box drawing, stats HUD, CSV, snapshots
├── config/
│   └── config.py       # All configuration settings
├── hf_spaces/
│   ├── app.py          # Gradio web UI for HuggingFace Spaces
│   └── requirements.txt
├── models/
│   └── classes.txt     # One detection class per line
├── static/
│   └── output/         # Snapshots, CSV log, and recordings (auto-created)
├── .gitignore
├── deploy_to_hf.py     # One-click HuggingFace Spaces deployment
├── README.md
├── requirements.txt
├── run.bat             # Windows launcher
└── run.py              # Python entry point
```

## HuggingFace Spaces (Cloud Demo)

A Gradio web UI version is available in `hf_spaces/`. To deploy:
```bash
python deploy_to_hf.py
```
Supports image upload, live webcam streaming, and video file processing in the browser — no local install required.
