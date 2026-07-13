# config/config.py
import os

# YOLO Configuration
# yolov8n = nano (Super fast, but low accuracy/guesses wrong)
# yolov8s = small (Perfect middle ground: high accuracy, smooth speed)
# yolov8m = medium (Highly accurate but causes lag on standard computers)
MODEL_NAME = 'yolov8s-world.pt'

# Read custom classes from models/classes.txt
# Wrapped in try/except so a missing file gives a clear, actionable error.
_classes_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'classes.txt')
try:
    with open(_classes_file, 'r') as _f:
        CUSTOM_CLASSES = [line.strip() for line in _f if line.strip()]
    if not CUSTOM_CLASSES:
        raise ValueError("classes.txt is empty — please add at least one class name.")
    print(f"[Config] Loaded {len(CUSTOM_CLASSES)} custom classes from classes.txt")
except FileNotFoundError:
    raise FileNotFoundError(
        f"\n[Config] ERROR: 'models/classes.txt' not found at:\n  {_classes_file}\n"
        f"Create this file and add one object class per line (e.g. 'person', 'car')."
    )

IMAGE_SIZE = 960                # Higher resolution for better small object accuracy (default was 640)
CONFIDENCE_THRESHOLD = 0.40     # Lowered to 40% to detect small objects that have lower confidence
IOU_THRESHOLD = 0.45            # Non-Maximum Suppression threshold (tighter box overlap)

# Alert Configuration
# Leaving ALERT_CLASSES empty [] means you get voice & mobile alerts for ALL detected objects
ALERT_CLASSES = []         # [] = alert on ALL objects
COOLDOWN_TIME = 60         # 60s between ntfy notifications per class (saves daily quota)

# Notification Settings
# Set NTFY_TOPIC in your environment (recommended) or change the fallback string below.
# To set env var (Windows): setx NTFY_TOPIC "your-topic-name"
# To set env var (Linux/Mac): export NTFY_TOPIC="your-topic-name"
NTFY_TOPIC = os.environ.get('NTFY_TOPIC', 'rajiv-notification')

# Logging and Snapshot Settings
SAVE_SNAPSHOTS = True
SNAPSHOT_DIR = 'static/output/snapshots'
LOG_CSV = True
CSV_LOG_FILE = 'static/output/detection_log.csv'
SHOW_STATS = True  # Show FPS and object count on screen

# ── Recording Settings ───────────────────────────────────────────────────────
# RECORD_CLIPS      : Save a short video clip whenever an alert triggers.
# CONTINUOUS_RECORD : Record the entire session to one long video file.
# RECORDING_DIR     : Parent folder — clips/ and sessions/ sub-folders created automatically.
# CLIP_DURATION_SEC : How many seconds to record AFTER the detection event.
# PRE_BUFFER_SEC    : Seconds of footage BEFORE the trigger included in the clip.
# CLIP_COOLDOWN_SEC : Minimum gap (seconds) between clips for the same class.
RECORD_CLIPS      = True
CONTINUOUS_RECORD = False                    # Set True to record full session
RECORDING_DIR     = 'static/output/recordings'
CLIP_DURATION_SEC = 15       # seconds of video after detection
PRE_BUFFER_SEC    = 3        # seconds of pre-detection buffer included
CLIP_COOLDOWN_SEC = 30       # min gap between clips for the same object class
