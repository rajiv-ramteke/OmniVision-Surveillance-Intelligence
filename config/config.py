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
        # Strip comments (lines starting with #) and blank lines
        CUSTOM_CLASSES = [
            line.strip() for line in _f
            if line.strip() and not line.strip().startswith('#')
        ]
    if not CUSTOM_CLASSES:
        raise ValueError("classes.txt is empty — please add at least one class name.")
    print(f"[Config] Loaded {len(CUSTOM_CLASSES)} custom classes from classes.txt")
except FileNotFoundError:
    raise FileNotFoundError(
        f"\n[Config] ERROR: 'models/classes.txt' not found at:\n  {_classes_file}\n"
        f"Create this file and add one object class per line (e.g. 'person', 'car')."
    )

# ── Detection Accuracy Tuning ────────────────────────────────────────────────
# IMAGE_SIZE        : Higher = better accuracy for small objects (slower on CPU)
# CONFIDENCE_THRESHOLD: 0.70 = only very confident detections shown (no false positives)
# IOU_THRESHOLD     : 0.50 = tighter NMS, removes overlapping duplicate boxes
IMAGE_SIZE = 960
CONFIDENCE_THRESHOLD = 0.55
IOU_THRESHOLD = 0.50           # Tightened: removes duplicate/overlapping boxes better

# FRAME_SKIP: Run inference every N frames (1 = every frame, 2 = every other frame)
# Higher value = faster display FPS but slightly delayed detection update
FRAME_SKIP = 1                 # 1 = max accuracy, 2 = max speed

# Alert Configuration
# Leaving ALERT_CLASSES empty [] means you get voice & mobile alerts for ALL detected objects
ALERT_CLASSES = []         # [] = alert on ALL objects
COOLDOWN_TIME = 60         # 60s between ntfy notifications per class (saves daily quota)

# Notification Settings
# Set NTFY_TOPIC in your environment (recommended) or change the fallback string below.
# To set env var (Windows): setx NTFY_TOPIC "your-topic-name"
# To set env var (Linux/Mac): export NTFY_TOPIC="your-topic-name"
NTFY_TOPIC = os.environ.get('NTFY_TOPIC', 'my_notify')

# Logging and Snapshot Settings
SAVE_SNAPSHOTS = True
SNAPSHOT_DIR = 'static/output/snapshots'
LOG_CSV = True
CSV_LOG_FILE = 'static/output/detection_log.csv'
SHOW_STATS = True  # Show FPS and object count on screen

# ── Image Processing Settings ────────────────────────────────────────────────
NIGHT_VISION_ENHANCEMENT = True  # Auto-brightens image in low light using CLAHE

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
