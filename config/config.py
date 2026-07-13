# config/config.py

# YOLO Configuration
# yolov8n = nano (Super fast, but low accuracy/guesses wrong)
# yolov8s = small (Perfect middle ground: high accuracy, smooth speed)
# yolov8m = medium (Highly accurate but causes lag on standard computers)
MODEL_NAME = 'yolov8s-world.pt'

# Read custom classes from models/classes.txt
import os
classes_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'classes.txt')
with open(classes_file, 'r') as f:
    CUSTOM_CLASSES = [line.strip() for line in f if line.strip()]

IMAGE_SIZE = 960                # Higher resolution for better small object accuracy (default was 640)
CONFIDENCE_THRESHOLD = 0.40     # Lowered to 40% to detect small objects that have lower confidence
IOU_THRESHOLD = 0.45            # Non-Maximum Suppression threshold (tighter box overlap)

# Alert Configuration
# Leaving ALERT_CLASSES empty [] means you get voice & mobile alerts for ALL detected objects
ALERT_CLASSES = []         # [] = alert on ALL objects
COOLDOWN_TIME = 60         # 60s between ntfy notifications per class (saves daily quota)

# Notification Settings
# Add your ntfy topic here to enable mobile notifications (e.g., 'my_security_alerts_123')
NTFY_TOPIC = 'rajiv-notification'

# Logging and Snapshot Settings
SAVE_SNAPSHOTS = True
SNAPSHOT_DIR = 'static/output/snapshots'
LOG_CSV = True
CSV_LOG_FILE = 'static/output/detection_log.csv'
SHOW_STATS = True  # Show FPS and object count on screen

# ── Google Drive Cloud Backup ────────────────────────────────────────────────
# Set GOOGLE_DRIVE_UPLOAD = True to auto-upload every alert snapshot to Drive.
# CREDENTIALS_FILE: path to your downloaded credentials.json from Google Cloud.
# GOOGLE_DRIVE_FOLDER: name of the folder that will be created in your Drive.
GOOGLE_DRIVE_UPLOAD   = False          # Change to True after setting up credentials
CREDENTIALS_FILE      = 'credentials.json'
GOOGLE_DRIVE_FOLDER   = 'Security_Snapshots'
