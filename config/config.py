# config/config.py

# YOLO Configuration
# yolov8n = nano (fastest, lowest accuracy)
# yolov8s = small (good balance of speed and accuracy) ← recommended
# yolov8m = medium (higher accuracy, slightly slower)
MODEL_NAME = 'yolov8s.pt'       # Upgraded from nano to small for better accuracy
CONFIDENCE_THRESHOLD = 0.65     # Raised from 0.5 → filters out false positives
IOU_THRESHOLD = 0.45            # Non-Maximum Suppression threshold (lower = stricter)

# Alert Configuration
# Leave ALERT_CLASSES empty [] to get notifications for ALL detected objects
# Or specify a list e.g. ['person', 'car', 'dog'] to filter specific classes
ALERT_CLASSES = []         # [] = alert on ALL objects
COOLDOWN_TIME = 2          # Minimum seconds between alerts per object class

# Notification Settings
# Add your ntfy topic here to enable mobile notifications (e.g., 'my_security_alerts_123')
NTFY_TOPIC = 'notify-rajiv'

# Logging and Snapshot Settings
SAVE_SNAPSHOTS = True
SNAPSHOT_DIR = 'static/output/snapshots'
LOG_CSV = True
CSV_LOG_FILE = 'static/output/detection_log.csv'
SHOW_STATS = True  # Show FPS and object count on screen
