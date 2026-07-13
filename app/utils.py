import cv2
import os
import csv
import datetime
import numpy as np

# ── Colour palette — one consistent colour per class ────────────────────────
_CLASS_COLORS = {}

def _get_class_color(class_name: str):
    """Returns a consistent, bright BGR colour for each object class."""
    if class_name not in _CLASS_COLORS:
        # Hash the class name so the same class always gets the same colour
        h = hash(class_name) & 0xFF
        # Use HSV with fixed saturation/value for vivid, easily readable colours
        hsv = np.uint8([[[h, 220, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        _CLASS_COLORS[class_name] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    return _CLASS_COLORS[class_name]


def draw_bounding_box(frame, box, class_name, confidence, track_id=None):
    """Draws a smooth, coloured bounding box and label on the given frame."""
    x1, y1, x2, y2 = map(int, box)
    color = _get_class_color(class_name)

    # Main bounding box (thicker for visibility)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner decorators for a modern look
    corner_len = max(10, (x2 - x1) // 8)
    thickness  = 3
    for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        dx = corner_len if cx == x1 else -corner_len
        dy = corner_len if cy == y1 else -corner_len
        cv2.line(frame, (cx, cy), (cx + dx, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy), color, thickness)

    # Label pill background
    id_text = f" #{track_id}" if track_id is not None else ""
    label = f" {class_name}{id_text}  {confidence:.0%} "
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    font_thick = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
    label_y1 = max(y1 - th - 8, 0)
    label_y2 = label_y1 + th + 8

    # Semi-transparent pill
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, label_y1), (x1 + tw + 4, label_y2), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Label text in black for contrast
    cv2.putText(frame, label, (x1 + 2, label_y2 - 5),
                font, font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)
    return frame


def draw_stats(frame, fps, object_counts):
    """Draws a semi-transparent stats panel on the top-left of the frame."""
    h, w = frame.shape[:2]
    panel_w  = 220
    row_h    = 26
    rows     = 1 + len(object_counts)   # FPS row + one row per class
    panel_h  = rows * row_h + 20

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    font_thick = 1

    # FPS in bright cyan
    fps_color = (0, 255, 200) if fps >= 20 else (0, 180, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (16, 8 + row_h),
                font, font_scale, fps_color, font_thick, cv2.LINE_AA)

    # Object counts with per-class colour
    y = 8 + row_h * 2
    for cls, count in sorted(object_counts.items()):
        color = _get_class_color(cls)
        text  = f"  {cls.capitalize()}: {count}"
        cv2.putText(frame, text, (16, y),
                    font, font_scale, color, font_thick, cv2.LINE_AA)
        y += row_h

    return frame


def save_snapshot(frame, snapshot_dir, class_name):
    """Saves a snapshot of the current frame to the specified directory."""
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{class_name}_{timestamp}.jpg"
    filepath  = os.path.join(snapshot_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath


def log_detection_csv(csv_file, class_name, confidence):
    """Logs the detection event to a CSV file."""
    # Safely resolve the directory — handles both 'dir/file.csv' and 'file.csv'
    csv_dir = os.path.dirname(os.path.abspath(csv_file))
    os.makedirs(csv_dir, exist_ok=True)

    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Class', 'Confidence'])  # Header
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, class_name, f"{confidence:.2f}"])
