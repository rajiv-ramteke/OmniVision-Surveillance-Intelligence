import cv2
import os
import csv
import datetime

def draw_bounding_box(frame, box, class_name, confidence):
    """Draws a bounding box and label on the given frame."""
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{class_name} {confidence:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame

def draw_stats(frame, fps, object_counts):
    """Draws FPS and object counts on the top-left of the frame."""
    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw counts
    y_offset = 60
    for cls, count in object_counts.items():
        text = f"{cls.capitalize()}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25
    return frame

def save_snapshot(frame, snapshot_dir, class_name):
    """Saves a snapshot of the current frame to the specified directory."""
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{class_name}_{timestamp}.jpg"
    filepath = os.path.join(snapshot_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath

def log_detection_csv(csv_file, class_name, confidence):
    """Logs the detection event to a CSV file."""
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Class', 'Confidence']) # Header
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, class_name, f"{confidence:.2f}"])
