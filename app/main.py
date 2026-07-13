import cv2
import sys
import os
import time
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from app.camera import CameraStream
from app.detection import ObjectDetector
from app.alerts import AlertSystem
from app.utils import draw_bounding_box, draw_stats, save_snapshot, log_detection_csv
from app.recorder import ClipRecorder, ContinuousRecorder


def main():
    print("Initializing Real-Time Object Detection System...")

    try:
        camera = CameraStream(source=0)  # 0 for default webcam
    except ValueError as e:
        print(e)
        return

    detector     = ObjectDetector()
    alert_system = AlertSystem()

    # ── Detect camera resolution & FPS for recorders ──────────────────────────
    cam_w   = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h   = int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = camera.cap.get(cv2.CAP_PROP_FPS) or 20.0

    # ── Initialise recorders (only if enabled in config) ──────────────────────
    clip_recorder = ClipRecorder(cam_fps, cam_w, cam_h) if config.RECORD_CLIPS else None
    cont_recorder = ContinuousRecorder(cam_fps, cam_w, cam_h) if config.CONTINUOUS_RECORD else None

    print("System started successfully. Press 'q' to quit.")

    # ── Smooth FPS using a rolling average over the last 30 frames ───────────
    frame_times = deque(maxlen=30)
    prev_time   = time.time()
    last_frame_time = time.time()

    try:
        while True:
            ret, frame = camera.get_frame()
            if not ret or frame is None:
                # Frame not ready yet — yield briefly and retry
                time.sleep(0.005)
                # If no frame has been received for 3 seconds, assume camera is off/disconnected
                if time.time() - last_frame_time > 3.0:
                    print("\nCamera disconnected or turned off. Stopping application...")
                    break
                continue

            last_frame_time = time.time()

            # ── Feed raw frame to recorders (before annotation) ───────────────
            if clip_recorder:
                clip_recorder.push_frame(frame)
            if cont_recorder:
                cont_recorder.push_frame(frame)

            # ── Run detection (async — returns last inference result instantly) ──
            detections = detector.detect(frame)

            # ── FPS calculation ───────────────────────────────────────────────
            current_time = time.time()
            frame_times.append(current_time - prev_time)
            prev_time = current_time
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

            # ── Process detections ───────────────────────────────────────────
            object_counts = {}
            for det in detections:
                class_name = det["class_name"]
                confidence = det["confidence"]
                box        = det["box"]
                track_id   = det.get("track_id")

                # Update counts
                object_counts[class_name] = object_counts.get(class_name, 0) + 1

                # Draw bounding box
                frame = draw_bounding_box(frame, box, class_name, confidence, track_id)

                # ── SOUND: speak name on EVERY detection (0.5 s per-class cooldown)
                if not config.ALERT_CLASSES or class_name in config.ALERT_CLASSES:
                    alert_system.sound_alert(class_name)

                # ── HEAVY ALERTS: CSV / snapshot / ntfy — cooldown managed inside trigger_alert()
                if not config.ALERT_CLASSES or class_name in config.ALERT_CLASSES:
                    current_t  = time.time()
                    last_alert = alert_system.last_alert_time.get(class_name, 0)

                    if current_t - last_alert >= config.COOLDOWN_TIME:
                        if config.LOG_CSV:
                            log_detection_csv(config.CSV_LOG_FILE, class_name, confidence)
                        if config.SAVE_SNAPSHOTS:
                            save_snapshot(frame, config.SNAPSHOT_DIR, class_name)
                        if clip_recorder:
                            clip_recorder.trigger(class_name)
                        # trigger_alert sends the ntfy notification and records the timestamp
                        alert_system.trigger_alert(class_name, confidence)

            # ── Draw live stats overlay ───────────────────────────────────────
            if config.SHOW_STATS:
                frame = draw_stats(frame, fps, object_counts)

            # ── Display frame ─────────────────────────────────────────────────
            cv2.imshow('Real-Time Object Detection System', frame)

            # Exit condition — waitKey(1) keeps the window responsive
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Cleaning up...")
        detector.stop()
        if clip_recorder:
            clip_recorder.stop()
        if cont_recorder:
            cont_recorder.stop()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
