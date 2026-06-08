import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from app.camera import CameraStream
from app.detection import ObjectDetector
from app.alerts import AlertSystem
from app.utils import draw_bounding_box

def main():
    print("Initializing Real-Time Object Detection System...")
    
    try:
        camera = CameraStream(source=0) # 0 for default webcam
    except ValueError as e:
        print(e)
        return

    detector = ObjectDetector()
    alert_system = AlertSystem()
    
    print("System started successfully. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = camera.get_frame()
            if not ret:
                print("Failed to grab frame. Exiting...")
                break
                
            # Run detection
            detections = detector.detect(frame)
            
            # Process detections
            for det in detections:
                class_name = det["class_name"]
                confidence = det["confidence"]
                box = det["box"]
                
                # Draw bounding box
                frame = draw_bounding_box(frame, box, class_name, confidence)
                
                # Check for alerts:
                # If ALERT_CLASSES is empty, alert on ALL detected objects
                # Otherwise only alert on the specified classes
                if not config.ALERT_CLASSES or class_name in config.ALERT_CLASSES:
                    alert_system.trigger_alert(class_name, confidence)
                    
            # Display frame
            cv2.imshow('Real-Time Object Detection System', frame)
            
            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Cleaning up...")
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
