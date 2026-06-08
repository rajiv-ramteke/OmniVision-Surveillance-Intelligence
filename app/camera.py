import cv2
import time

class CameraStream:
    def __init__(self, source=0):
        """
        Initializes the camera stream. 
        source: 0 for default webcam, or a string for a video file/RTSP URL.
        Uses DirectShow backend on Windows to avoid black screen issues.
        """
        # cv2.CAP_DSHOW is the Windows DirectShow backend - fixes black screen
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")
        
        # Set resolution explicitly to help camera initialize
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Warm-up: discard first several frames so camera auto-exposure adjusts
        print("Warming up camera...")
        for _ in range(10):
            self.cap.read()
        print("Camera ready.")
            
    def get_frame(self):
        """
        Reads a frame from the stream. Returns (success, frame).
        """
        ret, frame = self.cap.read()
        return ret, frame
        
    def release(self):
        """
        Releases the camera resource.
        """
        if self.cap.isOpened():
            self.cap.release()
