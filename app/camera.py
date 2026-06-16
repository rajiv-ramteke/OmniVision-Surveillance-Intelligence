import cv2
import time
import threading

class CameraStream:
    def __init__(self, source=0):
        """
        Initializes the camera stream with a background reader thread.
        This eliminates the I/O bottleneck so the main loop never waits
        on cap.read() — it always gets the latest frame instantly.
        source: 0 for default webcam, or a string for a video file/RTSP URL.
        """
        # cv2.CAP_DSHOW is the Windows DirectShow backend - fixes black screen
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")

        # Set resolution explicitly to help camera initialize
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Reduce the internal capture buffer so we always get the freshest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm-up: discard first several frames so camera auto-exposure adjusts
        print("Warming up camera...")
        for _ in range(10):
            self.cap.read()
        print("Camera ready.")

        # Threaded frame-reading state
        self._ret = False
        self._frame = None
        self._lock = threading.Lock()
        self._stopped = False

        # Start the background reader thread
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        # Give the thread a moment to grab the first frame
        time.sleep(0.1)

    def _reader(self):
        """Background thread that continuously reads frames from the camera."""
        while not self._stopped:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame

    def get_frame(self):
        """
        Returns the most recent frame grabbed by the background thread.
        Returns (success, frame) — never blocks on I/O.
        """
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def release(self):
        """Stops the reader thread and releases the camera resource."""
        self._stopped = True
        time.sleep(0.2)
        if self.cap.isOpened():
            self.cap.release()
