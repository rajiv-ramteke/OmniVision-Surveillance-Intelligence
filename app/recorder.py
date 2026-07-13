"""
app/recorder.py
───────────────
Handles two video-recording modes:

  1. CLIP_ON_DETECTION  (RECORD_CLIPS = True in config)
     When an object alert fires, a short clip (CLIP_DURATION_SEC seconds) is
     saved around that moment.  The recorder keeps a rolling pre-buffer so the
     clip actually starts a couple of seconds BEFORE the detection.

  2. CONTINUOUS_RECORD  (CONTINUOUS_RECORD = True in config)
     The full session is written to a single .mp4 file that is created at
     startup and closed cleanly on exit.

Both modes write to sub-folders inside RECORDING_DIR and are completely
non-blocking — frame writing is offloaded to a daemon thread.
"""

import cv2
import os
import queue
import threading
import datetime
import collections
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_writer(filepath: str, fps: float, width: int, height: int):
    """Creates an OpenCV VideoWriter using the mp4v codec."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[Recorder] ✗ Could not open VideoWriter for: {filepath}")
        return None
    return writer


def _timestamp_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ─────────────────────────────────────────────────────────────────────────────
#  Clip recorder  (triggered on detection)
# ─────────────────────────────────────────────────────────────────────────────

class ClipRecorder:
    """
    Saves short video clips around detection events.

    • Maintains a rolling pre-buffer (PRE_BUFFER_SEC seconds of frames).
    • On trigger(), dumps the buffer + records for CLIP_DURATION_SEC more
      seconds, then stops.
    • Multiple triggers while a clip is already recording simply extend the
      recording window by resetting the countdown.
    """

    def __init__(self, fps: float, width: int, height: int):
        self._fps    = max(fps, 1.0)
        self._width  = width
        self._height = height

        pre_buf_frames = int(config.PRE_BUFFER_SEC * self._fps)
        self._pre_buf  = collections.deque(maxlen=pre_buf_frames)

        self._writer       = None
        self._frames_left  = 0          # frames still to write after trigger
        self._lock         = threading.Lock()

        # Async write queue — keeps main loop unblocked
        self._write_queue  = queue.Queue(maxsize=512)
        self._worker       = threading.Thread(target=self._write_worker, daemon=True,
                                              name="ClipWriter")
        self._worker.start()

        # Per-class cooldown so one class doesn't spam many files
        self._last_clip_time = {}

        clip_dir = os.path.join(config.RECORDING_DIR, 'clips')
        os.makedirs(clip_dir, exist_ok=True)
        print(f"[ClipRecorder] Ready — clips → {clip_dir}")

    # ── internal ─────────────────────────────────────────────────────────────

    def _write_worker(self):
        """Daemon thread: drains the write queue and writes frames to disk."""
        while True:
            item = self._write_queue.get()
            if item is None:
                break                   # poison pill
            frame, writer = item
            if writer and writer.isOpened():
                writer.write(frame)
            self._write_queue.task_done()

    def _new_clip_path(self, class_name: str) -> str:
        clip_dir = os.path.join(config.RECORDING_DIR, 'clips')
        return os.path.join(clip_dir, f"{class_name}_{_timestamp_str()}.mp4")

    def _close_current_clip(self):
        """
        Closes the active writer cleanly.
        Must be called while NOT holding self._lock.
        Sends a poison pill to the CURRENT worker, waits for it to drain,
        then starts a fresh worker for the next clip.
        """
        with self._lock:
            writer = self._writer
            self._writer = None
            self._frames_left = 0

        if writer is None:
            return

        # Send poison pill to the current worker and wait for it to finish
        self._write_queue.put(None)
        self._worker.join(timeout=5)

        # Release the writer only AFTER the worker has drained the queue
        writer.release()

        # Start a fresh queue and worker for the next clip
        self._write_queue = queue.Queue(maxsize=512)
        self._worker = threading.Thread(target=self._write_worker,
                                        daemon=True, name="ClipWriter")
        self._worker.start()

    # ── public API ────────────────────────────────────────────────────────────

    def push_frame(self, frame):
        """
        Call on EVERY frame from the main loop.
        • Always adds the frame to the pre-buffer.
        • If a clip is active, queues the frame for writing.
        """
        # Always keep pre-buffer fresh
        self._pre_buf.append(frame.copy())

        with self._lock:
            if self._writer is None:
                return
            if self._frames_left <= 0:
                # Clip just finished — close in background to avoid blocking
                threading.Thread(target=self._close_current_clip, daemon=True).start()
                return
            self._frames_left -= 1

        self._write_queue.put((frame.copy(), self._writer))

    def trigger(self, class_name: str):
        """
        Trigger a clip save for the given class.
        Respects CLIP_COOLDOWN_SEC so one class doesn't create too many files.
        """
        now = datetime.datetime.now().timestamp()
        if now - self._last_clip_time.get(class_name, 0) < config.CLIP_COOLDOWN_SEC:
            # Extend the current clip instead of starting a new one
            with self._lock:
                if self._writer is not None:
                    self._frames_left = max(
                        self._frames_left,
                        int(config.CLIP_DURATION_SEC * self._fps)
                    )
            return

        self._last_clip_time[class_name] = now

        with self._lock:
            if self._writer is None:
                # Open a new writer
                path   = self._new_clip_path(class_name)
                writer = _make_writer(path, self._fps, self._width, self._height)
                if writer is None:
                    return
                self._writer      = writer
                self._frames_left = int(config.CLIP_DURATION_SEC * self._fps)

                # Flush pre-buffer frames first (these are BEFORE the trigger)
                for buffered_frame in list(self._pre_buf):
                    self._write_queue.put((buffered_frame, self._writer))

                print(f"[ClipRecorder] 🔴 Recording clip → {path}")
            else:
                # Extend ongoing clip
                self._frames_left = int(config.CLIP_DURATION_SEC * self._fps)

    def stop(self):
        """Cleanly stop — flush queue and release any open writer."""
        self._write_queue.put(None)
        self._worker.join(timeout=5)
        with self._lock:
            if self._writer:
                self._writer.release()
                self._writer = None


# ─────────────────────────────────────────────────────────────────────────────
#  Continuous recorder
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousRecorder:
    """
    Records the entire session to a single .mp4 file.
    Frame writing is asynchronous — main loop never blocks.
    """

    def __init__(self, fps: float, width: int, height: int):
        self._fps    = max(fps, 1.0)
        self._width  = width
        self._height = height

        path = os.path.join(
            config.RECORDING_DIR, 'sessions',
            f"session_{_timestamp_str()}.mp4"
        )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        self._writer = _make_writer(path, self._fps, self._width, self._height)
        self._queue  = queue.Queue(maxsize=1024)
        self._worker = threading.Thread(target=self._write_worker, daemon=True,
                                        name="ContWriter")
        self._worker.start()

        if self._writer:
            print(f"[ContinuousRecorder] 🔴 Recording session → {path}")
        else:
            print("[ContinuousRecorder] ✗ Failed to start — check RECORDING_DIR")

    def _write_worker(self):
        while True:
            frame = self._queue.get()
            if frame is None:
                break
            if self._writer and self._writer.isOpened():
                self._writer.write(frame)
            self._queue.task_done()

    def push_frame(self, frame):
        """Non-blocking push — drops frame if queue is full to avoid lag."""
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # drop frame — display loop must never lag

    def stop(self):
        """Flush and close the output file."""
        self._queue.put(None)
        self._worker.join(timeout=5)
        if self._writer:
            self._writer.release()
            self._writer = None
        print("[ContinuousRecorder] ✓ Session recording saved.")
