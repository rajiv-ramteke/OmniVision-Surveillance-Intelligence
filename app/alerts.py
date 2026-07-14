import time
import pyttsx3
import threading
import queue
import requests
import os
import sys

# Add root project dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config


class AlertSystem:
    def __init__(self):
        # Cooldown for FULL alerts (snapshots, CSV, ntfy) — long
        self.last_alert_time = {}
        # Cooldown for SOUND ONLY (voice) — short, fires on every detection
        self._last_sound_time = {}
        self._sound_cooldown  = 0.5   # seconds between voice alerts per class

        # ── Voice queue: maxsize=1 so we NEVER queue a backlog.
        # Newest object name always replaces any stale queued item.
        self._voice_queue = queue.Queue(maxsize=1)
        self._voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self._voice_thread.start()

        print("[Voice] Text-to-speech engine ready — speaks object names aloud.")

        # Send a startup ping so you know ntfy is connected
        if config.NTFY_TOPIC:
            threading.Thread(target=self._send_startup_ping, daemon=True).start()

    # ── Voice worker ─────────────────────────────────────────────────────────
    def _voice_worker(self):
        """
        Dedicated thread: speaks the object name aloud using Windows TTS.
        Uses a maxsize=1 queue so it always speaks the *latest* detected
        object and never builds up a long backlog.
        """
        engine = pyttsx3.init()
        engine.setProperty('rate', 200)    # Faster = snappier announcements
        engine.setProperty('volume', 1.0)  # Max volume

        # Try to pick a clear English voice if available
        voices = engine.getProperty('voices')
        for v in voices:
            if 'english' in v.name.lower() or 'zira' in v.name.lower() or 'david' in v.name.lower():
                engine.setProperty('voice', v.id)
                break

        while True:
            text = self._voice_queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[Voice] Error: {e}")
            finally:
                self._voice_queue.task_done()

    def _speak(self, object_name):
        """
        Non-blocking speech: speak just the object name.
        If the voice thread is still busy, replace the queued item with
        the newest name so we never lag behind detections.
        """
        # Drain the queue first (discard stale item)
        while not self._voice_queue.empty():
            try:
                self._voice_queue.get_nowait()
                self._voice_queue.task_done()
            except queue.Empty:
                break
        # Now put the fresh name
        try:
            self._voice_queue.put_nowait(object_name)
        except queue.Full:
            pass  # still busy — skip this one

    # ── Public API ────────────────────────────────────────────────────────────

    def sound_alert(self, class_name):
        """
        Called on EVERY detection frame.
        Speaks the object name with a SHORT per-class cooldown (0.5 s).
        """
        now = time.time()
        if now - self._last_sound_time.get(class_name, 0) < self._sound_cooldown:
            return  # too soon for this class — skip
        self._last_sound_time[class_name] = now
        self._speak(class_name)

    def trigger_alert(self, class_name, confidence=None):
        """
        Called for heavy alerts: ntfy push notification.
        Uses the longer COOLDOWN_TIME from config (default 60 s).
        CSV logging and snapshots are handled in main.py before this call.
        """
        current_time = time.time()

        if current_time - self.last_alert_time.get(class_name, 0) < config.COOLDOWN_TIME:
            return  # Still in cooldown

        self.last_alert_time[class_name] = current_time

        # Mobile notification in background
        threading.Thread(
            target=self._send_mobile_notification,
            args=(class_name, confidence),
            daemon=True
        ).start()

    # ── Mobile notification ───────────────────────────────────────────────────

    # Danger-level classes get 'urgent' priority on ntfy
    _URGENT_CLASSES = {'knife', 'gun', 'fire', 'scissors'}

    # Full emoji tag map for ntfy
    _TAG_MAP = {
        'person':       'bust_in_silhouette',
        'car':          'red_car',
        'truck':        'truck',
        'bus':          'bus',
        'motorcycle':   'motorcycle',
        'bicycle':      'bike',
        'dog':          'dog',
        'cat':          'cat',
        'knife':        'hocho',
        'gun':          'rotating_light',
        'fire':         'fire',
        'scissors':     'scissors',
        'cell phone':   'iphone',
        'laptop':       'computer',
        'backpack':     'school_satchel',
        'handbag':      'handbag',
        'bottle':       'bottle_with_popping_cork',
        'cup':          'coffee',
        'book':         'books',
        'keyboard':     'keyboard',
        'mouse':        'mouse',
        'tv':           'tv',
        'remote':       'joystick',
        'clock':        'clock1',
        'chair':        'chair',
        'umbrella':     'umbrella',
        'suitcase':     'luggage',
    }

    def _send_startup_ping(self):
        """Sends a one-time startup notification so you know ntfy is working."""
        try:
            url = f"https://ntfy.envs.net/{config.NTFY_TOPIC}"
            import datetime
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            headers = {
                "Title":    "Object Detection System Online",
                "Tags":     "white_check_mark,camera",
                "Priority": "default",
            }
            if hasattr(config, 'NTFY_TOKEN') and config.NTFY_TOKEN:
                headers["Authorization"] = f"Bearer {config.NTFY_TOKEN}"
                
            requests.post(
                url,
                data=f"System ONLINE at {ts} - Watching for objects...".encode('utf-8'),
                headers=headers,
                timeout=8
            )
            print(f"[ntfy] Startup ping sent to ntfy.envs.net/{config.NTFY_TOPIC}")
        except Exception as e:
            print(f"[ntfy] Startup ping failed: {e}")

    def _send_mobile_notification(self, class_name, confidence=None):
        """Sends a push notification to ntfy with retry on failure."""
        import datetime
        conf_str  = f" — {int(confidence * 100)}% confident" if confidence else ""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        conf_pct  = int(confidence * 100) if confidence else 0

        print(f"\n[ALERT] {class_name.upper()} detected{conf_str}")

        if not config.NTFY_TOPIC:
            return

        url      = f"https://ntfy.envs.net/{config.NTFY_TOPIC}"
        message  = f"[ALERT] {class_name.capitalize()} detected at {timestamp}{conf_str}."
        tag      = self._TAG_MAP.get(class_name, 'eyes')
        priority = "urgent" if class_name in self._URGENT_CLASSES else "high"

        headers = {
            "Title":    f"{class_name.capitalize()} Detected!",
            "Tags":     f"warning,{tag}",
            "Priority": priority,
        }
        
        if hasattr(config, 'NTFY_TOKEN') and config.NTFY_TOKEN:
            headers["Authorization"] = f"Bearer {config.NTFY_TOKEN}"

        # Try up to 2 times in case of transient network error
        for attempt in range(1, 3):
            try:
                response = requests.post(
                    url,
                    data=message.encode('utf-8'),
                    headers=headers,
                    timeout=6          # don't hang the thread
                )
                if response.status_code == 200:
                    print(f"[ntfy] ✓ Sent: {class_name} ({conf_pct}%)")
                    return
                else:
                    print(f"[ntfy] Attempt {attempt} failed ({response.status_code}): {response.text[:80]}")
            except requests.exceptions.Timeout:
                print(f"[ntfy] Attempt {attempt} timed out")
            except Exception as e:
                print(f"[ntfy] Attempt {attempt} error: {e}")

        print(f"[ntfy] ✗ All attempts failed for: {class_name}")
