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
        self.last_alert_time = {}

        # pyttsx3 is NOT thread-safe — run it on a single dedicated worker thread
        self._voice_queue = queue.Queue()
        self._voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self._voice_thread.start()
        print("[Voice] Text-to-speech engine ready.")
        
    def _voice_worker(self):
        """Dedicated background thread that processes voice announcements one at a time."""
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)   # Speaking speed
        engine.setProperty('volume', 1.0) # Max volume
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

    def trigger_alert(self, class_name, confidence=None):
        current_time = time.time()
        
        # Check cooldown per class
        if class_name in self.last_alert_time:
            if current_time - self.last_alert_time[class_name] < config.COOLDOWN_TIME:
                return # Still in cooldown
                
        self.last_alert_time[class_name] = current_time
        
        # Enqueue voice announcement (safe — one thread handles all speech)
        self._voice_queue.put(f"{class_name} detected")
        
        # Send mobile notification asynchronously
        threading.Thread(
            target=self._send_mobile_notification,
            args=(class_name, confidence),
            daemon=True
        ).start()
        
            
    def _send_mobile_notification(self, class_name, confidence=None):
        conf_str = f" ({int(confidence * 100)}% confidence)" if confidence else ""
        print(f"\n[ALERT] {class_name.upper()} detected{conf_str}")
        
        if config.NTFY_TOPIC:
            try:
                url = f"https://ntfy.sh/{config.NTFY_TOPIC}"
                message = f"{class_name.capitalize()} detected{conf_str}."

                # Pick an emoji tag based on the object class
                tag_map = {
                    'person': 'bust_in_silhouette',
                    'car': 'red_car',
                    'dog': 'dog',
                    'cat': 'cat',
                    'knife': 'hocho',
                    'gun': 'rotating_light',
                    'fire': 'fire',
                    'cell phone': 'iphone',
                    'laptop': 'computer',
                    'backpack': 'school_satchel',
                }
                tag = tag_map.get(class_name, 'eyes')  # default fallback emoji
                
                headers = {
                    "Title": f"Security Alert: {class_name.capitalize()} Detected",
                    "Tags": f"warning,{tag}",
                    "Priority": "high"
                }
                response = requests.post(url, data=message.encode('utf-8'), headers=headers)
                if response.status_code == 200:
                    print(f"[ntfy] Mobile notification sent for: {class_name}")
                else:
                    print(f"[ntfy] Failed to send notification: {response.text}")
            except Exception as e:
                print(f"[ntfy] Notification error: {e}")
