"""
app/cloud_storage.py
────────────────────
Handles automatic background upload of detection snapshots to Google Drive.

How it works:
  1. After a snapshot is saved locally, call `upload_snapshot(filepath)`.
  2. This queues the file for upload on a daemon thread — it NEVER blocks
     the main detection loop.
  3. On first run, it opens a browser for OAuth consent (one-time login).
     After that, a token.json is saved so it never asks again.

Setup (one time only):
  1. Go to: https://console.cloud.google.com/
  2. Create a new project > Enable "Google Drive API"
  3. Create OAuth 2.0 credentials (Desktop App) > Download as credentials.json
  4. Place credentials.json in the root project folder.
  5. Set GOOGLE_DRIVE_UPLOAD = True in config/config.py
"""

import os
import sys
import queue
import threading
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# ── Lazy imports: only load Google libraries if Drive is enabled ──────────────
_drive_service = None
_folder_id     = None
_upload_queue  = queue.Queue()
_initialized   = False


def _authenticate():
    """Performs OAuth2 authentication and returns a Google Drive service."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    token_path = 'token.json'
    creds = None

    # Load saved token if available
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    # If no valid credentials, run OAuth flow (opens browser once)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(config.CREDENTIALS_FILE):
                print(
                    f"\n[Drive] ERROR: '{config.CREDENTIALS_FILE}' not found!\n"
                    "  Please download it from Google Cloud Console and place it\n"
                    "  in the project root folder. Drive upload is DISABLED.\n"
                )
                return None
            flow = InstalledAppFlow.from_client_secrets_file(
                config.CREDENTIALS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save token for future runs
        with open(token_path, 'w') as f:
            f.write(creds.to_json())

    return build('drive', 'v3', credentials=creds)


def _get_or_create_folder(service, folder_name):
    """Returns the Drive folder ID, creating it if it doesn't exist."""
    # Search for existing folder
    query = (
        f"name='{folder_name}' "
        f"and mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])

    if files:
        folder_id = files[0]['id']
        print(f"[Drive] Using existing folder '{folder_name}' (id={folder_id})")
        return folder_id

    # Create new folder
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    folder_id = folder.get('id')
    print(f"[Drive] Created new folder '{folder_name}' (id={folder_id})")
    return folder_id


def _upload_worker():
    """Background daemon thread that processes the upload queue."""
    global _drive_service, _folder_id, _initialized

    print("[Drive] Upload worker started. Authenticating with Google Drive...")
    _drive_service = _authenticate()
    if _drive_service is None:
        print("[Drive] Authentication failed. Upload worker exiting.")
        return

    _folder_id = _get_or_create_folder(_drive_service, config.GOOGLE_DRIVE_FOLDER)
    _initialized = True
    print(f"[Drive] ✓ Ready. Snapshots will be uploaded to '{config.GOOGLE_DRIVE_FOLDER}'")

    from googleapiclient.http import MediaFileUpload
    import mimetypes

    while True:
        filepath = _upload_queue.get()
        if filepath is None:
            break  # Poison pill — stop the thread
        try:
            filename  = os.path.basename(filepath)
            mime_type = mimetypes.guess_type(filepath)[0] or 'image/jpeg'

            file_metadata = {
                'name':    filename,
                'parents': [_folder_id]
            }
            media = MediaFileUpload(filepath, mimetype=mime_type, resumable=False)
            uploaded = _drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,webViewLink'
            ).execute()

            print(f"[Drive] ✓ Uploaded: {filename}  →  {uploaded.get('webViewLink', '')}")
        except Exception:
            print(f"[Drive] ✗ Upload failed for: {filepath}")
            traceback.print_exc()
        finally:
            _upload_queue.task_done()


# ── Public API ────────────────────────────────────────────────────────────────

_worker_thread = None


def start():
    """
    Call once at app startup (only if GOOGLE_DRIVE_UPLOAD is True).
    Launches the background upload thread and handles authentication.
    """
    global _worker_thread
    if not config.GOOGLE_DRIVE_UPLOAD:
        return

    _worker_thread = threading.Thread(target=_upload_worker, daemon=True, name="DriveUploader")
    _worker_thread.start()


def upload_snapshot(filepath: str):
    """
    Non-blocking: queues a snapshot file for background upload to Google Drive.
    Call this right after save_snapshot() in main.py.
    Does nothing if Drive upload is disabled or not yet initialized.
    """
    if not config.GOOGLE_DRIVE_UPLOAD:
        return

    if not os.path.exists(filepath):
        print(f"[Drive] File not found, skipping upload: {filepath}")
        return

    _upload_queue.put(filepath)
