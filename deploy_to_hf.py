"""
deploy_to_hf.py
---------------
Deploys the project to HuggingFace Spaces as a FREE Static Space.
Static Spaces host pure HTML/CSS/JS — no Python server, no PRO subscription needed.

Usage:
    python deploy_to_hf.py
"""

import subprocess
import sys
import os

# Config
SPACE_NAME  = "OmniVision-Intelligence-System"
SPACE_DIR   = os.path.join(os.path.dirname(__file__), "hf_spaces")

# Only upload these files for a static space (skip app.py, requirements.txt, Dockerfile)
STATIC_FILES = ["index.html", "README.md", "yolov8s.onnx"]


def main():
    print("=" * 60)
    print("  HuggingFace Spaces -- Static Space Deployment")
    print("=" * 60)

    # 1. Make sure huggingface_hub is installed
    try:
        import huggingface_hub
        print("[OK] huggingface_hub " + huggingface_hub.__version__ + " found")
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        import huggingface_hub

    from huggingface_hub import HfApi, login

    # 2. Login securely in terminal
    print("\n[AUTH] Enter your HuggingFace token below.")
    print("       Get it from: https://huggingface.co/settings/tokens")
    print("       Token is NOT saved anywhere.\n")
    login()

    api = HfApi()
    try:
        user_info = api.whoami()
        hf_username = user_info["name"]
    except Exception:
        # Fallback if token is read-only
        hf_username = input("Enter your HuggingFace username: ").strip()

    # 3. Create Static Space (free for everyone)
    repo_id = hf_username + "/" + SPACE_NAME
    print("\n[SPACE] Creating Static Space: " + repo_id)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="static",
            private=False,
            exist_ok=True,
        )
        print("[OK] Space ready: https://huggingface.co/spaces/" + repo_id)
    except Exception as e:
        print("[FAIL] Could not create Space: " + str(e))
        sys.exit(1)

    # 4. Upload static files only
    print("\n[UPLOAD] Uploading static files...")
    count = 0
    for filename in STATIC_FILES:
        filepath = os.path.join(SPACE_DIR, filename)
        if not os.path.isfile(filepath):
            print("   [SKIP] " + filename + " not found")
            continue
        print("   -> " + filename)
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="space",
        )
        count += 1

    print("\n[DONE] " + str(count) + " files uploaded.")
    print("[LIVE] https://huggingface.co/spaces/" + repo_id)
    print("       Static Spaces load instantly — no build step needed!")


if __name__ == "__main__":
    main()
