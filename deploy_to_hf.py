"""
deploy_to_hf.py
---------------
Run this script to deploy the project to HuggingFace Spaces.
Your token is entered securely in the terminal -- never stored in code.

Usage:
    python deploy_to_hf.py
"""

import subprocess
import sys
import os

# Config
HF_USERNAME = "rajiv-ramteke"
SPACE_NAME  = "Real-Time-Object-Detection"
SPACE_DIR   = os.path.join(os.path.dirname(__file__), "hf_spaces")


def main():
    print("=" * 60)
    print("  HuggingFace Spaces -- Deployment Script")
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

    # 3. Create Space
    repo_id = HF_USERNAME + "/" + SPACE_NAME
    print("\n[SPACE] Creating Space: " + repo_id)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True,
        )
        print("[OK] Space ready: https://huggingface.co/spaces/" + repo_id)
    except Exception as e:
        print("[FAIL] Could not create Space: " + str(e))
        sys.exit(1)

    # 4. Upload files
    print("\n[UPLOAD] Uploading files from: " + SPACE_DIR)
    count = 0
    for filename in os.listdir(SPACE_DIR):
        filepath = os.path.join(SPACE_DIR, filename)
        if os.path.isfile(filepath):
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
    print("       (First build takes 2-3 minutes)")


if __name__ == "__main__":
    main()
