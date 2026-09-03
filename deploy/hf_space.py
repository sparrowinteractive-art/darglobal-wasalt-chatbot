"""Deploy the backend container to a Hugging Face Space (Docker SDK).

Usage:
    HF_TOKEN=hf_xxx python deploy/hf_space.py [--space <user>/<name>]

Creates the Space if needed, uploads the project (without the venv, git
history, web frontend and local index), sets the API keys as Space secrets,
and prints the public URL. The Space builds the Dockerfile itself.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

IGNORE = [
    ".venv/**", ".git/**", "__pycache__/**", "**/__pycache__/**", "web/**", "data/index/**",
    "data/raw/*.log", "data/raw/*.txt", ".env", "deploy/**", "*.png", "tests_*.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", help="<user>/<name>; defaults to <you>/darglobal-wasalt-chatbot")
    ap.add_argument("--cors", default=os.getenv("CORS_ORIGINS", "*"))
    args = ap.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    api = HfApi(token=token)
    me = api.whoami()["name"]
    repo_id = args.space or f"{me}/darglobal-wasalt-chatbot"

    url = api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    print("space:", url)

    secrets = {
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
        "SARVAM_API_KEY": os.getenv("SARVAM_API_KEY", ""),
    }
    for k, v in secrets.items():
        if v and "..." not in v:
            api.add_space_secret(repo_id, k, v)
            print("secret set:", k)
    api.add_space_variable(repo_id, "CORS_ORIGINS", args.cors)
    api.add_space_variable(repo_id, "PORT", "8000")
    api.add_space_variable(repo_id, "OPENROUTER_MODELS", os.getenv("OPENROUTER_MODELS", "google/gemma-4-31b-it:free,z-ai/glm-5.2:free,nvidia/nemotron-3-super-120b-a12b:free,minimax/minimax-m2.7:free"))
    api.add_space_variable(repo_id, "SARVAM_MODELS", os.getenv("SARVAM_MODELS", "sarvam-105b-conversations"))

    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(ROOT), ignore_patterns=IGNORE, commit_message="Deploy chatbot backend")
    print("uploaded; build starting")
    host = f"https://{repo_id.replace('/', '-').replace('_', '-').lower()}.hf.space"
    print("public API URL (once built):", host)
    print("health:", host + "/health")


if __name__ == "__main__":
    main()
