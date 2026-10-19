"""
exp16_wm/code_version.py — which code produced a result (Exp-16): the git commit, written into every results.json / eval file.

In a git checkout: `git rev-parse HEAD` (+ '-dirty' when tracked files have uncommitted changes). In a cloud code bundle (no .git):
the GIT_COMMIT file cloud/launch.py puts at the bundle root. Otherwise 'unknown'.
"""

import os
import subprocess

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMIT_FILE = "GIT_COMMIT"


def code_version(repo_dir: str = REPO_DIR) -> str:
    try:
        head = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    path = os.path.join(repo_dir, COMMIT_FILE)
    if os.path.exists(path):
        return open(path).read().strip()
    return "unknown"
