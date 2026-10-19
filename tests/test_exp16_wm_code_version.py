"""Tests for exp16_wm.code_version and the GIT_COMMIT file of the cloud bundle."""

import os
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cloud.launch import make_bundle
from exp16_wm.code_version import code_version


def _repo(tmp: Path) -> Path:
    r = tmp / "repo"; r.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(r), *a], check=True, capture_output=True)
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    (r / "a.py").write_text("x = 1\n"); run("add", "a.py"); run("commit", "-q", "-m", "c")
    return r


def test_git_checkout_gives_head_and_marks_dirty(tmp_path):
    r = _repo(tmp_path)
    head = subprocess.run(["git", "-C", str(r), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert code_version(str(r)) == head
    (r / "a.py").write_text("x = 2\n")
    assert code_version(str(r)) == head + "-dirty"


def test_bundle_carries_the_commit_and_an_unpacked_bundle_reports_it(tmp_path):
    r = _repo(tmp_path)
    head = subprocess.run(["git", "-C", str(r), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = make_bundle(r, tmp_path / "b.tar.gz")
    d = tmp_path / "unpacked"; d.mkdir()
    with tarfile.open(out) as t:
        t.extractall(d)
    assert (d / "GIT_COMMIT").read_text().strip() == head
    assert code_version(str(d)) == head                                  # no .git there: falls back to the file


def test_unknown_without_git_or_file(tmp_path):
    assert code_version(str(tmp_path)) == "unknown"
