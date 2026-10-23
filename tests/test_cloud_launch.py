"""tests/test_cloud_launch.py — cloud/launch.py: user-data generation, validation, code bundle."""

import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cloud.launch import CMD_DELIMITER, MAX_HOURS_CAP, SPOT_POLL_SECONDS, build_user_data, is_dirty, make_bundle, run_aws  # noqa: E402

ARGS = dict(bucket="my-results-bucket", job="smoke-1", command="python -c 'print(1)'", max_hours=2.0, region="us-east-1")


def test_user_data_happy_path() -> None:
    script = build_user_data(**ARGS)
    assert script.startswith("#!/bin/bash")
    assert "shutdown -h +120" in script  # hard deadline, 2 h
    assert "trap finish EXIT" in script  # upload + shutdown even when the command fails
    assert "shutdown -h now" in script
    assert "s3://$BUCKET/runs/$JOB/" in script and "BUCKET=my-results-bucket" in script and "JOB=smoke-1" in script
    assert "python -c 'print(1)'" in script
    assert "setup_cloud.sh" in script


def test_user_data_deadline_rounds_up() -> None:
    assert "shutdown -h +31" in build_user_data(**{**ARGS, "max_hours": 0.51})


@pytest.mark.parametrize("hours", [0, -1, MAX_HOURS_CAP + 0.1, float("nan")])
def test_user_data_rejects_bad_deadline(hours: float) -> None:
    with pytest.raises(ValueError):
        build_user_data(**{**ARGS, "max_hours": hours})


@pytest.mark.parametrize("job", ["", "has space", "../escape", "a/b", "x" * 41, "-lead"])
def test_user_data_rejects_bad_job(job: str) -> None:
    with pytest.raises(ValueError):
        build_user_data(**{**ARGS, "job": job})


@pytest.mark.parametrize("command", ["", "   \n", f"echo hi\n{CMD_DELIMITER}\nrm -rf /"])
def test_user_data_rejects_bad_command(command: str) -> None:
    with pytest.raises(ValueError):
        build_user_data(**{**ARGS, "command": command})


@pytest.mark.parametrize("field,value", [("bucket", "Bad_Bucket"), ("bucket", "x"), ("region", "us east 1"), ("region", "")])
def test_user_data_rejects_bad_bucket_or_region(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        build_user_data(**{**ARGS, field: value})


def _fake_run(returncode: int, stderr: str):  # noqa: ANN202
    def run(cmd, **kwargs):  # noqa: ANN001, ANN003, ANN202
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

    return run


def test_run_aws_failure_surfaces_aws_error_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run(254, "An error occurred (InsufficientInstanceCapacity): no capacity"))
    with pytest.raises(RuntimeError, match="InsufficientInstanceCapacity"):
        run_aws(["ec2", "run-instances"], "us-east-1")


def test_run_aws_check_false_returns_failure_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _fake_run(254, "DryRunOperation"))
    res = run_aws(["ec2", "run-instances"], "us-east-1", check=False)
    assert res.returncode == 254 and "DryRunOperation" in res.stderr


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "tracked.txt").write_text("tracked")
    (tmp_path / "untracked_dir").mkdir()
    (tmp_path / "untracked_dir" / "data.bin").write_bytes(b"\x00\x01")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_bundle_has_tracked_and_extras_only(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    out = tmp_path_factory.mktemp("out") / "b.tar.gz"
    make_bundle(repo, out, extras=["untracked_dir"])
    with tarfile.open(out) as tar:
        names = set(tar.getnames())
    assert "tracked.txt" in names
    assert "untracked_dir/data.bin" in names


def test_bundle_without_extras_skips_untracked(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    out = tmp_path_factory.mktemp("out") / "b.tar.gz"
    make_bundle(repo, out)
    with tarfile.open(out) as tar:
        assert "untracked_dir/data.bin" not in tar.getnames()


def test_bundle_missing_extra_raises(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(FileNotFoundError):
        make_bundle(repo, tmp_path_factory.mktemp("out") / "b.tar.gz", extras=["nope"])


@pytest.mark.parametrize("extra", ["", " ", "."])
def test_bundle_rejects_extra_naming_the_repo_root(repo: Path, tmp_path_factory: pytest.TempPathFactory, extra: str) -> None:
    with pytest.raises(ValueError):
        make_bundle(repo, tmp_path_factory.mktemp("out") / "b.tar.gz", extras=[extra])


def test_is_dirty_detects_tracked_changes_only(repo: Path) -> None:
    assert not is_dirty(repo)  # untracked files do not count
    (repo / "tracked.txt").write_text("changed")
    assert is_dirty(repo)


def test_user_data_uploads_progress_and_handles_spot_notice() -> None:
    script = build_user_data(**ARGS, sync_seconds=120)
    assert "watch_progress &" in script and "WATCH_PID=$!" in script          # runs in the background from the start
    assert script.index("watch_progress &") < script.index("aws s3 cp \"s3://$BUCKET/code/")
    assert '[ "$n" -ge 120 ]' in script                                         # periodic upload interval
    assert "meta-data/spot/instance-action" in script and "api/token" in script  # IMDSv2 interruption notice
    assert "spot-interrupted" in script
    assert 'kill "$WATCH_PID"' in script                                        # finish() stops the watcher before the final upload
    assert script.count("sync_out") >= 4                                        # definition + periodic + notice + finish


@pytest.mark.parametrize("seconds", [SPOT_POLL_SECONDS - 1, 3601])
def test_user_data_rejects_bad_sync_interval(seconds: int) -> None:
    with pytest.raises(ValueError):
        build_user_data(**ARGS, sync_seconds=seconds)


def test_user_data_is_valid_bash() -> None:
    res = subprocess.run(["bash", "-n"], input=build_user_data(**ARGS), text=True, capture_output=True)
    assert res.returncode == 0, res.stderr



def test_user_data_restores_earlier_outputs_and_marks_running_before_the_command() -> None:
    script = build_user_data(**ARGS)
    restore = script.index('aws s3 sync "s3://$BUCKET/runs/$JOB/out" "$OUT"')
    assert script.index("echo running > /tmp/status_run") < restore < script.index("bash cloud_cmd.sh")
    assert restore < script.index('aws s3 cp "s3://$BUCKET/code/')                  # before the code and the command


def test_rank_placements_keeps_the_latest_price_per_pool_and_orders_cheapest_first() -> None:
    from cloud.launch import rank_placements
    hist = [
        {"InstanceType": "c7a.16xlarge", "AvailabilityZone": "us-east-1a", "SpotPrice": "0.90", "Timestamp": "2026-10-19T10:00:00Z"},
        {"InstanceType": "c7a.16xlarge", "AvailabilityZone": "us-east-1a", "SpotPrice": "1.20", "Timestamp": "2026-10-19T11:00:00Z"},
        {"InstanceType": "m7a.16xlarge", "AvailabilityZone": "us-east-1f", "SpotPrice": "0.70", "Timestamp": "2026-10-19T09:00:00Z"},
        {"InstanceType": "m7i.16xlarge", "AvailabilityZone": "us-east-1b", "SpotPrice": "0.10", "Timestamp": "2026-10-19T11:00:00Z"},
    ]
    r = rank_placements(hist, ["c7a.16xlarge", "m7a.16xlarge"])
    assert r == [("m7a.16xlarge", "us-east-1f", 0.70), ("c7a.16xlarge", "us-east-1a", 1.20)]   # latest c7a price; m7i not allowed
    assert rank_placements([], ["c7a.16xlarge"]) == []


def test_placement_overrides_set_type_and_subnet() -> None:
    from cloud.launch import placement_overrides
    assert placement_overrides("m7a.16xlarge", "subnet-1") == ["--instance-type", "m7a.16xlarge", "--subnet-id", "subnet-1"]
    assert placement_overrides("c7a.16xlarge", None) == ["--instance-type", "c7a.16xlarge"]


def test_avoid_placements_moves_reclaimed_pools_last() -> None:
    from cloud.launch import avoid_placements
    places = [("m7a.16xlarge", "us-east-1f", 0.7), ("c7a.16xlarge", "us-east-1b", 0.94), ("c7a.16xlarge", "us-east-1c", 0.99)]
    assert avoid_placements(places, ["c7a.16xlarge@us-east-1b"]) == [places[0], places[2], places[1]]
    assert avoid_placements(places, []) == places


def test_setup_falls_back_to_pypi_panda3d_when_the_archive_is_down() -> None:
    script = (Path(__file__).resolve().parent.parent / "cloud" / "setup_cloud.sh").read_text()
    assert "--extra-index-url https://archive.panda3d.org/" in script
    i_try, i_fb = script.index("if ! uv pip install --extra-index-url"), script.index('"panda3d==1.10.16"')
    assert i_try < i_fb < script.index("uv pip install --no-deps pooltool-billiards==0.5.0")

