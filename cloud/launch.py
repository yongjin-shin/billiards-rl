"""
cloud/launch.py — bundle the code, upload it to S3, and start one self-terminating Spot instance (roadmap: cloud execution plan).

The instance (cloud/stack.yaml launch template) downloads the bundle, runs cloud/setup_cloud.sh, runs your command, uploads
$OUT_DIR and the log to s3://<bucket>/runs/<job>/, and shuts itself down (shutdown behavior = terminate). A hard deadline
(--max-hours, capped at MAX_HOURS_CAP) ends it even if the command hangs.

Dry run is the default: it checks the arguments and asks EC2 whether the launch would be permitted, without uploading or
starting anything. Add --go to really launch.

Usage:
    AWS_PROFILE=agent-toolkit python cloud/launch.py --job smoke --command "python -c 'import pooltool'" --max-hours 0.5
    AWS_PROFILE=agent-toolkit python cloud/launch.py --job smoke --command "..." --max-hours 0.5 --go
"""

import argparse
import json
import math
import re
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
MAX_HOURS_CAP = 12.0
CMD_DELIMITER = "__BILLIARDS_JOB_CMD_EOF__"
JOB_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}")
BUCKET_PATTERN = re.compile(r"[a-z0-9][a-z0-9.-]{2,62}")
REGION_PATTERN = re.compile(r"[a-z]{2}(-[a-z]+)+-\d")


def validate_job(job: str) -> str:
    if not JOB_PATTERN.fullmatch(job):
        raise ValueError(f"job must match {JOB_PATTERN.pattern}, got {job!r}")
    return job


def validate_max_hours(max_hours: float) -> float:
    if not 0 < max_hours <= MAX_HOURS_CAP:
        raise ValueError(f"max_hours must be in (0, {MAX_HOURS_CAP}], got {max_hours}")
    return max_hours


def build_user_data(bucket: str, job: str, command: str, max_hours: float, region: str) -> str:
    """Bash script run by cloud-init as root on the instance. It always uploads the log and shuts the instance down."""
    validate_job(job)
    validate_max_hours(max_hours)
    if not BUCKET_PATTERN.fullmatch(bucket):
        raise ValueError(f"bucket looks invalid: {bucket!r}")
    if not REGION_PATTERN.fullmatch(region):
        raise ValueError(f"region looks invalid: {region!r}")
    if not command.strip():
        raise ValueError("command is empty")
    if CMD_DELIMITER in command:
        raise ValueError(f"command must not contain {CMD_DELIMITER}")
    minutes = math.ceil(max_hours * 60)
    return f"""#!/bin/bash
set -uo pipefail
exec > >(tee -a /var/log/job.log) 2>&1
export AWS_DEFAULT_REGION={region}
BUCKET={bucket}
JOB={job}
WORK=/opt/job
OUT=$WORK/cloud_out
STATUS=1

# Hard deadline: ends the instance (shutdown behavior = terminate) even if everything below hangs.
shutdown -h +{minutes}

finish() {{
  echo "[job] exit status $STATUS at $(date -u +%FT%TZ)"
  mkdir -p "$OUT"
  aws s3 cp /var/log/job.log "s3://$BUCKET/runs/$JOB/job.log" || true
  aws s3 sync "$OUT" "s3://$BUCKET/runs/$JOB/out" --only-show-errors || true
  echo "$STATUS" > /tmp/status
  aws s3 cp /tmp/status "s3://$BUCKET/runs/$JOB/STATUS" || true
  shutdown -h now
}}
trap finish EXIT

mkdir -p "$WORK" "$OUT" && cd "$WORK"
aws s3 cp "s3://$BUCKET/code/$JOB.tar.gz" code.tar.gz && tar xzf code.tar.gz && rm code.tar.gz
cat > cloud_cmd.sh <<'{CMD_DELIMITER}'
{command}
{CMD_DELIMITER}
export WANDB_MODE=disabled OUT_DIR="$OUT" PYTHONUNBUFFERED=1
( bash cloud/setup_cloud.sh && source .venv/bin/activate && bash cloud_cmd.sh )
STATUS=$?
"""


def is_dirty(repo_dir: Path) -> bool:
    """True if tracked files have uncommitted changes (the bundle is built from HEAD, so they would be missing)."""
    out = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True
    ).stdout
    return bool(out.strip())


def make_bundle(repo_dir: Path, out_path: Path, extras: Sequence[str] = ()) -> Path:
    """tar.gz of `git archive HEAD` plus extra (e.g. untracked data) paths given relative to repo_dir."""
    for extra in extras:
        if not (repo_dir / extra).exists():
            raise FileNotFoundError(f"--include path not found: {extra}")
    proc = subprocess.Popen(["git", "-C", str(repo_dir), "archive", "--format=tar", "HEAD"], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    with tarfile.open(fileobj=proc.stdout, mode="r|") as src, tarfile.open(out_path, "w:gz") as dst:
        for member in src:
            dst.addfile(member, src.extractfile(member) if member.isfile() else None)
        for extra in extras:
            dst.add(repo_dir / extra, arcname=extra)
    if proc.wait() != 0:
        raise RuntimeError("git archive failed")
    return out_path


def run_aws(args: Sequence[str], region: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["aws", *args, "--region", region], capture_output=True, text=True, check=check)


def stack_outputs(stack: str, region: str) -> dict[str, str]:
    out = run_aws(["cloudformation", "describe-stacks", "--stack-name", stack, "--query", "Stacks[0].Outputs", "--output", "json"], region)
    return {o["OutputKey"]: o["OutputValue"] for o in json.loads(out.stdout)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True, help="name; S3 keys code/<job>.tar.gz and runs/<job>/")
    ap.add_argument("--command", required=True, help="shell command run on the instance inside the venv, from the repo root")
    ap.add_argument("--max-hours", type=float, required=True, help=f"hard deadline, 0 < h <= {MAX_HOURS_CAP}")
    ap.add_argument("--include", action="append", default=[], help="extra untracked path (relative to repo) to bundle; repeatable")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--stack", default="billiards-rl-cloud")
    ap.add_argument("--go", action="store_true", help="really upload and launch (default is a dry run)")
    ap.add_argument("--allow-dirty", action="store_true", help="bundle HEAD even though tracked files have uncommitted changes")
    args = ap.parse_args(argv)

    validate_job(args.job)
    validate_max_hours(args.max_hours)
    if is_dirty(REPO_DIR) and not args.allow_dirty:
        print("tracked files have uncommitted changes; commit them (the bundle is built from HEAD) or pass --allow-dirty", file=sys.stderr)
        return 2

    outputs = stack_outputs(args.stack, args.region)
    bucket, template_id = outputs["BucketName"], outputs["LaunchTemplateId"]
    user_data = build_user_data(bucket, args.job, args.command, args.max_hours, args.region)

    with tempfile.TemporaryDirectory() as tmp:
        user_data_path = Path(tmp) / "user_data.sh"
        user_data_path.write_text(user_data)
        run_cmd = [
            "ec2", "run-instances", "--launch-template", f"LaunchTemplateId={template_id},Version=$Default", "--count", "1",
            "--user-data", f"file://{user_data_path}", "--query", "Instances[0].InstanceId", "--output", "text",
        ]  # fmt: skip
        print(f"job={args.job} region={args.region} bucket={bucket} max_hours={args.max_hours} include={args.include}")
        if not args.go:
            res = run_aws([*run_cmd, "--dry-run"], args.region, check=False)
            ok = "DryRunOperation" in res.stderr
            print("dry run:", "launch would be permitted (nothing uploaded or started)" if ok else res.stderr.strip())
            return 0 if ok else 1

        bundle = make_bundle(REPO_DIR, Path(tmp) / f"{args.job}.tar.gz", args.include)
        print(f"bundle {bundle.stat().st_size / 1e6:.1f} MB -> s3://{bucket}/code/{args.job}.tar.gz")
        run_aws(["s3", "cp", str(bundle), f"s3://{bucket}/code/{args.job}.tar.gz", "--only-show-errors"], args.region)
        res = run_aws(run_cmd, args.region)
        print(f"launched {res.stdout.strip()}  (ends by itself within {args.max_hours} h; results in s3://{bucket}/runs/{args.job}/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
