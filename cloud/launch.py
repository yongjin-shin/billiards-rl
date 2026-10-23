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
import io
import json
import math
import re
import subprocess
import sys
import tarfile
import tempfile
import time
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


SYNC_SECONDS = 300          # progress upload interval of $OUT_DIR and the log
SPOT_POLL_SECONDS = 15     # Spot interruption notice check (AWS gives about 2 minutes of warning)


def build_user_data(bucket: str, job: str, command: str, max_hours: float, region: str, sync_seconds: int = SYNC_SECONDS) -> str:
    """
    Bash script run by cloud-init as root on the instance. It always uploads the log and shuts the instance down. While the command
    runs it also uploads $OUT_DIR and the log every sync_seconds, and right away on a Spot interruption notice (STATUS
    'spot-interrupted'): a reclaimed Spot instance never reaches finish(), and the first P3 run lost everything that way.
    Relaunching the same job name resumes it: the earlier runs/<job>/out is restored into $OUT_DIR before the command (training
    with --ckpt-every continues from its checkpoint), and STATUS reads 'running' until this instance finishes.
    """
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
    if not SPOT_POLL_SECONDS <= sync_seconds <= 3600:
        raise ValueError(f"sync_seconds must be in [{SPOT_POLL_SECONDS}, 3600], got {sync_seconds}")
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

sync_out() {{
  mkdir -p "$OUT"
  aws s3 cp /var/log/job.log "s3://$BUCKET/runs/$JOB/job.log" --only-show-errors || true
  aws s3 sync "$OUT" "s3://$BUCKET/runs/$JOB/out" --only-show-errors || true
}}

# Progress uploads every {sync_seconds} s, and at once on a Spot interruption notice (IMDSv2 spot/instance-action).
watch_progress() {{
  local n=0 token
  while true; do
    sleep {SPOT_POLL_SECONDS}; n=$((n + {SPOT_POLL_SECONDS}))
    token=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
    if curl -s -f -m 2 -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/spot/instance-action >/dev/null; then
      echo "[job] spot interruption notice at $(date -u +%FT%TZ)"
      sync_out
      echo spot-interrupted > /tmp/status_spot
      aws s3 cp /tmp/status_spot "s3://$BUCKET/runs/$JOB/STATUS" || true
      return
    fi
    if [ "$n" -ge {sync_seconds} ]; then sync_out; n=0; fi
  done
}}

finish() {{
  echo "[job] exit status $STATUS at $(date -u +%FT%TZ)"
  [ -n "${{WATCH_PID:-}}" ] && kill "$WATCH_PID" 2>/dev/null
  sync_out
  echo "$STATUS" > /tmp/status
  aws s3 cp /tmp/status "s3://$BUCKET/runs/$JOB/STATUS" || true
  shutdown -h now
}}
trap finish EXIT

mkdir -p "$WORK" "$OUT" && cd "$WORK"
watch_progress &
WATCH_PID=$!
echo running > /tmp/status_run && aws s3 cp /tmp/status_run "s3://$BUCKET/runs/$JOB/STATUS" --only-show-errors || true
aws s3 sync "s3://$BUCKET/runs/$JOB/out" "$OUT" --only-show-errors || true    # resume: outputs of an earlier, interrupted instance
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
        if not extra.strip() or (repo_dir / extra).resolve() == repo_dir.resolve():   # "" (a trailing newline) = the whole repo + logs
            raise ValueError(f"--include path must name a sub-path, got {extra!r}")
        if not (repo_dir / extra).exists():
            raise FileNotFoundError(f"--include path not found: {extra}")
    proc = subprocess.Popen(["git", "-C", str(repo_dir), "archive", "--format=tar", "HEAD"], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    with tarfile.open(fileobj=proc.stdout, mode="r|") as src, tarfile.open(out_path, "w:gz") as dst:
        for member in src:
            dst.addfile(member, src.extractfile(member) if member.isfile() else None)
        for extra in extras:
            dst.add(repo_dir / extra, arcname=extra)
        commit = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        info = tarfile.TarInfo("GIT_COMMIT")                      # the bundle has no .git: results record this (exp16_wm.code_version)
        data = (commit + "\n").encode()
        info.size = len(data)
        dst.addfile(info, io.BytesIO(data))
    if proc.wait() != 0:
        raise RuntimeError("git archive failed")
    return out_path


def run_aws(args: Sequence[str], region: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run the aws CLI; on failure (check=True) raise with AWS's own error text instead of a bare exit status."""
    res = subprocess.run(["aws", *args, "--region", region], capture_output=True, text=True, check=False)
    if check and res.returncode != 0:
        raise RuntimeError(f"aws {' '.join(args[:2])} failed (exit {res.returncode}): {res.stderr.strip()}")
    return res


CAPACITY_ERRORS = ("InsufficientInstanceCapacity", "capacity-not-available", "SpotMaxPriceTooLow", "Unsupported")


def rank_placements(history: list[dict], types: Sequence[str]) -> list[tuple[str, str, float]]:
    """Spot price history records (InstanceType, AvailabilityZone, SpotPrice, Timestamp) → (type, AZ, latest price) for the allowed
    types, cheapest first. The cheapest pool is usually the one with the most spare capacity, so the least likely to be reclaimed."""
    latest: dict[tuple[str, str], tuple[str, float]] = {}
    for r in history:
        key = (r["InstanceType"], r["AvailabilityZone"])
        if r["InstanceType"] in types and (key not in latest or r["Timestamp"] > latest[key][0]):
            latest[key] = (r["Timestamp"], float(r["SpotPrice"]))
    return sorted(((t, az, p) for (t, az), (_, p) in latest.items()), key=lambda x: (x[2], x[0], x[1]))


def avoid_placements(places: list[tuple[str, str, float]], avoid: Sequence[str]) -> list[tuple[str, str, float]]:
    """Move pools given as 'type@az' (e.g. ones that were just reclaimed) to the end, keeping the price order otherwise."""
    bad = set(avoid)
    return [p for p in places if f"{p[0]}@{p[1]}" not in bad] + [p for p in places if f"{p[0]}@{p[1]}" in bad]


def placement_overrides(instance_type: str, subnet_id: str | None) -> list[str]:
    """run-instances arguments that override the launch template's instance type and pick the AZ through its default subnet."""
    return ["--instance-type", instance_type] + (["--subnet-id", subnet_id] if subnet_id else [])


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
    ap.add_argument("--avoid", action="append", default=[], help="type@az pool to try last (e.g. just reclaimed); repeatable")
    ap.add_argument("--instance-types", nargs="+", default=["c7a.16xlarge"],
                    help="allowed types (same CPU family, e.g. c7a/m7a/r7a.16xlarge); the cheapest (type, AZ) Spot pool is tried first, "
                         "then the next on a capacity error. Every type must be allowed by the stack's AgentWritePolicy")
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
        hist = json.loads(run_aws(["ec2", "describe-spot-price-history", "--instance-types", *args.instance_types, "--product-descriptions",
                                   "Linux/UNIX", "--start-time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 3 * 3600)),
                                   "--query", "SpotPriceHistory", "--output", "json"], args.region).stdout)
        subnets = {s_["AvailabilityZone"]: s_["SubnetId"] for s_ in json.loads(run_aws(
            ["ec2", "describe-subnets", "--filters", "Name=default-for-az,Values=true", "--query", "Subnets", "--output", "json"], args.region).stdout)}
        places = [(t, az, p) for t, az, p in rank_placements(hist, args.instance_types) if az in subnets] or \
                 [(args.instance_types[0], None, float("nan"))]
        places = avoid_placements(places, args.avoid)
        print("placements (cheapest first):", ", ".join(f"{t}@{az} ${p:.3f}" for t, az, p in places[:5]))
        if not args.go:
            t, az, _ = places[0]
            res = run_aws([*run_cmd, *placement_overrides(t, subnets.get(az)), "--dry-run"], args.region, check=False)
            ok = "DryRunOperation" in res.stderr
            print("dry run:", "launch would be permitted (nothing uploaded or started)" if ok else res.stderr.strip())
            return 0 if ok else 1

        bundle = make_bundle(REPO_DIR, Path(tmp) / f"{args.job}.tar.gz", args.include)
        print(f"bundle {bundle.stat().st_size / 1e6:.1f} MB -> s3://{bucket}/code/{args.job}.tar.gz")
        run_aws(["s3", "cp", str(bundle), f"s3://{bucket}/code/{args.job}.tar.gz", "--only-show-errors"], args.region)
        for t, az, p in places:
            res = run_aws([*run_cmd, *placement_overrides(t, subnets.get(az))], args.region, check=False)
            if res.returncode == 0:
                print(f"launched {res.stdout.strip()} {t}@{az} (${p:.3f}/h)  (ends by itself within {args.max_hours} h; results in s3://{bucket}/runs/{args.job}/)")
                return 0
            if not any(e in res.stderr for e in CAPACITY_ERRORS):
                raise RuntimeError(f"aws ec2 run-instances failed: {res.stderr.strip()}")
            print(f"  no capacity for {t}@{az}, trying the next placement", flush=True)
        raise RuntimeError("no Spot capacity in any allowed placement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
