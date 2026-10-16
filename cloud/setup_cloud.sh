#!/usr/bin/env bash
# cloud/setup_cloud.sh — build the CPU training environment on the cloud instance (Amazon Linux 2023, x86_64).
# Runs from the extracted code bundle:  bash cloud/setup_cloud.sh
# Result: ./.venv with Python 3.13 and the versions in requirements-cloud.txt, then a short smoke test.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_VERSION="3.13"
SMOKE_TESTS="${SMOKE_TESTS:-tests/test_exp16_wm_sim_identity.py tests/test_exp16_wm_geom.py}"

echo "[1/4] uv"
if ! command -v uv &>/dev/null; then
  command -v pip3 &>/dev/null || dnf install -y -q python3-pip
  pip3 install --quiet uv
fi
uv --version

echo "[2/4] .venv (Python ${PYTHON_VERSION})"
uv venv --python "${PYTHON_VERSION}" .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python --version

echo "[3/4] packages (torch CPU wheel first, then the pinned rest)"
uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.10.0"
uv pip install -r requirements-cloud.txt
python - <<'PY'
import numba, numpy, pooltool, torch
print("torch", torch.__version__, "| threads", torch.get_num_threads(), "| numpy", numpy.__version__, "| numba", numba.__version__)
PY

echo "[4/4] smoke tests: ${SMOKE_TESTS}"
# shellcheck disable=SC2086
python -m pytest ${SMOKE_TESTS} -q -x

echo "setup_cloud: done"
