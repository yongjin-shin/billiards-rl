#!/usr/bin/env bash
# =============================================================================
# setup.sh — billiards-rl project setup
# Uses Homebrew to install Python 3.13 + virtualenv + pooltool-billiards
# Run once from the project root:  bash setup.sh
# =============================================================================

set -e  # exit on any error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_VERSION="3.13"

echo "================================================"
echo "  billiards-rl setup"
echo "  Project: $PROJECT_DIR"
echo "================================================"

# --- 1. Homebrew check -------------------------------------------------------
if ! command -v brew &>/dev/null; then
  echo "[1/4] Homebrew not found. Installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add Homebrew to PATH for the rest of this script (Apple Silicon vs Intel)
  if [ -f "/opt/homebrew/bin/brew" ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [ -f "/usr/local/bin/brew" ]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
fi
echo "[1/4] Homebrew found: $(brew --version | head -1)"

# --- 2. Install Python 3.13 via Homebrew ------------------------------------
echo "[2/4] Installing Python $PYTHON_VERSION via Homebrew..."
brew install python@$PYTHON_VERSION

PYTHON_BIN="$(brew --prefix)/bin/python$PYTHON_VERSION"
if [ ! -f "$PYTHON_BIN" ]; then
  echo "[ERROR] python$PYTHON_VERSION not found at $PYTHON_BIN"
  exit 1
fi
echo "      Python: $($PYTHON_BIN --version)"

# --- 3. Create virtual environment ------------------------------------------
echo "[3/4] Creating .venv with Python $PYTHON_VERSION..."
if [ -d "$VENV_DIR" ]; then
  echo "      .venv already exists — skipping creation"
else
  $PYTHON_BIN -m venv "$VENV_DIR"
  echo "      .venv created at $VENV_DIR"
fi

# --- 4. Install packages into venv ------------------------------------------
echo "[4/4] Installing packages into .venv..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
pip install -r "$PROJECT_DIR/requirements.txt" --quiet
echo "      Installed: $(pip show pooltool-billiards | grep Version)"
echo "      SB3: $(pip show stable-baselines3 | grep Version)"
echo "      sb3-contrib: $(pip show sb3-contrib | grep Version)"

# --- Done --------------------------------------------------------------------
echo ""
echo "================================================"
echo "  Setup complete!"
echo ""
echo "  Activate your env:   source .venv/bin/activate"
echo "  Test the simulator:  python simulator.py"
echo "  Train an agent:      python train.py --algo TQC"
echo "  Compare results:     bash run_comparison.sh"
echo "================================================"
