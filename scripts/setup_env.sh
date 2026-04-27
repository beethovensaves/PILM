#!/usr/bin/env bash
# Bootstrap a Python environment for PILM.
#
# Usage:
#   ./scripts/setup_env.sh                  # Phase 0/1: core only
#   ./scripts/setup_env.sh acoustic ml      # add extras
#   ./scripts/setup_env.sh all dev          # everything + dev tools
#
# Creates ./.venv at the repo root and installs the package in editable mode.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN not found. Install Python 3.11+ or set PYTHON_BIN." >&2
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

EXTRAS_ARG=""
if [ "$#" -gt 0 ]; then
    EXTRAS=$(IFS=,; echo "$*")
    EXTRAS_ARG="[$EXTRAS]"
fi

echo "Installing pilm${EXTRAS_ARG} in editable mode ..."
pip install -e ".${EXTRAS_ARG}"

echo
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
echo "Phase-2+ extras (run later): ./scripts/setup_env.sh acoustic ml speaker"
echo "Phase-4+ extras (run later): ./scripts/setup_env.sh acoustic ml speaker training viz"
echo
echo "Note: Montreal Forced Aligner is not pip-installable. When you hit Phase 3,"
echo "      install it via conda:  conda install -c conda-forge montreal-forced-aligner"
