#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Project root: ${PROJECT_ROOT}"
echo "Using Python: ${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} not found." >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Virtual environment already exists at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
python -m pip install -e "${PROJECT_ROOT}[dev]"

echo
echo "Virtual environment is ready."
echo "Activate it with:"
echo "  source .venv/bin/activate"
echo
echo "Run scripts with:"
echo "  .venv/bin/python scripts/prepare_matches.py --summary"
echo "  bdp metadata validate --summary"
