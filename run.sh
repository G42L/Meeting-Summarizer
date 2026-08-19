#!/usr/bin/env bash
set -euo pipefail

VENV_PYTHON="venv/bin/python3"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "error: $VENV_PYTHON not found or not executable." >&2
    echo "Run this from the project root, and make sure the venv exists (python3 -m venv venv)." >&2
    exit 1
fi

"$VENV_PYTHON" run.py
