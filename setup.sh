#!/usr/bin/env bash

set -euo pipefail

cmake --preset release
cmake --build --preset release -j2

if [ ! -f ./.venv/bin/activate ]; then
    echo "Creating virtual environment (for build_assumptions.py)..."
    python3 -m venv ./.venv
fi

. ./.venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
