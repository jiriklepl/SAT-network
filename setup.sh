#!/usr/bin/env bash

set -euo pipefail

if [ ! -f ./.venv/bin/activate ]; then
    echo "Creating virtual environment..."
    python3 -m venv ./.venv
fi

. ./.venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
