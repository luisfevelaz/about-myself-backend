#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart about-myself
sudo systemctl status about-myself --no-pager
