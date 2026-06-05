#!/usr/bin/env bash
# Airheads — event-day launcher
# Double-click this file (or run it from terminal) to start the mirror.
# No terminal window needed on event day.

set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate
exec python mirror.py
