#!/usr/bin/env bash
# Camera-only live view (colour + thermal) WITHOUT the drone app, e.g. to aim/focus the cameras:
#   ./live.sh   -> http://<jetson>:8090
# (When the app is running, use its own page instead: http://<jetson>:8000/live)
cd "$(dirname "$0")"
if pgrep -f '^(\S*/)?\.venv/bin/python main\.py' >/dev/null; then
  echo "drone-brain is running and owns the cameras: open http://<jetson>:8000/live instead"; exit 1
fi
exec .venv/bin/python scripts/live_view.py --port 8090 "$@"
