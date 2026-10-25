#!/usr/bin/env bash
# System check (read-only: never changes anything on the Pixhawk).
#   ./check.sh            everything (host, Gemma, cameras, YOLO, CLIP, Pixhawk, web, files)
#   ./check.sh quick      skip cameras / YOLO / CLIP (fast)
#   ./check.sh vehicle    Pixhawk only: link, GPS, EKF, pre-arm, battery, fence + failsafe params
#   ./check.sh camera     colour + thermal cameras only
# If drone-brain is running it reports what the running app sees; otherwise it opens the devices itself.
cd "$(dirname "$0")"
PIX=(--mavlink-url serial:/dev/ttyACM0 --baud 115200)
case "${1:-all}" in
  all)     exec .venv/bin/python main.py --check "${PIX[@]}" ;;
  quick)   exec .venv/bin/python main.py --check --quick "${PIX[@]}" ;;
  vehicle) exec .venv/bin/python main.py --check --only vehicle "${PIX[@]}" ;;
  camera)  exec .venv/bin/python main.py --check --only camera,thermal ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac
