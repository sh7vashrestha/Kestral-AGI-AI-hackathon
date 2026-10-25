#!/usr/bin/env bash
# Stop drone-brain started by start.sh (graceful: drone-brain commands RTL first if a mission is flying),
# then Gemma and SITL, and switch the WiFi back from the hotspot.   Usage: ./stop.sh [--keep-hotspot]
cd "$(dirname "$0")"
MAIN='^(\S*/)?\.venv/bin/python main\.py'          # anchored: never matches a shell that merely mentions it
pids=$(pgrep -f "$MAIN")
if [[ -n $pids ]]; then
  kill -INT $pids; echo "drone-brain: stopping"
  for _ in $(seq 15); do pgrep -f "$MAIN" >/dev/null || break; sleep 1; done
fi
pkill -f '^(\S*/)?bash (\S*/)?start\.sh( |$)' 2>/dev/null
pkill -x arducopter 2>/dev/null && echo "SITL stopped"
pkill -x llama-server 2>/dev/null && echo "Gemma (llama-server) stopped"
if [[ "${1:-}" != --keep-hotspot ]] && nmcli -t -f NAME con show --active | grep -qx jetson-hotspot; then
  prev=$(cat data/prev_wifi 2>/dev/null)
  if [[ -n $prev ]]; then nmcli con up "$prev" && echo "WiFi back on '$prev'"; else nmcli con down jetson-hotspot; fi
fi
echo "stopped"
