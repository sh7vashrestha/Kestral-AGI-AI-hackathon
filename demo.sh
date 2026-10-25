#!/usr/bin/env bash
# Ground demo: the full phone app (Gemma chat plan -> Confirm -> GO -> alerts -> summary) with NOTHING flying.
#
#   ./demo.sh               pretend drone (loopback mock, hovers at home), real RGB + thermal cameras
#   ./demo.sh --hotspot     also switch WiFi to the jetson-hotspot network (drops the current WiFi/VS Code link!)
#
# GO starts the live camera search: YOLO finds people, CLIP scores each one against the target, the best match is
# boxed red on the Live page (with its photo and temperature). Abort ends the search and Gemma writes the summary.
# The Pixhawk is never opened. The flying setup (../drone-brain, start.sh, the drone-brain service) is not touched;
# stop that service first, because it holds the cameras.
# Ctrl+C stops everything this script started.
set -uo pipefail
cd "$(dirname "$0")"
HOTSPOT=0
for a in "$@"; do
  case "$a" in
    --hotspot) HOTSPOT=1 ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) echo "unknown argument: $a (see ./demo.sh --help)"; exit 2 ;;
  esac
done
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # CLIP is cached in models/: no internet lookups
mkdir -p data
STARTED_LLAMA=0

cleanup() { if [[ $STARTED_LLAMA == 1 ]]; then pkill -x llama-server && echo "llama-server stopped"; fi; }
trap cleanup EXIT
trap 'exit 130' INT TERM

[[ -x $PY ]] || { echo "no .venv in $(pwd)"; exit 1; }
if systemctl is-active --quiet drone-brain 2>/dev/null; then
  echo "The drone-brain service is running and holds the cameras. Stop it first:"
  echo "    sudo systemctl stop drone-brain        (start it again later: sudo systemctl start drone-brain)"
  exit 1
fi

if [[ $HOTSPOT == 1 ]]; then
  echo "== WiFi hotspot"
  nmcli -t -f NAME,TYPE con show --active | grep 802-11-wireless | cut -d: -f1 | grep -vx jetson-hotspot > data/prev_wifi || true
  nmcli con up jetson-hotspot || echo "WARNING: hotspot did not come up"
  echo "   phone: join WiFi '$(nmcli -g 802-11-wireless.ssid con show jetson-hotspot)'"
fi

echo "== Gemma planner (llama-server)"
if curl -sf http://127.0.0.1:8080/health >/dev/null; then
  echo "already running"
elif scripts/start_llama.sh text; then
  STARTED_LLAMA=1
else
  echo "WARNING: Gemma did not start (data/llama-server.log). Map planning still works; chat planning and the"
  echo "         written summary will not."
fi

PORT=$(grep -E '^WEB_PORT=' .env 2>/dev/null | cut -d= -f2); PORT=${PORT:-8000}
URL="http://%s:$PORT"
if grep -qE '^HTTPS=true' .env 2>/dev/null; then
  HPORT=$(grep -E '^HTTPS_PORT=' .env | cut -d= -f2); URL="https://%s:${HPORT:-8443}"
  [[ -f data/tls/server.crt ]] || scripts/make_cert.sh
fi
echo "== drone-brain DEMO (nothing flies). Open on the phone:"
for ip in $(hostname -I); do [[ $ip == 172.17.* || $ip == *:* ]] || printf "     $URL\n" "$ip"; done
echo "   Plan in the chat (\"find the person in a red jacket\") -> Confirm -> GO -> Live page."
echo "   If RGB says 'No cameras available': sudo systemctl restart nvargus-daemon"
echo "   Ctrl+C to stop."
$PY main.py --demo
