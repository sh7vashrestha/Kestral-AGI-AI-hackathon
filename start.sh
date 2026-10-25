#!/usr/bin/env bash
# One command to bring drone-brain up.
#
#   ./start.sh            field   : real Pixhawk, full mode (Upload -> checks -> GO). Arming still needs
#                                   ALLOW_REAL_ARM=true + DRY_RUN=false in .env, set by a human.
#   ./start.sh bench              : real Pixhawk, upload only (no GO), bench home = KSU field (no GPS needed)
#   ./start.sh readonly           : real Pixhawk, read-only (telemetry + checks, nothing sent)
#   ./start.sh sim                : ArduPilot SITL + test images instead of the camera
#
# Options: --hotspot   switch WiFi to the jetson-hotspot network (drops the current WiFi/VS Code link!)
#          --no-check  skip the quick system check
#          --no-vision no cameras/YOLO/CLIP
# Ctrl+C (or ./stop.sh from another terminal / SSH) stops everything this script started
# (drone-brain sends RTL first if a mission is flying).
set -uo pipefail
cd "$(dirname "$0")"
MODE=field; HOTSPOT=0; CHECK=1; EXTRA=()
for a in "$@"; do
  case "$a" in
    field|bench|readonly|sim) MODE=$a ;;
    --hotspot) HOTSPOT=1 ;;
    --no-check) CHECK=0 ;;
    --no-vision) EXTRA+=(--no-vision) ;;
    -h|--help) sed -n 2,14p "$0"; exit 0 ;;
    *) echo "unknown argument: $a (see ./start.sh --help)"; exit 2 ;;
  esac
done
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # CLIP is cached in models/: no internet lookups (each offline start wasted ~30 s on retries)
PIXHAWK=(--mavlink-url serial:/dev/ttyACM0 --baud 115200)
mkdir -p data
STARTED_LLAMA=0; SITL_PID=""

cleanup() {
  if [[ -n "$SITL_PID" ]]; then pkill -x arducopter; kill "$SITL_PID" 2>/dev/null; echo "SITL stopped"; fi
  if [[ $STARTED_LLAMA == 1 ]]; then pkill -x llama-server && echo "llama-server stopped"; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

[[ -x $PY ]] || { echo "no .venv in $(pwd)"; exit 1; }

# 1. WiFi hotspot (optional)
if [[ $HOTSPOT == 1 ]]; then
  echo "== WiFi hotspot (stop.sh switches back to: $(nmcli -t -f NAME,TYPE con show --active | grep 802-11-wireless | cut -d: -f1))"
  nmcli -t -f NAME,TYPE con show --active | grep 802-11-wireless | cut -d: -f1 | grep -vx jetson-hotspot > data/prev_wifi || true
  nmcli con up jetson-hotspot || echo "WARNING: hotspot did not come up"
  echo "   phone: join WiFi '$(nmcli -g 802-11-wireless.ssid con show jetson-hotspot)'"
fi

# 2. Pixhawk on USB (not needed for sim)
if [[ $MODE != sim ]]; then
  echo "== Pixhawk"
  for _ in $(seq 10); do [[ -e /dev/ttyACM0 ]] && break; sleep 1; done
  [[ -e /dev/ttyACM0 ]] && echo "found /dev/ttyACM0" || \
    echo "WARNING: Pixhawk not found on /dev/ttyACM0 - starting anyway; it connects when you plug in the USB cable"
fi

# 3. Gemma (llama-server) - must start before the other GPU jobs
echo "== Gemma planner (llama-server)"
if curl -sf http://127.0.0.1:8080/health >/dev/null; then
  echo "already running"
elif scripts/start_llama.sh text; then
  STARTED_LLAMA=1
else
  echo "WARNING: Gemma did not start (data/llama-server.log). Map planning still works; chat planning and the"
  echo "         written flight summary will not (a template summary is used)."
fi

# 4. Simulator (sim mode only)
if [[ $MODE == sim ]]; then
  echo "== ArduPilot SITL (log: data/sitl.log)"
  SPEEDUP=${SPEEDUP:-5} scripts/sitl_run.sh > data/sitl.log 2>&1 &
  SITL_PID=$!
  sleep 5
fi

# 5. Quick system check (informational: indoors GPS/home/EKF are red, that is expected)
if [[ $CHECK == 1 && $MODE != sim ]]; then
  echo "== Quick system check"
  $PY main.py --check --quick --standalone "${PIXHAWK[@]}" | tail -25
fi

# 6. drone-brain
case $MODE in
  field)    ARGS=(--vehicle full "${PIXHAWK[@]}") ;;
  bench)    ARGS=(--vehicle upload "${PIXHAWK[@]}" --bench-home field) ;;
  readonly) ARGS=(--vehicle readonly "${PIXHAWK[@]}") ;;
  sim)      ARGS=(--vehicle sim --mavlink-url udpin:127.0.0.1:14550 --camera file:tests/assets/persons)
            export CAMERA_ROTATE=0 CAMERA_FLIP=none ;;   # test photos are upright; CAMERA_ROTATE=180 is the real camera's mount
esac
PORT=$(grep -E '^WEB_PORT=' .env 2>/dev/null | cut -d= -f2); PORT=${PORT:-8000}
URL="http://%s:$PORT"
if grep -qE '^HTTPS=true' .env; then
  HPORT=$(grep -E '^HTTPS_PORT=' .env | cut -d= -f2); URL="https://%s:${HPORT:-8443}"
  [[ -f data/tls/server.crt ]] || scripts/make_cert.sh
fi
echo "== drone-brain ($MODE). Open on the phone:"
for ip in $(hostname -I); do [[ $ip == 172.17.* || $ip == *:* ]] || printf "     $URL\n" "$ip"; done
grep -qE '^ALLOW_REAL_ARM=true' .env || [[ $MODE != field ]] || echo "   (ALLOW_REAL_ARM is not true in .env: GO will stay blocked - upload and checks still work)"
echo "   Ctrl+C to stop."
$PY main.py "${ARGS[@]}" "${EXTRA[@]}"
