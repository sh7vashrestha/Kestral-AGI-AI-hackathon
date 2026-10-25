#!/usr/bin/env bash
# Run real ArduPilot SITL (ArduCopter) for drone-brain. NOT run automatically.
#
# Build once (verified on the Orin Nano: ~16 min with -j2, no sudo needed; Python prereqs via
# `python3 -m pip install --user empy==3.3.4 pexpect future`):
#   git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/ArduPilot/ardupilot ~/ardupilot
#   cd ~/ardupilot && ./waf configure --board sitl && ./waf copter -j2     # -> build/sitl/bin/arducopter
#
# Usage:
#   scripts/sitl_run.sh                      # SITL -> udp 127.0.0.1:14550 (drone-brain listens there)
#   SPEEDUP=5 scripts/sitl_run.sh            # faster simulation
#   OUT_HOST=192.168.4.1 scripts/sitl_run.sh # SITL on a LAPTOP, MAVLink sent to the Jetson
#
# drone-brain side:
#   .venv/bin/python main.py --mavlink-url udpin:127.0.0.1:14550 --camera file:tests/assets/persons
#   (VEHICLE_MODE=sim; loopback counts as a simulator so the arm guard allows arming)
#
# The arducopter binary is run directly (no MAVProxy): SERIAL0 = udpclient to drone-brain.
# Its working directory (eeprom.bin, logs/) is $SITL_DIR (default ~/.cache/drone-brain-sitl).
#
# Environment overrides:
#   ARDUPILOT_DIR (default ~/ardupilot)   OUT_HOST (127.0.0.1)   OUT_PORT (14550)
#   HOME_LOC      (default KSU field "33.9371904,-84.5271293,300,0" = lat,lon,alt_amsl,heading)
#   SPEEDUP (1)   INSTANCE (0)   WIPE=1 to start from fresh parameters
set -euo pipefail

ARDUPILOT_DIR="${ARDUPILOT_DIR:-$HOME/ardupilot}"
OUT_HOST="${OUT_HOST:-127.0.0.1}"
OUT_PORT="${OUT_PORT:-14550}"
HOME_LOC="${HOME_LOC:-33.9371904,-84.5271293,300,0}"
SPEEDUP="${SPEEDUP:-1}"
INSTANCE="${INSTANCE:-0}"
SITL_DIR="${SITL_DIR:-$HOME/.cache/drone-brain-sitl}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BIN="$ARDUPILOT_DIR/build/sitl/bin/arducopter"
[[ -x "$BIN" ]] || { echo "missing $BIN (see the build steps at the top of this script)" >&2; exit 1; }
mkdir -p "$SITL_DIR"
cd "$SITL_DIR"
WIPE_ARG=()
[[ "${WIPE:-0}" == "1" ]] && WIPE_ARG=(--wipe)
# Never touches serial hardware: SERIAL0 is UDP only.
exec "$BIN" --model quad --speedup "$SPEEDUP" --home "$HOME_LOC" -I"$INSTANCE" "${WIPE_ARG[@]}" \
  --defaults "$ARDUPILOT_DIR/Tools/autotest/default_params/copter.parm,$HERE/sitl_dronebrain.parm" \
  --serial0 "udpclient:${OUT_HOST}:${OUT_PORT}"
