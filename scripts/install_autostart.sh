#!/usr/bin/env bash
# Start drone-brain automatically at boot: hotspot (KESTREL) + Gemma + drone-brain in field mode.
# Phones that join the hotspot get the app (captive portal: run scripts/setup_hotspot_portal.sh once first).
#   sudo scripts/install_autostart.sh            install + enable (starts at every boot)
#   sudo scripts/install_autostart.sh --remove   disable + remove (back to normal WiFi at boot)
# Day to day: sudo systemctl stop drone-brain / start drone-brain;  logs: journalctl -u drone-brain -f
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
UNIT=/etc/systemd/system/drone-brain.service
DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "${1:-}" == --remove ]]; then
  systemctl disable --now drone-brain 2>/dev/null || true
  rm -f "$UNIT"; systemctl daemon-reload
  echo "removed: drone-brain no longer starts at boot"
  exit 0
fi

cat > "$UNIT" <<EOF
[Unit]
Description=drone-brain (hotspot + Gemma + search app)
Wants=network-online.target
After=network-online.target NetworkManager.service nvargus-daemon.service

[Service]
User=drone
WorkingDirectory=$DIR
# The CSI camera service sometimes comes up stuck ("No cameras available"): restart it first ("+" = as root)
ExecStartPre=+/bin/systemctl restart nvargus-daemon
ExecStart="$DIR/start.sh" field --hotspot --no-check
# SIGINT = the same clean stop as Ctrl+C (RTL first if a mission is flying)
KillSignal=SIGINT
TimeoutStopSec=30
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable drone-brain
echo "installed: drone-brain starts at every boot (now: sudo systemctl start drone-brain)"
