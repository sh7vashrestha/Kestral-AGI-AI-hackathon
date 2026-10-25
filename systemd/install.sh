#!/usr/bin/env bash
# Install drone-brain systemd units. NOT run automatically. Requires root:
#   sudo bash systemd/install.sh            # copy units + daemon-reload (no enable)
#   sudo bash systemd/install.sh --enable   # also enable at boot (does not start)
#
# NOTE: drone-hotspot.service drops the client Wi-Fi when it starts. Before enabling,
# create hotspot/hotspot.env with a real PASSWORD. Enabling does not start anything;
# the units start at next boot (or `systemctl start drone-brain`).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $EUID -eq 0 ]] || { echo "run as root (sudo)" >&2; exit 1; }
UNITS=(drone-hotspot.service drone-llama.service drone-brain.service)
for u in "${UNITS[@]}"; do
    install -m 0644 "$HERE/$u" "/etc/systemd/system/$u"
    echo "installed $u"
done
systemctl daemon-reload
if [[ "${1:-}" == "--enable" ]]; then
    for u in "${UNITS[@]}"; do systemctl enable "$u"; done
    echo "enabled (will start at next boot)"
else
    echo "units installed; re-run with --enable to enable at boot"
fi
