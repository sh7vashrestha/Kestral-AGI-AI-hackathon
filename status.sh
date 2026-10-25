#!/usr/bin/env bash
# What is running right now, and where to open the app.
cd "$(dirname "$0")"
on() { printf "  %-22s %s\n" "$1" "$2"; }
echo "KESTREL status ($(date +%T))"
pgrep -f '^(\S*/)?\.venv/bin/python main\.py' >/dev/null && on "app (drone-brain)" "running" || on "app (drone-brain)" "stopped"
curl -sf -m 2 http://127.0.0.1:8080/health >/dev/null && on "Gemma" "running" || on "Gemma" "stopped"
pgrep -x arducopter >/dev/null && on "simulator (SITL)" "running"
[[ -e /dev/ttyACM0 ]] && on "Pixhawk USB" "connected" || on "Pixhawk USB" "not connected"
wifi=$(nmcli -t -f NAME,TYPE con show --active 2>/dev/null | grep 802-11-wireless | cut -d: -f1)
on "WiFi" "${wifi:-none}$([[ $wifi == jetson-hotspot ]] && echo " (hotspot KESTREL)")"
systemctl is-enabled drone-brain >/dev/null 2>&1 && on "start at boot" "enabled" || on "start at boot" "off (sudo scripts/install_autostart.sh)"
if curl -sf -m 2 -o /dev/null http://127.0.0.1:8000/; then
  echo "  open on the phone:"
  U="http://%s:8000"; grep -qE '^HTTPS=true' .env && U="https://%s:8443"
  for ip in $(hostname -I); do [[ $ip == 172.17.* || $ip == *:* ]] || printf "    $U   (camera: $U/live)\n" "$ip" "$ip"; done
  grep -qE '^HTTPS=true' .env && echo "    (first time on a phone: install http://<address>:8000/kestrel-ca.crt, see README: HTTPS)"
fi
