#!/usr/bin/env bash
# Switch the Jetson's WiFi:   ./hotspot.sh on   -> KESTREL hotspot (phones join it; app at http://10.42.0.1)
#                             ./hotspot.sh off  -> back to the previous WiFi
#                             ./hotspot.sh pass -> show the hotspot password (asks for sudo)
# Turning it on drops the Jetson's normal WiFi (and VS Code if it uses WiFi; the cable keeps working).
cd "$(dirname "$0")"
mkdir -p data
case "${1:-}" in
  on)   nmcli -t -f NAME,TYPE con show --active | grep 802-11-wireless | cut -d: -f1 | grep -vx jetson-hotspot > data/prev_wifi || true
        exec nmcli con up jetson-hotspot ;;
  off)  prev=$(cat data/prev_wifi 2>/dev/null)
        if [[ -n $prev ]]; then exec nmcli con up "$prev"; else exec nmcli con down jetson-hotspot; fi ;;
  pass) exec sudo nmcli -s -g 802-11-wireless-security.psk con show jetson-hotspot ;;
  *) sed -n 2,5p "$0"; exit 2 ;;
esac
