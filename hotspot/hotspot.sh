#!/usr/bin/env bash
# drone-brain Wi-Fi hotspot control (NetworkManager shared mode).
#
#   hotspot.sh up       create/activate the hotspot (phone connects to http://10.42.0.1:8000)
#   hotspot.sh down     deactivate the hotspot (the connection profile is kept)
#   hotspot.sh status   show state; read-only, safe any time
#
# WARNING: the Jetson has ONE Wi-Fi radio. Bringing the hotspot UP DROPS the current
# client Wi-Fi connection (and with it any internet / ssh over Wi-Fi). `down` does not
# automatically reconnect; run `nmcli con up <your-network>` or let autoconnect do it.
#
# Config: hotspot/hotspot.env (see hotspot.env.example).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${HOTSPOT_ENV:-$HERE/hotspot.env}"

IFACE="wlP1p1s0"
SSID="drone-brain"
PASSWORD=""
CON_NAME="drone-brain-hotspot"
ADDRESS="10.42.0.1/24"

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

die() { echo "hotspot.sh: $*" >&2; exit 1; }

usage() { echo "usage: $0 {up|down|status}" >&2; exit 2; }

con_exists() { nmcli -t -f NAME con show | grep -Fxq -- "$CON_NAME"; }

cmd_status() {
    echo "interface: $IFACE   ssid: $SSID   connection: $CON_NAME"
    if nmcli -t -f NAME con show --active | grep -Fxq -- "$CON_NAME"; then
        echo "state: UP"
        nmcli -t -f IP4.ADDRESS con show "$CON_NAME" 2>/dev/null || true
    else
        echo "state: DOWN"
        echo "active on $IFACE:"
        nmcli -t -f GENERAL.CONNECTION dev show "$IFACE" 2>/dev/null || true
    fi
}

cmd_up() {
    [[ -f "$ENV_FILE" ]] || die "missing $ENV_FILE (copy hotspot.env.example and set PASSWORD)"
    [[ -n "$PASSWORD" ]] || die "PASSWORD not set in $ENV_FILE"
    [[ "$PASSWORD" != "CHANGE-ME-before-use" ]] || die "change the placeholder PASSWORD in $ENV_FILE"
    (( ${#PASSWORD} >= 8 && ${#PASSWORD} <= 63 )) || die "PASSWORD must be 8-63 characters (WPA2)"
    [[ -n "$SSID" ]] || die "SSID empty"

    echo "WARNING: bringing up the hotspot on $IFACE drops the current client Wi-Fi connection." >&2

    if ! con_exists; then
        nmcli dev wifi hotspot ifname "$IFACE" con-name "$CON_NAME" ssid "$SSID" password "$PASSWORD"
    fi
    # Idempotent (re)configuration of the profile.
    nmcli con modify "$CON_NAME" \
        802-11-wireless.ssid "$SSID" \
        802-11-wireless-security.psk "$PASSWORD" \
        ipv4.method shared \
        ipv4.addresses "$ADDRESS" \
        ipv6.method ignore \
        connection.autoconnect yes \
        connection.autoconnect-priority 100
    nmcli con up "$CON_NAME" ifname "$IFACE"
    cmd_status
}

cmd_down() {
    if con_exists; then
        nmcli con down "$CON_NAME" || true
    else
        echo "connection $CON_NAME does not exist; nothing to do"
    fi
}

command -v nmcli >/dev/null || die "nmcli not found"

case "${1:-}" in
    up) cmd_up ;;
    down) cmd_down ;;
    status) cmd_status ;;
    *) usage ;;
esac
