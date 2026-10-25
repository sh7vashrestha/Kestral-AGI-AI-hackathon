#!/usr/bin/env bash
# One-time system setup for the field hotspot (run it yourself:  sudo scripts/setup_hotspot_portal.sh).
#  1. dnsmasq: on the jetson-hotspot network every DNS name resolves to the Jetson (10.42.0.1), so a phone's
#     "is there internet?" check reaches drone-brain and the phone shows "Sign in to network" -> the app.
#     Only devices joined to the hotspot are affected (NetworkManager's shared-mode dnsmasq).
#  2. NetworkManager dispatcher: while jetson-hotspot is up, TCP port 80 on that interface goes to drone-brain
#     (8000); the rule is removed when the hotspot goes down.
#  3. polkit: the 'drone' user may switch NetworkManager connections, so ./start.sh --hotspot and ./stop.sh
#     work without sudo. Skip with --no-polkit.
#  4. Restarts nvargus-daemon (the CSI camera service was stuck: "No cameras available").
# Undo:  sudo scripts/setup_hotspot_portal.sh --remove
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
DNS=/etc/NetworkManager/dnsmasq-shared.d/drone-captive.conf
HOOK=/etc/NetworkManager/dispatcher.d/90-drone-captive
PK=/etc/polkit-1/localauthority/50-local.d/10-drone-network.pkla

if [[ "${1:-}" == --remove ]]; then
  rm -f "$DNS" "$HOOK" "$PK"
  echo "removed $DNS $HOOK $PK (takes effect the next time the hotspot starts)"
  exit 0
fi

mkdir -p "$(dirname "$DNS")"
cat > "$DNS" <<'EOF'
# drone-brain captive portal (scripts/setup_hotspot_portal.sh)
address=/#/10.42.0.1
EOF

cat > "$HOOK" <<'EOF'
#!/bin/sh
# drone-brain captive portal (scripts/setup_hotspot_portal.sh): port 80 -> 8000 while jetson-hotspot is up.
[ "$CONNECTION_ID" = "jetson-hotspot" ] || exit 0
RULE="PREROUTING -i $1 -p tcp --dport 80 -j REDIRECT --to-ports 8000"
case "$2" in
  up)   /usr/sbin/iptables -t nat -C $RULE 2>/dev/null || /usr/sbin/iptables -t nat -A $RULE ;;
  down) /usr/sbin/iptables -t nat -D $RULE 2>/dev/null || true ;;
esac
EOF
chmod 755 "$HOOK"

if [[ "${1:-}" != --no-polkit ]]; then
  cat > "$PK" <<'EOF'
[Let drone control NetworkManager (drone-brain start.sh --hotspot)]
Identity=unix-user:drone
Action=org.freedesktop.NetworkManager.*
ResultAny=yes
ResultInactive=yes
ResultActive=yes
EOF
fi

systemctl restart nvargus-daemon
echo "done: captive portal ready (CAPTIVE_PORTAL=true in drone-brain/.env), camera service restarted"
