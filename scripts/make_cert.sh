#!/usr/bin/env bash
# HTTPS for the phone page: a small local certificate authority ("KESTREL local CA") + a server certificate.
#   scripts/make_cert.sh          create (keeps an existing CA, so phones that trust it keep working)
#   scripts/make_cert.sh --new-ca also replace the CA (every phone must install the new one)
# Install data/tls/kestrel-ca.crt on each phone once (README: HTTPS), then set HTTPS=true in .env.
# The CA is name-constrained to private addresses and .local names: even if its key leaked it could not
# impersonate real websites. The server certificate is re-made on every run (new IPs are picked up).
set -euo pipefail
cd "$(dirname "$0")/.."
D=data/tls; mkdir -p "$D"; chmod 700 "$D"
if [[ "${1:-}" == --new-ca || ! -f $D/kestrel-ca.key ]]; then
  openssl req -x509 -new -newkey rsa:2048 -nodes -days 3650 -sha256 -keyout "$D/kestrel-ca.key" -out "$D/kestrel-ca.crt" \
    -subj "/CN=KESTREL local CA" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:0" -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -addext "nameConstraints=critical,permitted;IP:10.0.0.0/255.0.0.0,permitted;IP:192.168.0.0/255.255.0.0,permitted;IP:172.16.0.0/255.240.0.0,permitted;IP:127.0.0.0/255.0.0.0,permitted;DNS:.local,permitted;DNS:localhost" \
    2>/dev/null
  echo "new CA: $D/kestrel-ca.crt (install it on every phone)"
fi
SAN="DNS:kestrel.local,DNS:localhost,IP:127.0.0.1,IP:10.42.0.1"
for ip in $(hostname -I); do [[ $ip == *:* || $ip == 10.42.0.1 ]] || SAN+=",IP:$ip"; done
openssl req -new -newkey rsa:2048 -nodes -keyout "$D/server.key" -out "$D/server.csr" -subj "/CN=KESTREL" 2>/dev/null
# 800 days: iOS rejects server certificates valid for more than 825 days
openssl x509 -req -in "$D/server.csr" -CA "$D/kestrel-ca.crt" -CAkey "$D/kestrel-ca.key" -CAcreateserial -days 800 -sha256 \
  -out "$D/server.crt" -extfile <(printf "subjectAltName=%s\nbasicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n" "$SAN") 2>/dev/null
rm -f "$D/server.csr"; chmod 600 "$D"/*.key
echo "server certificate for: $SAN"
