# KESTREL (drone-brain)

Offline AI search drone. An operator marks an area on a phone (or describes it in chat), says who to look
for, and a Jetson Orin Nano on the drone plans the survey, uploads it to the Pixhawk, runs the checks, flies
it after a human GO, finds the person with YOLO + thermal + CLIP, and writes a flight summary with photos.
Everything runs on the drone. No internet is needed.

Hardware: Jetson Orin Nano 8 GB · Pixhawk 6X (ArduCopter, USB) · IMX477 camera (CSI) · FLIR Lepton
thermal (USB) · phone on the Jetson's WiFi hotspot `KESTREL`.

## Quick start

All commands are run in `~/Desktop/Flight Plan/drone-brain`.

| I want to... | Command |
|---|---|
| Start everything for a flight | `./start.sh` |
| Start with the Jetson's own WiFi (field) | `./start.sh --hotspot` |
| Test uploads indoors (no GPS) | `./start.sh bench` |
| Practise a whole flight on a simulated drone | `./start.sh sim` |
| Stop everything | `./stop.sh` |
| See what is running and the address to open | `./status.sh` |
| Check the hardware | `./check.sh` (or `quick`, `vehicle`, `camera`) |

Then open the address that `start.sh` / `status.sh` prints on the phone, for example
`https://10.42.0.1:8443` on the hotspot (install the certificate once, see HTTPS). Chrome ⋮ → *Add to Home screen* (Safari: Share → *Add to Home Screen*)
gives a KESTREL app icon.

## Ground demo (nothing flies)

`./demo.sh` runs the same phone app without flying: a pretend drone on loopback stands in for the Pixhawk (which
is never opened), and the real RGB + thermal cameras do the search.

1. `sudo systemctl stop drone-brain` (the service holds the cameras), then `./demo.sh`.
2. On the phone, in the chat: *"find the person in a red jacket"*. Gemma writes the plan and the CLIP prompts.
3. Confirm, then GO (*start live search*). Open **Live**: every person is boxed with their CLIP score and
   temperature, and the best match turns red with its photo above the video.
4. Abort (or 10 min, `DEMO_MAX_S`) ends the search, and Gemma writes the summary.

`scripts/demo_search.py` is a lighter standalone version: camera + search bar only, no Gemma, no plan.

## Scripts

| Script | What it does |
|---|---|
| `./start.sh [field\|bench\|readonly\|sim] [--hotspot] [--no-check] [--no-vision]` | Starts Gemma first, then the app. `field` (default): real Pixhawk, Upload → checks → GO. `bench`: upload only, KSU field as stand-in home, no GPS needed. `readonly`: watch only, nothing is sent. `sim`: ArduPilot simulator + test photos instead of the camera. A missing Pixhawk is only a warning. |
| `./stop.sh [--keep-hotspot]` | Clean stop (RTL first if a mission is flying), stops Gemma and the simulator, and switches the WiFi back from the hotspot. |
| `./status.sh` | App / Gemma / simulator / Pixhawk / WiFi / start-at-boot, plus the addresses to open. |
| `./check.sh [all\|quick\|vehicle\|camera]` | Read-only system check with a "what to do" list. `vehicle` = GPS, EKF, pre-arm, battery, fence and failsafe parameters. |
| `./logs.sh [app\|gemma\|sim]` | Follows a log (Ctrl+C to leave). |
| `./test.sh [all\|quick\|gemma\|sim]` | Tests: all offline tests (~10 min), a quick subset (~1 min), live Gemma planner tests, simulator flights. |
| `./hotspot.sh on\|off\|pass` | Switches the WiFi to the `KESTREL` hotspot or back, and shows the hotspot password. |
| `./live.sh` | Camera-only live view (colour + thermal) on port 8090 when the app is not running. With the app running use `/live` on port 8000. |
| `scripts/make_cert.sh [--new-ca]` | Makes the HTTPS certificate (see HTTPS). `start.sh` runs it when none exists. |
| `scripts/start_llama.sh` | Starts Gemma on its own (`start.sh` already does this). It falls back to leaner settings when GPU memory is tight. |

### One-time setup (needs `sudo`)

| Script | What it does |
|---|---|
| `sudo scripts/setup_hotspot_portal.sh` | Hotspot sign-in pop-up (phones joining `KESTREL` are sent to the app), lets `start.sh --hotspot` switch WiFi, and restarts the camera service. Undo: `--remove`. |
| `sudo scripts/install_autostart.sh` | Starts hotspot + Gemma + app at every boot. Then `sudo systemctl stop\|start drone-brain`, logs: `journalctl -u drone-brain -f`. Undo (normal WiFi at boot): `--remove`. |

## A flight, step by step

1. **Plan**: tap 3–8 corners on the map (Survey tab), or chat, for example "100x50 north, man in a red
   jacket", "200x20 in front of the drone", "100x100 around home". Defaults: 20 m altitude, 3 m/s. The
   language model only reads your words; Python computes every waypoint and checks the limits.
2. **Upload**: tap *Upload plan to drone* twice. The mission is sent and read back item by item. Nothing arms.
3. **Review**: check the path on the phone or in Mission Planner (*Read WPs*) while the checklist runs: link,
   GPS, EKF, pre-arm, RC, battery, fence, failsafes, home, mission unchanged, altitude, legal, real-arming switch.
4. **GO**: tap twice. Every check runs again and the mission is re-read, then AUTO → arm → start.
5. **Search**: target alerts with photos and positions. **In flight the abort is the RC transmitter's RTL
   switch**: the phone is usually out of WiFi range.
6. **After landing**: reconnect to the hotspot. The flight summary shows Gemma's text and the best photos.

## Safety

- The app can **never arm a real Pixhawk** unless a human sets `ALLOW_REAL_ARM=true` and `DRY_RUN=false`
  in `.env`. Until then GO stays blocked (the simulator is always allowed).
- Two human gates: Upload, then GO. ABORT / RTL is on every screen.
- Limits: altitude between 15 m (obstacle clearance) and the lowest of `MAX_ALT_M`, `LAANC_CEILING_M` and
  120 m; max 10 m/s; path ≤ 3 km; area ≤ 20 ha. Keep `MAX_ALT_M` below the Pixhawk's `FENCE_ALT_MAX`.
- If the RC pilot changes mode, the app stops commanding and never fights the pilot.
- A mission left on the Pixhawk can be flown by RC (arm + AUTO). Keep the props off on the bench, or clear it
  (Vehicle tab → *Clear mission on Pixhawk*).

## Settings (`.env`)

| Setting | Meaning |
|---|---|
| `DEFAULT_ALT_M=20`, `DEFAULT_SPEED_MS=3` | Plan defaults (a chat request can override within the limits) |
| `MAX_ALT_M`, `LAANC_CEILING_M` | Altitude ceiling: keep it below `FENCE_ALT_MAX` and your LAANC authorisation |
| `ALLOW_REAL_ARM`, `DRY_RUN` | Real arming switches (human only) |
| `CAMERA_ROTATE=180` | Colour camera mounted upside down to match the thermal camera |
| `FOCAL_LENGTH_MM`, `CAMERA_HFOV_DEG`, `CAMERA_VFOV_DEG` | Real lens values (6 mm is still assumed) |
| `CAPTIVE_PORTAL=true` | Hotspot sign-in pop-up |
| `CLIP_USE_TRT=false` | CLIP runs in PyTorch (the TensorRT engine gave wrong results on this Jetson) |

## HTTPS

The page is served over HTTPS (`HTTPS=true` in `.env`) at `https://10.42.0.1:8443` (hotspot). Plain HTTP only
redirects there, so the hotspot pop-up and `http://10.42.0.1` still work. The WiFi itself is already encrypted
(WPA2); HTTPS also encrypts the page itself.

The certificate comes from a small local authority made on the Jetson (`scripts/make_cert.sh`, files in `data/tls/`).
It only covers private addresses (10.x, 192.168.x, .local), so it cannot be misused for real websites.
**Each phone installs it once** (otherwise the browser shows "Not secure"):

- **iPhone:** Safari → `http://10.42.0.1/kestrel-ca.crt` → *Allow* → Settings → *Profile Downloaded* → *Install*.
  Then Settings → General → About → *Certificate Trust Settings* → turn on *KESTREL local CA*.
- **Android (Samsung):** Chrome → `http://10.42.0.1/kestrel-ca.crt` (it is saved). Settings → Security and privacy →
  More security settings → *Install from device storage* → *CA certificate* → pick `kestrel-ca.crt`.

New IP address (e.g. another wired network): run `scripts/make_cert.sh` again; phones keep working (same authority).
`scripts/make_cert.sh --new-ca` replaces the authority, and then every phone must install it again. Turn HTTPS off:
`HTTPS=false` in `.env`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Camera offline" / `No cameras available (argus)` | `sudo systemctl restart nvargus-daemon` (or reboot). A black image means the lens cap is on. |
| Chat says the AI assistant is not available | Gemma did not fit in GPU memory. Close extra VS Code windows, then `./stop.sh` and `./start.sh` (Gemma must start before the app). |
| Phone joins `KESTREL` but says "no internet" and drops off | Normal: the hotspot has no internet. Turn off mobile data, or Wi-Fi → Intelligent Wi-Fi → "Switch to mobile data" (Samsung), and choose "Keep Wi-Fi connection". Then open `http://10.42.0.1`. |
| No sign-in pop-up on the phone | Some Android phones only show a "Sign in to network" notification. Open `http://10.42.0.1` or use the home-screen icon. |
| Upload refused: no home position | Field mode needs a GPS fix (outdoors). Indoors use `./start.sh bench`. |
| GO stays grey | The checklist under it names the first blocking item. |
| VS Code disconnected after `--hotspot` | The Jetson left your WiFi. Use the cable, or join `KESTREL` and connect to `drone@10.42.0.1`. |

## Files

- `data/flights/<time>/`: one folder per flight: `flight.json`, mission files, alert photos, `summary.md`.
- `data/plans/<id>/`: every plan: `mission.waypoints` (planned) and `readback.waypoints` (on the Pixhawk).
  Both open in Mission Planner via *Plan → Load WP File*.
- `data/maps/usa.pmtiles`: offline map of the lower 48 states (19 GB, OpenStreetMap). The page uses the one `.pmtiles`
  file in that folder (others: rename to `.off`). Another area (needs internet once):
  `.venv/bin/python scripts/fetch_map.py --center LAT,LON --radius-km 5 --out data/maps/area.pmtiles`.
- `data/start.log`, `data/llama-server.log`, `data/sitl.log`: logs (`./logs.sh`).
- Code: `main.py` (entry point), `drone_brain/` (all modules + `webapp/`, the phone app), `scripts/` (tools, mock
  vehicle, SITL), `tests/`. Models go in `models/` (`scripts/export_models.py`), runtime files in `data/`.
- Design notes and decisions: `DECISIONS.md`, `notes/`.
