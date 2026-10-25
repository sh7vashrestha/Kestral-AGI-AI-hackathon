# System check (`python main.py --check`)

One command checks everything, read-only: `python main.py --check` or `.venv/bin/python scripts/system_check.py`.
Flags: `--quick` (skip camera/thermal/YOLO/CLIP), `--json`, `--only host,llm,planner,camera,thermal,yolo,clip,vehicle,web,files`,
`--standalone` (ignore a running main.py), `--mavlink-url/--baud`. Exit code 1 if any check failed; report in `data/system_check.json`
(native argus/CUDA noise goes to `data/system_check.log`). Code: `syscheck.py`.

* Standalone: groups run in parallel threads (~30 s). The vehicle is opened with the read-only backend (`run_vehicle_check`, no duplicated
  logic); if MAVLINK_URL is the SITL default and /dev/ttyACM0 exists, it also probes the Pixhawk there (read-only) and warns.
* If main.py is running (port answers) the CLI asks it: `GET /api/system/check?quick=&only=` (router in `syscheck.make_router`, wired in
  `main.build_app`); devices are untouched, lines say "[running app]". An older main.py without that endpoint: the CLI falls back to its
  `/api/status` + `/api/vehicle/check`. Busy serial port / thermal device (scan of /proc/*/fd) => skipped with the pid.
* YOLO/CLIP are not loaded standalone if < 1.2 / 1.0 GB RAM is free (llama-server holds ~4 GB; the skip message says so).
* Phone UI: Vehicle tab -> "System" card (Quick / Full check).

## Fixes
* camera.py: "opened" only after the first real frame (open + first frame within 15 s); backoff 0.5 -> 30 s; failure logged once, summary every
  60 s with the `sudo systemctl restart nvargus-daemon` hint; `Camera.state/status()/fps()`. `/api/status` -> `vision.camera_status`,
  `vision.camera_problem`; status bar shows CAM unavailable + a notice. Pipeline `alive()` is True while the camera is starting/unavailable
  (no watchdog restart loop); the watchdog logs once per distinct state, then once a minute.
* Vehicle: no heartbeat -> one message with the fix (SITL / --mock-vehicle / readonly serial), `status.vehicle_hint`, status bar "VEHICLE none";
  quiet retry with backoff to 30 s, summary per minute; any exception (busy serial port) is handled.
* STATUSTEXT: `mission.StatusTextLog` keeps each distinct text once with `count`, `first_ts`, `ts` (last seen); logged once, summary every
  10 min; phone feed shows `x N`.
* `POST /api/client_log` (30/min, 429 beyond); `static/js/clientlog.js` forwards window errors, unhandled rejections, one "page load (type)" line.
  MapView degrades to the no-map layout on WebGL context loss (3 s grace), WebGL errors or constructor failure.

## Page-reload loop finding
Not reproducible: headless Firefox against `main.py --mock-vehicle` for 66 s = 1 page load, no reloads (no Chromium/playwright on the Jetson).
Software GL there could not create a WebGL context: the existing fallback hid that silently, now it is reported via client_log and the
no-map layout is shown. app.js has no reload code, so the ~2 s reload is client-side (browser / VS Code webview auto-reloading a crashed
WebGL tab). Next time look for `client[...] page load (reload|navigate)` and `WebGL context lost` in the server log.
