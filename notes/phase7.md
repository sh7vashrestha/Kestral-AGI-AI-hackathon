# Phase 7 - main.py orchestrator, real controller, vehicle modes, files, SITL

## Progress
- `main.py` (CLI + asyncio/uvicorn wiring, `build_app()` used by tests), `controller_real.py` (`RealController`, full
  Controller protocol + legal checklist, upload/clear, vehicle check), `pipeline.py` (`VisionPipeline` / `NullPipeline`),
  `clip_prompts.py` (Gemma -> CLIP prompt lists, cached), `plan_files.py` (WPL 110 + GeoJSON), `vehicle_check.py`,
  `webapp/files.py` (router), `scripts/hardware_check.py`, `scripts/sitl_dronebrain.parm`, `scripts/sitl_run.sh` (rewritten).
- Edited: `mission.py` (access levels readonly/upload/control, HOME_POSITION tracking, get_param, preflight_check,
  mission_type/fence, clear, RUN_PREARM_CHECKS, raw message cache, STATUSTEXT log), `safety.py` (effective ceiling/floor),
  `storage.py` (flight summary, `PlanStore`), `matcher.py` (ClipBackend `cutoff`), `scripts/mock_vehicle.py` (movable
  home, home reset at arming, preflight params, fence mission type, prearm/EKF/RC/EXTENDED_SYS_STATE/AUTOPILOT_VERSION),
  `webapp/server.py` (only `/api/limits` uses the effective band), phone UI (legal modal, pre-flight table + downloads,
  Vehicle tab, upload label, clear-mission button). ADDs in config.py / models.py / .env.example.
- Tests: `tests/test_phase7.py` (45), `tests/test_e2e.py` (12, mock vehicle over loopback UDP, WebSocket),
  `tests/test_e2e_sitl.py` (2, `-m sitl`). `pytest -q`: all pass; `pytest -m sitl`: 2 passed (85 s).

## How to run
```
# full demo, no hardware (fake ArduCopter at the KSU field; open http://<jetson>:8000/)
.venv/bin/python main.py --mock-vehicle [--mock-speed 5] [--camera file:tests/assets/persons] [--no-vision]
# ArduPilot SITL (terminal 1) + drone-brain (terminal 2)
SPEEDUP=1 scripts/sitl_run.sh
.venv/bin/python main.py --mavlink-url udpin:127.0.0.1:14550 --camera file:tests/assets/persons
# real Pixhawk, read-only (telemetry, params, pre-arm report; nothing is ever changed)
.venv/bin/python main.py --vehicle readonly --mavlink-url serial:/dev/ttyACM0 --baud 115200 --no-vision
.venv/bin/python scripts/hardware_check.py          # 25 s report -> data/hardware_check.json, data/vehicle_check.json
# real Pixhawk, upload-only (Confirm uploads + verifies the mission for review in Mission Planner/QGC; never flies)
.venv/bin/python main.py --vehicle upload --mavlink-url serial:/dev/ttyACM0 --baud 115200 --no-vision
```
Start order: llama-server first (GPU memory), then main.py. Port 8000 must be free (the `webapp.server --fake`
demo uses it; pass `--port`).

## Vehicle modes (VEHICLE_MODE / --vehicle)
| mode | link | allowed to send | Confirm | Abort |
|---|---|---|---|---|
| sim (default) | loopback / wildcard+DRY_RUN only (main.py refuses a serial port) | everything | fly_plan | RTL |
| readonly | any | heartbeat, stream/interval requests, REQUEST_MESSAGE, PARAM_REQUEST_READ/LIST, RUN_PREARM_CHECKS, mission download (MISSION_REQUEST_LIST/_INT + closing ACK: a read) | "Read-only mode: plan saved, not uploaded" | "use the RC transmitter" |
| upload | any | readonly + MISSION_COUNT/ITEM_INT/CLEAR_ALL | upload + read-back verify (refused while armed) | "use the RC transmitter" |
| full | any | everything; arm still needs ALLOW_REAL_ARM=true + DRY_RUN=false | fly_plan | RTL |
Two layers: every method checks the access level before any I/O (`ReadOnlyViolation`), and `_send()` has a per-level
allow-list of outgoing messages/commands. Unit-tested for each refused method in readonly and upload, plus on the
mock that no MISSION_*/PARAM_SET/SET_MODE/arm/takeoff/RTL command ever reached it.

## Home (from the Pixhawk only, live)
- HOME_POSITION requested at connect (REQUEST_MESSAGE 242), every 5 s while unknown, 1 s/5 s after the first 3D fix,
  0.3 s/2 s after an arming transition (ArduPilot resets home at arming), and streamed at 0.2 Hz. `HomeFix(lat, lon,
  amsl_m, ts, version)`; `on_home_changed(old, new)` callback.
- Controller checks it every 2 Hz tick: pushes `status` (map H marker/fence move), logs `home_changed` to telemetry.jsonl in
  flight, error-alerts a mid-flight change (a change within 20 s of Confirm is reported as "home reset at arming").
- Pending plan whose home moved > HOME_SHIFT_REPLAN_M (5 m) or > HOME_ALT_SHIFT_M (2 m) vertically is rebuilt from the
  same request (new plan_id, safety + preflight re-run, chat "Home moved 12 m - plan recalculated").
- Confirm re-checks plan home vs current Pixhawk home (else recalculates and refuses), and refuses if the drone sits more
  than 5 m from its home (arming would move home under the plan).
- No home / no 3D fix -> planning refused "Waiting for Pixhawk home position - needs 3D GPS fix".
- No hard-coded home in the controller; the field coordinate is only the mock/SITL start.

## Files
- `data/plans/<plan_id>/`: plan.json, mission.waypoints (QGC WPL 110: home line frame 0 AMSL, TAKEOFF, DO_CHANGE_SPEED,
  waypoints frame 3, RTL), mission.geojson (area Polygon, route LineString, home Point), preflight.json; after an upload:
  upload.json, readback.waypoints (what the Pixhawk returned).
- `data/flights/<UTC ts>/`: frames/ (every processed frame + `_thermal`), crops/, detections/matches/alerts/telemetry.jsonl
  (telemetry 1 Hz + events: state, home_changed, rtl, link_lost), mission.waypoints, mission.geojson, flight.json
  (plan, targets, summary: plan_id, start/end, end_reason, distance, max alt, alerts, home changes, link losses, vision).
- Endpoints: GET /api/plans, /api/plans/{id}, /api/plans/{id}/{file}, /api/flights, /api/flights/{id}/{file},
  /api/legal, POST /api/legal/ack, /api/preflight, /api/vehicle/check[?cached=1], POST /api/vehicle/clear_mission.

## Decisions
- P7.1 Controller tasks live in the app lifespan (`attach`/`detach`), so TestClient and main.py run identical code.
  Watchdog `_supervise`: restart with 1..30 s backoff; telemetry task is critical (fails while uploading/flying -> RTL).
  Vision watchdog: worker dead or no frame for > 5 s -> RTL if flying, then restart (camera dropped, models kept).
- P7.2 Link loss while uploading/flying -> RTL attempt + error alert; RTL re-sent on link restore if not in RTL/LAND.
- P7.3 State machine: idle -> planned -> uploading -> flying -> rtl -> landed (landed when disarmed after being armed;
  mission-end RTL in AUTO is also caught by the disarm). Flight record closed on landed/shutdown.
- P7.4 Legal checklist: per server session (controller memory), all 4 items must be ticked; Confirm refuses before.
- P7.5 Effective ceiling = min(MAX_ALT_M, LAANC_CEILING_M, 120); floor = max(MIN_ALT_M, OBSTACLE_CLEARANCE_M);
  floor > ceiling blocks every plan. /api/limits returns both with sources (UI slider uses them).
- P7.6 Preflight (PARAM_REQUEST_READ only): table from phase7-requirements.md; plus plan-specific: plan altitude must be
  below FENCE_ALT_MAX and the area inside FENCE_RADIUS. WPNAV_SPEED and FS_THR_ENABLE=2 are warnings. ArduCopter 4.7+
  names RTL_ALT_M (m) / WP_SPD (m/s) are read when RTL_ALT / WPNAV_SPEED do not answer (the real Pixhawk runs 4.7.2).
- P7.7 Images never go to the LLM (user policy): CLIP is the only matcher; VLM backend / VLM_CONFIRM_TOP are ignored
  with a warning. Gemma writes the CLIP prompt lists (cached per target, fallback default prompts). CLIP_MATCH_CUTOFF
  (default 0.5) is a setting - UNTUNED. CLIP load retried 3x (CUDA OOM), and again in the background at each flight start.
- P7.8 Fence polygon upload (upload mode, UPLOAD_FENCE_POLYGON=false by default): area as inclusion polygon
  (mission_type FENCE, cmd 5001), verified by read-back; FENCE_* params are never touched. Opt-in because it replaces any
  fence polygon already on the Pixhawk.

## Measurements
- Mock e2e (x10 sim speed): connect 1.1 s, plan 1.1 s, upload+verify+AUTO+arm+start 0.05 s, first alerts +0.3 s,
  landed 4.8 s after start.
- SITL (ArduCopter 4.8.0-dev, built here in 16 min with -j2, speedup 5): ready (3D fix + home + pre-arm OK) 5 s,
  plan 10.9 s (includes 2 x ~5 s timeouts for RTL_ALT/WPNAV_SPEED which 4.8 renamed), upload+verify+AUTO+arm+start 10.3 s,
  alerts +0.6 s, landed 20 s later; max alt 20.0 m; all waypoints flown. Upload-only on SITL with fence: OK.
- Real vision on the Jetson (mock vehicle, file camera, llama-server resident): YOLO26n TRT ~30-60 ms/frame; CLIP TRT
  engine fails to deserialize ("built with another TensorRT version") -> torch FP16 fallback works; one start-up run hit
  CUDA OOM (retry added). Two transient `NVML_SUCCESS == r INTERNAL ASSERT` CUDA allocator errors during the run
  (handled: frame skipped / matcher backoff).

## Real Pixhawk (read-only, /dev/ttyACM0 @115200, indoors, USB power) - data/hardware_check.json
ArduCopter 4.7.2 official, quad X, STABILIZE, disarmed, ON_GROUND/STANDBY; GPS NO FIX 0 sats; no home; battery 0.0 V
(USB); EKF compass variance 1.0 (BAD), const-pos mode; pre-arm: "RC not found", "Compasses inconsistent",
"Battery 1 unhealthy"; existing 9-item mission on board. Verdict NOT READY (expected indoors).
Params: RTL_ALT_M 15 OK, FENCE_ENABLE 0 BAD, FENCE_TYPE 0 BAD, FENCE_ALT_MAX 100 BAD (> 60), FENCE_RADIUS 300 OK,
FENCE_ACTION 0 BAD, BATT_FS_LOW_ACT 0 BAD, FS_THR_ENABLE 1 OK, WP_SPD 10 m/s OK.
Human must set in Mission Planner: FENCE_ENABLE=1, FENCE_TYPE>=3, FENCE_ALT_MAX<=60 (or LAANC), FENCE_ACTION=1,
BATT_FS_LOW_ACT=2 (+ BATT_LOW_VOLT/BATT_MONITOR), calibrate compasses, bind RC.

## Bench home (upload mode ONLY, indoor test without GPS)
- `--bench-home LAT,LON[,AMSL]|field` / `BENCH_HOME` (empty default; `field` = KSU soccer field centre, AMSL = FIELD_AMSL_M 300 m;
  FIELD_HOME / FIELD_AMSL_M now live in config.py, main.py re-exports them). `config.bench_home_for()` raises
  `BenchHomeError` unless VEHICLE_MODE=upload: `main.check_vehicle_mode` turns it into "refusing to start" (sim, readonly,
  full all refuse) and `RealController.__init__` refuses too. Only `_effective_home()` reads it; the arming paths never do.
- Precedence: a real Pixhawk HOME_POSITION always wins (log + status `bench_home.note` + one chat line when it appears; a
  pending bench plan is rebuilt with the real home, and Confirm refuses a plan whose `home_source` changed). Without a real
  home the GPS-fix gate is skipped and planning, safety (geofence) and the upload use the bench home (item 0 = bench home).
- Recorded: `PlanProposal.home_source` ("pixhawk" | "bench") -> plan.json, `preflight.home_source` -> preflight.json,
  mission.geojson home feature, upload.json, snapshot `home_source` / `home.source` / `bench_home`.
- Preflight downgrade (only `home_source == "bench"`): every failed preflight ERROR row (fence / battery / RC-loss params,
  plan vs fence) becomes a warning row ("bench: would block a flight, does not block this upload"); the table still runs.
  safety.validate errors (altitude, geofence from the bench home, area, path, items) still block. Legal checklist unchanged.
  NOTE for the user: on the real Pixhawk FENCE_ENABLE=0 / BATT_FS_LOW_ACT=0 therefore do NOT block the bench upload.
- UI: chip "BENCH HOME \u2014 not from GPS (upload test only)" in a bar under the status bar, in the Preview, in the chat plan
  card and in the read-back card; the upload message repeats it and says ArduPilot replaces item 0 with its real home on
  fix/arm and that waypoints are absolute lat/lon while altitudes are relative to wherever it arms.

## Read-back view
- After an upload (and on "Read mission from Pixhawk") the controller keeps `readback` (list of {seq, command, name, frame,
  frame_name, lat, lon, alt, p1..p4, positional}) and `readback_info` {plan_id, ts, source "upload"|"read", files, hint, diff}
  in `snapshot()`; `last_upload` in the snapshot omits them, upload.json contains them.
- `mission_diff.compare_missions(planned, readback)`: row per seq with planned/readback views, horizontal diff in cm, altitude
  diff, ok flag, note; tolerances = `verify_mission` (POS_TOL_DEG 1e-6 deg, ALT_TOL_M 0.05 m; frames 3/6, 0/5, 10/11 are
  equivalent). Item 0 is compared by command only (ArduPilot substitutes its home); non-positional items (TAKEOFF,
  DO_CHANGE_SPEED, RTL) compare command + params (+ altitude). Summary text "12/12 items match, max error 0.4 cm".
- `POST /api/vehicle/read_mission` (controller `read_vehicle_mission`, upload + readonly + any mode, refused while
  uploading/flying) re-downloads and compares with the pending plan, else the last uploaded one; saves readback.waypoints.
  Readonly now allows `download_mission` (send allow-list: MISSION_REQUEST_LIST/_INT + MISSION_ACK; upload/clear/count/item
  still refused; guard test updated, new test that the mock receives no MISSION_COUNT/ITEM/CLEAR).
- Phone: Preview shows a card on top (summary notice, links `mission.waypoints` (planned) / `readback.waypoints` (on the
  Pixhawk) + "Mission Planner -> Plan -> Load WP File", collapsible "Planned vs uploaded" table, "Read mission from Pixhawk"
  button; also on the Vehicle tab); map layer = thin dashed contrasting line (near-white dark / near-black light) with amber
  numbered markers (red where the item differs) over the planned route, legend "Planned" / "On Pixhawk". A chat card appears
  per upload/read. In upload/readonly mode the UI no longer jumps to the Flight step while "uploading".
- Tests: `tests/test_bench_readback.py` (gating x4 modes, parsing, real home wins, home_source files, preflight downgrade only
  in upload+bench, safety still blocks, readonly download, diff tolerances/non-positional, e2e on the mock incl. tamper ->
  read -> rows differ). Screens: `data/ui_screens/upload_{dark,light}_*.png` (412 px). SITL not re-run (SITL has GPS).

## Open issues
- CLIP TensorRT engine needs re-export for the installed TensorRT; torch FP16 fallback is used meanwhile.
- CLIP cut-off untuned; the file: test source changes person every frame, so target_match needs real video.
- Thermal + RGB calibration not done (Phase 5); thermal was disabled in the runs here.
- Confirm refuses if the drone is > 5 m from its Pixhawk home (arming resets home) - the operator must arm where the
  Pixhawk set home, or wait for a home update.
- In sim mode MAVLINK_URL=udpin:0.0.0.0 counts as simulator only with DRY_RUN=true (unchanged Phase 4 guard).
- Legal ack is per server process, not per phone.
