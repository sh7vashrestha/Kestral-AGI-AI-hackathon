# Hardening C: vehicle and mission edge cases

Scope: `mission.py`, `controller_real.py` (vehicle part), `main.py`, mock vehicle, ArduPilot SITL.
Tests: `tests/test_vehicle_edges.py` (74 mock tests, ~2.5 min), `tests/test_e2e_sitl.py` (`pytest -m sitl`, 6 tests).
Result column: **PASS** = behaved as expected already; **FIXED** = bug found by the test and fixed; **DOC** = limitation, documented.

## Decisions

- **Pilot takes over (new state `manual`).** If the vehicle is armed and leaves AUTO for a non-RTL mode after we
  started our mission (RC mode switch, failsafe into ALT_HOLD, ...), the controller goes to state `manual`, alerts
  "PILOT CONTROL", and sends **nothing** again: no RTL on link loss, Abort answers "use the RC transmitter", no new
  plans (`manual` is a busy state). It ends at disarm (`landed`) and closes the flight record. A throttle-stick
  override inside AUTO does not change the mode and cannot be seen (DOC).
- **Vehicle-initiated RTL/LAND (battery, GPS/EKF, fence, pilot RTL switch)** -> state `rtl`; the app reports the mode
  change and every STATUSTEXT with severity <= 4 ("Vehicle: ...") during a flight, and "GPS fix LOST" once. It does
  not send its own RTL on top.
- **SIGTERM policy (sim/full).** Armed in AUTO with our mission -> command RTL first (bounded 2 s, in parallel with
  the 3 s hard exit); mid-upload -> cancel, never arm; pilot control / already RTL / readonly / upload mode -> nothing.
  Port released and process gone in 0.5 s (measured with a real `main.py` + external mock process; hard limit 3 s).
- **Upload refused while armed in every mode** (was only upload mode): a mission never changes under an armed vehicle.
- **Abort is cancel-aware.** Abort or link loss during upload calls `cancel_flight()`; `fly_plan` re-checks before
  verify, AUTO, arm and start, so the vehicle never arms after an abort.
- **One Confirm at a time** (`_confirming`): double tap / two phones can no longer both run the preflight await and
  both start a flight.

## Table

| # | Edge case | Expected | Result | Fix / note |
|---|---|---|---|---|
| 1 | Link lost while idle / planned | alert; plan and Confirm refused with "No MAVLink link" | PASS | |
| 2 | Link lost during upload | never arms, back to `planned`, clear message | FIXED | `_on_link_lost` now cancels the upload (before, a quick link flap let fly_plan go on to AUTO+arm after the RTL request) |
| 3 | Link lost during arming / takeoff | RTL queued, sent when the link returns | PASS | RTL retried via `_rtl_on_restore` |
| 4 | Link lost in survey, link flaps 3x | vehicle ends in RTL, no command storm | PASS | |
| 5 | Link lost in RTL / landed | alert only, no commands sent | PASS | |
| 6 | Heartbeat flapping x5 | every loss/restore reported once, link usable after | PASS | backend test |
| 7 | Vehicle disappears and a new one starts on the same port (serial replug equivalent) | lost then restored, commands work, no hang while down | PASS | |
| 8 | `recv_match` raises (adapter unplugged) | reader thread survives, reopens the link | PASS | |
| 9 | Random bytes in the stream, second GCS (sysid 255), second vehicle (sysid 7, other position/mode) | ignored; telemetry stays that of the first autopilot; upload and mode change still work | PASS | pymavlink `udpin` writes to every peer that sent a packet; harmless |
| 10 | MISSION_ACK NO_SPACE / INVALID_SEQUENCE / DENIED / ERROR / UNSUPPORTED_FRAME / CANCELLED | `CommandRejected` naming the code, nothing armed | PASS | |
| 11 | Mission bigger than vehicle capacity | NO_SPACE named | PASS | |
| 12 | Upload interrupted mid-way (vehicle NAKs after N items) | error, `fly_plan` never arms or changes mode | PASS | |
| 13 | Requests out of order (1,0,2,...) and duplicated | upload completes, verify OK | PASS | |
| 14 | Lost items / lost MISSION_COUNT | per-item retry (4x), bounded `LinkTimeout` "stalled" | PASS | |
| 15 | Read-back mismatch (altitudes +5 m) | "read-back verification failed", **never arms**, no AUTO_OPTIONS change | PASS | |
| 16 | Upload / clear while armed | refused before any I/O in every mode | FIXED | control mode was not guarded |
| 17 | Abort during upload | upload cancelled, never arms, <10 s | FIXED | cancel flag (see decisions) |
| 18 | Command ACK missing | bounded `LinkTimeout`, no hang | PASS | |
| 19 | Home missing | planning/Confirm refused, "needs 3D GPS fix" | PASS | |
| 20 | Home moves > 5 m after planning | plan rebuilt, old plan_id expired, 12 m offset refused at Confirm | PASS | existing e2e |
| 21 | Home altitude jump +6 m | plan rebuilt with new AMSL | PASS | |
| 22 | Home wobble 1.5 m / 0.5 m | plan kept | PASS | |
| 23 | Home moves mid-flight | alert "RTL will return to the NEW home", logged | PASS | existing e2e |
| 24 | Bench home vs real home | real home wins; bench only in `upload` mode | PASS | `test_bench_readback.py` + `test_bench_home_only_in_upload_mode` |
| 25 | GPS lost in flight (EKF failsafe -> LAND) | "GPS fix LOST" + "Vehicle: EKF failsafe", state `rtl`, landed | PASS | alerts are new |
| 26 | Battery failsafe -> RTL by the vehicle | statustext + mode-change alert, app sends no RTL of its own | FIXED | no alert existed |
| 27 | Geofence breach -> RTL | "Vehicle: Fence breach" alert, state `rtl` | FIXED | no alert existed |
| 28 | Pilot flips mode out of AUTO (LOITER) | state `manual`, alert, no commands, Abort and link loss do not fight, new plans blocked, landed at disarm | FIXED | state `manual` was missing: app stayed "flying" and would RTL on link loss |
| 29 | Pilot switches within 0.5 s of start | still detected | FIXED | `_saw_auto` is now set when `fly_plan` returns |
| 30 | Mission completes, lands, disarms | `landed`, flight record closed `end_reason=landed` | PASS | |
| 31 | Vehicle rejects arm ("PreArm: Compass not calibrated") | text reaches the operator, stays `planned`, Confirm can be repeated | PASS | |
| 32 | Abort during takeoff | RTL | PASS | |
| 33 | Abort while already RTL / already landed | "Already returning" / "disarmed, nothing to abort" | FIXED | disarmed-but-state-still-rtl window re-sent RTL |
| 34 | Abort with the link down | immediate "ABORT NOT SENT ... use the RC transmitter NOW"; RTL sent when the link returns | FIXED | used to block up to 25 s |
| 35 | Concurrent Abort from two phones | both answered, single RTL mode, no error | PASS | |
| 36 | Double Confirm / Confirm while an upload runs | exactly one flight, one ARM command | FIXED | race window across the preflight await |
| 37 | Confirm of a stale / superseded / unknown plan_id | "Unknown or expired plan" | PASS | empty id is a 422 from the API |
| 38 | readonly: every write message (MISSION_COUNT/ITEM/CLEAR_ALL, SET_MODE, PARAM_SET, COMMAND_LONG arm/mode/RTL/start/takeoff, COMMAND_INT, RC override) | `ReadOnlyViolation` before the socket is touched; reads (heartbeat, param read, mission request, PREARM_CHECKS, SET_MESSAGE_INTERVAL) pass | PASS | |
| 39 | upload: mode/arm/RTL/params/start refused; mission protocol allowed | | PASS | |
| 40 | full on a serial link without ALLOW_REAL_ARM / DRY_RUN=false | `ArmRefused`, nothing sent | PASS | |
| 41 | SIGTERM while flying (sim) | RTL commanded, process gone < 3 s, port free | FIXED | no RTL before; port free ok |
| 42 | SIGTERM during upload | upload cancelled, never arms | FIXED | |
| 43 | SIGTERM under pilot control / readonly / idle | no command | PASS | |
| 44 | RC throttle override inside AUTO | not visible in telemetry | DOC | pilot should flip the mode switch; failsafes (RC loss) are on the Pixhawk |
| 45 | Link loss while `manual` | no RTL (pilot has control) | PASS | by design |

## SITL (ArduCopter SITL, built from ~/ardupilot: reports 4.8.0-dev, the real Pixhawk is 4.7.2)

Speedup 5, wall-clock seconds, Orin Nano with llama-server resident.

| Scenario (real ArduCopter SITL, full app: WS -> plan -> Confirm -> upload/verify -> AUTO -> arm) | Result | Timings (wall s, speedup 5) |
|---|---|---|
| Full mission + RTL + land + disarm (`test_sitl_full_mission`) | PASS, alerts person + target_match, flight record `landed`, max alt 20 m | ready 5.0, plan 15.9, started 26.2, landed 46.3 |
| Upload-only mode, no arming (`test_sitl_upload_only`) | PASS (verify OK, fence polygon, clear) | 37.8 total |
| Abort mid-survey (seq >= 5) | PASS: ack in the same tick, vehicle mode RTL after 0.5 s, landed and disarmed after 11 s | confirm->started 10.3, airborne 12.3, total 43.7 |
| Pilot flips to LOITER (second MAVLink link) | PASS: "PILOT CONTROL" after 0.5 s, state `manual`, 4 s wall (20 s sim) of silence, SITL stays LOITER, Abort answers "pilot has taken over", pilot LAND -> `landed`, flight record closed | total 33.7 |
| Battery failsafe (`BATT_LOW_VOLT=11`, `SIM_BATT_VOLTAGE=9`, SITL only) | PASS: "Vehicle: Battery 1 is low 8.58V" after 2.5 s, vehicle RTL after 3.0 s, no RTL event from the app, landed 9 s later | total 38.7 |
| GPS loss (`SIM_GPS1_ENABLE=0`, SITL only) | PASS: "GPS fix LOST ... RC transmitter" after 0.5 s, vehicle went LAND by itself after 2.0 s, state `rtl`, landed | total 35.7 |

Whole SITL file: 6 passed in 240 s. SITL processes stopped afterwards. SITL parameters were changed only through the
test's own second link to the SITL it started (private TCP port); the app never sends PARAM_SET except `AUTO_OPTIONS`
inside `fly_plan` (sim/full only).

## Open / not covered

- UI: `webapp/static/app.js` lists `["flying","uploading","rtl"]` for showing the flight panel and the Abort button;
  the new `manual` state is not in those lists (not my file). Until added, in `manual` the phone shows the
  "new survey" button; the backend refuses planning in that state with "A mission is already in progress".
- Real-hardware only: serial reopen after USB unplug is covered by `_reconnect` + a monkeypatched read error, not by
  pulling a cable (the Pixhawk was not touched).
- Throttle override in AUTO, RC-loss failsafe and `FS_GCS_ENABLE` behaviour are Pixhawk configuration; the
  pre-flight check reads `FS_THR_ENABLE` only.
