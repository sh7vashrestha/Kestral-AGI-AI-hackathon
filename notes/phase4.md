# Phase 4 (part 1): mission.py, mock vehicle, tests

## Progress
- `mission.py`: `VehicleBackend` ABC + `ArduPilotBackend` (pymavlink), `create_backend(settings)` factory keyed on `settings.autopilot`.
- `scripts/mock_vehicle.py`: pure-pymavlink fake ArduCopter (loopback UDP only; refuses other hosts). Usable as module (`MockVehicle(...).start()/stop()`) and CLI.
- `scripts/sitl_run.sh`: documented ArduPilot SITL launcher (not run; no ArduPilot clone/build done).
- `tests/test_mission.py`: 20 tests, ~15 s total, all passing (3 consecutive runs). Run: `.venv/bin/pytest tests/test_mission.py`.
- `models.py`: ADDED `MissionItem` (seq, command, frame=6, param1..4, lat/lon degrees, alt m, autocontinue). No existing field changed.

### Public API (ArduPilotBackend, all of it also on the ABC)
`connect(timeout=10)`, `close()`, `telemetry() -> Telemetry`, `heartbeat_ok() -> bool`, `get_home(timeout=5) -> GeoPoint|None`,
`build_mission_items(plan) -> list[MissionItem]`, `upload_mission(items)`, `download_mission() -> list[MissionItem]`,
`verify_mission(items) -> bool`, `set_param(name, value)`, `set_mode(name)`, `arm()`, `disarm(force=False)`,
`start_mission()`, `rtl()`, `fly_plan(plan)`; attributes `on_link_lost`, `on_link_restored` (also constructor kwargs),
`reconnect_count`, `home_amsl_m`. Module helpers: `arm_guard(settings)`, `is_simulated_link(url, dry_run)`, `ARDUCOPTER_MODES`.
Errors: `VehicleError` > `LinkTimeout`, `CommandRejected(.result)`, `ArmRefused`.

### Behaviour worth knowing (for main.py / safety.py / webapp)
- Constructor opens nothing; `connect()` opens the link and blocks until the first autopilot HEARTBEAT (else `LinkTimeout`, link closed).
- Link callbacks run in a short-lived daemon thread (NOT the reader thread), so a callback may safely call `rtl()`. They fire on heartbeat age >= `HEARTBEAT_LOSS_S` and when heartbeats resume. RTL policy is main.py's decision.
- Reconnect: read errors trigger reopen after 1 s; on non-`udpin` links a reopen is also tried when the link has been lost for >10 s. A listening `udpin` socket needs none.
- `fly_plan` = heartbeat+GPS(>=3D) precheck -> build -> upload -> verify (raises if mismatch, does NOT arm) -> `AUTO_OPTIONS=3` -> mode AUTO -> arm (guarded) -> MISSION_START. `arm()` waits for the HEARTBEAT to show armed.
- `rtl()` uses the RTL flight mode (confirmed via HEARTBEAT); falls back to MAV_CMD_NAV_RETURN_TO_LAUNCH.
- `build_mission_items` raises `ValueError` for altitude > `HARD_MAX_ALT_M` (120), <= 0, or no waypoints. It does not replicate safety.py's checks (geofence, max path, ...).
- `verify_mission` compares command always, lat/lon (1e-6 deg) and alt (0.05 m) for seq >= 1. Slot 0 (home) is compared by command only because ArduPilot substitutes its real home.
- We send our heartbeat as MAV_TYPE_ONBOARD_CONTROLLER from sysid 245 / compid 191. ArduPilot's GCS failsafe (`FS_GCS_ENABLE`) only watches `SYSID_MYGCS` (default 255) and ignores us; if you ever enable it, either set `SYSID_MYGCS=245` or run a real GCS. Component 191 ids can be changed via the `source_system` kwarg.
- Mock fidelity: arm in AUTO fails unless `AUTO_OPTIONS&1`, takeoff in AUTO waits unless `AUTO_OPTIONS&2`, disarm while airborne fails unless forced (21196), unknown params are silently ignored (like ArduPilot), item 0 downloads as the real home. Fault injection: `mock.drop_inbound["MISSION_ITEM_INT"] = n`, `mock.mute = True` (link loss), `mock.gps_fix = 1`. Telemetry/timing is accelerated with `time_scale`.

## Decisions
- **P4.1 Arm guard.** `arm()` is allowed only if the link is a simulator or `settings.real_arm_permitted`. Simulator = `udp/udpin/udpout/tcp/tcpin` to loopback (127.x, localhost, ::1), or a wildcard listener (`0.0.0.0`) while `DRY_RUN=true` (SITL on a laptop sends to the Jetson, so a wildcard is needed). Serial devices, remote hosts and broadcast never qualify, even with `DRY_RUN=true`. The guard runs before any I/O. Code never writes `ALLOW_REAL_ARM`.
- **P4.2 MAVLink 2 forced.** `os.environ.setdefault("MAVLINK20","1")` before importing pymavlink (in mission.py and the mock) so pymavlink never swaps dialect modules mid-run (it does that on the first v2 packet otherwise, racing the sender threads).
- **P4.3 Single reader thread + one condition variable** guarding telemetry, home and bounded per-type protocol deques. Callers hold an RLock for a whole request/response conversation; replies are filtered by command/param/seq so stale replies are discarded.
- **P4.4 `serial:` prefix.** `MAVLINK_URL=serial:/dev/ttyTHS1` (as in config.py comments) is stripped to the device path for pymavlink (which would treat `serial:` as UDP). Only `udp*`, `tcp*`, `serial:` and `/dev/...` are accepted; a plain file path (pymavlink would open it as a log) is rejected.
- **P4.5 Mission shape.** Item 0 = home placeholder (NAV_WAYPOINT, frame GLOBAL), 1 = NAV_TAKEOFF (frame 6, lat/lon 0 = current position), 2 = DO_CHANGE_SPEED (groundspeed, plan.speed_ms, throttle -1), then NAV_WAYPOINTs (frame 6, yaw param NaN avoided: all zeros so JSON stays valid), last = NAV_RETURN_TO_LAUNCH (frame MISSION).
- **P4.6 AUTO_OPTIONS=3** (bit0 arm in AUTO, bit1 takeoff without throttle) is set per flight by `fly_plan`; it is a persistent vehicle parameter, remember that when switching to manual flying.

## Real-hardware settings (documentation only; nothing was touched)
Pixhawk TELEM2 to Jetson Orin Nano 40-pin header UART (`/dev/ttyTHS1`: pin 8 TX, pin 10 RX, GND pin 6; 3.3 V logic, do NOT connect 5 V):
- `.env`: `MAVLINK_URL=serial:/dev/ttyTHS1`, `MAVLINK_BAUD=921600`, plus a human deciding `DRY_RUN=false`/`ALLOW_REAL_ARM=true` when (and only when) flying for real.
- ArduPilot: `SERIAL2_PROTOCOL=2` (MAVLink 2), `SERIAL2_BAUD=921` (921600), `SERIAL2_OPTIONS=0`; reboot after changing. Stream rates are requested by mission.py, but `SR2_*` can also be set.
- Pixhawk TX/RX cross: Pixhawk TX to Jetson RX (pin 10), Pixhawk RX to Jetson TX (pin 8). Common ground is mandatory; the Pixhawk TELEM2 header is 5 V-powered on pin 1 but the signals are 3.3 V.
- Linux: user must be in `dialout` (already true on this machine). If a serial getty holds the port: `sudo systemctl disable --now nvgetty` (currently inactive on this unit, per Phase 0) and make sure no `serial-getty@ttyTHS1` is running. Test with `pymavlink` heartbeat only, props off.
- Failsafes to configure on the vehicle (not in software here): `FS_GCS_ENABLE` (see heartbeat note above), `BATT_FS_*`, `FENCE_*`, `RTL_ALT`, `FS_THR_ENABLE`, and an RC transmitter in range as manual override.

## PX4 differences (for a future `PX4Backend`)
- Mission item 0 is a normal item, not a home placeholder; there is no seq-0 home slot, and home is set by the vehicle (read HOME_POSITION, or send `MAV_CMD_DO_SET_HOME`). `build_mission_items` must not emit the placeholder.
- Modes are not a flat `custom_mode`: `custom_mode` = main_mode (byte 3) | sub_mode (byte 4) << 8 e.g. AUTO.MISSION = main 4 / sub 4, AUTO.RTL = main 4 / sub 5, POSCTL = main 3. Set via `MAV_CMD_DO_SET_MODE` with param1=1, param2=main, param3=sub. HEARTBEAT decoding needs the same split.
- `MAV_CMD_MISSION_START` exists but PX4 normally starts the mission by switching to AUTO.MISSION (the vehicle must be armed, and takeoff comes from the NAV_TAKEOFF item; there is no AUTO_OPTIONS). Arm first, then set AUTO.MISSION or send MISSION_START.
- Failsafe parameters: `COM_RCL_EXCEPT` (bitmask of modes where RC loss does not trigger failsafe, e.g. Mission = bit 0) must be set to allow flying a mission without a transmitter; also `NAV_RCL_ACT`, `NAV_DLL_ACT` (data link loss action, our heartbeat is MAV_TYPE_ONBOARD_CONTROLLER and PX4 needs a GCS heartbeat for DLL unless `COM_DL_LOSS_T`/`NAV_DLL_ACT` are disabled), and `COM_ARM_WO_GPS`.
- MAVLink on TELEM2: `MAV_1_CONFIG=102` (TELEM2), `MAV_1_MODE=2` (Onboard) or 0 (Normal), `SER_TEL2_BAUD=921600`, `MAV_1_RATE`; reboot. Parameter set uses the same `PARAM_SET`, but PX4 params have real types (INT32 vs REAL32) and the `param_type` field must be correct (PX4 bit-casts INT32 params in PARAM_SET; mission.py currently sends REAL32 which ArduPilot accepts for any numeric parameter).
- RTL is a mode (AUTO.RTL) or `MAV_CMD_NAV_RETURN_TO_LAUNCH`; PX4 climbs to `RTL_RETURN_ALT` first. Arm via `MAV_CMD_COMPONENT_ARM_DISARM`; force-arm param2 = 21196 same as ArduPilot.

## Open issues
- Not tested against real ArduPilot SITL yet (a later agent will build it). Things SITL may expose: `HOME_POSITION` only valid after GPS lock/arming, `SET_MESSAGE_INTERVAL` rates, whether the MISSION_START path needs `current` item handling, ArduPilot's item-0 handling.
- `Telemetry.mission_count` is only updated by our own upload (MISSION_CURRENT.total used when the vehicle supplies it).
- Reconnect logic for serial/TCP links is exercised only by reasoning, not by tests (UDP is connectionless).
