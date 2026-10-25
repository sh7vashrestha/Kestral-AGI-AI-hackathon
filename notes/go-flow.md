# Upload → Review + checks → GO (contract, 2026-10-20)

Operator flow requested by the user: (1) upload the flight plan, (2) the human reviews it
(phone overlay / Mission Planner) while drone-brain runs every essential check, (3) a separate
human **GO** arms and starts. Applies to `--vehicle sim` and `--vehicle full`. `upload` mode is
unchanged (upload only, never GO). `readonly` is unchanged.

Safety rules that do not change: real arming still needs `ALLOW_REAL_ARM=true` + `DRY_RUN=false`
set by a human (`mission.arm_guard`). Nothing here sets it. Abort/RTL is always available.

## States (controller_real.Controller.state)
`idle → planned → uploading → staged → (GO) arming → flying → rtl → landed`
- `staged` (NEW): mission uploaded + read-back verified, vehicle disarmed, waiting for GO.
  Not in BUSY_STATES for chat purposes, but a new plan / plan_area while staged replaces the
  staged plan only after the operator Aborts it (reply tells them so).
- `arming` (NEW): GO accepted, final checks + AUTO + arm + MISSION_START in progress. In
  BUSY_STATES. Abort here = `cancel_flight()`; if already armed → RTL (same as old "uploading").

## Confirm (sim/full) now = stage 1 only
Everything `_confirm_locked` already checks still runs, then `upload_for_review` (mission +
read-back verify). Verify OK → state `staged`, `self.staged = {plan_id, items, uploaded_ts,
home_version}`. Verify failed → state `planned`, error message. No arming, no mode change.

## Readiness (runs while `staged`)
Background task every `READINESS_INTERVAL_S` (5 s) + immediately after upload and on
`POST /api/readiness/refresh`. Snapshot field:
```
"readiness": {
  "ts": float, "running": bool, "go_allowed": bool, "go_reason": str,   # reason = first blocking failure, "" if allowed
  "items": [ {"id": str, "label": str, "status": "ok"|"warn"|"fail"|"pending",
              "detail": str, "blocking": bool} ]
} | null          # null when not staged
```
Item ids (all blocking unless noted): `link` (heartbeat), `disarmed`, `gps` (fix ≥3, sats, hdop),
`ekf`, `prearm` (SYS_STATUS prearm bit / PreArm STATUSTEXT), `battery` (voltage/%, warn below
`BATT_WARN`), `fence` (FENCE_* params), `failsafe` (BATT_FS_LOW_ACT, FS_THR_ENABLE), `home`
(unchanged since upload, drone within home_shift_replan_m), `mission` (vehicle mission still
equals the staged items — re-downloaded on refresh and at GO; a Mission Planner edit = fail),
`geofence_alt` (plan alt ≤ ceiling), `arm_guard` (ALLOW_REAL_ARM/DRY_RUN; on a real link
fail with "a human must set ALLOW_REAL_ARM=true in .env"), `vision` (warn only), `legal` (ack).
Reuse `vehicle_check.run_vehicle_check` / `preflight_check` — do not duplicate logic.
Bench home (upload mode only) never reaches here.

## GO
`POST /api/go {"plan_id": str}` and WS `{"type":"go","plan_id":...}` → `{"message": str}`.
Refuses unless: state `staged`, plan_id matches, fresh readiness run (re-run inside GO, incl.
mission re-download/compare) has `go_allowed`. Then backend `start_staged(items)`:
verify_mission → AUTO_OPTIONS=3 → AUTO → arm → MISSION_START, with the existing
`_check_cancel` points. Failures behave like the old fly_plan path (not armed → back to
`staged`/`planned`; armed → emergency RTL). `fly_plan` stays for existing callers/tests.

Abort while `staged` → state `planned` (plan kept, vehicle mission left as is), nothing sent.
Staged plan expires after `STAGED_MAX_AGE_S` (900 s) → `planned`, must re-confirm.
Vehicle armed by someone else while `staged` → `manual`, readiness null.

## In-flight text (user: "no need to broadcast the text")
During `arming/flying/rtl` informational `_chat("system", …)` messages are not broadcast;
they go to the flight record (events). Safety errors (`_err`), Abort/GO replies and target
alerts are still sent. Status/telemetry WS messages are unchanged.

## Flight summary by Gemma
On `_finish_flight` (landed / disarmed / aborted) `flight_summary.summarize(flight_dir, llm)`
runs as a background task: builds a compact text digest of flight.json (plan, duration,
distance, max alt, alerts with confidences/positions, mode changes, link losses, end reason)
and asks Gemma (text only, thinking off, ≤ 200 words, plain text). Saved as
`<flight_dir>/summary.md`; broadcast once as WS `{"type":"flight_summary","data":{"text",
"flight_dir","source":"gemma"|"template"}}`. LLM unavailable/timeout (60 s) → deterministic
template text, source "template". Never blocks or delays RTL/safety. Snapshot gets
`"last_flight_summary": {...} | null`. `GET /api/flights/last/summary` returns it.

## Update (user, 2026-10-20): summary is fetched after landing over WiFi, with images
The phone is expected to be OUT of WiFi range during the flight; the operator reconnects after
landing. So the summary must be persistent and pulled on reconnect, not only pushed once:
- `last_flight_summary` (snapshot, and `GET /api/flights/last/summary`) =
  `{"text", "flight_dir", "source", "ts", "images": [{"url", "caption", "conf", "lat", "lon", "t"}]}`.
- `images`: up to 6 of the flight's saved alert crops/frames (already written by the pipeline/storage
  under the flight dir), best `C_avg` first, one per track, served via the existing `/data/...` route.
  Captions are built in Python from alert metadata (confidence, position, time), NOT by Gemma.
- Gemma is TEXT ONLY: it gets the flight.json digest (alert list as text), never pixels.
- On every WS (re)connect the server already sends a status snapshot; the UI shows the summary card
  from `last_flight_summary` if it is newer than what this phone has seen (localStorage ts, try/catch).
- Safety note for the operator text: out of WiFi range the phone Abort cannot reach the drone; the RC
  transmitter (RTL switch) is the in-flight abort.
