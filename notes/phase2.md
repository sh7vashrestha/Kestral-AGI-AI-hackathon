# Phase 2: hotspot, systemd, web app

## Progress
- hotspot/hotspot.sh {up|down|status}, hotspot.env.example (hotspot.env is git-ignored via hotspot/.gitignore). `bash -n` OK. Never run `up` (it drops client WiFi).
- systemd/: drone-hotspot, drone-llama, drone-brain units + install.sh (not run). `systemd-analyze verify` clean for our units.
- webapp/: controller.py (Controller Protocol + FakeController), server.py (Hub, create_app, --fake dev server), static/{index.html,app.css,app.js}.
- tests/test_webapp.py: 18 tests pass. Fake server started, `/`, `/api/status`, chat/confirm over REST verified; headless Firefox render of the phone page against the fake server checked (status bar, plan card with SVG map, abort bar).

## Decisions
- Hotspot connection is named `drone-brain-hotspot` (not NM's default "Hotspot", which already exists on this machine). Autoconnect yes, priority 100, `ipv4.method shared`, `10.42.0.1/24`. `up` refuses the placeholder password.
- drone-hotspot.service runs as root (nmcli connection changes); llama/brain run as User=drone. Units do not `Requires=` each other (only Wants) so a failed hotspot does not stop the brain.
- Server echoes the user's chat line to all phones as `chat` role=user; every command result is broadcast to all clients, followed by a `status` snapshot. REST endpoints do the same and also return the primary WsMessage.
- Each WS command runs as its own task; abort is never queued behind chat/LLM (tested). Timeouts: chat/target 120 s, confirm/abort 30 s. A failing abort sends an `error` message telling the user to use the RC.
- Snapshot dict contract (documented in webapp/controller.py): telemetry, mission_state, target, alerts (oldest first), plan (pending PlanProposal or null). The UI hides the plan card if status shows a non-"planned" state and plan is null.
- `/data/...` serves only image files (jpg/jpeg/png/webp) under settings.data_dir, resolved path must stay inside data_dir, dotfiles refused.
- Client sends `{"type":"ping"}` every 5 s; server answers with a `telemetry` message (keepalive; client reconnects after 15 s of silence).
- Phone UI: Confirm and Abort both require a second tap in-page (no confirm()); Abort falls back to POST /api/abort when WS is down.
- Optional hooks (duck-typed, not in the Protocol): controller.attach(hub) on app startup and `await controller.detach()` on shutdown.
- No models.py/config.py changes.
