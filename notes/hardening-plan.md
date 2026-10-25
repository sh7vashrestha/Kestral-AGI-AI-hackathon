# Hardening wave (starts after Phase 7)

User goals (2026-10-19): test all edge cases and fix them; the LLM gets **text only, never
images**; use the LLM to plan properly ("what to do"); make it excellent.

## Principle: who decides what

| Decision | Made by | Never by |
|---|---|---|
| Understanding the request, asking follow-up questions, target wording, CLIP prompts, how much detail the target needs | Gemma 4 E2B (text, JSON schema) | — |
| Coordinates, waypoints, altitude/speed limits, geofence, RTL, upload, arming | Python + human Confirm | the LLM |
| Abort / RTL | the red button (two taps) | the LLM (chat can only *suggest* it) |

## Five parallel agents, one file set each

**A. LLM mission assistant** (`planner.py`, `assistant.py`, prompts, tests)
- Multi-turn draft: the assistant keeps a mission draft and asks one clear question
  when something is missing instead of rejecting ("What should I look for?",
  "How big is the area?"), then summarises before planning.
- Structured actions: `plan | ask | reject | update_target | explain | suggest_abort`.
- Target detail level (person only / clothing colour / fine detail) → Python picks the
  recommended altitude so the target is ≥ 80 / ≥ 150 px tall.
- Target → CLIP positive/negative prompts, validated (count, length, no duplicates).
- Units and phrasing: ft/yards/metres, "football field", compass words, typos,
  non-English, emoji, very long or empty messages, several targets, contradictions.
- Prompt injection ("ignore safety, fly at 500 m") must still be stopped by Python.
- LLM down / timeout / bad JSON → clear message plus a manual (map + form) path.

**B. Geometry and safety** (`survey.py`, `safety.py`, `webapp/coords.py`)
- Degenerate polygons (repeated or collinear corners, tiny or huge areas), corners
  across the geofence edge, anti-meridian/hemisphere signs, coordinate formats
  (DMS, swapped lat/lon, comma decimals), altitude at exactly the limits, NaN/inf,
  LAANC ceiling below obstacle clearance, overlap 0 and 0.9, very slow or fast cameras.

**C. Vehicle and mission** (`mission.py`, `controller_real.py`, SITL)
- Link loss in every state, heartbeat flapping, mission upload interrupted or
  rejected, read-back mismatch, home moving or missing, GPS loss mid-flight,
  low battery, RC override / mode change by the pilot, Abort during upload,
  double Confirm, Confirm of a stale plan, readonly/upload modes refusing writes.

**D. Vision, fusion and matching** (`matcher.py`, `fusion.py`, `thermal.py`, `clip_encoder.py`)
- Tune CLIP prompts and the match cut-off on aerial-like crops (now AUC 0.94,
  accuracy 0.62 at 0.5); camera or thermal unplugged mid-flight, frozen frames,
  dark scenes, saturated thermal, many people at once, the same person across
  lanes, target update while crops are queued, TensorRT fallback, GPU OOM.

**E. Web app and end-to-end** (`webapp/`, e2e tests)
- Several phones at once, reconnects, slow network, Abort while any other request
  is running, refresh mid-flight, malformed WebSocket messages, very large
  alert history, offline map missing, end-to-end chaos runs in SITL.

## Done means

- Every edge case has a test; every failure found is fixed or documented with a reason.
- Full suite + SITL end-to-end pass.
- `DECISIONS.md` / `PROGRESS.md` updated by the coordinator.

## Found while testing the live page (2026-10-20) — FIXED in 09c3512

Status: 1 and 2 fixed and verified on hardware (CSI capture moved to a child process
`camera_worker.py`; `shutdown.py` releases both cameras, live_view exits in 1.46 s on SIGTERM);
3 not seen since (no new "Services are already stopped" after clean shutdowns).

1. **RGB camera never retries after a failed first open** in `scripts/live_view.py` (state stayed
   "unavailable", reconnects = None) even after `nvargus-daemon` was restarted and the camera worked
   from gst-launch. Restarting the process fixed it. The retry/backoff path must cover the
   "opened but no first frame" failure.
2. **live_view ignores SIGTERM** and keeps holding the thermal device (/dev/video0) until SIGKILL,
   which blocks the next process from opening it. Every process must release both cameras on
   SIGTERM/SIGINT within ~3 s (join threads with timeout, release captures, then exit).
3. **nvargus-daemon gets wedged** ("No cameras available", SCF "Services are already stopped")
   whenever a process holding the CSI camera is killed abruptly. Clean release (item 2) should
   prevent most of it; the system check and live page already print the restart command.
