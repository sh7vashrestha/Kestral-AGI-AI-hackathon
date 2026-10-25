# Phase 3 notes (llm_client, planner, survey, safety)

## Progress
- llm_client.py, planner.py, survey.py, safety.py done; tests: test_survey, test_safety, test_llm_client (httpx.MockTransport), test_planner (14 messages, fake LLM end-to-end + `@pytest.mark.live`).
- pytest.ini created (testpaths=tests, marker `live`). tests/conftest.py provides `settings` (explicit, ignores .env: HFOV 90/VFOV 60 so h=20 -> W=40), `settings_factory`, `home`, `live_client` (skips when /health unreachable).
- Live tests NOT yet run (llama-server not up): `.venv/bin/pytest -m live -v`.
- No models.py changes.

## Decisions
- Lanes: n = ceil(width/s); lane centres spread evenly, first at (width/n)/2 inside the edge. Equals s/2 when width is a multiple of s; otherwise spacing < s (never thinner coverage, never outside area). `SurveyPlan.line_spacing_m` reports nominal s = W(1-overlap).
- Rect frame: u along direction_deg from home (near edge at offset_m), v across = clockwise (bearing+90) from centre line; first lane is the left-hand (v<0) lane starting at the near edge. Waypoint `seq` is 1-based over survey legs; mission.py renumbers.
- safety.validate(plan, settings, items=None): SurveyPlan has no RTL (mission.py adds it), so the RTL-at-end check is `validate_mission_items(items, home, settings)` on the final list (also run by validate() if `items=` passed). Items: `(command, lat, lon, alt)` tuples or `safety.MissionItem`; command "RTL" or MAV_CMD 20. Items at lat=lon=0 are treated as position-less (RTL, DO_CHANGE_SPEED).
- Safety recomputes path length from waypoints and uses max(claimed, recomputed); geofence judged on lat/lon.
- Planner LLM schema `LLMIntent` (ok/error + sizes, bearing, offset, altitude, speed, target text); no lat/lon. Missing size -> error; missing target -> "any person"; altitude default from settings.default_alt_m. LLM failures -> Intent(error=...). `update_target_from_message` raises `PlannerError` (user-facing text) since Target has no error field.
- llm_client: new instance per request (no shared connection/loop state), json_schema first then json_object+schema on HTTP 400, retries on timeout/connect/429/5xx and on invalid/empty replies; `reasoning_content` used when `content` is empty; last top-level JSON object wins.
- asyncio tests use explicit `@pytest.mark.asyncio` (strict mode).
