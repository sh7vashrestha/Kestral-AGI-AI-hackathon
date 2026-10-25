# Hardening E: web app and end-to-end

Tests: `tests/test_web_edges.py` (121 tests), `tests/test_e2e_chaos.py` (3, ~60 s), page tests (Playwright, 56 checks,
run by hand, see bottom). Abort latency (time from the phone sending Abort to `controller.abort()` being entered):

| setup | latency |
|---|---|
| TestClient WS, 4 chats stuck on a slow LLM + 150-message ping flood from a third phone | < 300 ms asserted (a few ms measured) |
| real uvicorn + websockets, stuck chats + 300-message flood | 2-10 ms |
| RealController + mock vehicle, upload (confirm) + vehicle check + preflight + chat running | 1.3 ms |

| Edge case | Expected | Result | Fix |
|---|---|---|---|
| WS malformed JSON, non-object, unknown type, missing/typed-wrong fields, NaN corners, 2500-char text | `error` message, socket survives | OK | none |
| WS binary frame | error, socket survives | BUG: `receive_text` raised, the phone was disconnected | `ws.receive()`, "binary frames are not supported" |
| WS message > 1 MB (also 5 MB on real uvicorn) | rejected, socket survives | BUG: no limit | `MAX_WS_MESSAGE_BYTES` check |
| 50+ msg/s flood | rate limit, Abort exempt | BUG: unlimited | token bucket 20/s burst 40 per phone, one "slow down" note/s, Abort never limited; max 6 requests in flight per phone (Abort exempt) |
| Disconnect mid-request / mid slow chat | Abort still completes; no crash | OK | none |
| Slow phone (TCP stuck) | others get messages at once, stuck one dropped after 3 s | OK | none |
| 10 phones; reconnect storm of 40 | all get broadcasts; hub empty afterwards | OK | none |
| Server restart with phones connected | page reconnects and resyncs status | OK (real uvicorn test and browser) | none |
| Abort during slow chat / flood / upload / checks, two phones at once, REST fallback, hung or failing controller | reaches controller < 300 ms; failure says "use the RC transmitter" | OK (see table above) | none |
| REST bad bodies (types, empty, too long, missing, non-JSON, wrong method) | 422/400/405 | OK | none |
| NaN / Infinity in `/api/plan_area` | 422 or an unflyable plan | BUG: NaN made FastAPI answer **500** (it echoes the input into the error JSON) | custom `RequestValidationError` handler; Infinity altitude/speed yields an unsafe plan, never flyable |
| `/api/parse_coords` swapped lat/lon | helpful swap message | OK after fix requested by B | passes `home=` from the controller snapshot; snapshot errors tolerated |
| Traversal on /data, /maps, /tiles, /api/plans, /api/flights (`..`, `%2e%2e`, `%2f`, absolute, `%00`, double-encoded, symlinks out of data_dir) | 404 | OK for /data, /maps, /tiles; plan/flight files followed a symlink out of the data dir | `_file(root=)` resolves and refuses symlinks leaving the plans/flights dir; corrupt plan.json now 500 "unreadable" instead of an exception |
| PMTiles ranges: open-ended, suffix, suffix > size, end beyond EOF, start beyond EOF, junk, multi-range, HEAD | 206/416 with correct Content-Range | OK (multi-range is refused with 416, which maplibre/pmtiles never sends) | none |
| 10 000 alerts in the controller | snapshot bounded | OK: snapshot sends the last 20, RealController keeps `deque(maxlen=50)` | UI: alert list capped at 200 cards, 300 map pins, `S.alerts` capped |
| Missing offline map / data dir / calibration | no crash, `has_map=false`, finite limits | OK | none |
| `/api/client_log` flood | 429 after 30/min; log lines stay single line | OK | none |
| Per-phone chat session (A's request) | each phone has its own draft | `CHAT_SESSION` is set per WebSocket (`ws-<id>`), REST uses optional `session` field else "default" | server.py, test added |
| Page: refresh mid-flight | FLYING state and telemetry restored | OK | none |
| Page: WS drop | clear banner, Abort reachable | BUG: only a small "offline" dot | `#conn-banner` after 1.5 s offline ("Abort still works over HTTP, else RC"); first version never appeared during reconnect retries (timer reset each retry), fixed |
| Page: Abort with WS down, server up | REST fallback reaches controller | OK | none |
| Page: server down then back | banner, reconnect, status resync | OK | none |
| Page: browser offline | page still navigable | OK | none |
| Page: WebGL unavailable | notice, paste still works, Abort visible | OK | none |
| Page: 320 px, 568x320 and 844x390 landscape, all tabs | no horizontal overflow, Abort fully on screen | OK | none |
| Page: 2000-char chat message with a 700-char unbroken word | no overflow | OK | none |
| Page: 60 rapid tab switches | exactly one panel visible | OK | none |
| Page: wizard with garbage/NaN/out-of-range/2-point/24 kB/swapped/emoji paste | toast, no corners | OK; a list-style 422 detail would have shown `[object Object]` | `app.js` formats pydantic detail |
| Page: bench chip, read-back table with mismatches and null fields | renders | OK; latent `null.toFixed` crash possible | `rbDescr`/diff cells tolerate null/NaN |
| Page: 700 alerts | DOM bounded | BUG: unbounded | cap, see above |
| Page: console errors on every screen | none | none (browser network lines for the deliberate 422 and outages are filtered) | none |
| E2E chaos, seeds 1 and 7 (WS drops, refreshes, link mute, LLM failure, flood, garbage, Abort at random moments) | vehicle ends RTL/landed, no ERROR logs, every alert reaches the steady phone, files written | OK | none |

## Not changed / open
- uvicorn's own `ws_max_size` stays at 16 MB (main.py owns the uvicorn config); the 1 MB limit is enforced in the handler.
- Full `pytest -q` run aborted in this session with a native crash from the torch/GPU tests (other agents' load); my files pass alone.
- Page tests live outside the repo (Playwright is not in the project venv): `page_edges.py` starts `python -m webapp.server --fake --port 8768`, drives Chromium (swiftshader), restarts the server mid-run and fails on any console error. Result 56/56. The page exposes `window.__dbg` for them.
