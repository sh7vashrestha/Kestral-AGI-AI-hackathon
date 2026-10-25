# DECISIONS

Decision log for drone-brain. Each entry: what was decided, why, and the evidence.
Per-phase detail lives in `notes/phaseN*.md`; measured raw data in `data/bench/` (not in git).

## Project setup

- **D0.2 Python env.** `.venv` created with `--system-site-packages` to reuse JetPack torch 2.8.0 (CUDA), TensorRT 10.3, OpenCV 4.5.4 (GStreamer=YES), ultralytics 8.4.19 and pymavlink 2.4.49. Only pure-Python web/pydantic deps are pip-installed; never pip-install torch/opencv/numpy/tensorrt.
- **D0.3 Shared contracts first.** `config.py` (the single settings object) and `models.py` (all cross-module pydantic models) were written before any module, so agents could build modules in parallel against one interface. Shared files are add-only.
- **D0.4 Mission area convention (word mode).** Rectangle; near edge centred `offset_m` from home along `direction_deg`; extends `length_m` along the bearing and `width_m` across it.
- **D0.5 Safety switches.** Real arming requires `ALLOW_REAL_ARM=true` *and* `DRY_RUN=false`, both set by a human (`Settings.real_arm_permitted`). `mission.arm()` additionally refuses serial/remote links unless that is true.
- **D0.6 Live LLM tests opt-in.** `pytest.ini` deselects `-m live` by default; run them with `pytest -m live` while llama-server is up. Reason: they take minutes and failed runs blocked the normal suite.

## Memory (7.4 GiB shared CPU/GPU) — the dominant constraint

- **D1.1 Run headless in the field and for LLM work.** With the GNOME desktop, VS Code and Firefox running, only ~3.4–3.7 GB was available and Gemma could not load. `multi-user.target` frees ~1 GB (desktop) and is how the drone will boot anyway. Revert: `sudo systemctl set-default graphical.target`.
- **D1.2 Replace zram with an NVMe swapfile.** JetPack's `nvzramconfig` keeps 6 × 634 MB compressed swap *in RAM*, competing with CUDA. Disabled it; added a 32 GB `/swapfile` on NVMe. Note: swap does not extend GPU memory — CUDA/NvMap buffers must be real RAM — but it lets idle CPU pages move out.
- **D1.3 CUDA allocations fail well below "free".** On this Jetson `cudaMalloc` failed (NvMap error 12) for 24–125 MiB buffers while 3.4 GB was reported free. Only freeing real RAM fixed it. Lesson: budget against `MemAvailable`, not CUDA "free".

## LLM (llama.cpp `llama-server`, one shared instance)

- **D1.4 `--fit off`.** llama.cpp's default auto-fit (`--fit on`) aborts inside `mtmd_get_memory_usage` → `clip_model_loader::load_tensors` (`ggml_backend_buffer_set_usage`) for the Gemma 4 mmproj. Always pass `--fit off` (set in `gemma4-vlm/.env` `EXTRA_SERVER_ARGS`).
- **D1.5 Gemma 4 mmproj contains an audio encoder.** The 941 MiB `gemma-4-E2B-it-mmproj.gguf` holds vision (gemma4v, 16 layers, 224 px, patch 16) *and* audio encoders. On the GPU it OOMs; on the CPU (`--no-mmproj-offload`) it works. A vision-only copy (`scripts/strip_mmproj_audio.py`, original untouched) also works on CPU.
- **D1.6 Planner model = Gemma 4 E2B, text-only (no mmproj).** Live planner tests (14 messages + target update, 2 runs each):

  | Model | Live tests passed | Suite time |
  |---|---|---|
  | | Gemma 4 E2B Q4_0 QAT, text-only | Gemma 3 1B (unsloth Q4_K_M) |
  |---|---|---|
  | Live planner tests, run 1 / run 2 | **16/16, 16/16** | 12/16, 13/16 |
  | Planning call latency | 2.3–4.7 s (mean ≈ 3.1 s) | — |
  | Speed | 32.2 tok/s | 38.0 tok/s |
  | Target → CLIP prompt JSON | 5/5 valid, 4.3 s | 5/5 valid, 3.4 s |
  | RAM after load (used / available) | ≈ 5.4 / 1.5–1.9 GB* | ≈ 3.1 / 4.2 GB* |

  \*measured while other agents' GPU jobs were running. Gemma 3 1B got bearings wrong, copied the example altitude and missed a refusal (safety-relevant). Agreed rule (≤ 1 extra failure) → **E2B**. Memory breakdown: CUDA model buffer 1342 MiB + 2152 MiB CPU (per-layer embeddings, pageable) + KV 54 MiB + compute 126 MiB at ctx 4096. Gemma 3 1B stays downloaded (`gemma4-vlm/models/gemma-3-1b/`) as a commented fallback in `gemma4-vlm/.env`.
- **D1.8 Disable Gemma 4 "thinking" per request.** By default Gemma 4 writes a "Thinking Process" before answering: 112–145 tokens and ~4 s for a trivial JSON reply. `chat_template_kwargs: {"enable_thinking": false}` cuts that to ~12 tokens and 0.5 s; `reasoning_budget: 0` and `reasoning_format` had no effect. `planner.py` sends the kwarg on every call; no server flag needed.
- **D1.9 One slot, start the LLM first.** `-np 1` (single slot): with the default 4 slots the CUDA buffer allocation failed whenever another process held GPU memory. Even then, starting llama-server while YOLO/CLIP exports run can fail, so llama-server starts **before** other GPU jobs (systemd order: hotspot → llama-server → drone-brain). `MMPROJ_FILE="none"` starts text-only (guard added in `gemma4-vlm/common.sh` and `run_server.sh`).
- **D1.7 LLM never emits coordinates.** Planner output is intent JSON (sizes, bearing, altitude, target) constrained by `response_format` JSON schema and validated with pydantic; Python builds waypoints; safety checks and a human Confirm gate every upload. Corner coordinates from the map/paste come from the human and are parsed by Python (`webapp/coords.py`), never by the LLM.

## Matching a person to the target description

- **D6.1 Gemma vision is not the per-crop matcher.** Vision mmproj on CPU (audio stripped), target "person wearing a red jacket": JSON valid 100 %, is_human 6/6, but it answered *matches* for 5/6 crops including a white shirt, a green top, a light jacket and a black suit → poor discrimination. **Latency for a never-seen crop ≈ 59 s** (p50 59 s, range 51–66 s, n = 11, from the server log). The 1.8 s figures in `data/bench/vlm_benchmark.json` are for crops already in llama-server's prompt cache and must not be used for sizing. With Δt = 1 s, vision can only ever be an occasional confirm step.
- **D6.5 If the optional VLM confirm is enabled:** `VLM_QUEUE_SIZE=2` (max backlog ≈ 2 min), one request in flight, keep best crop per track, drop oldest, `LLM_TIMEOUT_S ≥ 120` for image calls (60 s is fine for text).
- **D6.2 CLIP is the primary matcher; backends are pluggable.** `MATCH_BACKEND=clip|vlm`. CLIP scores a crop against positive/negative prompts (softmax over 100 × cosine); ~10–30 ms per crop expected on GPU, ~0.2 GB. Gemma (text) turns the free-text target into CLIP prompt lists. Optional `VLM_CONFIRM_TOP` sends a track's best crop once to Gemma vision for a human-readable reason (off by default; needs the mmproj loaded).
- **D6.3 Scoring.** `C_match = match_confidence` if human and matches else 0; `C_final = C_person × C_match`; `C_avg` = mean of the track's last 5 `C_final`; "target match" alert when `C_avg ≥ MATCH_THRESHOLD` (0.7) over ≥ `MIN_FRAMES` (2). "Person" alerts fire immediately, once per track. Target update resets match history and discards in-flight results.
- **D6.4 Bounded VLM/match queue.** One pending job per track (best `c × area` crop), at most `VLM_QUEUE_SIZE`, drop oldest; producers never block.

## Sensors (verified on hardware 2026-10-19)

- **D5.1 RGB camera = CSI IMX477** (`nvarguscamerasrc sensor-id=0`, overlay `imx477-A`; modes 3840×2160@30 and 1920×1080@60, the latter 2×2-binned). The earlier USB Logitech C930e is no longer attached. `CAMERA_SOURCE=csi:0`. Optics for the 1080p mode: active width 5.952 mm (3840 × 1.55 µm). **Lens unknown — assumed 6 mm** (HFOV ≈ 52.8°, VFOV ≈ 31.2°, ~20 m ground width at 20 m AGL). Must be confirmed by the user and set in `.env`.
- **D5.2 First "noise" frames were darkness, not a fault.** A grey speckled frame at max auto-gain; a fixed low-gain frame had mean 19.7/255 → no light (lens cap). After uncovering: sharp image.
- **D5.3 Thermal = PureThermal + FLIR Lepton 3.x, radiometric.** USB UVC `/dev/video0`, Y16 160×122 (last 2 rows telemetry) → °C = raw/100 − 273.15. Measured room ~20–21 °C, body 36–39 °C. **Stuck hot pixel** at the same position in every frame → masked. Assumed HFOV 57°.
- **D5.4 Fuse thermal with YOLO** (`fusion.py`): heat boosts person confidence `C_person = 1 − (1 − C_yolo)(1 − w·C_heat)`; heat-only blobs trigger a low-threshold YOLO re-check of that region, else a 🔥 "heat" alert; dark scenes switch to thermal-only. Thermal↔RGB alignment via a homography from `scripts/calibrate_thermal.py` (default: FOV-ratio scaling). Cameras must be mounted nadir side by side.

- **D5.5 CLIP model = OpenCLIP ViT-B/32 (laion2b).** Torch FP16: 17 ms per crop (batch 1), 29.5 ms (batch 4), ~350 MB GPU. TensorRT FP16 engine (183 MB): 4.5–8 ms, but it often fails to deserialize under memory pressure → `ClipEncoder` retries, then falls back to torch FP16 for the rest of the run. MobileCLIP not tried (needs timm + FastViT export). Accuracy on `tests/assets/match/labels.json` with default prompts: ranking good (mean per-target AUC 0.94) but only 90/144 (0.62) correct at the 0.5 cut-off → prompts and cut-off (`CLIP_MATCH_CUTOFF`) must be tuned before relying on "target match".
- **D5.6 YOLO26n TensorRT FP16** exported on the Jetson (~8 min) to `models/yolo26n.engine`: 22 ms mean / 25 ms p95 per 1080p frame at imgsz 640.
- **D5.7 Pipeline budget fits Δt = 1 s with a large margin:** YOLO + fusion + 1 CLIP crop ≈ 100 ms, with 3 crops ≈ 140 ms (measured with llama-server resident and other jobs loading the CPU). RGB 30 fps, thermal 8.5–8.7 fps.
- **D5.8 GPU allocation order.** Create `ClipEncoder` and `Detector` before starting the camera, and start llama-server before both. Large CUDA allocations (> ~100 MB) failed intermittently with NvMap error 12 when RAM was full or fragmented (TensorRT OOM, CUDA error 700, camera dying in the same process). `drop_caches` would help but needs root.
- **D5.9 Extra pip packages installed with `--no-deps`** (torch untouched, still 2.8.0 CUDA): onnx, onnxslim, onnxruntime, open_clip_torch, huggingface_hub 0.36.0, safetensors, ml_dtypes, a newer protobuf in the venv. Ultralytics auto-installed its `clip` package into `~/.local`.

- **D5.10 Camera orientation is a setting applied at capture.** `CAMERA_ROTATE=180` (user: line the main camera up with the thermal sensor), `CAMERA_FLIP=none`, `THERMAL_ROTATE=0`, `THERMAL_FLIP=none`; rotation is clockwise and applied before the flip. CSI uses the hardware `nvvidconv flip-method` (mapping measured on the real camera, all 12 combinations verified, `orient.py`); other sources use cv2. All optics use effective values (`eff_hfov_deg`, …): at 90/270 the frame is 1080×1920 and W/L swap (100×60 m at 20 m: 5 lanes at 0/180 vs 8 lanes at 90/270). The image's horizontal axis is across-track; the drone flies "up" the image. Saved thermal calibrations record the orientations and are ignored if they change.
- **D5.11 CSI capture runs in a child process.** libargus keeps its daemon connection inside the process, so after an `nvargus-daemon` restart no in-process retry can reopen the camera. `camera_worker.py` sends frames over a pipe; each retry is a fresh process (backoff 0.5–30 s, also on a 5 s stall).
- **D5.12 Clean shutdown everywhere** (`shutdown.py`, used by main.py, live_view, system_check): on SIGTERM/SIGINT stop the web server, stop both cameras (worker SIGTERM, SIGKILL after 2 s), force-exit by ~3 s. Open MJPEG streams had been blocking uvicorn's graceful shutdown, which left cameras held and wedged `nvargus-daemon`.

## Vehicle

- **D4.1 Pixhawk 6X runs ArduPilot ArduCopter** (USB by-id `ArduPilot_Pixhawk6X`, HEARTBEAT autopilot=3, type=2 quad). Bench: STABILIZE, disarmed; on USB it sends only HEARTBEAT/TIMESYNC until streams are requested. TELEM2 (`/dev/ttyTHS1`) showed 0 bytes — wiring/`SERIAL2_PROTOCOL` not yet set up; USB `/dev/ttyACM0` used for bench work.
- **D4.2 Development talks only to the mock vehicle / SITL.** The real Pixhawk is touched only by a read-only mode (`VEHICLE_MODE=readonly`: receive, request streams, read params; refuses upload/mode/arm/param-set before any I/O).
- **D4.3 Home always comes from the Pixhawk** (`HOME_POSITION`, live). If home moves > 5 m after planning, the plan is rebuilt and re-checked; Confirm re-verifies the plan's home against the current home. No GPS fix → no planning.
- **D4.4 Altitudes are relative to home** (`MAV_FRAME_GLOBAL_RELATIVE_ALT_INT`). Arm on the field so home = field level. Effective ceiling = `min(MAX_ALT_M, LAANC_CEILING_M, 120)`; minimum = `max(MIN_ALT_M, OBSTACLE_CLEARANCE_M=15)`. Pre-flight reads `RTL_ALT`, fence and failsafe params and blocks Confirm on mismatch (details: `notes/phase7-requirements.md`).

## Planning UI and maps

- **D2.1 Area by 4 tapped corners on an offline map** (plus "use drone position" and "paste coordinates"); word mode kept as the Chat tab. Phone browser geolocation is unavailable (plain http), so corners come from the map, the Pixhawk GPS, or pasted text.
- **D2.2 Polygon survey.** Convex hull of the corners (concave → hull + warning; self-intersecting → rejected); lanes along the longest edge, spacing `W(1 − overlap)`, first lane half a spacing inside, clipped to the polygon; of the 4 zig-zag orders the shortest home→route→home wins.
- **D2.3 Offline basemap = Protomaps PMTiles extract** (OSM/ODbL) rendered by vendored MapLibre GL 4.7.1 + pmtiles 3.2.1; no text labels, so no glyphs needed. Satellite only from a user-supplied `.mbtiles`; nothing bulk-downloaded from imagery providers.
- **D2.4 Flying site = KSU Marietta campus soccer field** ("Nuesoft Technologies Field", OSM way 180244662, centre 33.9371904, −84.5271293). Map extract radius 1.5 km, 1.5 MB (data to z15, overzoomed). Dobbins ARB is nearby: LAANC authorization and KSU UAS approval required; legal checklist shown before the first Confirm.

## Hardening (2026-10-20; tables in `notes/hardening-A..E.md`)

- **D10.1 Gemma is a multi-turn planning assistant** (`assistant.py`): one action per turn (`plan | ask | reject | update_target | explain | suggest_abort | smalltalk`), one question when information is missing, a draft per phone. Python owns units, sizes, distances, injection/arm/abort phrases and all numbers; chat can only *suggest* Abort.
- **D10.2 Automatic altitude from target detail:** highest whole metre giving ≥ 80 / 120 / 160 px (person / clothing colour / fine detail), clamped to [obstacle floor, ceiling] — ≈ 41 / 27 / 20 m with the assumed lens. An operator-requested altitude inside the limits is kept.
- **D10.3 CLIP prompts and thresholds tuned for precision:** contrastive negatives (other colours/sleeves), no generic person positive; `CLIP_MATCH_CUTOFF=0.6`, `MATCH_THRESHOLD=0.7`, `MIN_FRAMES=3`, `MATCH_STRONG_THRESHOLD=0.93`, `MATCH_MIN_PX=20`. Alert precision 0.81 → 0.95, recall 0.34 → 0.22 on degraded ground photos (no real aerial data yet).
- **D10.4 Multi-frame speed limit:** speed = min(requested, L(1−overlap)/Δt, L/(MIN_FRAMES·Δt)), so each ground point is seen in MIN_FRAMES frames (≈ 3.7 m/s at 20 m). The specified no-gap rule is unchanged; this only ever slows the drone.
- **D10.5 Pilot takeover = `manual` state:** leaving AUTO (pilot or failsafe) → alert, the app sends nothing, Abort tells the operator to use the RC, new plans blocked until disarm. Abort/link loss cancel an upload before it can arm; a second Confirm is refused while one is in progress; upload/clear refused while armed in every mode.
- **D10.6 Bench home** exists only in upload mode, is labelled everywhere, is recorded as `home_source: "bench"`, and a real Pixhawk home always replaces it.
- **D10.7 Web limits:** 1 MB per WebSocket message, 20 msg/s (burst 40) and 6 in-flight requests per phone; Abort is never limited. Alert list capped at 200 cards / 300 map pins.

## Upload → review → GO (2026-10-20; contract `notes/go-flow.md`)

- **D11.1 Confirm no longer flies (sim/full).** Confirm = every old check + upload + read-back verify → state `staged` (disarmed, no mode change). A separate human **GO** (`POST /api/go`, WS `go`) re-runs the readiness checks — including a fresh mission download compared with the staged items — and only then does `start_staged`: verify again → `AUTO_OPTIONS=3` → AUTO → arm (arm guard unchanged) → MISSION_START. `upload`/`readonly` modes are unchanged and never GO.
- **D11.2 Readiness reuses `vehicle_check.run_vehicle_check`** (findings tagged per item id; light run every 5 s = telemetry + SYS_STATUS pre-arm bit, deep run at staging / refresh / GO / every 30 s = + RUN_PREARM_CHECKS, param reads, mission re-download). Every item is blocking except `vision`; an extra `rc` item (RC_CHANNELS received) is blocking because the RC transmitter is the in-flight abort when the phone is out of WiFi range. A real (non-simulated) link fails `arm_guard` until a human sets `ALLOW_REAL_ARM=true` in `.env`.
- **D11.3 Staged plan rules:** Abort while staged → `planned`, nothing sent (vehicle mission left as is); a new plan / home re-plan / mission clear while staged is refused until Abort; home moved too far → readiness `home` fails (no silent re-plan); staged > 15 min → `planned`; armed by RC/GCS while staged → `manual`. Abort while `arming` cancels `start_staged`; already armed → RTL.
- **D11.4 Quiet flight, summary after landing.** Informational system chat in `arming/flying/rtl` goes to the flight record (`telemetry.jsonl` `note` events), not the phones; errors, replies and alerts still go out. After a flight that armed, `flight_summary.py` asks Gemma (text only, thinking off, ≤ 200 words, 60 s timeout → template) in a background task; `summary.md` + `summary.json` are saved, pushed once (`flight_summary`) and kept in the snapshot / `GET /api/flights/last/summary`, reloaded on restart. Up to 6 alert crops (best `C_avg` per track) with captions built in Python.

## Process

- **D9.1 Work split across parallel agents** against the shared contracts; the coordinator merges notes into this file and `PROGRESS.md`, checks cross-module compatibility (e.g. `safety.validate_mission_items` had to accept `models.MissionItem`, fixed in 81b1538) and commits.
- **D9.2 Decide before delegating.** Re-scoping a running agent four times (vision → CLIP → Gemma 4 only → smaller model) made Phase 1 slow. Model/approach choices are now settled before an agent starts.
