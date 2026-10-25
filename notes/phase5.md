# Phase 5 - cameras, thermal, YOLO, CLIP, fusion

## Progress
- camera.py (csi/v4l2/file, latest-frame, reconnect+backoff, open timeout), thermal.py (Y16 radiometric, stuck-pixel
  masks, hotspots, colorize), detector.py (YOLO26n TRT FP16), clip_encoder.py (OpenCLIP ViT-B/32), fusion.py
  (homography, fusion, heat-only alerts, dark mode), scripts/{camera_test,calibrate_thermal,export_models}.py, tests.
- Real hardware verified: IMX477 1920x1080 ~30 fps, PureThermal 160x120 ~8.6 fps, YOLO engine exported on device.
- NOT done: real thermal<->RGB calibration (needs the two cameras mounted side by side, see procedure); webapp heat badge
  (scope moved to another agent; Alert.level now accepts "heat").

## Public API for main.py
```
cam = Camera(s).start(wait_s=20); tc = ThermalCamera(s).start()
clip = ClipEncoder(s)          # create FIRST (see memory issue), then Detector
det = Detector(s)              # models/yolo26n.engine if present else .pt
fuser = Fuser(s, load_homography(s), detector=det, storage=storage, stuck_mask=lambda: tc.mask)
for frame, ts, fid in cam.frames():                     # latest frame every 1/capture_fps s
    temp, tts = tc.nearest(ts)                          # (None, 0.0) if older than THERMAL_MAX_AGE_S
    res = fuser.process(frame, ts, fid, telemetry, temp, tts)   # runs det.detect itself unless detections= given
    matcher.submit(frame, res.detections)               # c_person is used by matcher when present
    for a in res.alerts: on_alert(a)                    # level="heat" (rate limited per ground position)
```
`Detector.detect(frame, ts, frame_id, telemetry)` / `detect_roi(frame, roi_bbox, ..., conf)`;
`find_hotspots(temp_c, s, alt_m=None, mask=None)`; `colorize(temp_c)`.

## Decisions
- Model additions only: `Detection.c_heat`, `Detection.c_person` (default None), `HotSpot`, `Alert.level` += "heat".
  matcher.py: tiny `_conf(det)` = `c_person` if set else `c_yolo`, used for the threshold filter, job score, person alert
  score and the C_final product (C_final = c_person * C_match). Tracker/ground logic unchanged. All old tests pass.
- Heat alert: Alert(level="heat", track_id>=1_000_000, c_yolo=0, c_final=hotspot score, image = [thermal crop | RGB crop]).
- c_heat = clamp((delta-2)/8) of the mean of the 3 warmest pixels (<= THERMAL_MAX_C) in the projected YOLO box;
  C_person = 1-(1-C_yolo)(1-w*C_heat), w = THERMAL_WEIGHT 0.5. Thermal older than THERMAL_MAX_AGE_S is ignored entirely.
- Hotspot = connected component of (T - 31x31 local median) >= 4 C, peak in 24..42 C, area within altitude-scaled person
  range (0.3-1.8 m via thermal HFOV 57 deg); no altitude -> 2 px .. 40% of image.
- Stuck pixels: static `THERMAL_STUCK_PIXELS=67,64` (x,y) + dynamic StuckPixelDetector (EMA of "isolated single pixel
  deviating > 5 C from its 3x3 median"; flagged after >=40 frames above 0.8). It re-detected (x=67,y=64) on its own.
  Masked pixels are median-filled; find_hotspots always applies the static list, so a stuck pixel can never be a hotspot.
- Dark mode: mean gray < RGB_DARK_THRESHOLD (25) -> YOLO/CLIP skipped (detections=[]), only heat alerts; transitions logged.
- CLIP: OpenCLIP ViT-B/32 laion2b_s34b_b79k (MobileCLIP needs timm + TorchScript FastViT export - not attempted).
  Crops are padded to a square with grey instead of centre-cropped (mean AUC on match labels 0.94 pad vs 0.91 crop).
  Image tower: TensorRT FP16 engine `models/clip_image.engine` (batch 1-4) if it loads, otherwise (automatic fallback,
  also after a runtime TRT failure) torch FP16 on CUDA. Text tower: torch FP16, strings cached.
- Engines/weights live in `models/` (gitignored). `scripts/export_models.py yolo|clip|all` (YOLO build ~8 min, CLIP ~3 min
  on this Orin with llama-server resident).
- Extra pip (all `--no-deps`, torch/numpy/opencv/tensorrt untouched, torch still 2.8.0 + CUDA): onnx, onnxslim, onnxruntime,
  open_clip_torch, huggingface_hub==0.36.0, safetensors, ml_dtypes, venv-local protobuf 7 (system one too old for onnx).
  ultralytics auto-installed `clip` (ultralytics/CLIP) into ~/.local when first importing text_model - harmless.

## Measurements (Orin Nano 8 GB, llama-server resident, other agents' jobs loading the CPU, so pessimistic)
- RGB IMX477 1920x1080: ~30 fps from the GStreamer pipeline (camera_test: 20 frames in 0.8 s, mean brightness 62 indoors).
- Thermal: 8.5-8.7 fps (native ~9), room 21-23 C, stuck pixel found.
- YOLO26n TRT FP16, 1920x1080 frame, imgsz 640: 22 ms mean / 25 ms p95 per frame (includes letterbox + NMS-free decode).
- CLIP ViT-B/32 per crop (incl. CPU resize/pad): torch FP16 17 ms (batch 1), 29.5 ms (batch 4); raw TRT FP16 engine
  4.5-8 ms (batch 1), 8 ms (batch 4), 12.8 ms (batch 8 profile, now max 4) when it loads. Torch GPU memory ~320-350 MB
  (text 126 MB + image tower); engine file 183 MB.
- Fusion (hotspots + projection, 160x120): ~5 ms alone, 24 ms mean in the loaded run.
- Pipeline per frame (YOLO + fusion + CLIP x1) ~100 ms; with 3 CLIP crops ~140 ms -> comfortably fits 1 fps.
- Memory during the 30 s run (tegrastats, whole system): RAM 2.4-7.0 GB of 7.6 GB (llama-server alone is ~4 GB RES);
  GR3D avg 25 %.
- CLIP accuracy on tests/assets/match/labels.json with matcher.default_prompts (positive prompt + 5 generic negatives,
  threshold 0.5): 90/144 = 0.62 per-decision; ranking is good: mean per-target AUC 0.94 (red jacket 1.00,
  long sleeve 0.89, hat 0.91, green top 0.90, white short sleeve 1.00). Threshold/prompts need tuning (planner's
  ClipPrompts with better negatives should help); do not rely on the 0.5 cut-off.
- Sanity prompts "a person wearing a white shirt"/"dark jacket": white-cap runner 0.98 white, black suit 0.97 dark.

## Known issue: GPU memory on this board (important for main.py)
- Jetson CUDA allocations of >~100 MB can fail with `NvMapMemAllocInternalTagged error 12` when system memory is
  fragmented (tegrastats lfb ~4 MB) or llama-server + other jobs fill RAM. Symptoms seen: TRT engine
  "cannot deserialize", Myelin "OutOfMemory 170980096", CUDA error 700, and the CSI camera dying
  ("IImageNativeBuffer not supported", then the process aborted) while a CLIP engine load was failing.
- Mitigations in code: retries with sleep on load, TRT runner pre-allocates its max-shape workspace at load, any TRT runtime
  error permanently falls back to torch FP16, non-finite outputs raise. Create ClipEncoder and Detector BEFORE starting the
  camera, and keep the text tower on the GPU before creating the TRT context.
- Practically the CLIP image tower often ends up in torch FP16 (17 ms) - fine for 1 fps. Ask the user to run
  `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` before flight if TRT loading fails repeatedly (we cannot).

## Thermal <-> RGB calibration procedure (scripts/calibrate_thermal.py)
1. Mount both cameras rigidly side by side, nadir (or looking at a plain wall from ~1-3 m; the homography is exact only
   for a plane, parallax at flight altitude of 20 m+ is negligible for a baseline of a few cm).
2. `.venv/bin/python scripts/calibrate_thermal.py --auto --points 8`: keep the object out of view for 3 s (RGB reference),
   then move a hand/hot mug to 8 different places spread over the image (including near the corners) when prompted,
   keep the rest of the scene static and cool. RGB point = centroid of frame-difference blob, thermal = hottest blob.
3. RANSAC homography, per-point error and RMS printed (RGB px), saved to `data/calibration/thermal_to_rgb.json`
   (stores the RGB size; rescaled if the RGB resolution changes). Check with `scripts/camera_test.py` (overlay.jpg).
4. Manual alternative: `--pairs pairs.json` with {"pairs":[{"thermal":[x,y],"rgb":[x,y]},...]}; `--reset` goes back to the
   default FOV-ratio mapping (pinhole scale about the image centres, CAMERA_*/THERMAL_*_FOV). Tested with synthetic pairs
   (RMS 0.85 px vs 40 px for the uncalibrated default with a 40 px offset). Real-camera auto calibration has NOT been run
   (no mounted rig/warm target set up in this session).
- Default optics are assumptions: IMX477 with 6 mm lens (HFOV 52.8/VFOV 31.2), Lepton 3.5 HFOV 57/VFOV 44. Set the real lens in .env.

## Camera rotation and effective optics (2026-10-20)

`CAMERA_ROTATE/FLIP` (default 180) and `THERMAL_ROTATE/FLIP` are applied at capture (see notes/live-view.md). Everything that
used the sensor FOV/size now uses the effective values of the corrected image: `Settings.eff_hfov_deg`, `eff_vfov_deg`,
`eff_camera_width/height`, `eff_image_width_px`, `eff_sensor_width_mm` (the sensor axis that ends up horizontal; needs
`SENSOR_HEIGHT_MM`), and `eff_thermal_hfov_deg/vfov_deg/size`. Used by `survey.build_survey` / polygon survey (W across-track =
image horizontal axis, L along-track, GSD, person_px, max altitude), `tracker.detection_ground`, fusion
(`default_homography`, calibration size/orientation), `/api/limits` (hfov/vfov/sensor width/image width for the planner UI),
detector warm-up and the system check's expected frame size.

Default plan, 100 x 60 m at 20 m, IMX477 + assumed 6 mm lens, 25 % overlap, 1 fps, 5 m/s:

| CAMERA_ROTATE | frame | footprint W x L | line spacing | v_max | GSD | person px | lanes |
|---|---|---|---|---|---|---|---|
| 0 or 180 (default 180) | 1920x1080 | 19.9 x 11.2 m | 14.9 m | 8.4 m/s | 1.03 cm | 165 | 5 |
| 90 or 270 | 1080x1920 | 11.2 x 19.9 m | 8.4 m | 14.9 m/s | 1.03 cm | 165 | 8 |

(Pixels are square so GSD does not change; 90/270 only trade lane spacing for along-track speed.) Tests: `tests/test_orient.py`,
`tests/test_camera_robust.py`.
