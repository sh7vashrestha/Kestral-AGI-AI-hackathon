# Hardening D: vision (cameras, thermal, YOLO, CLIP, matcher, pipeline)

Tests: `tests/test_vision_edges.py` (47 tests, fakes only). Reproduce tuning: `CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/tune_clip.py`.
Real-hardware soak: `scripts/vision_soak.py`.

## 1. CLIP tuning

Data: `tests/assets/match/labels.json` (18 persons, 6 with aerial-like copies, 3 non-person crops) + synthetic variants
(aerial downscales 48/70 px, blur+dark, bright, rot180, hue-shifted skier crops so colour targets have positives AND
negatives in both splits) = 174 images, 5 targets (the trivial "any person" target is excluded). Split by PERSON:
dev (picks settings) 84 images, test (report only) 90 images. All numbers CPU FP32 (GPU was busy); FP16 differs slightly.
**Caveat: the "aerial" crops are degraded ground photos, not real top-down views; real overhead performance will be worse. The
dataset is tiny (18 people); treat differences of a few points as noise. A field test with a person at altitude is still required.**

What was wrong: the default prompts were 1 positive + 5 negatives of which 2 were GENERIC PERSON prompts ("a photo of a person"). The
target person looks like "a person" more than anything, so those negatives swallowed the probability: recall at cut-off 0.5 was 0.15.
Contrastive negatives (same person, other colour / other sleeve length / bare head) separate the target from other people instead.

| | before (legacy default prompts) | after (`default_prompts`, deterministic, no LLM) |
|---|---|---|
| mean per-target AUC dev / test | 0.861 / 0.748 | **0.923 / 0.777** |
| red jacket AUC dev / test | 1.00 / 0.31 | 1.00 / 0.52 (test has only 2 hue-shifted positives) |
| long sleeve | 0.59 / 0.84 | 0.87 / 0.86 |
| hat or cap | 0.77 / 0.77 | 0.77 / 0.65 |
| green top | 1.00 / 0.84 | 1.00 / 0.93 |
| white short sleeve + cap | 0.94 / 0.98 | 0.98 / 0.92 |
| per-image at cut-off 0.5 (pooled) dev: acc / precision / recall | 0.74 / 0.78 / 0.15 | 0.87 / 0.82 / 0.70 |
| same, test | 0.72 / 0.66 / 0.16 | 0.78 / 0.63 / 0.62 |
| at the chosen cut-off 0.6, dev / test: precision / recall | | 0.84 / 0.61, 0.65 / 0.54 |

Other findings: logit scale 40..140 changes nothing in ranking (kept 100); sum vs mean vs max aggregation of the positive prompts: same AUC
(kept sum); the A-built LLM prompts (aerial first, LLM contrast negatives) score the same as the deterministic ones (dev 0.920 / test 0.754 vs
0.923 / 0.777) and adding contrastive negatives to them is neutral-to-slightly-better, so `clip_prompts.validate_prompts` now pads missing
negatives with contrastive ones first. A's "no generic person positive" decision is confirmed. Attribute targets are weak: **"hat or cap"
(AUC 0.65-0.77) is effectively not usable** (alert-level recall 0.02), sleeve length is mid, colour targets are good.

Alert-level simulation (C_final = c_yolo (0.82-0.97) x C_match, mean of the last 5, 4-frame tracks, per person, dev+test):

| cut-off | threshold | min_frames | strong | precision | recall | red jacket P/R | green P/R |
|---|---|---|---|---|---|---|---|
| 0.5 (old) | 0.7 | 2 (old) | - | 0.81 | 0.34 | 0.86/0.67 | 0.60/0.95 |
| 0.6 | 0.7 | 2 | - | 0.80 | 0.33 | 0.88/0.67 | 0.60/0.94 |
| **0.6** | **0.7** | **3** | **0.93** | **0.95** | **0.22** | 0.95/0.67 | 0.93/0.69 |
| 0.6 | 0.7 | 3 | 0.90 | 0.88 | 0.22 | 0.93/0.67 | 0.77/0.74 |

Chosen (precision first, recall floor 0.65 on colour targets): `CLIP_MATCH_CUTOFF=0.6`, `MATCH_THRESHOLD=0.7` (unchanged; with c_yolo >= 0.8 it
already demands conf of about 0.8), `MIN_FRAMES=3`, new `MATCH_STRONG_THRESHOLD=0.93` (a track with >= 2 results and C_avg >= 0.93 alerts
without the third frame; 0.9 gave too many 2-frame false alerts), new `MATCH_MIN_PX=20` (smaller boxes: person alert only, no CLIP),
`VLM_QUEUE_SIZE` 4 -> 32 (a 30-person frame must fit). Updated in config.py, .env.example and .env.
**Flight consequence (open for the planner owner):** at the default 5 m/s with an 11 m along-track footprint a ground point is in ~2 frames, so
most "target match" alerts will need the strong rule or a second pass. For target search fly slower (<= footprint/3 per second, e.g. 3 m/s) or raise
the frame rate. Person alerts are unaffected.

## 2. CLIP TensorRT engine

The engine is fine for TensorRT 10.3 (`models/clip_image.engine`, 183 MB, device memory 2.6 MB; nothing to re-export, workspace is a build-time
setting and irrelevant at load). The failures are **GPU memory**: deserialisation needs one 170 MB allocation ("NvMap error 12", "Myelin OutOfMemory").
- 10 separate-process load cycles with llama-server resident and ~2.3 GB available: **10/10 TensorRT on the first attempt**, 6.5-7.2 s each.
- With 1.1 GB of CUDA ballast (MemAvailable ~1.2 GB): 3 cycles could not even create the CUDA context (existing retry in `ClipEncoder`), 3 cycles
  ran the 4 TensorRT attempts, failed and **fell back to torch FP16 (works, 14 s load)**; the earlier unlucky first attempt in a normal run
  also succeeded on the 2nd attempt.
- Kept: TensorRT default with automatic torch fallback. Changes: misleading "built with another TensorRT?" message now names memory, gc between
  attempts, CUDA tensors for mean/std also retried. In the soak the engine loaded (`clip: tensorrt`).

## 3. Edge cases

R = fixed by code change in this pass, T = test only (already behaved), D = documented limit.

| Edge case | Expected | Result | Fix |
|---|---|---|---|
| Camera unplugged / argus failure mid-run | worker released, reopened, ids continue | pass (also existing test_camera_robust) | T |
| Frozen frames (same textured picture) | detected as a stall, capture reopened after 5 s | **fixed**: never detected before | R `_FrozenWatch` (camera.py) |
| Frozen black frames (lens cap) | not a reopen loop; dark mode | pass | T |
| Black frame | dark mode, YOLO skipped | pass | T |
| Overexposed / blown-out frame (mean >= 235, std < 3) | YOLO skipped, logged | **fixed** (YOLO ran on white) | R `FusionResult.flat` |
| Rotation 180 in frames, boxes and crops | frame = rotated source, crop of a box on it is the marked region | pass | T |
| Thermal stuck-pixel mask under all 12 rotate/flip combos | same physical pixel | pass | T |
| Very slow frames (0.3 s) | each frame once, no dupes | pass | T |
| Frame size change mid-run | published; homography follows | **fixed**: H stayed at the configured size | R `Fuser._fit_frame_size` |
| Thermal unplugged mid-run | reads fail, device released, reconnect, frame stale for fusion | pass | T |
| Thermal zero / 655 C / NaN / -273 C frames, NaN or inf pixels | no hotspot, no exception, a real person still found | **fixed**: NaN reached uint8 casts | R `delta_map` sanitises |
| All-hot scene (45 C tarmac) | no false alert | pass; a 36 C person on 45 C ground is colder than the background and cannot be seen (physics) | T, D |
| Cold night (3 C ground, 28 C person) | found | pass | T |
| New stuck pixel appears | never a hotspot, masked after ~45 frames | pass | T |
| Thermal older than 0.5 s vs RGB | ignored | pass | T |
| Dark mode flapping around the threshold | no flapping | **fixed**: entered/left every frame | R hysteresis (leave above 35, 3 frames) |
| 0 / 1 / 30 people | 0 alerts / 1 / 30 person alerts, 30 tracks, all scored | **fixed**: queue 4 dropped the best 26 | R queue 32, best crops offered last |
| Queue overflow | non-blocking, best crops kept | pass (2000-job flood < 2 s) | T |
| Same person on 2 lanes (90 s apart) | one track, one person alert | **fixed**: tracks expired after 30 s | R GPS tracks live 15 min, max 400 tracks |
| People at image edges | clamped crops | **fixed**: YOLO boxes beyond the image / NaN / sub-pixel passed through | R detector clamps and drops |
| Box entirely outside the image | no crash | pass | T |
| Tiny people (< 20 px) | tracked, person alert, no CLIP | **fixed**: CLIP scored 12 px blobs | R `MATCH_MIN_PX` |
| YOLO engine missing (YOLO_MODEL=...engine) | use the .pt | **fixed**: vision was DISABLED | R `resolve_weights` |
| YOLO engine fails to load | fall back to .pt | **fixed** | R Detector |
| CUDA OOM during YOLO inference | free cache, retry once; if still failing the frame is an error, pipeline lives | **fixed**: no retry | R, T (pipeline keeps going, size change too) |
| Hotspot re-check (YOLO ROI) raises | heat alert still sent | **fixed**: exception killed the frame | R |
| Target update with crops queued / in flight | queue cleared, in-flight result discarded | pass | T |
| Empty / garbage / emoji / 400x-long CLIP prompts, builder exception | sanitised or default prompts | **fixed**: empty positives crashed `argmax`, LLM exception during lazy prepare killed the job | R `ClipBackend.sanitize` |
| Empty / generic / emoji / 500-char target text | prompts exist | pass | T |
| NaN CLIP output | confidence 0, never poisons C_avg | **fixed** (NaN stayed in match_hist) | R |
| Matcher worker exception (CUDA error) | worker survives, errors counted, backoff after 3 | pass; backoff added | T, R |
| Concurrent lazy prepare + update_target | one LLM call | **fixed**: two LLM calls | R `_prep_lock` |
| Alerts rate limit / min_frames / strong rule | as configured | pass | T |
| Heat blob on or near a YOLO person (this or a recent frame) | no extra heat alert | **fixed** (second frame produced a heat alert) | R person memory 10 s |
| Watchdog restart (hung camera, crashed worker) | alive() false, restart() | pass | T |
| SIGTERM | cameras free < 3 s | pass (subprocess test; real: below) | T |
| Storage garbage / read-only dir | returns None, no raise | pass | T |
| Cluster of 30 people within 8 m | merged into fewer tracks by GPS grouping (`TRACKER_GPS_RADIUS_M=8`) | D: a deliberate trade for GPS error; a duplicate alert is less harmful than a missed person, lower it to 4-5 if clusters are expected |
| Hat-only / sleeve-only targets | weak (see tuning) | D | |
| Right after another process released the CSI camera | first open can fail once ("Failed to create CaptureSession" / stall 5 s), then recovers in 2-3 s | D (worker retry handles it) | |

Counts: 41 edge cases tested (47 test functions incl. parametrisation), **19 fixed**, 22 already correct, 4 documented limits.

## 4. Real hardware (Orin Nano, llama-server resident, IMX477 csi:0 rotate 180 + PureThermal, YOLO engine, CLIP TensorRT)

`scripts/vision_soak.py --minutes 5 --inject-box --idle` (one synthetic 220x420 box per frame pushes real camera crops through CLIP + prompt
building through llama-server; capture 1 fps; room scene without people, so person detections are only the injected box):
- 299 frames, 298 processed, **0 errors, 0 pipeline restarts**; camera: 1 reconnect at start (first open after the previous process, recovered in 3 s).
- Loop time (fusion + YOLO + CLIP + storage): **p50 94 ms, p95 118 ms, max 135 ms**. Matcher: 296 results, mean 31 ms, queue 0, 0 errors.
- Thermal 8.7 fps, 2649 frames, 1 stuck pixel masked; 28 heat alerts (warm objects in the room, rate limited), 2 person alerts.
- tegrastats: RAM used peak **6839 MB of 7607** (mean 6784), swap peak 1158 MB, GR3D mean 4.5 % / peak 97 %; process RSS 1998 MB flat (no leak).
  Memory is the margin: llama-server 4 GB + this process 2 GB leaves ~0.7 GB.
- SIGTERM: `vp.stop()+release_all` 0.2 s, process gone **2.6 s** after the signal; no process held /dev/video*; both cameras reopened immediately
  (RGB first frame and thermal at 2.3 s after start, 0 reconnects). nvargus-daemon logs `SCF BadParameter` bursts at every open that follows a recent close
  (also before this work); no "Services are already stopped" and the daemon stayed healthy.
- llama-server was never stopped; `curl /health` ok at the end.
