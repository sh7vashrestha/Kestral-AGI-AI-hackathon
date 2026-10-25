# Phase 1 notes (LLM/VLM serving, test assets, benchmarks)

## Progress
- Test assets: `tests/assets/persons/` (18 person crops p01-p18 + 6 aerial-like a01-a06), `tests/assets/nonperson/` (3), `tests/assets/match/labels.json` (attributes per crop, 6 target descriptions with expected true/false per crop), `tests/assets/SOURCES.md`. 668 KB. Rebuild with `scripts/make_test_assets.py`.
- `scripts/strip_mmproj_audio.py`: writes `gemma-4-E2B-it-mmproj-vision.gguf` (375 MB, 660 vision tensors, `clip.has_audio_encoder=false`; original 987 MB untouched).
- `scripts/vlm_benchmark.py`, `scripts/clip_prompts_bench.py`, `scripts/live_planner_bench.sh`, `scripts/start_llama.sh` (modes text/vision-cpu/vision-gpu, `MODEL_GGUF=` override).
- planner.py: `extra={"chat_template_kwargs":{"enable_thinking":False}}`, new optional `direction` compass word (converted in code), 2 more few-shot examples, stricter target prompt (animals rejected). tests/test_planner.py unchanged. `pytest -q`: 236 passed; `-m live`: 16/16.
- gemma4-vlm: `.env` now text-only default; `common.sh`/`run_server.sh` accept `MMPROJ_FILE="none"` (guard added; backups were taken). Gemma 3 1B (unsloth Q4_K_M, 806 MB) downloaded to `gemma4-vlm/models/gemma-3-1b/` (google QAT repo is gated: HTTP 401).

## Decisions
- Planner model: **Gemma 4 E2B text-only** (rule: Gemma 3 1B must match E2B on accept/reject cases).
- Gemma 4 emits a "Thinking Process" by default (112-145 tokens for a trivial reply, 4 s). `chat_template_kwargs {"enable_thinking": false}` removes it (12 tokens, 0.5 s). `reasoning_budget:0` and `reasoning_format` do NOT help. No server flag needed; planner passes it per request. llm_max_tokens 512 is fine (JSON plan ~100 tokens).
- Server flags: `--fit off -np 1` (+ `-c 4096`). With 4 slots the CUDA alloc of KV/compute buffers failed whenever other processes held GPU memory (happened repeatedly while YOLO/CLIP export and chromium ran); `-np 1` loads reliably. Retrying the start (3x) is sometimes still needed under load.
- Vision projector: stripped vision-only mmproj on GPU loads but the server CRASHES (cudaMalloc OOM, abort) on the first image. Stripped mmproj on CPU works: RAM saving ~600 MB vs original (375 MB vs 987 MB file). Kept CPU+stripped as the (optional) vision config, not default.
- Queue policy (if Gemma vision stays as confirm path): t_vlm = ~59 s (p50 59, range 51-66, n=11 never-seen crops, CPU image encode; generation itself ~30 tok/s) vs dt = 1 s. Server cache hits for already-seen crops take 1.8 s (a bench artefact: the 6-crop benchmark JSON is the cached case). Policy: bounded queue (settings.vlm_queue_size), keep best crop per track (highest c_yolo x size), drop-oldest when full, one in-flight request, never block the detect loop. Throughput ~1 crop/min, so with queue 4 the max backlog is 4 crops = ~4 min latency; recommend VLM_QUEUE_SIZE=2 (<= 2 min) and treating the VLM as confirm-only after CLIP/fusion passes threshold. Keep llm_timeout_s=60 for text; use >=120 s for image calls (not changed in config.py).
- Gemma vision quality on tiny aerial-like crops is poor: for target "person wearing a red jacket" it said matches=True for 5 of 6 crops (white shirt, green top, light jacket, black suit were wrong); is_human was right 6/6. Conclusion: not usable as the matcher; CLIP primary is the right call.

## Measurements
| | Gemma 4 E2B text-only | Gemma 3 1B Q4_K_M text-only |
|---|---|---|
| live planner (run1/run2, final prompt) | 16/16, 16/16 | 12/16, 13/16 (bearing/altitude copy errors, 'no verb' refusal, safety-limit case) |
| per planning call | 2.3-4.7 s (mean ~3.1) | ~2 s |
| tok/s (CLIP-prompt task) | 32.2 | 38.0 |
| CLIP-prompt JSON valid / mean latency | 5/5, 4.3 s | 5/5, 3.4 s |
| RAM after load (`free` used / available) | ~5.4 GB / ~1.5-1.9 GB | ~3.1 GB / ~4.2 GB |
tegrastats peaks were contaminated by other agents' concurrent jobs (6.2-7.2 GB used for E2B, 4.5-6.4 for 1B); use the free-after-load row. Raw files in `data/bench/` (`live_*`, `clip_prompts_*.json`, `vlm_benchmark*.json`, tegrastats).
- Gemma 4 vision via llama-server, stripped mmproj on CPU: JSON valid 100%, is_human 6/6, cold ~59 s/crop.
- Not done: Gemma 3 270M (skipped by instruction); full 24-crop VLM sweep (cold cost ~24 min).

## .env changes (gemma4-vlm/.env)
MMPROJ_FILE="none" (commented alternatives: stripped vision / original); commented Gemma 3 1B model lines; EXTRA_SERVER_ARGS="--fit off -np 1 --no-mmproj-offload". Before: MMPROJ original, `--fit off --no-mmproj-offload`.

## Open issues
- GPU memory contention: llama-server may fail to start if another GPU job runs; start it first.
- Suggest: VLM_QUEUE_SIZE=2 default; llm_timeout_s 60 ok for text.
