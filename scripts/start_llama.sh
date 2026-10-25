#!/usr/bin/env bash
# Start llama-server detached. Usage: scripts/start_llama.sh text|vision-cpu|vision-gpu|vision-full-cpu [extra args]
# MODEL_GGUF=<path> overrides the model file. Stops any running llama-server first. Log: data/llama-server.log
set -euo pipefail
MODE="${1:-text}"; shift || true
G="/home/drone/Desktop/Flight Plan/gemma4-vlm"
M="$G/models/gemma-4-E2B-it-qat-q4_0"
LOG="$(cd "$(dirname "$0")/.." && pwd)/data/llama-server.log"
for p in $(pgrep -x llama-server || true); do kill "$p" || true; done
for _ in $(seq 30); do pgrep -x llama-server >/dev/null || break; sleep 1; done
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" HF_HUB_OFFLINE=1
# -np 1: one slot (smaller KV/compute buffers). The 1.8 GB per-layer embedding table (more than half the model) is only
# looked up, so it stays in plain RAM (-ot ...=CPU --no-host): on the Orin, CUDA + pinned allocations hit a cap near
# 3.3 GB well below "free" memory, and the all-GPU load failed there with "cudaMalloc failed: out of memory" (2026-10-20).
ARGS=(-m "${MODEL_GGUF:-$M/gemma-4-E2B_q4_0-it.gguf}" -ngl 99 -c 4096 --host 127.0.0.1 --port 8080 --jinja --fit off -np 1
      -ot "per_layer_token_embd\.weight=CPU" --no-host -lm dio)
# -lm dio: read the model with direct I/O instead of mmap. With mmap the 3.1 GB file filled the page cache, which the
# Orin's GPU allocator does not reclaim (NvMap error 12): every setting failed with 5.5 GB "available" (2026-10-21).
case "$MODE" in
  text) ;;
  vision-gpu) ARGS+=(--mmproj "$M/gemma-4-E2B-it-mmproj-vision.gguf") ;;
  vision-cpu) ARGS+=(--mmproj "$M/gemma-4-E2B-it-mmproj-vision.gguf" --no-mmproj-offload) ;;
  vision-full-cpu) ARGS+=(--mmproj "$M/gemma-4-E2B-it-mmproj.gguf" --no-mmproj-offload) ;;
  *) echo "bad mode"; exit 2 ;;
esac
# The app's YOLO/CLIP hold GPU memory: Gemma must load before drone-brain starts.
APP=$(pgrep -f '^(\S*/)?\.venv/bin/python main\.py' || true)
[[ -n $APP ]] && echo "WARNING: drone-brain is already running (pid $APP) and holds GPU memory; if Gemma does not fit: ./stop.sh, then ./start.sh"
# GPU memory on the Orin is capped (~3.3 GB for CUDA) and shared with VS Code, the app's YOLO/CLIP, ...:
# if the full setting does not fit, retry with a smaller compute batch, then a shorter context.
for LEAN in "" "-ub 256" "-ub 128 -c 2048"; do
  echo "=== $(date -Is) start mode=$MODE ${ARGS[*]} $LEAN $*" >> "$LOG"
  # Also drop any cached copy of the model file (no root needed), e.g. from an older mmap load.
  dd if="${ARGS[1]}" iflag=nocache count=0 status=none 2>/dev/null || true
  # shellcheck disable=SC2086
  setsid nohup "$G/llama.cpp/build/bin/llama-server" "${ARGS[@]}" $LEAN "$@" >> "$LOG" 2>&1 < /dev/null &
  for _ in $(seq 240); do
    curl -sf http://127.0.0.1:8080/health >/dev/null && { echo "ready mode=$MODE ${LEAN:-(full)}"; exit 0; }
    pgrep -x llama-server >/dev/null || { echo "llama-server did not fit ${LEAN:-(full)}: $(grep -E ' E ' "$LOG" | tail -1 | cut -c1-100)"; break; }
    sleep 2
  done
  pgrep -x llama-server >/dev/null && { echo "not ready"; exit 1; }
done
echo "Gemma did not fit in GPU memory (close VS Code windows / stop other GPU jobs, then retry)"; exit 1
