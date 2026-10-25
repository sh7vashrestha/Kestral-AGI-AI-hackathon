#!/usr/bin/env bash
# Follow the logs (Ctrl+C to leave).   ./logs.sh [app|gemma|sim]
cd "$(dirname "$0")"
case "${1:-app}" in
  app)   exec tail -n 60 -F data/start.log ;;
  gemma) exec tail -n 60 -F data/llama-server.log ;;
  sim)   exec tail -n 60 -F data/sitl.log ;;
  *) echo "usage: ./logs.sh [app|gemma|sim]"; exit 2 ;;
esac
