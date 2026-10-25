#!/usr/bin/env bash
# Run the tests.
#   ./test.sh            all offline tests (~10 min; mocks, no hardware, no Gemma)
#   ./test.sh quick      planning + web + GO flow only (~1 min)
#   ./test.sh gemma      live planner tests against the running Gemma (start it first: scripts/start_llama.sh)
#   ./test.sh sim        ArduPilot SITL flights (~4 min; needs ~/ardupilot built)
cd "$(dirname "$0")"
PT=(.venv/bin/python -m pytest -q -p no:cacheprovider)
case "${1:-all}" in
  all)   exec "${PT[@]}" -m "not live and not sitl" ;;
  quick) exec "${PT[@]}" -m "not live and not sitl" tests/test_assistant.py tests/test_survey.py tests/test_safety.py \
              tests/test_webapp.py tests/test_webapp_map.py tests/test_go_flow.py ;;
  gemma) exec "${PT[@]}" -m live ;;
  sim)   exec "${PT[@]}" -m sitl tests/test_e2e_sitl.py ;;
  *) sed -n 2,7p "$0"; exit 2 ;;
esac
