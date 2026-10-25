#!/usr/bin/env python3
"""Check the whole system in one go (read-only). Same as `python main.py --check`.

    .venv/bin/python scripts/system_check.py [--quick] [--json] [--only vehicle,camera,...] [--standalone]

Groups: host llm planner camera thermal yolo clip vehicle web files. Exit code 1 if any check failed.
Saves data/system_check.json. If main.py is running, the running app is asked instead (devices stay untouched).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "drone_brain")]

import shutdown  # noqa: E402
from syscheck import cli  # noqa: E402

if __name__ == "__main__":
    shutdown.install(exit_code=130)      # SIGINT/SIGTERM: release both cameras, exit within ~3 s
    sys.exit(cli())
