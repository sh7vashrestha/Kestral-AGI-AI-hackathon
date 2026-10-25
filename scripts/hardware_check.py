#!/usr/bin/env python3
"""READ-ONLY check of the real Pixhawk (Phase 7). Never uploads, never changes mode/params, never arms.

    .venv/bin/python scripts/hardware_check.py [--url serial:/dev/ttyACM0] [--baud 115200] [--seconds 25]

Opens the link with VEHICLE_MODE=readonly (every vehicle-changing call raises before any I/O), watches
telemetry, reads the pre-flight parameters, runs MAV_CMD_RUN_PREARM_CHECKS (does not arm) and writes
data/hardware_check.json (samples + summary) and data/vehicle_check.json (full report). Then closes the port.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "drone_brain")]

from config import get_settings  # noqa: E402
from mission import ArduPilotBackend  # noqa: E402
from vehicle_check import run_vehicle_check  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="serial:/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=25.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings().model_copy(update={"vehicle_mode": "readonly", "mavlink_url": args.url, "mavlink_baud": args.baud})
    be = ArduPilotBackend(s)
    assert be.access == "readonly" and be.readonly, "refusing: backend is not read-only"
    t0 = time.time()
    be.connect(timeout=15)
    samples = []
    try:
        while time.time() - t0 < args.seconds:
            tel = be.telemetry()
            samples.append({k: v for k, v in tel.model_dump().items() if v is not None})
            time.sleep(1.0)
        report = run_vehicle_check(be, s, prearm_wait_s=3.0, param_timeout_s=3.0)
        tel = be.telemetry()
        home = be.home_position()
        out = {
            "ts": time.time(), "url": args.url, "baud": args.baud, "duration_s": round(time.time() - t0, 1),
            "summary": {"mode": tel.mode, "armed": tel.armed, "gps_fix": tel.gps_fix, "satellites": tel.satellites,
                        "hdop": tel.hdop, "battery_v": tel.voltage_v, "battery_pct": tel.battery_pct,
                        "landed_state": tel.landed_state, "system_status": tel.system_status,
                        "vehicle_type": tel.vehicle_type, "lat": tel.lat, "lon": tel.lon, "alt_rel_m": tel.alt_rel_m,
                        "home": None if home is None else {"lat": home.lat, "lon": home.lon, "amsl_m": home.amsl_m},
                        "heartbeat_age_s": tel.heartbeat_age_s},
            "statustext": be.statustexts(),
            "vehicle_check": report,
            "samples": samples,
        }
    finally:
        be.close()
    data = Path(s.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    (data / "hardware_check.json").write_text(json.dumps(out, indent=1, default=str))
    (data / "vehicle_check.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(out["summary"], indent=1))
    print("verdict:", report["verdict"])
    for r in report["reasons"]:
        print("  -", r)
    for c in report.get("preflight", {}).get("checks", []):
        print(f"  {'OK ' if c['ok'] else 'BAD'} {c['param']:<16} {str(c['value']):>8}  {c['message']}")
    print("pre-arm:", report.get("prearm"))
    print("saved", data / "hardware_check.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
