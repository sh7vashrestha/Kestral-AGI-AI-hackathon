"""Read-only vehicle health report (Phase 7): GPS, EKF, sensors, pre-arm, battery, RC, firmware, parameters.

    report = run_vehicle_check(backend, settings)      # dict, also saved to data/vehicle_check.json by callers

Uses only what read-only mode allows: received messages (SYS_STATUS, GPS_RAW_INT, EKF_STATUS_REPORT,
RC_CHANNELS, AUTOPILOT_VERSION, HEARTBEAT, EXTENDED_SYS_STATE, HOME_POSITION), PARAM_REQUEST_READ and
MAV_CMD_RUN_PREARM_CHECKS (401, which does not arm). Verdict READY / NOT READY with reasons.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from config import Settings

log = logging.getLogger(__name__)

SENSOR_BITS = {
    0x1: "3D gyro", 0x2: "3D accel", 0x4: "3D mag", 0x8: "baro", 0x10: "diff pressure", 0x20: "GPS",
    0x40: "optical flow", 0x80: "vision position", 0x100: "laser position", 0x200: "ground truth",
    0x400: "rate control", 0x800: "attitude stabilization", 0x1000: "yaw position", 0x2000: "altitude control",
    0x4000: "XY position control", 0x8000: "motor outputs", 0x10000: "RC receiver", 0x20000: "gyro 2",
    0x40000: "accel 2", 0x80000: "mag 2", 0x100000: "geofence", 0x200000: "AHRS", 0x400000: "terrain",
    0x800000: "reverse motor", 0x1000000: "logging", 0x2000000: "battery", 0x4000000: "proximity",
    0x8000000: "satcom", 0x10000000: "pre-arm check", 0x20000000: "obstacle avoidance", 0x40000000: "propulsion",
}
PREARM_BIT = 0x10000000
EKF_FLAGS = {
    0x1: "attitude", 0x2: "velocity horiz", 0x4: "velocity vert", 0x8: "pos horiz rel", 0x10: "pos horiz abs",
    0x20: "pos vert abs", 0x40: "pos vert agl", 0x80: "const pos mode", 0x100: "pred pos horiz rel",
    0x200: "pred pos horiz abs", 0x400: "uninitialized", 0x8000: "GPS glitch",
}
EKF_NEEDED = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20
GPS_FIX = {0: "NO GPS", 1: "NO FIX", 2: "2D", 3: "3D", 4: "DGPS", 5: "RTK float", 6: "RTK fixed", 7: "static", 8: "PPP"}
FW_TYPES = {0: "dev", 64: "alpha", 128: "beta", 192: "rc", 255: "official"}
FRAME_CLASS = {0: "undefined", 1: "quad", 2: "hexa", 3: "octa", 4: "octaquad", 5: "Y6", 6: "heli", 7: "tri",
               8: "single", 9: "coax", 10: "bicopter", 11: "heli dual", 12: "dodecahexa", 13: "heli quad", 14: "deca"}
FRAME_TYPE = {0: "plus", 1: "X", 2: "V", 3: "H", 4: "V-tail", 5: "A-tail", 10: "Y6B", 11: "Y6F", 12: "BetaFlightX",
              13: "DJIX", 14: "ClockwiseX", 15: "I", 18: "BetaFlightXReversed"}


def decode_bits(mask: int, names: dict[int, str]) -> list[str]:
    return [n for b, n in names.items() if mask & b]


def variance_level(v: Optional[float]) -> str:
    if v is None:
        return "UNKNOWN"
    return "OK" if v < 0.5 else ("WARN" if v < 0.8 else "BAD")


def sensors_report(sys_status: Optional[dict]) -> dict:
    if not sys_status:
        return {"available": False}
    pres = int(sys_status.get("onboard_control_sensors_present", 0))
    en = int(sys_status.get("onboard_control_sensors_enabled", 0))
    hl = int(sys_status.get("onboard_control_sensors_health", 0))
    failing = decode_bits(en & ~hl & pres, SENSOR_BITS)
    return {"available": True, "present": decode_bits(pres, SENSOR_BITS), "enabled": decode_bits(en, SENSOR_BITS),
            "unhealthy": failing, "prearm_bit_present": bool(pres & PREARM_BIT),
            "prearm_ok": bool(hl & PREARM_BIT) if pres & PREARM_BIT else None}


def ekf_report(ekf: Optional[dict]) -> dict:
    if not ekf:
        return {"available": False, "level": "UNKNOWN"}
    flags = int(ekf.get("flags", 0))
    var = {k: round(float(ekf.get(f"{k}_variance", 0.0)), 3) for k in ("velocity", "pos_horiz", "pos_vert", "compass", "terrain_alt")}
    levels = {k: variance_level(v) for k, v in var.items() if k != "terrain_alt"}
    missing = decode_bits(EKF_NEEDED & ~flags, EKF_FLAGS)
    bad_flags = decode_bits(flags & (0x400 | 0x8000 | 0x80), EKF_FLAGS)
    worst = "OK"
    if "WARN" in levels.values() or missing:
        worst = "WARN"
    if "BAD" in levels.values() or bad_flags:
        worst = "BAD"
    return {"available": True, "flags": decode_bits(flags, EKF_FLAGS), "missing": missing, "bad_flags": bad_flags,
            "variances": var, "levels": levels, "level": worst}


def firmware_report(av: Optional[dict]) -> dict:
    if not av:
        return {"available": False}
    v = int(av.get("flight_sw_version", 0))
    return {"available": True, "version": f"{(v >> 24) & 0xFF}.{(v >> 16) & 0xFF}.{(v >> 8) & 0xFF}",
            "type": FW_TYPES.get(v & 0xFF, str(v & 0xFF)),
            "git": bytes(av.get("flight_custom_version") or []).hex()[:8] or None,
            "board_version": av.get("board_version"), "vendor_id": av.get("vendor_id"), "product_id": av.get("product_id")}


# Readiness items while a mission is staged (notes/go-flow.md). (id, label, blocking). Order = phone display order.
READINESS_ITEMS = (
    ("link", "MAVLink link", True), ("disarmed", "Vehicle disarmed", True), ("gps", "GPS", True),
    ("ekf", "EKF", True), ("prearm", "Pre-arm checks", True), ("rc", "RC receiver", True),
    ("battery", "Battery", True), ("fence", "Geofence", True), ("failsafe", "Failsafes", True),
    ("home", "Home unchanged", True), ("mission", "Mission on vehicle = staged plan", True),
    ("geofence_alt", "Altitude vs ceiling", True), ("arm_guard", "Real-flight lock", True),
    ("vision", "Vision pipeline", False), ("legal", "Legal checklist", True),
)
# preflight_check rows -> readiness item
PARAM_ITEM = {"FENCE_ENABLE": "fence", "FENCE_TYPE": "fence", "FENCE_RADIUS": "fence", "FENCE_ACTION": "fence",
              "FENCE_ALT_MAX": "geofence_alt", "RTL_ALT": "failsafe", "RTL_ALT_M": "failsafe",
              "BATT_FS_LOW_ACT": "failsafe", "FS_BATT_ENABLE": "failsafe", "FS_THR_ENABLE": "failsafe",
              "WPNAV_SPEED": "failsafe", "WP_SPD": "failsafe"}


def run_vehicle_check(backend: Any, settings: Settings, *, prearm_wait_s: float = 3.0,
                      param_timeout_s: float = 3.0, plan: Any = None, run_prearm: bool = True,
                      read_frame: bool = True, preflight: Any = None) -> dict:
    """The full report. Readiness (staged mission) re-uses it in a light form: `run_prearm=False` (SYS_STATUS
    pre-arm bit only, no 401 command), `read_frame=False`, `preflight=<earlier PreflightResult>` (no param reads).
    `plan` makes the parameter cross-check plan-specific (fence radius / ceiling vs the plan).
    rep["findings"] = {item id: {"fail": [...], "warn": [...]}} - the same reasons/warnings, sorted by READINESS_ITEMS id."""
    from mission import preflight_check, MAV_TYPES

    t0 = time.time()
    tel = backend.telemetry()
    reasons: list[str] = []
    warnings: list[str] = []
    findings: dict[str, dict[str, list[str]]] = {}

    def fail(item: str, text: str, detail: Optional[str] = None) -> None:
        reasons.append(text)
        findings.setdefault(item, {"fail": [], "warn": []})["fail"].append(detail or text)

    def warn(item: str, text: str) -> None:
        warnings.append(text)
        findings.setdefault(item, {"fail": [], "warn": []})["warn"].append(text)

    rep: dict[str, Any] = {"ts": t0, "link": {"connected": tel.connected, "heartbeat_age_s": tel.heartbeat_age_s,
                                              "url": settings.mavlink_url, "vehicle_mode": settings.vehicle_mode},
                           "findings": findings}
    if not tel.connected:
        fail("link", "no MAVLink heartbeat from the Pixhawk")
        rep.update(verdict="NOT READY", reasons=reasons, warnings=[])
        return rep
    raw = backend.raw

    def r(t: str) -> Optional[dict]:
        x = raw(t)
        return x[1] if x else None

    hb = r("HEARTBEAT") or {}
    rep["vehicle"] = {"type": MAV_TYPES.get(int(hb.get("type", -1)), str(hb.get("type"))), "mode": tel.mode,
                      "armed": tel.armed, "system_status": tel.system_status, "landed_state": tel.landed_state}
    # GPS / home
    gps = r("GPS_RAW_INT") or {}
    home = backend.home_position()
    rep["gps"] = {"fix_type": tel.gps_fix, "fix": GPS_FIX.get(tel.gps_fix, str(tel.gps_fix)), "satellites": tel.satellites,
                  "hdop": tel.hdop, "lat": tel.lat, "lon": tel.lon,
                  "home": None if home is None else {"lat": home.lat, "lon": home.lon, "amsl_m": home.amsl_m}}
    if (tel.gps_fix or 0) < 3:
        fail("gps", f"GPS: no 3D fix ({rep['gps']['fix']}, {tel.satellites or 0} satellites)")
    else:
        if (tel.satellites or 0) < 8:
            warn("gps", f"GPS: only {tel.satellites} satellites")
        if tel.hdop is not None and tel.hdop > 2.0:
            fail("gps", f"GPS: HDOP {tel.hdop} > 2.0")
    if home is None:
        fail("home", "home position not set (needs 3D fix)")
    # EKF
    rep["ekf"] = ekf_report(r("EKF_STATUS_REPORT"))
    if rep["ekf"]["level"] == "BAD":
        fail("ekf", "EKF: " + ", ".join([f"{k} {v}" for k, v in rep["ekf"].get("levels", {}).items() if v == "BAD"]
                                        + rep["ekf"].get("bad_flags", [])))
    elif rep["ekf"]["level"] == "WARN":
        warn("ekf", "EKF: " + ", ".join([f"{k} {v}" for k, v in rep["ekf"].get("levels", {}).items() if v == "WARN"]
                                        + [f"missing {m}" for m in rep["ekf"].get("missing", [])]))
    elif rep["ekf"]["level"] == "UNKNOWN":
        warn("ekf", "EKF: no EKF_STATUS_REPORT received")
    # sensors
    rep["sensors"] = sensors_report(r("SYS_STATUS"))
    if rep["sensors"].get("unhealthy"):
        fail("prearm", "unhealthy sensors: " + ", ".join(rep["sensors"]["unhealthy"]))
    # battery
    rep["battery"] = {"voltage_v": tel.voltage_v, "pct": tel.battery_pct}
    if tel.battery_pct is not None and tel.battery_pct < 30:
        fail("battery", f"battery {tel.battery_pct:.0f}% < 30%")
    elif tel.battery_pct is not None and tel.battery_pct < settings.batt_warn_pct:
        findings.setdefault("battery", {"fail": [], "warn": []})["warn"].append(
            f"battery {tel.battery_pct:.0f}% < {settings.batt_warn_pct:.0f}% (BATT_WARN_PCT)")   # readiness only
    # RC
    rc = r("RC_CHANNELS")
    rep["rc"] = {"received": rc is not None, "chancount": rc.get("chancount") if rc else None,
                 "rssi": None if not rc or rc.get("rssi", 255) == 255 else rc.get("rssi"),
                 "healthy": "RC receiver" not in rep["sensors"].get("unhealthy", [])}
    if not rc or not rc.get("chancount"):
        fail("rc", "RC: no RC channels received (transmitter off / receiver not bound?)")
    # firmware + frame
    rep["firmware"] = firmware_report(r("AUTOPILOT_VERSION"))
    frame: dict[str, Any] = {"FRAME_CLASS": None, "FRAME_TYPE": None}
    for p in (("FRAME_CLASS", "FRAME_TYPE") if read_frame else ()):
        try:
            frame[p] = backend.get_param(p, timeout=param_timeout_s)
        except Exception as exc:  # noqa: BLE001
            frame[p] = None
            log.warning("vehicle check: %s read failed: %s", p, exc)
    rep["frame"] = {"class": FRAME_CLASS.get(int(frame["FRAME_CLASS"]), str(frame["FRAME_CLASS"])) if frame["FRAME_CLASS"] is not None else None,
                    "type": FRAME_TYPE.get(int(frame["FRAME_TYPE"]), str(frame["FRAME_TYPE"])) if frame["FRAME_TYPE"] is not None else None,
                    **frame}
    # pre-arm (MAV_CMD_RUN_PREARM_CHECKS does not arm)
    if run_prearm:
        try:
            pa = backend.run_prearm_checks(wait_s=prearm_wait_s)
        except Exception as exc:  # noqa: BLE001
            pa = {"ack": f"error: {exc}", "messages": []}
    else:
        pa = {"ack": "skipped", "messages": []}
    pa["prearm_bit_ok"] = rep["sensors"].get("prearm_ok")
    rep["prearm"] = pa
    if pa["messages"]:
        texts = list(dict.fromkeys(str(m.get("text", m)) if isinstance(m, dict) else str(m) for m in pa["messages"]))
        fail("prearm", f"pre-arm: {len(pa['messages'])} failure message(s)", "; ".join(texts[:3]))
    if pa["prearm_bit_ok"] is False:
        recent: list[str] = []
        fn = getattr(backend, "statustexts", None)
        if not pa["messages"] and callable(fn):      # light run: the vehicle's own periodic PreArm lines say why
            try:
                recent = [d["text"] for d in fn(since=t0 - 60.0) if str(d.get("text", "")).startswith(("PreArm", "Arm"))]
            except Exception:  # noqa: BLE001
                recent = []
        fail("prearm", "pre-arm check health bit: FAIL",
             "pre-arm check health bit: FAIL" + (f" ({recent[-1]})" if recent else ""))
    # parameter cross-check
    pre = preflight if preflight is not None else preflight_check(backend, settings, plan, timeout=param_timeout_s)
    rep["preflight"] = pre.model_dump()        # readiness re-uses it: PreflightResult.model_validate(rep["preflight"])
    for row in pre.checks:
        if not row.ok:
            (fail if row.severity == "error" else warn)(PARAM_ITEM.get(row.param, "fence"), f"Pixhawk {row.param}: {row.message}")
    for e in pre.errors:                 # errors without a table row (e.g. the parameter read itself failed)
        if e not in reasons:
            fail("failsafe", e)
    if tel.armed:
        warn("disarmed", "vehicle is ARMED")
    rep.update(verdict="READY" if not reasons else "NOT READY", reasons=reasons, warnings=warnings,
               duration_s=round(time.time() - t0, 1))
    return rep


def _ok_detail(iid: str, rep: dict) -> str:
    """What an item that passed shows on the phone (from the same report)."""
    g, b = rep.get("gps") or {}, rep.get("battery") or {}
    if iid == "link":
        age = (rep.get("link") or {}).get("heartbeat_age_s")
        return f"heartbeat {age:.1f} s ago" if isinstance(age, (int, float)) else "heartbeat OK"
    if iid == "gps":
        return f"{g.get('fix')} fix, {g.get('satellites') or 0} satellites, HDOP {g.get('hdop') if g.get('hdop') is not None else '-'}"
    if iid == "ekf":
        return "EKF healthy" if (rep.get("ekf") or {}).get("level") == "OK" else ""
    if iid == "prearm":
        pa = rep.get("prearm") or {}
        return "pre-arm checks pass" + ("" if pa.get("ack") in (None, "skipped") else f" (RUN_PREARM_CHECKS {pa.get('ack')})")
    if iid == "rc":
        rc = rep.get("rc") or {}
        return f"{rc.get('chancount')} channels" + (f", RSSI {rc['rssi']}" if rc.get("rssi") is not None else "")
    if iid == "battery":
        v, p = b.get("voltage_v"), b.get("pct")
        return ", ".join(x for x in (f"{v:.1f} V" if isinstance(v, (int, float)) else "",
                                     f"{p:.0f}%" if isinstance(p, (int, float)) else "") if x) or "no battery data"
    if iid in ("fence", "failsafe"):
        pre = rep.get("preflight") or {}
        return (f"fence {pre.get('fence')}" if iid == "fence" and pre.get("fence") else
                f"RTL at {pre['rtl_alt_m']:.0f} m, failsafes set" if iid == "failsafe" and pre.get("rtl_alt_m") else "parameters OK")
    return ""


def build_readiness(rep: dict, extra: dict[str, tuple[str, str]]) -> dict:
    """snapshot["readiness"] body from a run_vehicle_check report + the caller's own items.

    `extra` = {item id: (status, detail)} for what the report cannot know (disarmed, home vs upload, mission
    re-download, plan vs ceiling, arm_guard, vision, legal). A "fail"/"warn" in `extra` is added to the report's
    findings for that id; an "ok" detail replaces the default. A non-blocking item never fails (shown as warn).
    No link -> every vehicle item is "pending". GO is allowed only when no blocking item is fail/pending."""
    findings = rep.get("findings") or {}
    linked = bool((rep.get("link") or {}).get("connected"))
    no_link_ids = {"disarmed", "gps", "ekf", "prearm", "rc", "battery", "fence", "failsafe", "home", "mission"}
    items: list[dict] = []
    for iid, label, blocking in READINESS_ITEMS:
        f = findings.get(iid) or {}
        fails, warns = list(f.get("fail", [])), list(f.get("warn", []))
        ok_detail = _ok_detail(iid, rep)
        pending = None
        st, d = extra.get(iid, (None, ""))
        if st == "fail":
            fails.append(d)
        elif st == "warn":
            warns.append(d)
        elif st == "pending":
            pending = d
        elif st == "ok" and d:
            ok_detail = d
        if not linked and iid in no_link_ids:
            fails, warns, pending = [], [], "no MAVLink link"
        if fails:
            status, detail = ("fail" if blocking else "warn"), "; ".join(fails)
        elif pending is not None:
            status, detail = "pending", pending
        elif warns:
            status, detail = "warn", "; ".join(warns)
        else:
            status, detail = "ok", ok_detail
        items.append({"id": iid, "label": label, "status": status, "detail": detail, "blocking": blocking})
    block = next((i for i in items if i["blocking"] and i["status"] in ("fail", "pending")), None)
    return {"go_allowed": block is None, "go_reason": "" if block is None else f"{block['label']}: {block['detail']}",
            "items": items}
