"""Planned mission vs the mission read back from the Pixhawk (pure functions, no I/O).

    views = readback_view(items)                  # [{seq, command, name, frame, lat, lon, alt, p1..p4}, ...] for the phone
    diff  = compare_missions(planned, readback)   # {rows: [...], summary: {...}} -> the "Planned vs uploaded" table

Tolerances are verify_mission's (POS_TOL_DEG for lat/lon, ALT_TOL_M for altitude). Item 0 (ArduPilot's home slot) is
compared by command only: the autopilot substitutes its real home there.
Non-positional items (TAKEOFF, DO_CHANGE_SPEED, RTL, ...) compare command + params (+ altitude, e.g. the TAKEOFF height).
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

from models import MissionItem

POS_TOL_DEG = 1e-6          # keep in step with mission.POS_TOL_DEG (imported lazily there; no pymavlink here)
ALT_TOL_M = 0.05
PARAM_TOL = 1e-3
M_PER_DEG = 111_320.0

COMMAND_NAMES = {
    16: "WAYPOINT", 17: "LOITER_UNLIM", 18: "LOITER_TURNS", 19: "LOITER_TIME", 20: "RTL", 21: "LAND", 22: "TAKEOFF",
    82: "SPLINE_WAYPOINT", 115: "CONDITION_YAW", 177: "DO_JUMP", 178: "DO_CHANGE_SPEED", 183: "DO_SET_SERVO",
    189: "DO_LAND_START", 5001: "FENCE_POLYGON_INCLUSION", 5002: "FENCE_POLYGON_EXCLUSION",
}
FRAME_NAMES = {0: "AMSL", 2: "MISSION", 3: "REL_ALT", 5: "AMSL", 6: "REL_ALT", 10: "TERRAIN", 11: "TERRAIN"}
_FRAME_CLASS = {0: "abs", 5: "abs", 3: "rel", 6: "rel", 10: "terrain", 11: "terrain", 2: "mission"}


def command_name(cmd: int) -> str:
    return COMMAND_NAMES.get(int(cmd), f"CMD_{int(cmd)}")


def frame_name(frame: int) -> str:
    return FRAME_NAMES.get(int(frame), f"FRAME_{int(frame)}")


def _frame_class(frame: int) -> Any:
    return _FRAME_CLASS.get(int(frame), int(frame))


def is_positional(it: MissionItem) -> bool:
    """Carries a lat/lon the vehicle will fly to (the DO_*, RTL and TAKEOFF-in-place items do not)."""
    return abs(it.lat) > 1e-9 or abs(it.lon) > 1e-9


def item_view(it: MissionItem) -> dict[str, Any]:
    return {"seq": int(it.seq), "command": int(it.command), "name": command_name(it.command), "frame": int(it.frame),
            "frame_name": frame_name(it.frame), "lat": round(float(it.lat), 7), "lon": round(float(it.lon), 7),
            "alt": round(float(it.alt), 3), "p1": round(float(it.param1), 3), "p2": round(float(it.param2), 3),
            "p3": round(float(it.param3), 3), "p4": round(float(it.param4), 3), "positional": is_positional(it)}


def readback_view(items: Iterable[MissionItem]) -> list[dict[str, Any]]:
    return [item_view(i) for i in sorted(items, key=lambda i: i.seq)]


def horizontal_cm(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    dn = (b_lat - a_lat) * M_PER_DEG
    de = (b_lon - a_lon) * M_PER_DEG * math.cos(math.radians((a_lat + b_lat) / 2))
    return math.hypot(dn, de) * 100.0


def _row(want: Optional[MissionItem], got: Optional[MissionItem]) -> dict[str, Any]:
    ref = want or got
    assert ref is not None
    row: dict[str, Any] = {"seq": int(ref.seq), "command": int(ref.command), "name": command_name(ref.command),
                           "planned": item_view(want) if want else None, "readback": item_view(got) if got else None,
                           "positional": False, "d_h_cm": None, "d_alt_m": None, "ok": False, "note": ""}
    if want is None:
        row["note"] = "extra item on the Pixhawk (not in the plan)"
        return row
    if got is None:
        row["note"] = "missing on the Pixhawk"
        return row
    row["positional"] = is_positional(want)
    if want.command != got.command:
        row["note"] = f"command {command_name(got.command)} on the Pixhawk, plan has {command_name(want.command)}"
        return row
    if want.seq == 0:
        # home slot: ArduPilot substitutes its real home, so only the command is compared
        if is_positional(got):
            row["d_h_cm"] = round(horizontal_cm(want.lat, want.lon, got.lat, got.lon), 1)
        row["ok"] = True
        row["note"] = "home slot: ArduPilot substitutes its own home"
        return row
    problems: list[str] = []
    vals = [getattr(it, n) for it in (want, got) for n in ("lat", "lon", "alt", "param1", "param2", "param3", "param4")]
    if not all(math.isfinite(v) for v in vals):
        row["note"] = "non-finite (NaN/infinite) number in the plan or on the Pixhawk"
        return row
    # the frame only matters for items with a position: ArduPilot stores DO_CHANGE_SPEED / RTL with frame 0
    if (row["positional"] or is_positional(got)) and _frame_class(want.frame) != _frame_class(got.frame):
        problems.append(f"frame {frame_name(got.frame)} vs {frame_name(want.frame)}")
    d_alt = got.alt - want.alt
    row["d_alt_m"] = round(d_alt, 3)
    if abs(d_alt) > ALT_TOL_M:
        problems.append(f"altitude {d_alt:+.2f} m")
    if row["positional"] or is_positional(got):
        row["positional"] = True
        row["d_h_cm"] = round(horizontal_cm(want.lat, want.lon, got.lat, got.lon), 2)
        if abs(got.lat - want.lat) > POS_TOL_DEG or abs(got.lon - want.lon) > POS_TOL_DEG:
            problems.append(f"position off by {row['d_h_cm']:.1f} cm")
    else:
        for n in ("param1", "param2", "param3", "param4"):
            a, b = getattr(want, n), getattr(got, n)
            if abs(a - b) > PARAM_TOL:
                problems.append(f"{n.replace('param', 'p')} {b:g} vs {a:g}")
    row["ok"] = not problems
    row["note"] = "; ".join(problems)
    return row


def compare_missions(planned: Optional[Iterable[MissionItem]], readback: Iterable[MissionItem],
                     plan_id: Optional[str] = None) -> dict[str, Any]:
    """Row per sequence number. planned=None -> no comparison (read-back only)."""
    readback = list(readback)
    got = {i.seq: i for i in readback}
    if planned is None:
        return {"plan_id": plan_id, "compared": False, "rows": [], "summary": {
            "total": len(got), "matched": None, "max_h_cm": None, "max_alt_m": None, "all_ok": None,
            "text": f"{len(got)} items on the Pixhawk (no plan to compare with)"}}
    planned = list(planned)
    want = {i.seq: i for i in planned}
    dup = len(planned) != len(want) or len(readback) != len(got)
    rows = [_row(want.get(s), got.get(s)) for s in sorted(set(want) | set(got))]
    matched = sum(1 for r in rows if r["ok"])
    hs = [r["d_h_cm"] for r in rows if r["d_h_cm"] is not None and r["ok"] is not None and r["seq"] != 0]
    als = [abs(r["d_alt_m"]) for r in rows if r["d_alt_m"] is not None]
    max_h = max(hs) if hs else 0.0
    max_a = max(als) if als else 0.0
    all_ok = matched == len(rows) and bool(rows) and not dup
    text = f"{matched}/{len(rows)} items match, max error {max_h:.1f} cm" + (f", max altitude error {max_a:.2f} m" if max_a > 0.0005 else "")
    if dup:
        text += " - DUPLICATE sequence numbers in the list"
    if not all_ok:
        text += f" - {len(rows) - matched} DIFFER"
    return {"plan_id": plan_id, "compared": True, "rows": rows,
            "summary": {"total": len(rows), "matched": matched, "max_h_cm": round(max_h, 2), "max_alt_m": round(max_a, 3),
                        "all_ok": all_ok, "text": text}}
