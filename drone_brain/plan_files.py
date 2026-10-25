"""Mission export files (Phase 7): QGroundControl WPL 110 `.waypoints` and GeoJSON.

    text = to_wpl(items, home_amsl_m)        # loadable in Mission Planner / QGroundControl
    rows = parse_wpl(text)                   # list[dict] (round-trip check / tests)
    gj   = to_geojson(plan)                  # FeatureCollection: area Polygon, route LineString, home Point

WPL 110 columns (tab separated):
    INDEX CURRENT_WP COORD_FRAME COMMAND P1 P2 P3 P4 LAT LON ALT AUTOCONTINUE
Line 0 is home (frame 0 = absolute altitude AMSL). Positional items use frame 3
(MAV_FRAME_GLOBAL_RELATIVE_ALT, altitude relative to home); the uploaded mission uses the
_INT variant (6) which is the same frame on the wire. Non-positional commands
(DO_CHANGE_SPEED, RTL) use frame 2 (MAV_FRAME_MISSION) as QGC writes them.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from models import MissionItem, SurveyPlan
from survey import local_to_geo, rotate

WPL_HEADER = "QGC WPL 110"
_FRAME_FILE = {6: 3, 5: 0, 0: 0, 2: 2, 3: 3}


def _num(v: float) -> str:
    if v is None or not math.isfinite(float(v)):
        return "0"
    t = f"{float(v):.8f}".rstrip("0").rstrip(".")
    return "0" if t in ("", "-0") else t


def to_wpl(items: list[MissionItem], home_amsl_m: Optional[float] = None) -> str:
    """Mission items (as built by mission.build_mission_items) -> WPL 110 text."""
    lines = [WPL_HEADER]
    for it in sorted(items, key=lambda i: i.seq):
        if it.seq == 0:
            frame, alt, current = 0, (home_amsl_m or 0.0), 1
        else:
            frame, alt, current = _FRAME_FILE.get(it.frame, it.frame), it.alt, 0
        cols = [str(it.seq), str(current), str(frame), str(it.command),
                _num(it.param1), _num(it.param2), _num(it.param3), _num(it.param4),
                f"{it.lat:.8f}", f"{it.lon:.8f}", _num(alt), str(it.autocontinue)]
        lines.append("\t".join(cols))
    return "\n".join(lines) + "\n"


def parse_wpl(text: str) -> list[dict[str, Any]]:
    """Parse WPL 110 text; raises ValueError on a malformed file."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("QGC WPL 110"):
        raise ValueError("not a QGC WPL 110 file")
    rows = []
    for n, ln in enumerate(lines[1:], start=1):
        c = ln.split("\t")
        if len(c) != 12:
            raise ValueError(f"line {n}: expected 12 tab-separated columns, got {len(c)}")
        rows.append(dict(seq=int(c[0]), current=int(c[1]), frame=int(c[2]), command=int(c[3]),
                         param1=float(c[4]), param2=float(c[5]), param3=float(c[6]), param4=float(c[7]),
                         lat=float(c[8]), lon=float(c[9]), alt=float(c[10]), autocontinue=int(c[11])))
        if rows[-1]["seq"] != n - 1:
            raise ValueError(f"line {n}: index {rows[-1]['seq']} out of sequence")
    return rows


def area_outline(plan: SurveyPlan) -> list[tuple[float, float]]:
    """Search area as (lat, lon) corners: polygon corners, or the rectangle's 4 corners."""
    m = plan.mission
    if m.shape == "polygon" and m.corners:
        return [(c.lat, c.lon) for c in m.corners]
    a0 = -(m.length_m or 0.0) / 2.0 if m.centered else m.offset_m
    a1 = a0 + (m.length_m or 0.0)
    w = (m.width_m or 0.0) / 2
    out = []
    for along, across in ((a0, -w), (a1, -w), (a1, w), (a0, w)):
        n, e = rotate(along, across, m.direction_deg)
        out.append(local_to_geo(n, e, plan.home))
    return out


def to_geojson(plan: SurveyPlan, *, plan_id: Optional[str] = None,
               home_amsl_m: Optional[float] = None, home_source: str = "pixhawk") -> dict[str, Any]:
    home = [plan.home.lon, plan.home.lat]
    ring = [[lon, lat] for lat, lon in area_outline(plan)]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    route = [home] + [[w.lon, w.lat] for w in plan.waypoints] + [home]
    props = {"plan_id": plan_id, "altitude_m": plan.altitude_m, "speed_ms": plan.speed_ms,
             "path_length_m": plan.path_length_m, "area_m2": plan.area_m2, "lanes": plan.lane_count}
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"name": "search_area", **props},
             "geometry": {"type": "Polygon", "coordinates": [ring]}},
            {"type": "Feature", "properties": {"name": "route", "altitude_rel_m": plan.altitude_m},
             "geometry": {"type": "LineString", "coordinates": route}},
            {"type": "Feature", "properties": {"name": "home", "amsl_m": home_amsl_m, "home_source": home_source},
             "geometry": {"type": "Point", "coordinates": home}},
        ],
    }


def validate_geojson(gj: dict[str, Any]) -> None:
    """Minimal structural validation (RFC 7946 shapes we emit). Raises ValueError."""
    if gj.get("type") != "FeatureCollection" or not isinstance(gj.get("features"), list):
        raise ValueError("not a FeatureCollection")
    for f in gj["features"]:
        g = f.get("geometry") or {}
        t, c = g.get("type"), g.get("coordinates")
        if t == "Point":
            pts = [c]
        elif t == "LineString":
            pts = c
            if len(c) < 2:
                raise ValueError("LineString needs >= 2 positions")
        elif t == "Polygon":
            if not c or len(c[0]) < 4 or c[0][0] != c[0][-1]:
                raise ValueError("Polygon ring must be closed with >= 4 positions")
            pts = c[0]
        else:
            raise ValueError(f"unexpected geometry {t}")
        for lon, lat in pts:
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError(f"position out of range: {lon}, {lat}")
