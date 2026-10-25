"""Hard safety validation of a SurveyPlan / final mission item list.

Nothing here talks to hardware. Reject (ok=False) on any violation; warnings pass through.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Union

from pydantic import BaseModel

from config import HARD_MAX_ALT_M, Settings, get_settings
import models
from models import GeoPoint, SafetyResult, SurveyPlan
from survey import geo_to_local, path_length, polygon_area

MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
_RTL_NAMES = {"RTL", "RETURN_TO_LAUNCH", "MAV_CMD_NAV_RETURN_TO_LAUNCH"}
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_TAKEOFF = 22
_TAKEOFF_NAMES = {"TAKEOFF", "MAV_CMD_NAV_TAKEOFF"}
_WAYPOINT_NAMES = {"WAYPOINT", "MAV_CMD_NAV_WAYPOINT"}
_REL_FRAMES = {3, 6}           # GLOBAL_RELATIVE_ALT(_INT): altitude above home, the only frame the limits make sense for
_NO_POSITION_FRAMES = {2}      # MAV_FRAME_MISSION, used by non-positional items
# ArduPilot Copter on a Pixhawk stores roughly 700 mission items; stay well below it. A lawnmower
# of 300 lanes is far beyond the 3 km MAX_PATH_M anyway, so this only catches absurd settings.
MAX_MISSION_ITEMS = 600


class MissionItem(BaseModel):
    """Minimal view of one uploaded mission item (command name or MAV_CMD number)."""
    command: Union[str, int]
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0


ItemLike = Union[MissionItem, models.MissionItem, tuple[Union[str, int], float, float, float]]


def _normalise(it: ItemLike) -> Optional[MissionItem]:
    """Convert any accepted item form to the local view. Returns None for the
    ArduPilot home placeholder (models.MissionItem with seq 0), which the
    autopilot overwrites with the real home and is never flown to."""
    if isinstance(it, MissionItem):
        return it
    if isinstance(it, models.MissionItem):
        if it.seq == 0:
            return None
        return MissionItem(command=it.command, lat=it.lat, lon=it.lon, alt_m=it.alt)
    return MissionItem(command=it[0], lat=it[1], lon=it[2], alt_m=it[3])


def ceiling_info(settings: Settings) -> tuple[float, str]:
    """(effective ceiling m, source) = min(MAX_ALT_M, LAANC_CEILING_M if set, 120 m)."""
    cands = [(HARD_MAX_ALT_M, "120 m hard limit"), (settings.max_alt_m, "MAX_ALT_M")]
    laanc = getattr(settings, "laanc_ceiling_m", None)
    if laanc is not None:
        cands.append((laanc, "LAANC"))
    # ties: prefer the most specific source (LAANC > MAX_ALT_M > hard limit)
    order = {"LAANC": 0, "MAX_ALT_M": 1, "120 m hard limit": 2}
    alt, src = min(cands, key=lambda c: (c[0], order[c[1]]))
    return float(alt), src


def effective_max_alt(settings: Settings) -> float:
    return ceiling_info(settings)[0]


def effective_min_alt(settings: Settings) -> float:
    """MIN_ALT_M raised to OBSTACLE_CLEARANCE_M (lanes and RTL must clear poles/goals/trees)."""
    return max(settings.min_alt_m, getattr(settings, "obstacle_clearance_m", 0.0))


def limits_info(settings: Settings) -> dict:
    """Altitude band for the UI / plan preview, with where each bound comes from."""
    ceil, src = ceiling_info(settings)
    floor = effective_min_alt(settings)
    floor_src = "OBSTACLE_CLEARANCE_M" if getattr(settings, "obstacle_clearance_m", 0.0) > settings.min_alt_m else "MIN_ALT_M"
    return {"effective_max_alt_m": ceil, "ceiling_source": src, "laanc_ceiling_m": getattr(settings, "laanc_ceiling_m", None),
            "effective_min_alt_m": floor, "floor_source": floor_src,
            "obstacle_clearance_m": getattr(settings, "obstacle_clearance_m", 0.0),
            "flyable": floor <= ceil}


def _is_rtl(cmd: Union[str, int]) -> bool:
    if isinstance(cmd, int):
        return cmd == MAV_CMD_NAV_RETURN_TO_LAUNCH
    return cmd.strip().upper() in _RTL_NAMES or cmd.strip() == str(MAV_CMD_NAV_RETURN_TO_LAUNCH)


def _is_cmd(cmd: Union[str, int], num: int, names: set) -> bool:
    if isinstance(cmd, int):
        return cmd == num
    c = str(cmd).strip()
    return c.upper() in names or c == str(num)


def _finite(*vals: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)


def _check_alt(alt: float, label: str, settings: Settings, errors: list[str]) -> None:
    ceil, src = ceiling_info(settings)
    floor = effective_min_alt(settings)
    if alt > HARD_MAX_ALT_M:
        errors.append(f"{label}: altitude {alt:.0f} m exceeds the absolute limit of {HARD_MAX_ALT_M:.0f} m.")
    elif alt > settings.max_alt_m:
        errors.append(f"{label}: altitude {alt:.0f} m exceeds the allowed maximum of {settings.max_alt_m:.0f} m.")
    elif alt > ceil:
        errors.append(f"{label}: altitude {alt:.0f} m exceeds the {src} ceiling of {ceil:.0f} m.")
    elif alt < settings.min_alt_m:
        errors.append(f"{label}: altitude {alt:.0f} m is below the minimum of {settings.min_alt_m:.0f} m.")
    elif alt < floor:
        errors.append(f"{label}: altitude {alt:.0f} m is below the obstacle clearance of {floor:.0f} m.")


def _check_band(settings: Settings, errors: list[str]) -> None:
    ceil, src = ceiling_info(settings)
    floor = effective_min_alt(settings)
    if floor > ceil:
        errors.append(f"Obstacle clearance {floor:.0f} m is above the {src} ceiling of {ceil:.0f} m: "
                      f"this area cannot be flown legally and safely.")


def _dedupe(msgs: list[str]) -> list[str]:
    return list(dict.fromkeys(msgs))


def validate(plan: SurveyPlan, settings: Optional[Settings] = None,
             items: Optional[Iterable[ItemLike]] = None) -> SafetyResult:
    """Validate a plan. If ``items`` (final mission list incl. RTL) is given it is checked too
    (must end with RTL; altitude and geofence per item)."""
    s = settings or get_settings()
    errors: list[str] = []

    if not plan.waypoints:
        errors.append("The plan has no waypoints.")
    elif len(plan.waypoints) + 4 > MAX_MISSION_ITEMS:
        errors.append(f"The plan needs {len(plan.waypoints) + 4} mission items; the autopilot memory is "
                      f"limited, so at most {MAX_MISSION_ITEMS} are allowed. Use a smaller area or a higher altitude.")

    scalars = [plan.altitude_m, plan.speed_ms, plan.path_length_m, plan.home.lat, plan.home.lon,
               plan.v_max_ms, plan.line_spacing_m]
    wp_vals = [v for w in plan.waypoints for v in (w.lat, w.lon, w.alt_m, w.north_m, w.east_m)]
    if not _finite(*scalars, *wp_vals):
        errors.append("The plan contains invalid (NaN/infinite) numbers.")
        return SafetyResult(ok=False, errors=_dedupe(errors), warnings=list(plan.warnings))

    _check_band(s, errors)
    _check_alt(plan.altitude_m, "Plan", s, errors)
    for w in plan.waypoints:
        _check_alt(w.alt_m, f"Waypoint {w.seq}", s, errors)

    # geofence: judged on the lat/lon that would actually be uploaded
    for w in plan.waypoints:
        n, e = geo_to_local(w.lat, w.lon, plan.home)
        d = math.hypot(n, e)
        if d > s.geofence_radius_m:
            errors.append(f"Waypoint {w.seq} is {d:.0f} m from home, beyond the {s.geofence_radius_m:.0f} m geofence.")

    # polygon area: every tapped corner inside the geofence, area within limits
    m = plan.mission
    if m.shape == "polygon" and m.corners:
        ne = []
        for i, c in enumerate(m.corners, start=1):
            if not _finite(c.lat, c.lon):
                errors.append(f"Corner {i} has invalid coordinates.")
                continue
            n, e = geo_to_local(c.lat, c.lon, plan.home)
            ne.append((n, e))
            d = math.hypot(n, e)
            if d > s.geofence_radius_m:
                errors.append(f"Corner {i} is {d:.0f} m from home, beyond the {s.geofence_radius_m:.0f} m geofence.")
        if len(ne) == len(m.corners):
            area = max(polygon_area(ne), plan.area_m2 or 0.0)
            if area > s.max_area_m2:
                errors.append(f"Area {area:.0f} m\u00b2 exceeds the {s.max_area_m2:.0f} m\u00b2 limit. Pick a smaller area.")
            elif area < s.min_area_m2:
                errors.append(f"Area {area:.0f} m\u00b2 is smaller than the {s.min_area_m2:.0f} m\u00b2 minimum.")

    # path length: the larger of the claimed value and one recomputed from the waypoints
    recomputed = path_length([(w.north_m, w.east_m) for w in plan.waypoints])
    plen = max(plan.path_length_m, recomputed)
    if plen > s.max_path_m:
        errors.append(f"Total path {plen:.0f} m exceeds the {s.max_path_m:.0f} m limit. Use a smaller area.")

    if plan.speed_ms > s.max_speed_ms:
        errors.append(f"Speed {plan.speed_ms:.1f} m/s exceeds the {s.max_speed_ms:.1f} m/s limit.")
    if plan.speed_ms <= 0:
        errors.append("Speed must be positive.")

    warnings = list(plan.warnings)
    if items is not None:
        r = validate_mission_items(items, plan.home, s)
        errors.extend(r.errors)
        warnings.extend(r.warnings)

    return SafetyResult(ok=not errors, errors=_dedupe(errors), warnings=_dedupe(warnings))


def validate_mission_items(items: Iterable[ItemLike], home: GeoPoint,
                           settings: Optional[Settings] = None) -> SafetyResult:
    """Check the final item list that will be uploaded: non-empty, ends with RTL, every
    positional item inside altitude band and geofence."""
    s = settings or get_settings()
    errors: list[str] = []
    raw = list(items)
    norm: list[MissionItem] = []
    for it in raw:
        n_it = _normalise(it)
        if n_it is not None:
            norm.append(n_it)

    if not norm:
        return SafetyResult(ok=False, errors=["The mission item list is empty."])

    if len(raw) > MAX_MISSION_ITEMS:
        errors.append(f"The mission has {len(raw)} items; at most {MAX_MISSION_ITEMS} are allowed.")

    # raw ArduPilot items: finite params, sane frames
    for it in raw:
        if not isinstance(it, models.MissionItem) or it.seq == 0:
            continue
        if not _finite(it.param1, it.param2, it.param3, it.param4, it.lat, it.lon, it.alt):
            errors.append(f"Mission item {it.seq} contains invalid (NaN/infinite) numbers.")
        positional = _is_cmd(it.command, MAV_CMD_NAV_WAYPOINT, _WAYPOINT_NAMES) or abs(it.lat) + abs(it.lon) > 0
        if positional and it.frame not in _REL_FRAMES:
            errors.append(f"Mission item {it.seq} uses altitude frame {it.frame}; only altitude relative to home "
                          f"(frame 3 or 6) is allowed.")

    rtl_idx = [i for i, it in enumerate(norm) if _is_rtl(it.command)]
    if not norm or not _is_rtl(norm[-1].command):
        errors.append("The mission does not end with Return-To-Launch (RTL).")
    elif len(rtl_idx) > 1:
        errors.append("The mission has an RTL before its last item; nothing after an RTL would ever be flown.")

    first_wp = next((i for i, it in enumerate(norm) if _is_cmd(it.command, MAV_CMD_NAV_WAYPOINT, _WAYPOINT_NAMES)), None)
    if first_wp is not None:
        takeoff = next((i for i, it in enumerate(norm) if _is_cmd(it.command, MAV_CMD_NAV_TAKEOFF, _TAKEOFF_NAMES)), None)
        if takeoff is None or takeoff > first_wp:
            errors.append("The mission has no TAKEOFF before its first waypoint.")

    for i, it in enumerate(norm):
        if not _finite(it.lat, it.lon, it.alt_m):
            errors.append(f"Mission item {i} contains invalid (NaN/infinite) numbers.")
            continue
        if _is_rtl(it.command):
            continue
        is_wp = _is_cmd(it.command, MAV_CMD_NAV_WAYPOINT, _WAYPOINT_NAMES)
        if it.lat == 0.0 and it.lon == 0.0 and not is_wp:
            # item without a position (e.g. DO_CHANGE_SPEED): only the ceiling applies
            if it.alt_m > effective_max_alt(s):
                _check_alt(it.alt_m, f"Mission item {i}", s, errors)
            continue
        _check_alt(it.alt_m, f"Mission item {i}", s, errors)
        n, e = geo_to_local(it.lat, it.lon, home)
        d = math.hypot(n, e)
        if d > s.geofence_radius_m:
            errors.append(f"Mission item {i} is {d:.0f} m from home, beyond the {s.geofence_radius_m:.0f} m geofence.")

    return SafetyResult(ok=not errors, errors=_dedupe(errors))
