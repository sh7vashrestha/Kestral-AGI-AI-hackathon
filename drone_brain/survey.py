"""Survey geometry: pure math, no I/O.

Conventions: local frame is (north_m, east_m) from HOME; bearings are degrees clockwise
from north. Flat-earth lat/lon conversion (fine for a few hundred metres).
"""
from __future__ import annotations

import math
from typing import Optional

from config import Settings, get_settings
from models import GeoPoint, LatLon, Mission, SurveyPlan, Waypoint

EARTH_RADIUS_M = 6378137.0
MAX_POLE_LAT_DEG = 89.0        # flat-earth cos(lat) scaling is meaningless (and divides by ~0) beyond this
FLAT_EARTH_WARN_M = 2000.0     # flat-earth error grows with distance from home; warn beyond this
MAX_LANES = 2000               # absurd-input guard (the safety layer enforces the real waypoint cap)


# ---------- camera / optics ---------------------------------------------------

def footprint(h_m: float, hfov_deg: float, vfov_deg: float) -> tuple[float, float]:
    """Ground footprint (W across-track, L along-track) at height h: 2·h·tan(FOV/2)."""
    w = 2.0 * h_m * math.tan(math.radians(hfov_deg) / 2.0)
    l = 2.0 * h_m * math.tan(math.radians(vfov_deg) / 2.0)
    return w, l


def line_spacing(footprint_w_m: float, overlap: float) -> float:
    """s = W·(1−overlap)."""
    return footprint_w_m * (1.0 - overlap)


def max_speed(footprint_l_m: float, overlap: float, frame_interval_s: float) -> float:
    """v_max = L·(1−overlap)/Δt — fastest speed that keeps consecutive frames overlapping."""
    return footprint_l_m * (1.0 - overlap) / frame_interval_s


def match_speed(footprint_l_m: float, min_frames: int, frame_interval_s: float) -> float:
    """Fastest speed at which a ground point stays in view for ≥ min_frames frames:
    frames ≈ L/(v·Δt) ≥ N  →  v ≤ L/(N·Δt). Needed for multi-frame target confirmation."""
    return footprint_l_m / (max(1, min_frames) * frame_interval_s)


def clamp_speed(requested: float, footprint_l_m: float, overlap: float, h_m: float,
                s: Settings, warnings: list[str]) -> tuple[float, float]:
    """Return (speed, v_max): the requested speed limited by the no-gap rule (v_max) and the
    multi-frame confirmation rule (match_speed); explains any reduction in `warnings`."""
    v_max = max_speed(footprint_l_m, overlap, s.frame_interval_s)
    v_match = match_speed(footprint_l_m, s.min_frames, s.frame_interval_s)
    speed = min(requested, v_max, v_match)
    if speed < requested:
        if v_match < v_max:
            why = (f"each spot on the ground stays in view for {s.min_frames} frames "
                   f"(needed to confirm a target match) at {h_m:.0f} m altitude")
        else:
            why = f"consecutive frames keep {overlap:.0%} overlap at {h_m:.0f} m altitude"
        warnings.append(f"Speed reduced from {requested:.1f} to {speed:.1f} m/s so {why}.")
    return speed, v_max


def gsd(h_m: float, sensor_width_mm: float, focal_length_mm: float, image_width_px: float) -> float:
    """Ground sample distance in m/px: (h·sensor_width)/(focal_length·image_width)."""
    return (h_m * sensor_width_mm) / (focal_length_mm * image_width_px)


def person_pixels(gsd_m: float, person_height_m: float) -> float:
    return person_height_m / gsd_m


def max_alt_for_person_px(min_px: float, person_height_m: float, sensor_width_mm: float,
                          focal_length_mm: float, image_width_px: float) -> float:
    """Highest altitude at which a person is still >= min_px tall."""
    gsd_max = person_height_m / min_px
    return gsd_max * focal_length_mm * image_width_px / sensor_width_mm


# ---------- frames --------------------------------------------------------------

def bearing_unit(bearing_deg: float) -> tuple[float, float]:
    """Unit vector (north, east) for a bearing."""
    b = math.radians(bearing_deg)
    return math.cos(b), math.sin(b)


def rotate(along_m: float, across_m: float, bearing_deg: float) -> tuple[float, float]:
    """Rect-frame (along the bearing, across = 90° clockwise of it) -> local (north, east)."""
    an, ae = bearing_unit(bearing_deg)
    cn, ce = bearing_unit(bearing_deg + 90.0)
    return along_m * an + across_m * cn, along_m * ae + across_m * ce


def _check_home_lat(lat: float) -> None:
    if not math.isfinite(lat) or abs(lat) > MAX_POLE_LAT_DEG:
        raise ValueError(f"Home latitude {lat} is invalid or too close to a pole (|lat| must be <= {MAX_POLE_LAT_DEG:g}).")


def wrap_lon(lon: float) -> float:
    """Wrap a longitude into [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


def local_to_geo(north_m: float, east_m: float, home: GeoPoint) -> tuple[float, float]:
    """Flat-earth: dlat = dn/R·180/π ; dlon = de/(R·cos(lat))·180/π. Longitude wraps at ±180°."""
    _check_home_lat(home.lat)
    dlat = north_m / EARTH_RADIUS_M * 180.0 / math.pi
    dlon = east_m / (EARTH_RADIUS_M * math.cos(math.radians(home.lat))) * 180.0 / math.pi
    return home.lat + dlat, wrap_lon(home.lon + dlon)


def geo_to_local(lat: float, lon: float, home: GeoPoint) -> tuple[float, float]:
    """Inverse of local_to_geo -> (north_m, east_m); the longitude difference is taken the short way
    round, so points either side of the ±180° anti-meridian work."""
    _check_home_lat(home.lat)
    north = math.radians(lat - home.lat) * EARTH_RADIUS_M
    east = math.radians(wrap_lon(lon - home.lon)) * EARTH_RADIUS_M * math.cos(math.radians(home.lat))
    return north, east


def path_length(points_ne: list[tuple[float, float]]) -> float:
    """home(0,0) -> points... -> home horizontal length."""
    if not points_ne:
        return 0.0
    pts = [(0.0, 0.0), *points_ne, (0.0, 0.0)]
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


# ---------- lawnmower -----------------------------------------------------------

def lane_offsets(width_m: float, spacing_m: float) -> list[float]:
    """Cross-track lane positions (relative to the area centre line, left -> right).

    n = ceil(width/s) lanes; the first lane sits half a lane spacing inside the edge. When
    width is not a multiple of s the lanes are spread evenly (spacing <= s), so coverage is
    never thinner than designed and no lane leaves the area."""
    n = max(1, math.ceil(width_m / spacing_m - 1e-9))
    if n > MAX_LANES:
        raise PolygonError(f"That area would need {n} lanes at this altitude; pick a much smaller area or fly higher.")
    step = width_m / n
    return [-width_m / 2.0 + step / 2.0 + k * step for k in range(n)]


def lawnmower_local(mission: Mission, spacing_m: float) -> list[tuple[float, float]]:
    """Waypoints (north, east) at the ends of each lane; lanes parallel to direction_deg,
    alternating direction, starting at the near edge on the left-hand lane."""
    pts: list[tuple[float, float]] = []
    for k, v in enumerate(lane_offsets(mission.width_m, spacing_m)):
        near = -mission.length_m / 2.0 if mission.centered else mission.offset_m
        far = near + mission.length_m
        ends = (near, far) if k % 2 == 0 else (far, near)
        for u in ends:
            pts.append(rotate(u, v, mission.direction_deg))
    return pts


def build_survey(mission: Mission, home: GeoPoint, settings: Optional[Settings] = None) -> SurveyPlan:
    s = settings or get_settings()
    h = mission.altitude_m
    warnings: list[str] = []
    if not math.isfinite(h) or h <= 0:
        raise PolygonError("Altitude must be a positive number.")

    w, l = footprint(h, s.eff_hfov_deg, s.eff_vfov_deg)
    spacing = line_spacing(w, s.survey_overlap)
    requested = mission.speed_ms if mission.speed_ms is not None else s.default_speed_ms
    speed, v_max = clamp_speed(requested, l, s.survey_overlap, h, s, warnings)

    g = gsd(h, s.eff_sensor_width_mm, s.focal_length_mm, s.eff_image_width_px)
    ppx = person_pixels(g, s.person_height_m)
    if ppx < s.min_person_px:
        h_ok = max_alt_for_person_px(s.min_person_px, s.person_height_m, s.eff_sensor_width_mm,
                                     s.focal_length_mm, s.eff_image_width_px)
        warnings.append(
            f"At {h:.0f} m a person is only ~{ppx:.0f} px tall (< {s.min_person_px} px); "
            f"detection may miss people. Fly at {h_ok:.0f} m or lower for >= {s.min_person_px} px."
        )

    local = lawnmower_local(mission, spacing)
    _far_warning(local, warnings)
    waypoints: list[Waypoint] = []
    for i, (n, e) in enumerate(local, start=1):
        lat, lon = local_to_geo(n, e, home)
        waypoints.append(Waypoint(seq=i, north_m=n, east_m=e, lat=lat, lon=lon, alt_m=h))

    return SurveyPlan(
        home=home,
        mission=mission,
        waypoints=waypoints,
        altitude_m=h,
        speed_ms=speed,
        v_max_ms=v_max,
        footprint_w_m=w,
        footprint_l_m=l,
        line_spacing_m=spacing,
        path_length_m=path_length(local),
        gsd_m=g,
        person_px=ppx,
        warnings=warnings,
        area_m2=mission.length_m * mission.width_m,
        lane_count=len(local) // 2,
        **flight_estimate(path_length(local), speed, len(local), h, s),
    )


def _far_warning(local: list[tuple[float, float]], warnings: list[str]) -> None:
    far = max((math.hypot(n, e) for n, e in local), default=0.0)
    if far > FLAT_EARTH_WARN_M:
        warnings.append(f"Part of the area is {far / 1000:.1f} km from home; the flat-earth map maths loses "
                        f"accuracy beyond about {FLAT_EARTH_WARN_M / 1000:.0f} km.")


# ---------- flight time estimate ------------------------------------------------

CLIMB_MS = 2.5       # assumed climb rate for the estimate
DESCENT_MS = 1.5     # assumed landing descent rate
TURN_S = 2.0         # allowance per waypoint (slow-down / turn)
FIXED_S = 20.0       # arming, mode changes, final touchdown


def flight_estimate(path_m: float, speed_ms: float, n_waypoints: int, alt_m: float,
                    settings: Settings) -> dict:
    """Estimated total flight time (s) and a human battery note. Keys match SurveyPlan fields."""
    t = path_m / max(speed_ms, 0.1) + alt_m / CLIMB_MS + alt_m / DESCENT_MS + n_waypoints * TURN_S + FIXED_S
    minutes = t / 60.0
    frac = minutes / max(settings.est_endurance_min, 1.0)
    note = (f"About {minutes:.1f} min in the air, roughly {frac:.0%} of an assumed "
            f"{settings.est_endurance_min:.0f} min endurance. Start with a full battery.")
    if frac > 0.7:
        note += " This is a long flight: consider a smaller area."
    return {"estimated_flight_s": t, "battery_note": note}


# ---------- polygon survey ------------------------------------------------------

class PolygonError(ValueError):
    """The tapped polygon cannot be surveyed; the message is user-facing."""


Pt = tuple[float, float]


def polygon_area(pts: list[Pt]) -> float:
    """Absolute shoelace area."""
    a = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def _cross(o: Pt, a: Pt, b: Pt) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(p1: Pt, p2: Pt, p3: Pt, p4: Pt) -> bool:
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 * d2 != 0 and d3 * d4 != 0


def is_self_intersecting(pts: list[Pt]) -> bool:
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue   # adjacent edges share a vertex
            if _segments_cross(a, b, pts[j], pts[(j + 1) % n]):
                return True
    return False


def convex_hull(pts: list[Pt]) -> list[Pt]:
    """Andrew monotone chain, counter-clockwise, no collinear points."""
    p = sorted(set(pts))
    if len(p) <= 2:
        return p
    lower: list[Pt] = []
    for q in p:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: list[Pt] = []
    for q in reversed(p):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def _lane_chord(poly: list[Pt], v: float) -> Optional[tuple[float, float]]:
    """Intersection of the line y=v with a convex polygon in (u=x, v=y) coords -> (u_min, u_max)."""
    us: list[float] = []
    for (u1, v1), (u2, v2) in zip(poly, poly[1:] + poly[:1]):
        if v1 == v2:
            if abs(v1 - v) < 1e-9:
                us += [u1, u2]
            continue
        if min(v1, v2) - 1e-9 <= v <= max(v1, v2) + 1e-9:
            t = (v - v1) / (v2 - v1)
            us.append(u1 + t * (u2 - u1))
    if not us:
        return None
    return min(us), max(us)


def build_polygon_survey(corners: list[LatLon], altitude_m: float, home: GeoPoint,
                         settings: Optional[Settings] = None, speed_ms: Optional[float] = None,
                         overlap: Optional[float] = None) -> SurveyPlan:
    """Lawnmower survey of a polygon given by lat/lon corners in perimeter order.

    * Self-intersecting polygons (bow-tie) are rejected (PolygonError).
    * Concave polygons are replaced by their convex hull with a warning (lanes are straight
      chords; the hull only adds area, and its vertices are a subset of the tapped corners).
    * Sweep direction is the longest edge (fewest turns). Lanes spaced s = W(1-overlap),
      spread evenly so the first lane is half a spacing inside (see lane_offsets).
    * Of the four ways to traverse the zig-zag (first lane on either side, either direction)
      the one with the shortest home -> path -> home length is used, so the route starts at
      the corner nearest home.
    """
    s = settings or get_settings()
    if not 3 <= len(corners) <= 8:
        raise PolygonError("An area needs between 3 and 8 corners.")
    if altitude_m is None or not math.isfinite(altitude_m) or altitude_m <= 0:
        raise PolygonError("Altitude must be a positive number.")
    ov = s.survey_overlap if overlap is None else overlap
    if not (math.isfinite(ov) and 0.0 <= ov < 0.9):
        raise PolygonError("Overlap must be between 0 and 0.9.")
    if speed_ms is not None and not (math.isfinite(speed_ms) and speed_ms > 0):
        raise PolygonError("Speed must be a positive number.")
    if any(not (math.isfinite(c.lat) and math.isfinite(c.lon)) for c in corners):
        raise PolygonError("A corner has invalid coordinates.")

    ne = [geo_to_local(c.lat, c.lon, home) for c in corners]
    if len(set((round(n, 3), round(e, 3)) for n, e in ne)) < len(ne):
        raise PolygonError("Two corners are in the same place. Move one of them.")
    if is_self_intersecting(ne):
        raise PolygonError("The area outline crosses itself. Tap the corners in order around "
                           "the edge of the area (clockwise or counter-clockwise).")
    area = polygon_area(ne)
    if area < 1.0:
        raise PolygonError("The corners are in a line; they do not enclose an area.")

    warnings: list[str] = []
    hull = convex_hull(ne)
    if len(hull) < 3:
        raise PolygonError("The corners are in a line; they do not enclose an area.")
    if polygon_area(hull) > area * 1.001:
        extra = polygon_area(hull) - area
        warnings.append(
            f"The area is not convex; the survey covers its convex outline instead "
            f"(+{extra:.0f} m\u00b2 extra). Move a corner for a tighter fit."
        )
        area = polygon_area(hull)

    w, l = footprint(altitude_m, s.eff_hfov_deg, s.eff_vfov_deg)
    spacing = line_spacing(w, ov)
    requested = speed_ms if speed_ms is not None else s.default_speed_ms
    speed, v_max = clamp_speed(requested, l, ov, altitude_m, s, warnings)
    g = gsd(altitude_m, s.eff_sensor_width_mm, s.focal_length_mm, s.eff_image_width_px)
    ppx = person_pixels(g, s.person_height_m)
    if ppx < s.min_person_px:
        h_ok = max_alt_for_person_px(s.min_person_px, s.person_height_m, s.eff_sensor_width_mm,
                                     s.focal_length_mm, s.eff_image_width_px)
        warnings.append(
            f"At {altitude_m:.0f} m a person is only ~{ppx:.0f} px tall (< {s.min_person_px} px); "
            f"detection may miss people. Fly at {h_ok:.0f} m or lower for >= {s.min_person_px} px."
        )

    # sweep axis = longest edge of the hull; (u, v) = (along the edge, 90 deg clockwise of it)
    best_edge = max(zip(hull, hull[1:] + hull[:1]),
                    key=lambda ab: math.hypot(ab[1][0] - ab[0][0], ab[1][1] - ab[0][1]))
    bearing = math.degrees(math.atan2(best_edge[1][1] - best_edge[0][1],
                                      best_edge[1][0] - best_edge[0][0])) % 360.0

    def to_uv(p: Pt) -> Pt:
        an, ae = bearing_unit(bearing)
        cn, ce = bearing_unit(bearing + 90.0)
        return p[0] * an + p[1] * ae, p[0] * cn + p[1] * ce

    uv = [to_uv(p) for p in hull]
    vmin, vmax = min(p[1] for p in uv), max(p[1] for p in uv)
    extent = vmax - vmin
    n = max(1, math.ceil(extent / spacing - 1e-9))
    if n > MAX_LANES:
        raise PolygonError(f"That area would need {n} lanes at {altitude_m:.0f} m; pick a much smaller area or fly higher.")
    step = extent / n
    lanes: list[tuple[float, float, float]] = []   # (v, u_lo, u_hi)
    for k in range(n):
        v = vmin + step / 2.0 + k * step
        ch = _lane_chord(uv, v)
        if ch is not None:
            lanes.append((v, ch[0], ch[1]))
    if not lanes:
        raise PolygonError("The area is too thin to survey.")

    def zigzag(order: list[int], first_forward: bool) -> list[Pt]:
        pts: list[Pt] = []
        fwd = first_forward
        for idx in order:
            v, lo, hi = lanes[idx]
            ends = (lo, hi) if fwd else (hi, lo)
            for u in ends:
                pts.append((u, v))
            fwd = not fwd
        # chord of zero length -> duplicate points; drop consecutive duplicates
        out: list[Pt] = []
        for q in pts:
            if not out or math.hypot(q[0] - out[-1][0], q[1] - out[-1][1]) > 1e-6:
                out.append(q)
        return out

    def uv_to_ne(q: Pt) -> Pt:
        return rotate(q[0], q[1], bearing)

    candidates = []
    for order in (list(range(len(lanes))), list(range(len(lanes) - 1, -1, -1))):
        for fwd in (True, False):
            local = [uv_to_ne(q) for q in zigzag(order, fwd)]
            candidates.append((path_length(local), local))
    # equal length (symmetric area): prefer starting at the point nearest home
    plen, local = min(candidates, key=lambda c: (round(c[0], 3), math.hypot(*c[1][0])))

    waypoints: list[Waypoint] = []
    for i, (nn, ee) in enumerate(local, start=1):
        lat, lon = local_to_geo(nn, ee, home)
        waypoints.append(Waypoint(seq=i, north_m=nn, east_m=ee, lat=lat, lon=lon, alt_m=altitude_m))

    _far_warning(local, warnings)
    far_corner = max(math.hypot(n_, e_) for n_, e_ in ne)
    if far_corner > FLAT_EARTH_WARN_M and not any("flat-earth" in w for w in warnings):
        warnings.append(f"A corner is {far_corner / 1000:.1f} km from home; the flat-earth map maths loses accuracy "
                        f"beyond about {FLAT_EARTH_WARN_M / 1000:.0f} km.")
    mission = Mission(shape="polygon", corners=list(corners), altitude_m=altitude_m,
                      speed_ms=speed_ms, direction_deg=bearing)
    return SurveyPlan(
        home=home, mission=mission, waypoints=waypoints, altitude_m=altitude_m, speed_ms=speed,
        v_max_ms=v_max, footprint_w_m=w, footprint_l_m=l, line_spacing_m=spacing,
        path_length_m=plen, gsd_m=g, person_px=ppx, warnings=warnings, area_m2=area,
        lane_count=len(lanes), **flight_estimate(plen, speed, len(local), altitude_m, s),
    )
