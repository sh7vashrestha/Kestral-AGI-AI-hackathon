"""Polygon survey (Phase 2b): geometry, coverage, rejection, safety."""
from __future__ import annotations

import math

import pytest

from models import GeoPoint, LatLon, Mission
from safety import validate
from survey import (PolygonError, build_polygon_survey, geo_to_local, local_to_geo, polygon_area)


def corners_ne(home: GeoPoint, pts: list[tuple[float, float]]) -> list[LatLon]:
    out = []
    for n, e in pts:
        lat, lon = local_to_geo(n, e, home)
        out.append(LatLon(lat=lat, lon=lon))
    return out


def lanes_of(plan):
    wps = [(w.north_m, w.east_m) for w in plan.waypoints]
    return list(zip(wps[0::2], wps[1::2]))


def dist_point_seg(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def inside(poly, p, tol=1e-6):
    sign = 0
    for a, b in zip(poly, poly[1:] + poly[:1]):
        c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(c) <= tol * 100:
            continue
        sg = 1 if c > 0 else -1
        if sign and sg != sign:
            return False
        sign = sg
    return True


def check_coverage(plan, poly_ne, step=3.0):
    half = plan.footprint_w_m / 2.0
    lanes = lanes_of(plan)
    ns = [p[0] for p in poly_ne]
    es = [p[1] for p in poly_ne]
    n = min(ns)
    count = 0
    while n <= max(ns):
        e = min(es)
        while e <= max(es):
            if inside(poly_ne, (n, e)):
                count += 1
                d = min(dist_point_seg((n, e), a, b) for a, b in lanes)
                assert d <= half + 1e-6, f"point {(n, e)} is {d:.1f} m from nearest lane (> {half:.1f})"
            e += step
        n += step
    assert count > 20


def test_square(home, settings):
    sq = [(50, -50), (50, 50), (150, 50), (150, -50)]
    plan = build_polygon_survey(corners_ne(home, sq), 20, home, settings)
    assert plan.mission.shape == "polygon" and len(plan.mission.corners) == 4
    # h=20 -> W=40, s=30 -> ceil(100/30)=4 lanes
    assert plan.lane_count == 4 and len(plan.waypoints) == 8
    assert plan.area_m2 == pytest.approx(10000, rel=1e-3)
    assert plan.estimated_flight_s and plan.battery_note
    check_coverage(plan, sq)
    assert validate(plan, settings).ok


def test_rotated_rectangle(home, settings):
    a = math.radians(30)
    base = [(0, 0), (0, 120), (50, 120), (50, 0)]
    rot = [(n * math.cos(a) - e * math.sin(a) + 60, n * math.sin(a) + e * math.cos(a) - 20) for n, e in base]
    plan = build_polygon_survey(corners_ne(home, rot), 20, home, settings)
    # sweep along the long (120 m) edge -> 50 m across -> 2 lanes
    assert plan.lane_count == 2
    la = lanes_of(plan)[0]
    length = math.hypot(la[1][0] - la[0][0], la[1][1] - la[0][1])
    assert length == pytest.approx(120, abs=0.5)
    check_coverage(plan, rot)


def test_trapezoid(home, settings):
    tz = [(40, -20), (40, 20), (140, 60), (140, -60)]
    plan = build_polygon_survey(corners_ne(home, tz), 20, home, settings)
    poly = tz
    for a, b in lanes_of(plan):
        # slightly shrink toward the centroid to tolerate float error
        for p in (a, b):
            assert inside(poly, p, tol=1e-4)
    check_coverage(plan, tz)
    assert validate(plan, settings).ok


def test_thin_polygon_single_lane(home, settings):
    thin = [(60, -5), (60, 5), (160, 5), (160, -5)]
    plan = build_polygon_survey(corners_ne(home, thin), 20, home, settings)
    assert plan.lane_count == 1 and len(plan.waypoints) == 2
    (a, b), = lanes_of(plan)
    assert abs(a[1]) < 1e-3 and abs(b[1]) < 1e-3   # lane runs down the middle


def test_self_intersecting_rejected(home, settings):
    bow = [(50, -50), (150, 50), (150, -50), (50, 50)]
    with pytest.raises(PolygonError, match="crosses itself"):
        build_polygon_survey(corners_ne(home, bow), 20, home, settings)


def test_concave_uses_hull_with_warning(home, settings):
    arrow = [(50, 0), (150, -50), (110, 0), (150, 50)]
    plan = build_polygon_survey(corners_ne(home, arrow), 20, home, settings)
    assert any("not convex" in w for w in plan.warnings)
    assert plan.area_m2 > polygon_area(arrow)


def test_degenerate_rejected(home, settings):
    line = [(50, 0), (100, 0), (150, 0)]
    with pytest.raises(PolygonError):
        build_polygon_survey(corners_ne(home, line), 20, home, settings)
    dup = [(50, 0), (50, 0), (150, 50), (150, -50)]
    with pytest.raises(PolygonError, match="same place"):
        build_polygon_survey(corners_ne(home, dup), 20, home, settings)


def test_start_near_home(home, settings):
    sq = [(50, -50), (50, 50), (150, 50), (150, -50)]
    plan = build_polygon_survey(corners_ne(home, sq), 20, home, settings)
    first = plan.waypoints[0]
    d_first = math.hypot(first.north_m, first.east_m)
    # the route starts on the edge nearest home (north=50), at an outer lane
    assert first.north_m < 100 and d_first < 100
    # mirrored area on the far side: start flips
    sq2 = [(-n, e) for n, e in sq]
    plan2 = build_polygon_survey(corners_ne(home, sq2), 20, home, settings)
    f2 = plan2.waypoints[0]
    assert f2.north_m < 0
    assert plan.path_length_m == pytest.approx(plan2.path_length_m, rel=1e-3)


def test_overlap_and_speed_override(home, settings):
    sq = [(50, -50), (50, 50), (150, 50), (150, -50)]
    p = build_polygon_survey(corners_ne(home, sq), 20, home, settings, overlap=0.5, speed_ms=15)
    assert p.line_spacing_m == pytest.approx(20.0, rel=1e-3)
    assert p.lane_count == 5
    assert p.speed_ms <= p.v_max_ms
    assert any("Speed reduced" in w for w in p.warnings)


def test_high_altitude_person_px_warning(home, settings):
    sq = [(50, -50), (50, 50), (150, 50), (150, -50)]
    p = build_polygon_survey(corners_ne(home, sq), 40, home, settings)
    assert any("px" in w for w in p.warnings)


def test_mission_model_validation():
    with pytest.raises(ValueError):
        Mission(altitude_m=20)                       # rectangle needs sizes
    with pytest.raises(ValueError):
        Mission(shape="polygon", altitude_m=20)      # polygon needs corners
    assert Mission(length_m=10, width_m=10, altitude_m=20).shape == "rectangle"


# ---- safety ----

def test_safety_corner_outside_geofence(home, settings):
    far = [(250, -50), (250, 50), (350, 50), (350, -50)]
    plan = build_polygon_survey(corners_ne(home, far), 20, home, settings)
    r = validate(plan, settings)
    assert not r.ok and any("Corner" in e and "geofence" in e for e in r.errors)


def test_safety_area_too_big(home, settings_factory):
    s = settings_factory(max_area_m2=5000.0)
    sq = [(50, -50), (50, 50), (150, 50), (150, -50)]
    r = validate(build_polygon_survey(corners_ne(home, sq), 20, home, s), s)
    assert not r.ok and any("limit" in e and "m²" in e for e in r.errors)


def test_safety_area_too_small(home, settings):
    tiny = [(50, 0), (50, 5), (55, 5), (55, 0)]
    r = validate(build_polygon_survey(corners_ne(home, tiny), 20, home, settings), settings)
    assert not r.ok and any("minimum" in e for e in r.errors)


def test_safety_altitude_and_path(home, settings):
    sq = [(50, -50), (50, 50), (150, 50), (150, -50)]
    r = validate(build_polygon_survey(corners_ne(home, sq), 80, home, settings), settings)
    assert not r.ok and any("maximum" in e for e in r.errors)
    big = [(10, -250), (10, 250), (250, 250), (250, -250)]
    plan = build_polygon_survey(corners_ne(home, big), 5, home, settings)
    r = validate(plan, settings)
    assert not r.ok and any("Total path" in e for e in r.errors)
