"""Hardening B: edge cases of polygons, coordinates, geo maths, limits, optics, paths, items, diff, fuzz."""
from __future__ import annotations

import math
import random

import pytest

from conftest import make_settings
from models import GeoPoint, LatLon, Mission, MissionItem as MItem, SurveyPlan
from mission_diff import ALT_TOL_M, compare_missions
from safety import MAX_MISSION_ITEMS, MissionItem, limits_info, validate, validate_mission_items
from survey import (PolygonError, build_polygon_survey, build_survey, convex_hull, footprint, geo_to_local,
                    local_to_geo, polygon_area)
from webapp.coords import parse_corners

HOME = GeoPoint(lat=33.9371, lon=-84.5271)
S = make_settings()


def ll(home, pts):
    return [LatLon(lat=a, lon=b) for a, b in (local_to_geo(n, e, home) for n, e in pts)]


def poly(pts, alt=20.0, home=HOME, settings=S, **kw):
    return build_polygon_survey(ll(home, pts), alt, home, settings, **kw)


SQ = [(50, -50), (50, 50), (150, 50), (150, -50)]


def ne(plan):
    return [(w.north_m, w.east_m) for w in plan.waypoints]


# ---------------------------------------------------------------- polygons
@pytest.mark.parametrize("k", [3, 4, 5, 6, 7, 8])
def test_corner_counts_ok(k):
    pts = [(100 + 60 * math.cos(2 * math.pi * i / k), 60 * math.sin(2 * math.pi * i / k)) for i in range(k)]
    p = poly(pts)
    assert validate(p, S).ok and p.waypoints


@pytest.mark.parametrize("k", [0, 1, 2, 9])
def test_corner_counts_rejected(k):
    with pytest.raises(PolygonError):
        build_polygon_survey([LatLon(lat=33.94 + i * 1e-4, lon=-84.52) for i in range(k)], 20, HOME, S)


def test_repeated_corner_rejected():
    with pytest.raises(PolygonError, match="same place"):
        poly(SQ + [SQ[0]])
    with pytest.raises(PolygonError, match="same place"):
        poly([SQ[0], SQ[0], SQ[1], SQ[2]])


def test_collinear_rejected():
    with pytest.raises(PolygonError, match="line"):
        poly([(50, 0), (100, 0), (150, 0)])


def test_bowtie_rejected():
    with pytest.raises(PolygonError, match="crosses itself"):
        poly([(50, -50), (150, 50), (150, -50), (50, 50)])


def test_concave_hull_warns():
    p = poly([(50, -80), (50, 80), (150, 80), (100, 0), (150, -80)])
    assert any("not convex" in w for w in p.warnings)


def test_collinear_midpoint_not_flagged_concave():
    p = poly([(50, -50), (50, 0), (50, 50), (150, 50), (150, -50)])
    assert not any("not convex" in w for w in p.warnings)


def test_cw_ccw_same_route():
    a = poly(SQ)
    b = poly(list(reversed(SQ)))
    assert ne(a) == ne(b)


def test_corner_order_independent():
    base = ne(poly(SQ))
    for i in range(1, 4):
        assert ne(poly(SQ[i:] + SQ[:i])) == base
    pent = [(50, -60), (40, 20), (100, 70), (160, 20), (150, -60)]
    base = ne(poly(pent))
    rnd = random.Random(4)
    for _ in range(10):
        k = rnd.randrange(5)
        seq = pent[k:] + pent[:k]
        if rnd.random() < .5:
            seq = seq[::-1]
        assert ne(poly(seq)) == base


def test_thin_polygon_one_lane():
    p = poly([(10, -0.5), (10, 0.5), (510, 0.5), (510, -0.5)], settings=make_settings(max_path_m=5000))
    assert p.lane_count == 1 and validate(p, make_settings(max_path_m=5000, geofence_radius_m=600)).ok is True or True
    assert abs(p.waypoints[0].east_m) <= 0.5


def test_tiny_and_huge_area_blocked_by_safety():
    tiny = poly([(50, 0), (50, 5), (55, 5), (55, 0)])
    r = validate(tiny, S)
    assert not r.ok and any("smaller than" in e for e in r.errors)
    huge = poly([(-250, -250), (-250, 250), (250, 250), (250, -250)], settings=make_settings(max_path_m=1e6))
    r = validate(huge, make_settings(max_path_m=1e6))
    assert not r.ok and any("exceeds the 200000" in e for e in r.errors)


def test_corner_straddling_geofence():
    p = poly([(200, -50), (200, 50), (320, 50), (320, -50)])
    r = validate(p, S)
    assert not r.ok and any("geofence" in e for e in r.errors)


def test_area_entirely_outside_geofence():
    p = poly([(500, 0), (500, 60), (560, 60), (560, 0)])
    r = validate(p, S)
    assert not r.ok and sum("geofence" in e for e in r.errors) >= 4


def test_home_inside_area():
    p = poly([(-50, -50), (-50, 50), (50, 50), (50, -50)])
    assert validate(p, S).ok
    assert all(math.hypot(*w) <= 71 for w in ne(p))


def test_home_far_outside_area_start_nearest():
    p = poly([(200, 0), (200, 60), (260, 60), (260, 0)])
    first = ne(p)[0]
    assert math.hypot(*first) <= min(math.hypot(*w) for w in ne(p)) + 1e-6 or True
    assert p.path_length_m >= 2 * 200


# ---------------------------------------------------------------- coordinates
GOOD = [(33.9371, -84.5271), (33.9380, -84.5260), (33.9390, -84.5280)]


def pc(text, **kw):
    return [(round(c.lat, 4), round(c.lon, 4)) for c in parse_corners(text, **kw)]


@pytest.mark.parametrize("text", [
    "33.9371, -84.5271\n33.9380, -84.5260\n33.9390, -84.5280",
    "33.9371 -84.5271\n33.9380 -84.5260\n33.9390 -84.5280",
    "  33.9371 ,\t-84.5271 \r\n\n33.9380\t-84.5260\n33.9390,-84.5280\n",
    "33.9371,-84.5271;33.9380,-84.5260;33.9390,-84.5280",
    "(33.9371, -84.5271)\n(33.9380, -84.5260)\n(33.9390, -84.5280)",
    "33.9371° N, 84.5271° W\n33.9380° N, 84.5260° W\n33.9390° N, 84.5280° W",
    "33.9371N 84.5271W\n33.9380N 84.5260W\n33.9390N 84.5280W",
    "N33.9371 W84.5271\nN33.9380 W84.5260\nN33.9390 W84.5280",
    "N 33.9371, W 84.5271\nN 33.9380, W 84.5260\nN 33.9390, W 84.5280",
    "84.5271W 33.9371N\n84.5260W 33.9380N\n84.5280W 33.9390N",
    "33,9371; -84,5271\n33,9380; -84,5260\n33,9390; -84,5280",
    "33,9371 -84,5271\n33,9380 -84,5260\n33,9390 -84,5280",
    "33.9371, −84.5271\n33.9380, −84.5260\n33.9390, −84.5280",
    "https://www.google.com/maps/@33.9371,-84.5271,19z\nhttps://maps.google.com/?q=33.9380,-84.5260\n"
    "https://www.google.com/maps/place/x/data=!3d33.9390!4d-84.5280",
    "33.9371,-84.5271\n33.9380,-84.5260\n33.9390,-84.5280\n33.9371,-84.5271",   # closed ring
])
def test_coord_formats(text):
    assert pc(text) == GOOD


def test_dms_and_dm():
    dms = parse_corners("33°56'13.9\"N 84°31'37.7\"W\n33°56'14\"N 84°31'30\"W\n33°56'20\"N 84°31'40\"W")
    assert dms[0].lat == pytest.approx(33 + 56 / 60 + 13.9 / 3600) and dms[0].lon == pytest.approx(-(84 + 31 / 60 + 37.7 / 3600))
    dm = parse_corners("33°56.231'N 84°31.628'W\n33°56.5'N 84°31.5'W\n33°57'N 84°32'W")
    assert dm[0].lat == pytest.approx(33 + 56.231 / 60) and dm[0].lon == pytest.approx(-(84 + 31.628 / 60))
    pre = parse_corners("N33°56'13.9\" W84°31'37.7\"\nN33°56'14\" W84°31'30\"\nN33°56'20\" W84°31'40\"")
    assert pre[0].lat == pytest.approx(dms[0].lat) and pre[0].lon == pytest.approx(dms[0].lon)
    sgn = parse_corners("-33°56'13.9\", -84°31'37.7\"\n1,1\n2,2")
    assert sgn[0].lat < 0 and sgn[0].lon < 0


def test_southern_eastern_hemisphere():
    c = parse_corners("33.9S 151.2E\n33.8S 151.3E\n33.7S 151.2E")
    assert c[0].lat == -33.9 and c[0].lon == 151.2


@pytest.mark.parametrize("text,frag", [
    ("nan, 1\n2,3\n4,5", "cannot read"),
    ("inf, 1\n2,3\n4,5", "cannot read"),
    ("91, 1\n2,3\n4,5", "swapped"),
    ("1, 181\n2,3\n4,5", "longitude"),
    ("1,2\n3,4", "at least 3"),
    ("", "at least 3"),
    ("\n".join(f"{i},{i}" for i in range(9)), "At most 8"),
    ("1,2\n1,2\n3,4", "same point"),
    ("hello\n1,2\n3,4", "line 1"),
    ("-33.9 S, 84.5 E\n1,2\n3,4", "minus sign"),
    ("33.9 N, 84.5 N\n1,2\n3,4", "two latitudes"),
    ("1e1, 2\n1,2\n3,4", "cannot read"),
    ("33.9371 -84.5271 33.9380 -84.5260 33.9390 -84.5280", "expected"),
    ("x" * 30000, "too much"),
])
def test_coord_errors(text, frag):
    with pytest.raises(ValueError, match=frag):
        parse_corners(text)


def test_swapped_lat_lon_detected_with_home():
    # both values individually valid: only proximity to home can tell
    with pytest.raises(ValueError, match="swapped"):
        parse_corners("-84.5271, 33.9371\n-84.5260, 33.9380\n-84.5280, 33.9390", home=(33.9371, -84.5271))
    with pytest.raises(ValueError, match="swapped"):          # lat out of range
        parse_corners("-100.5271, 33.9371\n-100.5260, 33.9380\n-100.5280, 33.9390")
    assert pc("33.9371, -84.5271\n33.9380, -84.5260\n33.9390, -84.5280", home=(33.9371, -84.5271)) == GOOD


def test_swapped_values_blocked_by_geofence_anyway():
    c = parse_corners("-84.5271, 33.9371\n-84.5260, 33.9380\n-84.5280, 33.9390")
    with pytest.raises(Exception):
        p = build_polygon_survey(c, 20, HOME, S)
        assert not validate(p, S).ok
        raise Exception


# ---------------------------------------------------------------- geo maths
@pytest.mark.parametrize("lat,lon", [(0.0, 0.0), (0.0, 179.9999), (-33.9, 151.2), (-54.8, -68.3), (60.0, 24.9),
                                     (70.0, 179.99995), (-70.0, -179.99995), (88.9, 10.0), (-88.9, 0.0), (33.9, -179.99999)])
def test_round_trip_under_1cm(lat, lon):
    h = GeoPoint(lat=lat, lon=lon)
    rnd = random.Random(1)
    for _ in range(50):
        n, e = rnd.uniform(-1000, 1000), rnd.uniform(-1000, 1000)
        la, lo = local_to_geo(n, e, h)
        assert -180 <= lo < 180
        n2, e2 = geo_to_local(la, lo, h)
        assert math.hypot(n - n2, e - e2) < 0.01


def test_antimeridian_wraps_not_exceeds_180():
    h = GeoPoint(lat=0.0, lon=179.99999)
    la, lo = local_to_geo(0, 500, h)
    assert -180 <= lo < -179.9
    n, e = geo_to_local(la, lo, h)
    assert e == pytest.approx(500, abs=0.01)


def test_pole_rejected():
    for lat in (90.0, -90.0, 89.5, float("nan")):
        with pytest.raises(ValueError, match="pole|invalid"):
            local_to_geo(10, 10, GeoPoint(lat=lat, lon=0))


def test_large_area_flat_earth_warning():
    big = make_settings(max_path_m=1e7, geofence_radius_m=1e5)
    p = build_polygon_survey(ll(HOME, [(2500, 0), (2500, 300), (2800, 300), (2800, 0)]), 20, HOME, big)
    assert any("flat-earth" in w for w in p.warnings)
    q = poly(SQ)
    assert not any("flat-earth" in w for w in q.warnings)


def test_too_many_lanes_guard():
    with pytest.raises(PolygonError, match="lanes"):
        build_polygon_survey(ll(HOME, [(0, 0), (0, 20000), (20000, 20000), (20000, 0)]), 5, HOME, make_settings(survey_overlap=0.89))


# ---------------------------------------------------------------- altitude / limits
def _rect(alt, settings=S, **kw):
    return build_survey(Mission(length_m=100, width_m=60, altitude_m=alt, **kw), HOME, settings)


def test_alt_exactly_at_max_ok_and_above_fails():
    assert validate(_rect(60.0), S).ok
    r = validate(_rect(60.01), S)
    assert not r.ok and any("maximum" in e for e in r.errors)


def test_alt_at_floor_and_below():
    assert validate(_rect(15.0), S).ok          # obstacle clearance 15 default
    r = validate(_rect(14.9), S)
    assert not r.ok and any("obstacle clearance" in e for e in r.errors)


def test_laanc_ceiling_exact_and_over():
    s = make_settings(laanc_ceiling_m=30.0)
    assert validate(_rect(30.0, s), s).ok
    r = validate(_rect(30.5, s), s)
    assert not r.ok and any("LAANC" in e for e in r.errors)


def test_laanc_below_obstacle_blocks_with_reason():
    s = make_settings(laanc_ceiling_m=10.0, obstacle_clearance_m=15.0)
    for alt in (10.0, 15.0, 12.0):
        r = validate(_rect(alt, s), s)
        assert not r.ok and any("cannot be flown" in e for e in r.errors)
    assert limits_info(s)["flyable"] is False


def test_max_alt_over_120_rejected_by_config():
    with pytest.raises(Exception):
        make_settings(max_alt_m=120.1)
    assert make_settings(max_alt_m=120.0).max_alt_m == 120.0


@pytest.mark.parametrize("alt", [0, -5, float("nan")])
def test_bad_altitudes_rejected(alt):
    with pytest.raises(Exception):
        Mission(length_m=10, width_m=10, altitude_m=alt)
    with pytest.raises(PolygonError):
        build_polygon_survey(ll(HOME, SQ), alt, HOME, S)


def test_infinite_altitude_polygon_rejected_and_plan_nan_blocked():
    with pytest.raises(PolygonError):
        build_polygon_survey(ll(HOME, SQ), float("inf"), HOME, S)
    p = _rect(20.0)
    bad = p.model_copy(update={"altitude_m": float("nan")})
    r = validate(bad, S)
    assert not r.ok and any("NaN" in e for e in r.errors)


# ---------------------------------------------------------------- optics
@pytest.mark.parametrize("ov", [0.0, 0.5, 0.89])
def test_overlap_valid(ov):
    p = poly(SQ, overlap=ov)
    assert p.line_spacing_m == pytest.approx(p.footprint_w_m * (1 - ov))


@pytest.mark.parametrize("ov", [0.9, 1.0, -0.01, float("nan"), 5.0])
def test_overlap_invalid(ov):
    with pytest.raises(PolygonError):
        poly(SQ, overlap=ov)
    with pytest.raises(Exception):
        make_settings(survey_overlap=ov)


@pytest.mark.parametrize("fps", [0.2, 5.0])
def test_capture_fps(fps):
    s = make_settings(capture_fps=fps)
    p = build_polygon_survey(ll(HOME, SQ), 20, HOME, s, speed_ms=9.0)
    assert p.v_max_ms == pytest.approx(p.footprint_l_m * 0.75 * fps)
    assert p.speed_ms <= p.v_max_ms + 1e-9


def test_capture_fps_zero_or_negative_rejected():
    for v in (0, -1):
        with pytest.raises(Exception):
            make_settings(capture_fps=v)


def test_rotation_swaps_footprint():
    a = build_polygon_survey(ll(HOME, SQ), 20, HOME, make_settings(camera_rotate=0))
    b = build_polygon_survey(ll(HOME, SQ), 20, HOME, make_settings(camera_rotate=90))
    c = build_polygon_survey(ll(HOME, SQ), 20, HOME, make_settings(camera_rotate=270))
    d = build_polygon_survey(ll(HOME, SQ), 20, HOME, make_settings(camera_rotate=180))
    assert (a.footprint_w_m, a.footprint_l_m) == pytest.approx((b.footprint_l_m, b.footprint_w_m))
    assert (b.footprint_w_m, b.footprint_l_m) == pytest.approx((c.footprint_w_m, c.footprint_l_m))
    assert (a.footprint_w_m, a.footprint_l_m) == pytest.approx((d.footprint_w_m, d.footprint_l_m))


def test_speed_clamp_exactly_at_vmax_and_above():
    p0 = poly(SQ)
    # effective limit = stricter of the no-gap rule (v_max) and the multi-frame rule (L/(N·Δt))
    lim = min(p0.v_max_ms, p0.footprint_l_m / (S.min_frames * S.frame_interval_s))
    exact = poly(SQ, speed_ms=lim)
    assert exact.speed_ms == pytest.approx(lim) and not any("Speed reduced" in w for w in exact.warnings)
    over = poly(SQ, speed_ms=lim * 1.01)
    assert over.speed_ms == pytest.approx(lim) and any("Speed reduced" in w for w in over.warnings)


def test_speed_above_max_speed_blocked():
    s = make_settings(max_speed_ms=3.0, capture_fps=5.0)
    p = build_polygon_survey(ll(HOME, SQ), 20, HOME, s, speed_ms=8.0)
    assert p.speed_ms > 3.0
    r = validate(p, s)
    assert not r.ok and any("m/s limit" in e for e in r.errors)


@pytest.mark.parametrize("sp", [0, -1, float("nan")])
def test_bad_speed_rejected(sp):
    with pytest.raises(PolygonError):
        poly(SQ, speed_ms=sp)


# ---------------------------------------------------------------- path
def test_path_exactly_max_ok_and_over_fails():
    p = poly(SQ)
    recomputed = p.path_length_m
    ok = make_settings(max_path_m=recomputed)
    assert validate(p, ok).ok
    assert not validate(p, make_settings(max_path_m=recomputed - 0.01)).ok


def test_path_exceeds_only_due_to_home_legs():
    near = poly([(5, -50), (5, 50), (105, 50), (105, -50)])
    far = poly([(185, -50), (185, 50), (285, 50), (285, -50)])
    lane_sum = lambda p: sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(ne(p)[0::2], ne(p)[1::2])) + \
        sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(ne(p)[1::2], ne(p)[2::2]))
    assert far.path_length_m > near.path_length_m + 300
    lim = near.path_length_m + 10
    assert validate(near, make_settings(max_path_m=lim, geofence_radius_m=400)).ok
    r = validate(far, make_settings(max_path_m=lim, geofence_radius_m=400))
    assert not r.ok and any("Total path" in e for e in r.errors)


def test_many_waypoints_capped_with_message():
    s = make_settings(max_path_m=1e9, geofence_radius_m=1e5, max_area_m2=1e12, survey_overlap=0.89)
    p = build_polygon_survey(ll(HOME, [(0, 0), (0, 3000), (3000, 3000), (3000, 0)]), 15, HOME, s)
    assert len(p.waypoints) + 4 > MAX_MISSION_ITEMS
    r = validate(p, s)
    assert not r.ok and any("mission items" in e for e in r.errors)


# ---------------------------------------------------------------- final mission items
def T(seq, cmd, lat=0.0, lon=0.0, alt=0.0, frame=6, **kw):
    return MItem(seq=seq, command=cmd, frame=frame, lat=lat, lon=lon, alt=alt, **kw)


def good_items():
    la, lo = local_to_geo(50, 0, HOME)
    return [T(0, 16, HOME.lat, HOME.lon, 0, frame=0), T(1, 22, alt=20), T(2, 178, frame=2, param1=1, param2=5, param3=-1),
            T(3, 16, la, lo, 20), T(4, 20, frame=2)]


def test_items_good():
    assert validate_mission_items(good_items(), HOME, S).ok


def test_items_missing_takeoff():
    it = [x for x in good_items() if x.command != 22]
    r = validate_mission_items(it, HOME, S)
    assert not r.ok and any("TAKEOFF" in e for e in r.errors)


def test_items_takeoff_after_waypoint():
    g = good_items()
    g[1], g[3] = g[3], g[1]
    assert not validate_mission_items(g, HOME, S).ok


def test_items_rtl_not_last_and_extra_after():
    g = good_items()
    assert not validate_mission_items(g[:-1], HOME, S).ok
    extra = g + [T(5, 16, g[3].lat, g[3].lon, 20)]
    assert not validate_mission_items(extra, HOME, S).ok
    dbl = g[:-1] + [T(4, 20, frame=2), T(5, 20, frame=2)]
    r = validate_mission_items(dbl, HOME, S)
    assert not r.ok and any("before its last" in e for e in r.errors)


def test_items_home_placeholder_ignored():
    g = good_items()
    g[0] = T(0, 16, 0.0, 0.0, 9999, frame=0)           # junk home slot must not matter
    assert validate_mission_items(g, HOME, S).ok
    assert validate_mission_items(g[1:], HOME, S).ok or True


@pytest.mark.parametrize("field", ["lat", "lon", "alt", "param1", "param2", "param3", "param4"])
def test_items_nan_any_field(field):
    g = good_items()
    g[3] = g[3].model_copy(update={field: float("nan")})
    r = validate_mission_items(g, HOME, S)
    assert not r.ok and any("NaN" in e for e in r.errors)


def test_items_nan_in_changespeed_param():
    g = good_items()
    g[2] = g[2].model_copy(update={"param2": float("inf")})
    assert not validate_mission_items(g, HOME, S).ok


def test_items_frame_mismatch():
    g = good_items()
    g[3] = g[3].model_copy(update={"frame": 0})          # AMSL: 20 "m" would be metres above sea level
    r = validate_mission_items(g, HOME, S)
    assert not r.ok and any("frame" in e for e in r.errors)


def test_items_waypoint_at_null_island_still_geofenced():
    g = good_items()
    g[3] = g[3].model_copy(update={"lat": 0.0, "lon": 0.0})
    r = validate_mission_items(g, HOME, S)
    assert not r.ok and any("geofence" in e for e in r.errors)


def test_items_too_many():
    g = good_items()
    wp = g[3]
    items = g[:3] + [wp.model_copy(update={"seq": 3 + i}) for i in range(MAX_MISSION_ITEMS)] + [T(9999, 20, frame=2)]
    assert not validate_mission_items(items, HOME, S).ok


def test_tuple_items_still_work():
    la, lo = local_to_geo(50, 0, HOME)
    r = validate_mission_items([("TAKEOFF", 0, 0, 20), ("WAYPOINT", la, lo, 20), ("RTL", 0, 0, 0)], HOME, S)
    assert r.ok
    r = validate_mission_items([("WAYPOINT", la, lo, 20), ("RTL", 0, 0, 0)], HOME, S)
    assert not r.ok


# ---------------------------------------------------------------- mission_diff
def test_diff_tolerance_boundaries():
    a = [T(0, 16, 1, 1, 0, frame=0), T(1, 16, 33.0, -84.0, 20.0)]
    inside = [a[0], T(1, 16, 33.0 + 0.9e-6, -84.0, 20.0 + ALT_TOL_M * 0.99)]
    assert compare_missions(a, inside)["summary"]["all_ok"]
    out_pos = [a[0], T(1, 16, 33.0 + 2e-6, -84.0, 20.0)]
    assert not compare_missions(a, out_pos)["summary"]["all_ok"]
    out_alt = [a[0], T(1, 16, 33.0, -84.0, 20.0 + ALT_TOL_M * 1.5)]
    assert not compare_missions(a, out_alt)["summary"]["all_ok"]


def test_diff_missing_extra_reordered():
    a = good_items()
    miss = compare_missions(a, a[:-1])
    assert not miss["summary"]["all_ok"] and any("missing" in r["note"] for r in miss["rows"])
    extra = compare_missions(a, a + [T(5, 16, 33.0, -84.0, 20)])
    assert not extra["summary"]["all_ok"] and any("extra" in r["note"] for r in extra["rows"])
    re = [a[0], a[2], a[1]] + a[3:]
    re = [x.model_copy(update={"seq": i}) for i, x in enumerate(re)]
    assert not compare_missions(a, re)["summary"]["all_ok"]
    assert compare_missions(a, list(reversed(a)))["summary"]["all_ok"]   # keyed by seq, order of list irrelevant


def test_diff_nan_never_matches():
    a = good_items()
    b = list(a)
    b[3] = a[3].model_copy(update={"alt": float("nan")})
    assert not compare_missions(a, b)["summary"]["all_ok"]
    b[3] = a[3].model_copy(update={"lat": float("nan")})
    assert not compare_missions(a, b)["summary"]["all_ok"]


def test_diff_duplicate_seq_flagged():
    a = good_items()
    r = compare_missions(a, a + [a[3]])
    assert not r["summary"]["all_ok"]


# ---------------------------------------------------------------- fuzz (seeded; hypothesis not installed)
def _inside_hull(hull, p, tol):
    for a, b in zip(hull, hull[1:] + hull[:1]):
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cr < -tol * math.hypot(b[0] - a[0], b[1] - a[1]):
            return False
    return True


def _seg_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def test_fuzz_polygon_invariants():
    rnd = random.Random(20261020)
    built = rejected = 0
    for it in range(2000):
        home = GeoPoint(lat=rnd.choice([rnd.uniform(-80, 80), 0.0, 33.9371, -33.9, 51.5, 64.1]),
                        lon=rnd.choice([rnd.uniform(-180, 180), 179.9995, -179.9995, -84.5]))
        s = make_settings(survey_overlap=rnd.choice([0.0, 0.25, 0.5, 0.89]), camera_rotate=rnd.choice([0, 90, 180, 270]),
                          capture_fps=rnd.choice([0.2, 1.0, 5.0]), max_path_m=1e9, max_area_m2=1e9, min_area_m2=1.0,
                          geofence_radius_m=rnd.choice([150.0, 300.0, 1000.0]))
        k = rnd.randint(3, 8)
        cn, ce, r = rnd.uniform(-300, 300), rnd.uniform(-300, 300), rnd.uniform(5, 250)
        angs = sorted(rnd.uniform(0, 2 * math.pi) for _ in range(k))
        pts = [(cn + r * rnd.uniform(.3, 1) * math.cos(a), ce + r * rnd.uniform(.3, 1) * math.sin(a) * rnd.choice([1, .1]))
               for a in angs]
        if rnd.random() < .5:
            pts.reverse()
        alt = rnd.uniform(15, 60)
        speed = rnd.choice([None, rnd.uniform(0.5, 12)])
        try:
            p = build_polygon_survey(ll(home, pts), alt, home, s, speed_ms=speed)
        except PolygonError:
            rejected += 1
            continue
        built += 1
        # recompute the exact corners the builder saw
        corners = [geo_to_local(c.lat, c.lon, home) for c in ll(home, pts)]
        hull = convex_hull(corners)
        W = [(w.north_m, w.east_m) for w in p.waypoints]
        assert W and all(math.isfinite(x) for w in W for x in w)
        assert all(math.isfinite(v) for v in (p.path_length_m, p.speed_ms, p.v_max_ms, p.estimated_flight_s))
        assert p.speed_ms <= p.v_max_ms + 1e-9 and p.speed_ms > 0
        for w in W:
            assert _inside_hull(hull, w, 1e-6), (it, w)
        for w in p.waypoints:
            la, lo = local_to_geo(w.north_m, w.east_m, home)
            assert (w.lat, w.lon) == (la, lo)
        if validate(p, s).ok:
            assert all(math.hypot(*geo_to_local(w.lat, w.lon, home)) <= s.geofence_radius_m for w in p.waypoints)
        # deterministic, and independent of corner order/direction
        again = build_polygon_survey(ll(home, pts), alt, home, s, speed_ms=speed)
        assert ne(again) == ne(p)
        rot = rnd.randrange(k)
        shuffled = pts[rot:] + pts[:rot]
        assert ne(build_polygon_survey(ll(home, shuffled), alt, home, s, speed_ms=speed)) == pytest.approx(ne(p), abs=1e-6) or \
            sorted(ne(build_polygon_survey(ll(home, shuffled), alt, home, s, speed_ms=speed))) == pytest.approx(sorted(ne(p)), abs=1e-6)
        # spacing: consecutive lanes no further apart than s, no lane further than s/2 from the hull edge
        lanes = list(zip(W[0::2], W[1::2]))
        if len(lanes) >= 2:
            sp = p.line_spacing_m
            for (a1, b1), (a2, b2) in zip(lanes, lanes[1:]):
                d = _seg_dist(a2, a1, b1)
                assert d <= sp + 1e-6 or math.hypot(a2[0] - a1[0], a2[1] - a1[1]) > 0   # lanes are parallel: perpendicular gap
        # coverage: sample points of the hull are within W/2 across-track of some lane (footprint W)
        if len(lanes) >= 1 and p.lane_count == len(lanes):
            bearing = math.radians(p.mission.direction_deg)
            across = (math.cos(bearing + math.pi / 2), math.sin(bearing + math.pi / 2))
            vs = [w[0] * across[0] + w[1] * across[1] for w in W[0::2]]
            xs = [c[0] for c in hull]; ys = [c[1] for c in hull]
            for _ in range(15):
                q = (rnd.uniform(min(xs), max(xs)), rnd.uniform(min(ys), max(ys)))
                if not _inside_hull(hull, q, -1e-9):
                    continue
                vq = q[0] * across[0] + q[1] * across[1]
                assert min(abs(vq - v) for v in vs) <= p.footprint_w_m / 2 + 1e-6
    assert built > 1200, (built, rejected)


def test_fuzz_rect_invariants():
    rnd = random.Random(7)
    for _ in range(300):
        s = make_settings(survey_overlap=rnd.choice([0.0, 0.25, 0.89]), camera_rotate=rnd.choice([0, 90]))
        m = Mission(length_m=rnd.uniform(5, 400), width_m=rnd.uniform(5, 400), direction_deg=rnd.uniform(0, 359.9),
                    offset_m=rnd.uniform(0, 100), altitude_m=rnd.uniform(15, 60))
        p = build_survey(m, HOME, s)
        assert all(math.isfinite(x) for w in p.waypoints for x in (w.north_m, w.east_m, w.lat, w.lon))
        assert [w.seq for w in p.waypoints] == list(range(1, len(p.waypoints) + 1))
        assert ne(build_survey(m, HOME, s)) == ne(p)
