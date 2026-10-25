import math

import pytest

from models import GeoPoint, Mission
from survey import (build_survey, bearing_unit, footprint, geo_to_local, gsd, lane_offsets,
                    line_spacing, local_to_geo, max_alt_for_person_px, max_speed, path_length,
                    person_pixels, rotate)


def test_footprint_hand_numbers():
    w, l = footprint(20.0, 90.0, 60.0)
    assert w == pytest.approx(40.0)                        # 2*20*tan(45)
    assert l == pytest.approx(2 * 20 * math.tan(math.radians(30)))  # 23.094
    assert l == pytest.approx(23.0940, abs=1e-3)


def test_footprint_scales_linearly_with_altitude():
    w1, _ = footprint(10, 82.1, 52.2)
    w2, _ = footprint(40, 82.1, 52.2)
    assert w2 == pytest.approx(4 * w1)


def test_line_spacing():
    assert line_spacing(40.0, 0.25) == pytest.approx(30.0)
    assert line_spacing(40.0, 0.0) == pytest.approx(40.0)


def test_max_speed():
    assert max_speed(23.094, 0.25, 1.0) == pytest.approx(17.3205, abs=1e-3)
    assert max_speed(23.094, 0.25, 2.0) == pytest.approx(8.66, abs=1e-2)


def test_gsd_and_person_px():
    g = gsd(20.0, 4.8, 3.67, 1920)
    assert g == pytest.approx(96.0 / 7046.4, rel=1e-9)   # 0.013624 m/px
    assert person_pixels(g, 1.7) == pytest.approx(124.78, abs=0.05)


def test_max_alt_for_person_px():
    h = max_alt_for_person_px(80, 1.7, 4.8, 3.67, 1920)
    assert h == pytest.approx(31.19, abs=0.01)
    # round trip: at that altitude a person is exactly 80 px
    assert person_pixels(gsd(h, 4.8, 3.67, 1920), 1.7) == pytest.approx(80.0)


def test_bearing_unit_and_rotate():
    n, e = bearing_unit(90)
    assert (n, e) == pytest.approx((0.0, 1.0), abs=1e-12)
    # facing east, "across" (clockwise of heading) points south
    n, e = rotate(10.0, 5.0, 90.0)
    assert (n, e) == pytest.approx((-5.0, 10.0), abs=1e-9)
    n, e = rotate(10.0, 5.0, 0.0)
    assert (n, e) == pytest.approx((10.0, 5.0), abs=1e-9)


def test_local_to_geo_hand_numbers(home):
    lat, lon = local_to_geo(111.0, 0.0, home)
    assert lat - 47.0 == pytest.approx(111.0 / 6378137.0 * 180 / math.pi)
    assert lat - 47.0 == pytest.approx(0.000997, abs=1e-6)
    lat, lon = local_to_geo(0.0, 100.0, home)
    assert lat == 47.0
    assert lon - 8.0 == pytest.approx(100.0 / (6378137.0 * math.cos(math.radians(47.0))) * 180 / math.pi)


@pytest.mark.parametrize("n,e", [(0, 0), (250, -120), (-300, 300), (1.5, 0.2)])
def test_geo_round_trip(home, n, e):
    lat, lon = local_to_geo(n, e, home)
    n2, e2 = geo_to_local(lat, lon, home)
    assert n2 == pytest.approx(n, abs=1e-3)
    assert e2 == pytest.approx(e, abs=1e-3)


def test_path_length():
    assert path_length([]) == 0.0
    assert path_length([(3.0, 4.0)]) == pytest.approx(10.0)  # out and back


def test_lane_offsets():
    # width 60, s=30 -> 2 lanes, first s/2 inside edge
    assert lane_offsets(60, 30) == pytest.approx([-15.0, 15.0])
    assert lane_offsets(30, 30) == pytest.approx([0.0])
    assert lane_offsets(10, 30) == pytest.approx([0.0])          # narrower than a lane: centre line
    # width 70, s=30 -> 3 lanes, spread evenly (spacing 23.33 <= 30)
    offs = lane_offsets(70, 30)
    assert len(offs) == 3
    assert offs[0] == pytest.approx(-35 + 70 / 6)
    assert offs[1] - offs[0] <= 30 + 1e-9


def _flat(pts):
    return [c for p in pts for c in p]


def _mission(**kw):
    d = dict(length_m=100, width_m=60, direction_deg=0, offset_m=0, altitude_m=20)
    d.update(kw)
    return Mission(**d)


def test_build_survey_zigzag_north(home, settings):
    plan = build_survey(_mission(), home, settings)
    pts = [(w.north_m, w.east_m) for w in plan.waypoints]
    assert _flat(pts) == pytest.approx(_flat([(0, -15), (100, -15), (100, 15), (0, 15)]), abs=1e-9)
    assert [w.seq for w in plan.waypoints] == [1, 2, 3, 4]
    assert plan.footprint_w_m == pytest.approx(40.0)
    assert plan.line_spacing_m == pytest.approx(30.0)
    assert plan.v_max_ms == pytest.approx(17.3205, abs=1e-3)
    assert plan.speed_ms == 5.0 and plan.warnings == []   # default speed, no clamping
    # home->(0,-15)=15, 100, 30, 100, ->home 15
    assert plan.path_length_m == pytest.approx(260.0)
    assert all(w.alt_m == 20 for w in plan.waypoints)
    assert plan.person_px == pytest.approx(124.78, abs=0.05)


def test_build_survey_offset_and_direction_east(home, settings):
    plan = build_survey(_mission(direction_deg=90, offset_m=200, length_m=100, width_m=30), home, settings)
    # one lane centred on the east axis, from 200 m to 300 m east
    pts = [(w.north_m, w.east_m) for w in plan.waypoints]
    assert _flat(pts) == pytest.approx(_flat([(0, 200), (0, 300)]), abs=1e-9)
    assert plan.waypoints[0].lat == pytest.approx(home.lat)
    assert plan.waypoints[1].lon > home.lon


def test_lanes_alternate_and_stay_inside_rectangle(home, settings):
    m = _mission(direction_deg=37, offset_m=40, length_m=120, width_m=100)
    plan = build_survey(m, home, settings)
    assert len(plan.waypoints) % 2 == 0 and len(plan.waypoints) == 2 * 4  # ceil(100/30)=4 lanes
    ahead = bearing_unit(37)
    along = [w.north_m * ahead[0] + w.east_m * ahead[1] for w in plan.waypoints]
    # lane k runs near->far for even k, far->near for odd k
    for k in range(4):
        a, b = along[2 * k], along[2 * k + 1]
        assert a == pytest.approx(40 if k % 2 == 0 else 160)
        assert b == pytest.approx(160 if k % 2 == 0 else 40)
    # consecutive lane ends are one (<= s) spacing apart
    for i in range(1, len(plan.waypoints) - 2, 2):
        a, b = plan.waypoints[i], plan.waypoints[i + 1]
        assert math.hypot(a.north_m - b.north_m, a.east_m - b.east_m) <= 30 + 1e-9


def test_speed_clamped_with_warning(home, settings):
    plan = build_survey(_mission(speed_ms=25.0), home, settings)
    lim = min(plan.v_max_ms, plan.footprint_l_m / (settings.min_frames * settings.frame_interval_s))
    assert plan.speed_ms == pytest.approx(lim)
    assert any("Speed reduced" in w for w in plan.warnings)


def test_requested_slow_speed_kept(home, settings):
    plan = build_survey(_mission(speed_ms=3.0), home, settings)
    assert plan.speed_ms == 3.0 and not plan.warnings


def test_low_resolution_warning(home, settings):
    plan = build_survey(_mission(altitude_m=40), home, settings)
    assert plan.person_px == pytest.approx(62.39, abs=0.05)
    msg = [w for w in plan.warnings if "px" in w]
    assert msg and "31 m" in msg[0]


def test_frame_interval_changes_vmax(home, settings_factory):
    s = settings_factory(capture_fps=0.5)  # dt = 2 s
    plan = build_survey(_mission(), home, s)
    assert plan.v_max_ms == pytest.approx(8.66, abs=1e-2)


def test_match_speed_formula():
    from survey import match_speed
    # L = 12 m, 3 frames at 1 fps -> 4 m/s
    assert match_speed(12.0, 3, 1.0) == pytest.approx(4.0)
    assert match_speed(12.0, 0, 1.0) == pytest.approx(12.0)   # min_frames < 1 treated as 1


def test_speed_clamped_for_multi_frame_confirmation(settings_factory, home):
    from models import Mission
    from survey import build_survey, footprint
    s = settings_factory(min_frames=3, default_speed_ms=10.0, max_speed_ms=15.0)
    plan = build_survey(Mission(length_m=100, width_m=60, altitude_m=20), home, s)
    _, l = footprint(20, s.eff_hfov_deg, s.eff_vfov_deg)
    assert plan.speed_ms == pytest.approx(min(l / 3, plan.v_max_ms))
    assert any("frames" in w for w in plan.warnings)


def test_centred_rectangle_straddles_home(home, settings):
    from plan_files import area_outline
    from survey import geo_to_local
    plan = build_survey(_mission(length_m=200, width_m=20, direction_deg=90, centered=True, offset_m=70), home, settings)
    east = [w.east_m for w in plan.waypoints]
    assert min(east) == pytest.approx(-100) and max(east) == pytest.approx(100)    # offset ignored when centred
    corners = [geo_to_local(lat, lon, home) for lat, lon in area_outline(plan)]
    assert sorted(round(e) for _, e in corners) == [-100, -100, 100, 100]
