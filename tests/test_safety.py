import math

import pytest

from models import Mission
from safety import MissionItem, validate, validate_mission_items
from survey import build_survey


def _plan(home, settings, **kw):
    d = dict(length_m=100, width_m=60, direction_deg=0, offset_m=0, altitude_m=20)
    d.update(kw)
    return build_survey(Mission(**d), home, settings)


def test_valid_plan_passes(home, settings):
    r = validate(_plan(home, settings), settings)
    assert r.ok and r.errors == []


def test_warnings_pass_through(home, settings):
    r = validate(_plan(home, settings, altitude_m=40), settings)
    assert r.ok
    assert any("px" in w for w in r.warnings)


def test_altitude_above_max(home, settings):
    r = validate(_plan(home, settings, altitude_m=61), settings)
    assert not r.ok and any("maximum" in e for e in r.errors)


def test_altitude_above_hard_max_even_if_configured_higher(home, settings_factory):
    # settings validator forbids max_alt_m > 120, so bypass via model_copy to prove the hard cap
    s = settings_factory().model_copy(update={"max_alt_m": 500.0})
    r = validate(_plan(home, s, altitude_m=130), s)
    assert not r.ok and any("absolute limit" in e for e in r.errors)


def test_altitude_below_min(home, settings):
    r = validate(_plan(home, settings, altitude_m=3), settings)
    assert not r.ok and any("below the minimum" in e for e in r.errors)


def test_geofence(home, settings):
    r = validate(_plan(home, settings, offset_m=400, length_m=50, width_m=30), settings)
    assert not r.ok and any("geofence" in e for e in r.errors)


def test_geofence_boundary_inside_ok(home, settings):
    r = validate(_plan(home, settings, offset_m=100, length_m=100, width_m=30, direction_deg=90), settings)
    assert r.ok


def test_path_too_long(home, settings_factory):
    s = settings_factory(max_path_m=200.0)
    r = validate(_plan(home, s), s)
    assert not r.ok and any("path" in e.lower() for e in r.errors)


def test_path_length_recomputed_not_trusted(home, settings):
    plan = _plan(home, settings)
    plan.path_length_m = 1.0   # lying plan
    s = settings.model_copy(update={"max_path_m": 200.0})
    assert not validate(plan, s).ok


def test_speed_too_high(home, settings):
    plan = _plan(home, settings)
    plan.speed_ms = 12.0
    r = validate(plan, settings)
    assert not r.ok and any("Speed" in e for e in r.errors)


def test_zero_speed_rejected(home, settings):
    plan = _plan(home, settings)
    plan.speed_ms = 0.0
    assert not validate(plan, settings).ok


def test_empty_plan(home, settings):
    plan = _plan(home, settings)
    plan.waypoints = []
    r = validate(plan, settings)
    assert not r.ok and any("no waypoints" in e for e in r.errors)


@pytest.mark.parametrize("field", ["altitude_m", "speed_ms", "path_length_m"])
def test_nan_in_plan(home, settings, field):
    plan = _plan(home, settings)
    setattr(plan, field, math.nan)
    r = validate(plan, settings)
    assert not r.ok and any("NaN" in e for e in r.errors)


def test_nan_in_waypoint(home, settings):
    plan = _plan(home, settings)
    plan.waypoints[1].lat = math.nan
    assert not validate(plan, settings).ok


def test_waypoint_altitude_checked(home, settings):
    plan = _plan(home, settings)
    plan.waypoints[2].alt_m = 90.0
    r = validate(plan, settings)
    assert not r.ok and any("Waypoint 3" in e for e in r.errors)


# ---- final item list ----------------------------------------------------

def _items(home, rtl=True):
    items = [("TAKEOFF", home.lat, home.lon, 20.0), ("WAYPOINT", home.lat + 0.0005, home.lon, 20.0)]
    if rtl:
        items.append(("RTL", 0.0, 0.0, 0.0))
    return items


def test_items_end_with_rtl_ok(home, settings):
    assert validate_mission_items(_items(home), home, settings).ok


def test_items_numeric_rtl_command(home, settings):
    items = _items(home)[:-1] + [(20, 0.0, 0.0, 0.0)]
    assert validate_mission_items(items, home, settings).ok


def test_items_missing_rtl(home, settings):
    r = validate_mission_items(_items(home, rtl=False), home, settings)
    assert not r.ok and any("RTL" in e for e in r.errors)


def test_items_rtl_not_last(home, settings):
    items = _items(home) + [("WAYPOINT", home.lat, home.lon, 20.0)]
    assert not validate_mission_items(items, home, settings).ok


def test_items_empty(home, settings):
    assert not validate_mission_items([], home, settings).ok


def test_items_geofence_and_alt(home, settings):
    far = [("WAYPOINT", home.lat + 0.01, home.lon, 20.0), ("RTL", 0, 0, 0)]  # ~1.1 km
    assert not validate_mission_items(far, home, settings).ok
    high = [("WAYPOINT", home.lat, home.lon + 0.0001, 100.0), ("RTL", 0, 0, 0)]
    assert not validate_mission_items(high, home, settings).ok


def test_items_model_form_and_speed_cmd(home, settings):
    items = [MissionItem(command="DO_CHANGE_SPEED", alt_m=0.0), MissionItem(command="RTL")]
    assert validate_mission_items(items, home, settings).ok


def test_validate_with_items(home, settings):
    plan = _plan(home, settings)
    r = validate(plan, settings, items=_items(home, rtl=False))
    assert not r.ok and any("RTL" in e for e in r.errors)
    assert validate(plan, settings, items=_items(home)).ok


def test_validate_real_mission_items_from_mission_py(settings, home):
    """The exact list mission.py uploads (home placeholder, TAKEOFF, CHANGE_SPEED,
    waypoints, RTL) must pass safety, and fail without the final RTL."""
    from mission import ArduPilotBackend
    from models import Mission
    from safety import validate_mission_items
    from survey import build_survey

    plan = build_survey(Mission(length_m=100, width_m=50, altitude_m=20), home, settings)
    items = ArduPilotBackend(settings).build_mission_items(plan)
    assert validate_mission_items(items, home, settings).ok
    assert not validate_mission_items(items[:-1], home, settings).ok
