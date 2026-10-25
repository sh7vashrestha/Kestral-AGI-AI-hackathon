"""Phase 7 unit tests: read-only vehicle mode, effective ceiling, pre-flight evaluation, mission export files,
home tracking (mock vehicle home moves -> plan recalculated, confirm refuses a stale home)."""
from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import mission  # noqa: E402
import safety  # noqa: E402
from mission import ArduPilotBackend, ReadOnlyViolation, evaluate_preflight  # noqa: E402
from mock_vehicle import MockVehicle  # noqa: E402
from models import GeoPoint, LatLon, MissionItem  # noqa: E402
from plan_files import parse_wpl, to_geojson, to_wpl, validate_geojson  # noqa: E402
from conftest import make_settings  # noqa: E402

FIELD = (33.9371904, -84.5271293)
CORNERS = [{"lat": 33.93745, "lon": -84.52740}, {"lat": 33.93745, "lon": -84.52690},
           {"lat": 33.93790, "lon": -84.52690}, {"lat": 33.93790, "lon": -84.52740}]


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- read-only mode
RO_REFUSED = [
    ("upload_mission", ([MissionItem(seq=0, command=16)],)),
    ("set_mode", ("AUTO",)),
    ("arm", ()),
    ("disarm", ()),
    ("start_mission", ()),
    ("set_param", ("AUTO_OPTIONS", 3)),
    ("rtl", ()),
]


@pytest.mark.parametrize("method,args", RO_REFUSED)
def test_readonly_refuses_before_any_io(method, args):
    s = make_settings(vehicle_mode="readonly", mavlink_url="serial:/dev/ttyACM0")
    be = ArduPilotBackend(s)
    assert be.readonly
    be._conn = object()          # any I/O attempt would blow up on this, the guard must come first
    be._sysid = 1
    with pytest.raises(ReadOnlyViolation):
        getattr(be, method)(*args)


def test_readonly_refuses_fly_plan_and_raw_sends():
    from survey import build_survey
    from models import Mission
    s = make_settings(vehicle_mode="readonly")
    be = ArduPilotBackend(s)
    plan = build_survey(Mission(length_m=60, width_m=40, altitude_m=20, offset_m=10), GeoPoint(lat=FIELD[0], lon=FIELD[1]), s)
    with pytest.raises(ReadOnlyViolation):
        be.fly_plan(plan)
    be._conn = object()
    for meth, args in (("mission_count_send", (1, 1, 3, 0)), ("param_set_send", (1, 1, b"X", 1.0, 9)),
                       ("command_long_send", (1, 1, 400, 0, 1, 0, 0, 0, 0, 0, 0)),
                       ("command_long_send", (1, 1, 176, 0, 1, 3, 0, 0, 0, 0, 0))):
        with pytest.raises(ReadOnlyViolation):
            be._send(meth, *args)


def test_readonly_on_mock_reads_but_never_commands():
    port = free_udp_port()
    s = make_settings(vehicle_mode="readonly", mavlink_url=f"udpin:127.0.0.1:{port}")
    mock = MockVehicle(f"udpout:127.0.0.1:{port}", home=FIELD, time_scale=1).start()
    be = ArduPilotBackend(s)
    try:
        be.connect(timeout=10)
        assert be.get_param("RTL_ALT", timeout=3) == 2000.0
        res = mission.preflight_check(be, s, timeout=3)
        assert res.ok, res.errors
        deadline = time.time() + 5
        while be.home_position() is None and time.time() < deadline:
            time.sleep(0.1)
        assert be.home_position() is not None
        with pytest.raises(ReadOnlyViolation):
            be.rtl()
        time.sleep(0.5)
        bad = {"MISSION_COUNT", "MISSION_ITEM_INT", "PARAM_SET", "SET_MODE", "MISSION_REQUEST_LIST"}
        assert not bad & set(mock.received_types), mock.received_types
        assert mock.mode == 0 and not mock.armed and mock.params["AUTO_OPTIONS"] == 0.0
    finally:
        be.close()
        mock.stop()


def test_sim_mode_refuses_real_link():
    import main
    s = make_settings(vehicle_mode="sim", mavlink_url="serial:/dev/ttyACM0")
    with pytest.raises(SystemExit):
        main.check_vehicle_mode(s)
    main.check_vehicle_mode(make_settings(vehicle_mode="sim", mavlink_url="udpin:127.0.0.1:14550"))


# ---------------------------------------------------------------- ceiling / floor
def test_effective_ceiling_and_floor():
    s = make_settings(max_alt_m=60, laanc_ceiling_m=30, obstacle_clearance_m=15)
    assert safety.ceiling_info(s) == (30.0, "LAANC")
    assert safety.effective_min_alt(s) == 15.0
    lim = safety.limits_info(s)
    assert lim["flyable"] and lim["ceiling_source"] == "LAANC"
    assert safety.ceiling_info(make_settings(max_alt_m=60)) == (60.0, "MAX_ALT_M")
    bad = make_settings(laanc_ceiling_m=10, obstacle_clearance_m=15)
    assert not safety.limits_info(bad)["flyable"]


def test_validate_uses_laanc_ceiling_and_obstacle_floor():
    from survey import build_polygon_survey
    home = GeoPoint(lat=FIELD[0], lon=FIELD[1])
    corners = [LatLon(**c) for c in CORNERS]
    s = make_settings(laanc_ceiling_m=25)
    assert safety.validate(build_polygon_survey(corners, 20, home, s), s).ok
    r = safety.validate(build_polygon_survey(corners, 30, home, s), s)
    assert not r.ok and any("LAANC" in e for e in r.errors)
    r = safety.validate(build_polygon_survey(corners, 12, home, s), s)
    assert not r.ok and any("obstacle clearance" in e for e in r.errors)
    s2 = make_settings(laanc_ceiling_m=10, obstacle_clearance_m=15)
    r = safety.validate(build_polygon_survey(corners, 10, home, s2), s2)
    assert not r.ok and any("cannot be flown" in e for e in r.errors)


# ---------------------------------------------------------------- pre-flight evaluation
GOOD = {"RTL_ALT": 2000, "FENCE_ENABLE": 1, "FENCE_TYPE": 7, "FENCE_ALT_MAX": 50, "FENCE_RADIUS": 250,
        "FENCE_ACTION": 1, "BATT_FS_LOW_ACT": 2, "FS_THR_ENABLE": 1, "WPNAV_SPEED": 1000}


def test_preflight_good_params_pass():
    r = evaluate_preflight(dict(GOOD), make_settings())
    assert r.ok and not r.errors and not r.warnings
    assert r.rtl_alt_m == 20 and "enabled" in r.fence


@pytest.mark.parametrize("param,value,frag", [
    ("RTL_ALT", 1000, "below the obstacle clearance"), ("RTL_ALT", 9000, "above"), ("RTL_ALT", 0, "RTL_ALT=0"),
    ("FENCE_ENABLE", 0, "DISABLED"), ("FENCE_TYPE", 2, "must both"), ("FENCE_ALT_MAX", 100, "above"),
    ("FENCE_RADIUS", 500, "larger"), ("FENCE_ACTION", 0, "Report"), ("BATT_FS_LOW_ACT", 5, "Terminate"),
    ("BATT_FS_LOW_ACT", 0, "None"), ("FS_THR_ENABLE", 0, "DISABLED"), ("RTL_ALT", None, "could not be read"),
])
def test_preflight_blocks(param, value, frag):
    p = dict(GOOD)
    p[param] = value
    r = evaluate_preflight(p, make_settings())
    assert not r.ok
    assert any(frag in e for e in r.errors), r.errors


def test_preflight_laanc_ceiling_and_plan_checks():
    from survey import build_polygon_survey
    s = make_settings(laanc_ceiling_m=40)
    r = evaluate_preflight(dict(GOOD), s)           # FENCE_ALT_MAX 50 > LAANC 40
    assert not r.ok and r.ceiling_source == "LAANC"
    home = GeoPoint(lat=FIELD[0], lon=FIELD[1])
    plan = build_polygon_survey([LatLon(**c) for c in CORNERS], 20, home, make_settings())
    p = dict(GOOD, FENCE_RADIUS=40)                  # area reaches ~70 m from home
    r = evaluate_preflight(p, make_settings(), plan)
    assert not r.ok and any("outside the Pixhawk fence" in e for e in r.errors)
    r = evaluate_preflight(dict(GOOD, WPNAV_SPEED=200), make_settings(), plan)
    assert r.ok and any("WPNAV_SPEED" in w for w in r.warnings)
    r = evaluate_preflight(dict(GOOD, FS_THR_ENABLE=2), make_settings(), plan)
    assert r.ok and any("continue mission" in w for w in r.warnings)
    p = {k: v for k, v in GOOD.items() if k != "BATT_FS_LOW_ACT"}
    p["FS_BATT_ENABLE"] = 2
    assert evaluate_preflight(p, make_settings()).ok


# ---------------------------------------------------------------- export files
def test_wpl_and_geojson_roundtrip():
    from survey import build_polygon_survey
    s = make_settings()
    home = GeoPoint(lat=FIELD[0], lon=FIELD[1])
    plan = build_polygon_survey([LatLon(**c) for c in CORNERS], 20, home, s)
    items = ArduPilotBackend(s).build_mission_items(plan)
    text = to_wpl(items, 300.0)
    rows = parse_wpl(text)
    assert text.startswith("QGC WPL 110\n") and len(rows) == len(items)
    assert rows[0]["frame"] == 0 and rows[0]["alt"] == 300.0 and rows[0]["current"] == 1
    assert [r["command"] for r in rows[:3]] == [16, 22, 178] and rows[-1]["command"] == 20
    assert all(r["frame"] == 3 for r in rows if r["command"] in (16, 22) and r["seq"] > 0)
    assert rows[2]["param2"] == plan.speed_ms
    for r, it in zip(rows[3:-1], items[3:-1]):
        assert abs(r["lat"] - it.lat) < 1e-7 and abs(r["lon"] - it.lon) < 1e-7 and r["alt"] == it.alt
    gj = to_geojson(plan, plan_id="x", home_amsl_m=300.0)
    validate_geojson(json.loads(json.dumps(gj)))
    kinds = [f["geometry"]["type"] for f in gj["features"]]
    assert kinds == ["Polygon", "LineString", "Point"]
    with pytest.raises(ValueError):
        parse_wpl("nope")


def test_geojson_rectangle_outline():
    from models import Mission
    from survey import build_survey
    s = make_settings()
    plan = build_survey(Mission(length_m=60, width_m=40, altitude_m=20, offset_m=10, direction_deg=90),
                        GeoPoint(lat=FIELD[0], lon=FIELD[1]), s)
    gj = to_geojson(plan)
    validate_geojson(gj)
    ring = gj["features"][0]["geometry"]["coordinates"][0]
    assert len(ring) == 5 and all(lon > FIELD[1] for lon, _ in ring)     # east of home


# ---------------------------------------------------------------- home tracking with the mock
def test_backend_tracks_home_changes():
    port = free_udp_port()
    s = make_settings(mavlink_url=f"udpin:127.0.0.1:{port}")
    mock = MockVehicle(f"udpout:127.0.0.1:{port}", home=FIELD, time_scale=1).start()
    seen = []
    be = ArduPilotBackend(s, on_home_changed=lambda old, new: seen.append((old, new)))
    try:
        be.connect(timeout=10)
        deadline = time.time() + 5
        while be.home_position() is None and time.time() < deadline:
            time.sleep(0.05)
        h1 = be.home_position()
        assert h1 is not None and h1.version == 1
        mock.move_home(20.0, 0.0, d_alt_m=3.0)
        deadline = time.time() + 5
        while be.home_position().version == 1 and time.time() < deadline:
            time.sleep(0.05)
        h2 = be.home_position()
        assert h2.version == 2 and 19 < mission.home_shift_m(h1, h2) < 21 and abs(h2.amsl_m - h1.amsl_m - 3) < 0.01
        time.sleep(0.3)
        assert len(seen) == 2 and seen[1][0].version == h1.version and seen[1][0].lat == h1.lat   # (ts is refreshed by every HOME_POSITION)
    finally:
        be.close()
        mock.stop()


# ---------------------------------------------------------------- upload-only mode
UP_REFUSED = [m for m in RO_REFUSED if m[0] not in ("upload_mission",)]
CONTROL_CMDS = {176, 300, 400, 20, 22}       # set mode, mission start, arm/disarm, RTL, takeoff


@pytest.mark.parametrize("method,args", UP_REFUSED)
def test_upload_mode_refuses_control_before_io(method, args):
    be = ArduPilotBackend(make_settings(vehicle_mode="upload", mavlink_url="serial:/dev/ttyACM0"))
    assert be.access == "upload" and not be.readonly
    be._conn = object()
    be._sysid = 1
    with pytest.raises(ReadOnlyViolation):
        getattr(be, method)(*args)


def _plan():
    from survey import build_polygon_survey
    return build_polygon_survey([LatLon(**c) for c in CORNERS], 20, GeoPoint(lat=FIELD[0], lon=FIELD[1]), make_settings())


def test_upload_mode_refuses_fly_plan_and_raw_control():
    be = ArduPilotBackend(make_settings(vehicle_mode="upload"))
    with pytest.raises(ReadOnlyViolation):
        be.fly_plan(_plan())
    be._conn = object()
    for args in ((1, 1, 400, 0, 1, 0, 0, 0, 0, 0, 0), (1, 1, 300, 0, 0, 0, 0, 0, 0, 0, 0), (1, 1, 20, 0, 0, 0, 0, 0, 0, 0, 0)):
        with pytest.raises(ReadOnlyViolation):
            be._send("command_long_send", *args)
    with pytest.raises(ReadOnlyViolation):
        be._send("param_set_send", 1, 1, b"FENCE_ENABLE", 0.0, 9)
    with pytest.raises(ReadOnlyViolation):
        ArduPilotBackend(make_settings(vehicle_mode="readonly")).clear_mission()


@pytest.fixture
def upload_link():
    port = free_udp_port()
    s = make_settings(vehicle_mode="upload", mavlink_url=f"udpin:127.0.0.1:{port}")
    mock = MockVehicle(f"udpout:127.0.0.1:{port}", home=FIELD, time_scale=1).start()
    be = ArduPilotBackend(s)
    be.connect(timeout=10)
    deadline = time.time() + 5
    while (be.home_position() is None or be.telemetry().gps_fix is None) and time.time() < deadline:
        time.sleep(0.05)
    yield be, mock, s
    be.close()
    mock.stop()


def test_upload_for_review_on_mock(upload_link):
    be, mock, _ = upload_link
    from plan_files import area_outline
    plan = _plan()
    res = be.upload_for_review(plan, fence_corners=area_outline(plan))
    assert res["verified"] and res["items"] == len(plan.waypoints) + 4 == len(mock.mission)
    assert res["fence"] == {"vertices": 4, "verified": True} and len(mock.fence) == 4
    assert all(f.command == 5001 and f.p1 == 4 for f in mock.fence)
    assert len(res["readback"]) == res["items"]
    assert not mock.armed and mock.mode == 0 and mock.params["AUTO_OPTIONS"] == 0.0
    assert not CONTROL_CMDS & set(mock.commands_received), mock.commands_received
    assert "PARAM_SET" not in mock.received_types and "SET_MODE" not in mock.received_types
    be.clear_mission()
    assert mock.mission == [] and len(mock.fence) == 4   # fence untouched by a mission clear


def test_upload_refused_while_armed(upload_link):
    be, mock, _ = upload_link
    mock.armed = True
    assert be._wait_state(lambda t: t.armed, 5)
    with pytest.raises(mission.VehicleError, match="ARMED"):
        be.upload_for_review(_plan())
    with pytest.raises(mission.VehicleError, match="ARMED"):
        be.clear_mission()
    assert "MISSION_COUNT" not in mock.received_types
    mock.armed = False


# ---------------------------------------------------------------- vehicle check
def test_vehicle_check_ready_on_mock(upload_link):
    from vehicle_check import run_vehicle_check
    be, mock, s = upload_link
    time.sleep(1.5)                                    # EKF / RC / AUTOPILOT_VERSION streams
    rep = run_vehicle_check(be, s, prearm_wait_s=1.0)
    assert rep["verdict"] == "READY", rep["reasons"]
    assert rep["gps"]["fix"] == "3D" and rep["ekf"]["level"] == "OK" and rep["sensors"]["prearm_ok"] is True
    assert rep["firmware"]["version"] == "4.6.3" and rep["vehicle"]["type"] == "QUADROTOR"
    assert rep["rc"]["received"] and rep["preflight"]["ok"] and rep["prearm"]["ack"] == "ACCEPTED"
    tel = be.telemetry()
    assert tel.landed_state == "ON_GROUND" and tel.system_status == "STANDBY" and tel.hdop == 1.0
    assert 401 in mock.commands_received and not CONTROL_CMDS & set(mock.commands_received)


def test_vehicle_check_not_ready_without_gps(upload_link):
    from vehicle_check import run_vehicle_check
    be, mock, s = upload_link
    mock.gps_fix = 1
    mock.prearm_failures = ["PreArm: Throttle below failsafe"]
    time.sleep(1.5)
    rep = run_vehicle_check(be, s, prearm_wait_s=1.0)
    assert rep["verdict"] == "NOT READY"
    assert any("GPS" in r for r in rep["reasons"]) and any("pre-arm" in r for r in rep["reasons"])
    texts = [m["text"] for m in rep["prearm"]["messages"]]
    assert "PreArm: GPS 1: Bad fix" in texts and "PreArm: Throttle below failsafe" in texts
    assert rep["sensors"]["prearm_ok"] is False and "GPS" in rep["sensors"]["unhealthy"]


def test_decoders():
    from vehicle_check import decode_bits, ekf_report, SENSOR_BITS, variance_level
    assert decode_bits(0x21, SENSOR_BITS) == ["3D gyro", "GPS"]
    assert variance_level(0.3) == "OK" and variance_level(0.6) == "WARN" and variance_level(0.9) == "BAD"
    r = ekf_report({"flags": 0x3F, "velocity_variance": 0.1, "pos_horiz_variance": 0.9, "pos_vert_variance": 0.1,
                    "compass_variance": 0.1, "terrain_alt_variance": 0})
    assert r["level"] == "BAD" and r["levels"]["pos_horiz"] == "BAD"
    assert ekf_report({"flags": 0x1})["level"] == "WARN"


def test_preflight_arducopter_48_param_names():
    p = {k: v for k, v in GOOD.items() if k not in ("RTL_ALT", "WPNAV_SPEED")}
    p.update(RTL_ALT_M=20.0, WP_SPD=10.0)
    r = evaluate_preflight(p, make_settings())
    assert r.ok and r.rtl_alt_m == 20
    names = {c.param: c.value for c in r.checks}
    assert names["RTL_ALT_M"] == 20.0 and names["WP_SPD"] == 10.0
    p["RTL_ALT_M"] = 5.0
    assert any("RTL_ALT_M" in e for e in evaluate_preflight(p, make_settings()).errors)


def test_clip_prompt_builder_llm_cache_and_fallback():
    from clip_prompts import LLMClipPrompts, LlmPromptBuilder
    from models import Target

    class C:
        n = 0

        def chat_json(self, messages, cls, **kw):
            C.n += 1
            if "fail" in messages[-1]["content"]:
                raise RuntimeError("llm down")
            return LLMClipPrompts(positive=["a photo of a person in red", "red jacket person", "person, red coat"],
                                  negative=["a person in blue", "a person in green", "a rock"])
    b = LlmPromptBuilder(C())
    t = Target(description="person in a red jacket", attributes=["red jacket"])
    p1 = b(t)
    p2 = b(t)
    assert C.n == 1 and p1 is p2 and b.last_source == "cache" and len(p1.negative) == 3
    p3 = b(Target(description="fail please"))
    assert b.last_source == "default" and "a photo of fail please" in p3.positive   # Hardening D: aerial + plain positives
    b(Target(description="any person"))
    assert C.n == 2


def test_vision_stall_detected_and_restarted(monkeypatch):
    import threading
    import numpy as np
    import pipeline as pl

    monkeypatch.setattr(pl, "FRAME_STALL_S", 0.5)
    made = []

    class StallCam:
        def __init__(self):
            self.stop_ev = threading.Event()
            made.append(self)

        def start(self, wait_s=0):
            return self

        def stop(self):
            self.stop_ev.set()

        def frames(self):
            for i in range(2):
                yield np.zeros((120, 160, 3), np.uint8), time.time(), i
            self.stop_ev.wait()       # hung pipeline: no more frames, generator never ends by itself

    s = make_settings(capture_fps=10.0)
    vp = pl.VisionPipeline(s, on_alert=lambda a: None, telemetry_fn=lambda: None,
                           camera_factory=lambda _s, _src: StallCam(), detector_factory=lambda _s: object(),
                           backend_factory=lambda _s: pl._NoMatchBackend(), use_thermal=False)
    vp.start()
    deadline = time.time() + 5
    while vp.state != "ready" and time.time() < deadline:
        time.sleep(0.05)
    assert vp.state == "ready"
    time.sleep(1.0)
    assert not vp.alive()
    vp.restart()
    time.sleep(0.2)
    assert len(made) == 2 and made[0].stop_ev.is_set() and vp.frames >= 4 and vp.restarts == 1
    vp.stop()
