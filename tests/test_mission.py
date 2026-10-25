"""Phase 4 tests: mission.py against the in-process mock vehicle (localhost UDP only)."""
from __future__ import annotations

import math
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "drone_brain", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import Settings  # noqa: E402
from mission import (  # noqa: E402
    ArduPilotBackend, ArmRefused, CommandRejected, LinkTimeout, VehicleError,
    arm_guard, create_backend, is_simulated_link,
)
from mock_vehicle import MODE_AUTO, MODE_RTL, MockVehicle  # noqa: E402
from models import GeoPoint, Mission, SurveyPlan, Waypoint  # noqa: E402

HOME = (-35.363261, 149.165230)
M_LAT = 111_319.5


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_settings(url: str, **kw) -> Settings:
    base = dict(
        mavlink_url=url, heartbeat_loss_s=1.0, mavlink_timeout_s=2.0, dry_run=True,
        allow_real_arm=False, autopilot="ardupilot",
    )
    base.update(kw)
    return Settings(_env_file=None, **base)


def make_plan(alt: float = 20.0, speed: float = 5.0) -> SurveyPlan:
    """Hand-made 60 m square survey NE of home (no dependency on survey.py)."""
    pts_ne = [(20, 20), (80, 20), (80, 80), (20, 80)]
    wps = []
    for i, (n, e) in enumerate(pts_ne):
        wps.append(Waypoint(
            seq=i, north_m=n, east_m=e,
            lat=HOME[0] + n / M_LAT,
            lon=HOME[1] + e / (M_LAT * math.cos(math.radians(HOME[0]))),
            alt_m=alt,
        ))
    return SurveyPlan(
        home=GeoPoint(lat=HOME[0], lon=HOME[1]),
        mission=Mission(length_m=60, width_m=60, direction_deg=45, offset_m=20, altitude_m=alt),
        waypoints=wps, altitude_m=alt, speed_ms=speed, v_max_ms=8.0,
        footprint_w_m=30, footprint_l_m=20, line_spacing_m=20, path_length_m=300,
        gsd_m=0.02, person_px=85,
    )


@pytest.fixture
def mock():
    port = free_udp_port()
    m = MockVehicle(f"udpout:127.0.0.1:{port}", home=HOME, time_scale=40.0, hb_rate_hz=4.0)
    m.port = port  # type: ignore[attr-defined]
    m.start()
    yield m
    m.stop()


@pytest.fixture
def vehicle(mock):
    v = ArduPilotBackend(make_settings(f"udpin:127.0.0.1:{mock.port}"))
    v.connect(timeout=5.0)
    yield v
    v.close()


# ---------------------------------------------------------------- connection / telemetry
def test_connect_and_telemetry(mock, vehicle):
    assert vehicle.heartbeat_ok()
    assert mock.wait_until(lambda m: m.received_types.get("HEARTBEAT", 0) >= 1, 3), "mock never saw our heartbeat"
    deadline = time.time() + 5
    tel = vehicle.telemetry()
    while time.time() < deadline and (tel.lat is None or tel.battery_pct is None or tel.gps_fix is None
                                      or tel.groundspeed_ms is None or tel.mission_seq is None):
        time.sleep(0.1)
        tel = vehicle.telemetry()
    assert tel.connected and tel.mode == "STABILIZE" and not tel.armed
    assert tel.lat == pytest.approx(HOME[0], abs=1e-5)
    assert tel.lon == pytest.approx(HOME[1], abs=1e-5)
    assert tel.alt_rel_m == pytest.approx(0.0, abs=0.1)
    assert tel.gps_fix == 3 and tel.satellites == 12
    assert tel.voltage_v and 10 < tel.voltage_v < 30
    assert tel.battery_pct is not None and tel.mission_seq == 0
    assert tel.last_heartbeat_ts is not None
    home = vehicle.get_home(timeout=3)
    assert home is not None and home.lat == pytest.approx(HOME[0], abs=1e-6)


def test_connect_timeout_without_vehicle():
    v = ArduPilotBackend(make_settings(f"udpin:127.0.0.1:{free_udp_port()}"))
    t0 = time.time()
    with pytest.raises(LinkTimeout):
        v.connect(timeout=1.0)
    assert time.time() - t0 < 4
    assert not v.heartbeat_ok()


def test_factory_and_unsupported_autopilot():
    assert isinstance(create_backend(make_settings("udpin:127.0.0.1:1")), ArduPilotBackend)
    with pytest.raises(ValueError):
        create_backend(make_settings("udpin:127.0.0.1:1", autopilot="px4"))


# ---------------------------------------------------------------- mission build / protocol
def test_build_mission_items():
    v = ArduPilotBackend(make_settings("udpin:127.0.0.1:1"))
    plan = make_plan(alt=25.0, speed=4.0)
    items = v.build_mission_items(plan)
    assert [i.seq for i in items] == list(range(len(items))) and len(items) == 3 + 4 + 1
    assert items[0].command == 16 and items[0].lat == pytest.approx(HOME[0])
    assert items[1].command == 22 and items[1].alt == 25.0 and items[1].frame == 6
    assert items[2].command == 178 and items[2].param2 == 4.0
    assert all(i.command == 16 and i.frame == 6 and i.alt == 25.0 for i in items[3:-1])
    assert items[-1].command == 20
    with pytest.raises(ValueError):
        v.build_mission_items(make_plan(alt=130.0))


def test_upload_download_verify(mock, vehicle):
    items = vehicle.build_mission_items(make_plan())
    vehicle.upload_mission(items)
    assert len(mock.mission) == len(items)
    got = vehicle.download_mission()
    assert [g.command for g in got] == [i.command for i in items]
    assert got[3].lat == pytest.approx(items[3].lat, abs=1e-7)
    assert vehicle.verify_mission(items)
    assert vehicle.telemetry().mission_count == len(items)
    # a different mission must not verify
    tampered = [i.model_copy() for i in items]
    tampered[4].lat += 0.001
    assert not vehicle.verify_mission(tampered)


def test_upload_retries_dropped_item(mock):
    v = ArduPilotBackend(make_settings(f"udpin:127.0.0.1:{mock.port}", mavlink_timeout_s=0.5))
    v.connect(timeout=5)
    try:
        mock.drop_inbound["MISSION_ITEM_INT"] = 1
        items = v.build_mission_items(make_plan())
        v.upload_mission(items)
        assert v.verify_mission(items)
    finally:
        v.close()


def test_upload_rejects_bad_sequence(vehicle):
    with pytest.raises(ValueError):
        vehicle.upload_mission([])


# ---------------------------------------------------------------- params / modes / arming
def test_set_param_and_unknown_param(mock):
    v = ArduPilotBackend(make_settings(f"udpin:127.0.0.1:{mock.port}", mavlink_timeout_s=0.3))
    v.connect(timeout=5)
    try:
        v.set_param("AUTO_OPTIONS", 3)
        assert mock.params["AUTO_OPTIONS"] == 3
        with pytest.raises(LinkTimeout):
            v.set_param("NOT_A_PARAM", 1)
    finally:
        v.close()


def test_set_mode_and_unknown_mode(mock, vehicle):
    vehicle.set_mode("AUTO")
    assert vehicle.telemetry().mode == "AUTO" and mock.mode == MODE_AUTO
    vehicle.set_mode("loiter")
    assert vehicle.telemetry().mode == "LOITER"
    with pytest.raises(ValueError):
        vehicle.set_mode("WARP")


def test_arm_in_auto_needs_auto_options(mock, vehicle):
    vehicle.set_mode("AUTO")
    with pytest.raises(CommandRejected) as ei:
        vehicle.arm()
    assert "Mode not armable" in str(ei.value)  # STATUSTEXT surfaced in the error
    vehicle.set_param("AUTO_OPTIONS", 3)
    vehicle.arm()
    assert vehicle.telemetry().armed and mock.armed
    vehicle.disarm()
    assert not mock.armed


def test_disarm_in_flight_refused_unless_forced(mock, vehicle):
    vehicle.arm()
    mock.alt = 10.0
    with pytest.raises(CommandRejected):
        vehicle.disarm()
    vehicle.disarm(force=True)
    assert not mock.armed


def test_start_mission_denied_when_disarmed(vehicle):
    with pytest.raises(CommandRejected):
        vehicle.start_mission()


# ---------------------------------------------------------------- full flight
def test_fly_plan_completes_and_returns(mock, vehicle):
    plan = make_plan()
    vehicle.fly_plan(plan)
    assert mock.armed and mock.mode == MODE_AUTO and mock.mission_running
    assert mock.params["AUTO_OPTIONS"] == 3
    assert mock.cmd_speed == 5.0 or mock.seq <= 2
    # mission progresses (MISSION_CURRENT seq visible in telemetry)
    assert mock.wait_until(lambda m: m.seq >= 4, 20)
    assert mock.wait_until(lambda m: vehicle.telemetry().mission_seq is not None and vehicle.telemetry().mission_seq >= 3, 5)
    assert vehicle.telemetry().alt_rel_m > 5
    # reaches the final RTL item -> RTL mode -> lands -> disarmed
    assert mock.wait_until(lambda m: m.mode == MODE_RTL, 40)
    assert mock.wait_until(lambda m: not m.armed, 40)
    assert mock.alt == 0.0 and abs(mock.north) < 1 and abs(mock.east) < 1
    assert mock.mission_done
    assert mock.wait_until(lambda m: not vehicle.telemetry().armed, 5)
    assert vehicle.telemetry().mode == "RTL"


def test_rtl_midflight(mock, vehicle):
    vehicle.fly_plan(make_plan())
    assert mock.wait_until(lambda m: m.seq >= 3 and m.alt > 10, 20)
    vehicle.rtl()
    assert vehicle.telemetry().mode == "RTL"
    assert mock.wait_until(lambda m: not m.armed, 40)
    assert mock.seq < len(mock.mission) - 1, "RTL should have pre-empted the survey"


def test_fly_plan_refuses_without_gps_fix(mock, vehicle):
    mock.gps_fix = 1
    assert mock.wait_until(lambda m: vehicle.telemetry().gps_fix == 1, 5)
    with pytest.raises(VehicleError, match="GPS"):
        vehicle.fly_plan(make_plan())
    assert not mock.armed


# ---------------------------------------------------------------- link loss
def test_heartbeat_loss_and_restore_callbacks(mock):
    lost, restored = threading.Event(), threading.Event()
    v = ArduPilotBackend(
        make_settings(f"udpin:127.0.0.1:{mock.port}"),
        on_link_lost=lost.set, on_link_restored=restored.set,
    )
    v.connect(timeout=5)
    try:
        assert v.heartbeat_ok() and not lost.is_set()
        mock.mute = True
        assert lost.wait(4), "on_link_lost not fired"
        assert not v.heartbeat_ok() and not v.telemetry().connected
        mock.mute = False
        assert restored.wait(4), "on_link_restored not fired"
        assert v.heartbeat_ok()
    finally:
        v.close()


def test_heartbeat_loss_when_mock_stops(mock, vehicle):
    assert vehicle.heartbeat_ok()
    mock.stop()
    deadline = time.time() + 4
    while vehicle.heartbeat_ok() and time.time() < deadline:
        time.sleep(0.1)
    assert not vehicle.heartbeat_ok()
    with pytest.raises(LinkTimeout):
        # commands fail with a clear timeout, never hang
        vehicle.set_param("AUTO_OPTIONS", 3)


# ---------------------------------------------------------------- arming guard (safety)
def test_arm_guard_refuses_serial_even_in_dry_run():
    for url in ("serial:/dev/ttyTHS1", "/dev/ttyTHS1"):
        s = make_settings(url, dry_run=True)
        ok, reason = arm_guard(s)
        assert not ok and "refused" in reason
        v = ArduPilotBackend(s)  # never connected: guard fires before any I/O
        with pytest.raises(ArmRefused):
            v.arm()


def test_arm_guard_matrix():
    # loopback UDP/TCP is always a simulator
    for url in ("udpin:127.0.0.1:14550", "udpout:localhost:14550", "tcp:127.0.0.1:5760"):
        assert is_simulated_link(url, dry_run=False)
        assert arm_guard(make_settings(url, dry_run=False))[0]
    # wildcard listener only counts in dry-run
    assert arm_guard(make_settings("udpin:0.0.0.0:14550", dry_run=True))[0]
    assert not arm_guard(make_settings("udpin:0.0.0.0:14550", dry_run=False))[0]
    # remote hosts / broadcast are not simulators
    assert not arm_guard(make_settings("udpout:192.168.1.50:14550", dry_run=True))[0]
    assert not arm_guard(make_settings("udpbcast:192.168.1.255:14550", dry_run=True))[0]
    # real arming needs BOTH human switches
    assert not arm_guard(make_settings("serial:/dev/ttyTHS1", dry_run=False, allow_real_arm=False))[0]
    assert not arm_guard(make_settings("serial:/dev/ttyTHS1", dry_run=True, allow_real_arm=True))[0]
    assert arm_guard(make_settings("serial:/dev/ttyTHS1", dry_run=False, allow_real_arm=True))[0]


def test_serial_open_is_not_attempted_by_tests():
    # Constructing a backend never opens anything.
    v = ArduPilotBackend(make_settings("serial:/dev/ttyTHS1"))
    assert v._conn is None and not v.heartbeat_ok()
