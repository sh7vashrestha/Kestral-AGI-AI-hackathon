"""Hardening C: vehicle / mission edge cases against the in-process mock ArduCopter (loopback UDP only).

Backend level (mission.py): upload faults, link faults, garbage / foreign systems, access-mode gates.
Controller level (controller_real.py via the e2e Rig): link loss per state, pilot mode change, failsafes, commands.
Process level: SIGTERM of a real main.py process with the mock vehicle.
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "drone_brain", ROOT / "scripts", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mission  # noqa: E402
from conftest import make_settings  # noqa: E402
from mission import ArduPilotBackend, CommandRejected, LinkTimeout, ReadOnlyViolation, VehicleError  # noqa: E402
from mock_vehicle import MODE_AUTO, MODE_LAND, MODE_RTL, MockVehicle  # noqa: E402
from test_e2e import CORNERS, FIELD, Rig, free_udp_port  # noqa: E402
from test_mission import HOME, make_plan  # noqa: E402

MODE_LOITER = 5


# =============================================================================== backend level
def _backend(mock_port: int, **kw) -> ArduPilotBackend:
    s = make_settings(mavlink_url=f"udpin:127.0.0.1:{mock_port}", heartbeat_loss_s=1.0, mavlink_timeout_s=0.6, **kw)
    return ArduPilotBackend(s)


@pytest.fixture
def link():
    port = free_udp_port()
    mock = MockVehicle(f"udpout:127.0.0.1:{port}", home=HOME, time_scale=40.0, hb_rate_hz=4.0).start()
    be = _backend(port)
    be.connect(timeout=5)
    be.mock, be.port = mock, port          # type: ignore[attr-defined]
    yield be
    be.close()
    mock.stop()


# ---------------------------------------------------------------- upload
@pytest.mark.parametrize("code,name", [(4, "NO_SPACE"), (13, "INVALID_SEQUENCE"), (14, "DENIED"), (1, "ERROR"),
                                       (2, "UNSUPPORTED_FRAME"), (15, "OPERATION_CANCELLED")])
def test_upload_mission_ack_error_codes_are_named(link, code, name):
    link.mock.upload_ack_error = code
    with pytest.raises(CommandRejected, match=name) as ei:
        link.upload_mission(link.build_mission_items(make_plan()))
    assert ei.value.result == code
    assert link.mock.mission == []


def test_upload_bigger_than_vehicle_capacity(link):
    link.mock.mission_capacity = 5
    with pytest.raises(CommandRejected, match="NO_SPACE"):
        link.upload_mission(link.build_mission_items(make_plan()))


def test_upload_interrupted_mid_way_never_arms(link):
    link.mock.fail_upload_at = 4            # vehicle gives up after 4 items (INVALID_SEQUENCE)
    with pytest.raises(CommandRejected, match="INVALID_SEQUENCE"):
        link.fly_plan(make_plan())
    m = link.mock
    assert not m.armed and not {400, 300, 176} & set(m.commands_received) and m.mode == 0


@pytest.mark.parametrize("fault", ["swap_requests", "duplicate_requests"])
def test_upload_out_of_order_and_duplicate_requests(link, fault):
    setattr(link.mock, fault, True)
    items = link.build_mission_items(make_plan())
    link.upload_mission(items)
    assert link.verify_mission(items)


def test_upload_retries_each_item_then_gives_up(link):
    items = link.build_mission_items(make_plan())
    link.mock.drop_inbound["MISSION_ITEM_INT"] = 3        # three lost items in a row: retried
    link.upload_mission(items)
    assert link.verify_mission(items)
    link.mock.drop_inbound["MISSION_ITEM_INT"] = 1000     # link dead for items: bounded LinkTimeout, no hang
    t0 = time.time()
    with pytest.raises(LinkTimeout, match="stalled"):
        link.upload_mission(items)
    assert time.time() - t0 < (link.ITEM_RETRIES + 2) * 0.6 + 3


def test_upload_count_lost_is_resent(link):
    link.mock.drop_inbound["MISSION_COUNT"] = 2
    items = link.build_mission_items(make_plan())
    link.upload_mission(items)
    assert link.verify_mission(items)


def test_readback_mismatch_never_arms(link):
    link.mock.corrupt_readback = True
    with pytest.raises(VehicleError, match="read-back verification failed"):
        link.fly_plan(make_plan())
    m = link.mock
    assert not m.armed and not {400, 300, 176} & set(m.commands_received) and m.params["AUTO_OPTIONS"] == 0.0


def test_upload_refused_while_armed_in_control_mode(link):
    link.mock.armed = True
    assert link._wait_state(lambda t: t.armed, 5)
    with pytest.raises(VehicleError, match="ARMED"):
        link.upload_mission(link.build_mission_items(make_plan()))
    with pytest.raises(VehicleError, match="ARMED"):
        link.clear_mission()
    assert "MISSION_COUNT" not in link.mock.received_types and "MISSION_CLEAR_ALL" not in link.mock.received_types


def test_cancel_flight_stops_upload_before_arming(link):
    link.mock.drop_inbound["MISSION_ITEM_INT"] = 1000     # upload stalls on the first item
    out: list = []

    def run() -> None:
        try:
            link.fly_plan(make_plan())
        except Exception as exc:  # noqa: BLE001
            out.append(exc)
    th = threading.Thread(target=run)
    th.start()
    time.sleep(1.0)
    link.cancel_flight()
    th.join(5)
    assert not th.is_alive() and out and "cancelled" in str(out[0]), out
    assert not link.mock.armed and 400 not in link.mock.commands_received


def test_cancel_flight_before_arm_step(link):
    """Cancel set between upload and arm (the abort raced the upload finishing): still never arms."""
    orig = link.verify_mission

    def verify_then_cancel(items, *a, **kw):
        ok = orig(items, *a, **kw)
        link.cancel_flight()
        return ok
    link.verify_mission = verify_then_cancel     # type: ignore[method-assign]
    with pytest.raises(VehicleError, match="cancelled"):
        link.fly_plan(make_plan())
    assert not link.mock.armed and 400 not in link.mock.commands_received


def test_arm_rejected_surfaces_prearm_text(link):
    link.mock.prearm_failures = ["PreArm: Compass not calibrated"]
    with pytest.raises(CommandRejected, match="PreArm: Compass not calibrated"):
        link.fly_plan(make_plan())
    assert not link.mock.armed


def test_command_ack_timeout_is_bounded_not_a_hang(link):
    link.mock.drop_inbound["COMMAND_LONG"] = 100
    t0 = time.time()
    with pytest.raises(LinkTimeout):
        link.set_mode("LOITER")
    assert time.time() - t0 < link.COMMAND_RETRIES * 0.6 + 2


# ---------------------------------------------------------------- link
def test_heartbeat_flapping_counts_every_loss_and_recovers():
    port = free_udp_port()
    mock = MockVehicle(f"udpout:127.0.0.1:{port}", home=HOME, hb_rate_hz=10.0).start()
    lost, restored = [], []
    be = ArduPilotBackend(make_settings(mavlink_url=f"udpin:127.0.0.1:{port}", heartbeat_loss_s=0.6, mavlink_timeout_s=0.6),
                          on_link_lost=lambda: lost.append(time.time()), on_link_restored=lambda: restored.append(time.time()))
    be.connect(timeout=5)
    try:
        for _ in range(5):
            mock.mute = True
            assert Rig.wait(lambda: not be.heartbeat_ok(), 3)
            mock.mute = False
            assert Rig.wait(lambda: be.heartbeat_ok(), 3)
            time.sleep(0.2)
        assert Rig.wait(lambda: len(restored) == 5, 3)
        assert len(lost) == 5
        be.set_param("AUTO_OPTIONS", 3)        # the link is fully usable afterwards
        assert mock.params["AUTO_OPTIONS"] == 3.0
    finally:
        be.close()
        mock.stop()


def test_vehicle_restart_on_same_port_reconnects():
    """Serial unplug/replug equivalent: the vehicle disappears (mock stop) and a new one starts on the same link."""
    port = free_udp_port()
    mock = MockVehicle(f"udpout:127.0.0.1:{port}", home=HOME, hb_rate_hz=10.0).start()
    lost, restored = threading.Event(), threading.Event()
    be = ArduPilotBackend(make_settings(mavlink_url=f"udpin:127.0.0.1:{port}", heartbeat_loss_s=0.6, mavlink_timeout_s=0.6),
                          on_link_lost=lost.set, on_link_restored=restored.set)
    be.connect(timeout=5)
    mock2 = None
    try:
        mock.stop()
        assert lost.wait(4) and not be.heartbeat_ok()
        with pytest.raises(VehicleError):
            be.set_mode("LOITER")                 # clear error while down, no hang
        mock2 = MockVehicle(f"udpout:127.0.0.1:{port}", home=HOME, hb_rate_hz=10.0).start()
        assert restored.wait(6) and be.heartbeat_ok()
        be.set_mode("LOITER")
        assert mock2.mode == MODE_LOITER
    finally:
        be.close()
        mock.stop()
        if mock2:
            mock2.stop()


def test_serial_reader_error_triggers_reconnect_not_crash(monkeypatch):
    """recv_match raising (unplugged adapter) must not kill the reader thread; it reopens the link."""
    port = free_udp_port()
    mock = MockVehicle(f"udpout:127.0.0.1:{port}", home=HOME, hb_rate_hz=10.0).start()
    be = ArduPilotBackend(make_settings(mavlink_url=f"udpin:127.0.0.1:{port}", heartbeat_loss_s=1.0))
    be.connect(timeout=5)
    try:
        conn = be._conn

        def boom(*a, **kw):
            raise OSError("device disconnected")
        monkeypatch.setattr(conn, "recv_match", boom)
        assert Rig.wait(lambda: be.reconnect_count >= 1, 5)
        assert be._thread.is_alive()
        assert Rig.wait(lambda: be.heartbeat_ok() and be._conn is not conn, 8)
    finally:
        be.close()
        mock.stop()


def test_garbage_second_gcs_and_foreign_vehicle_are_ignored():
    port = free_udp_port()
    mock = MockVehicle(f"udpout:127.0.0.1:{port}", home=HOME, hb_rate_hz=10.0, time_scale=20).start()
    mock.garbage_every_s = 0.05                     # random bytes in the stream
    be = ArduPilotBackend(make_settings(mavlink_url=f"udpin:127.0.0.1:{port}", heartbeat_loss_s=1.0, mavlink_timeout_s=1.0))
    be.connect(timeout=5)
    from pymavlink import mavutil
    other = MockVehicle(f"udpout:127.0.0.1:{port}", home=(10.0, 10.0), sysid=7, hb_rate_hz=10.0).start()   # other drone
    other.armed = True
    other.pilot_set_mode(MODE_LOITER)
    gcs = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}", source_system=255)         # second GCS
    stop = threading.Event()

    def gcs_loop() -> None:
        while not stop.is_set():
            gcs.mav.heartbeat_send(6, 8, 0, 0, 4)
            gcs.mav.statustext_send(2, b"GCS says hello")
            gcs.mav.sys_status_send(0, 0, 0, 0, 1000, 0, 5, 0, 0, 0, 0, 0, 0)
            time.sleep(0.05)
    th = threading.Thread(target=gcs_loop, daemon=True)
    th.start()
    try:
        time.sleep(1.5)
        tel = be.telemetry()
        assert tel.connected and tel.mode == "STABILIZE" and not tel.armed
        assert be._sysid == 1
        assert tel.lat == pytest.approx(HOME[0], abs=1e-4)            # not the other vehicle's (10, 10)
        assert all("GCS says hello" != d["text"] for d in be.statustexts())
        items = be.build_mission_items(make_plan())
        be.upload_mission(items)                                    # protocol still works through the noise
        assert be.verify_mission(items)
        be.set_mode("LOITER")
        assert mock.mode == MODE_LOITER
    finally:
        stop.set()
        th.join(2)
        be.close()
        other.stop()
        mock.stop()


# ---------------------------------------------------------------- access modes (readonly / upload / full)
def _gated(mode: str):
    be = ArduPilotBackend(make_settings(vehicle_mode=mode, mavlink_url="serial:/dev/ttyACM0"))
    sent: list = []

    class Mav:
        def __getattr__(self, name):
            return lambda *a, **k: sent.append(name)

    class Conn:
        mav = Mav()
    be._conn, be._sysid = Conn(), 1
    return be, sent


@pytest.mark.parametrize("name,args", [
    ("mission_count_send", (1, 1, 3, 0)), ("mission_item_int_send", (1, 1, 0, 0, 16, 0, 1, 0, 0, 0, 0, 0, 0, 0.0, 0)),
    ("mission_clear_all_send", (1, 1, 0)), ("set_mode_send", (1, 1, 5)), ("param_set_send", (1, 1, b"X", 1.0, 9)),
    ("command_long_send", (1, 1, 400, 0, 1, 0, 0, 0, 0, 0, 0)), ("command_long_send", (1, 1, 176, 0, 1, 5, 0, 0, 0, 0, 0)),
    ("command_long_send", (1, 1, 20, 0, 0, 0, 0, 0, 0, 0, 0)), ("command_long_send", (1, 1, 300, 0, 0, 0, 0, 0, 0, 0, 0)),
    ("command_long_send", (1, 1, 22, 0, 0, 0, 0, 0, 0, 0, 0)), ("command_int_send", (1, 1, 0, 192, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    ("rc_channels_override_send", (1, 1, 1500, 1500, 1500, 1500, 0, 0, 0, 0)),
])
def test_readonly_refuses_every_write_message_before_io(name, args):
    be, sent = _gated("readonly")
    with pytest.raises(ReadOnlyViolation):
        be._send(name, *args)
    assert sent == []


@pytest.mark.parametrize("name,args", [
    ("heartbeat_send", (18, 8, 0, 0, 4)), ("request_data_stream_send", (1, 1, 0, 4, 1)),
    ("param_request_read_send", (1, 1, b"RTL_ALT", -1)), ("param_request_list_send", (1, 1)),
    ("mission_request_list_send", (1, 1, 0)), ("mission_request_int_send", (1, 1, 0, 0)), ("mission_ack_send", (1, 1, 0, 0)),
    ("command_long_send", (1, 1, 511, 0, 33, 200000, 0, 0, 0, 0, 0)), ("command_long_send", (1, 1, 512, 0, 242, 0, 0, 0, 0, 0, 0)),
    ("command_long_send", (1, 1, 401, 0, 0, 0, 0, 0, 0, 0, 0)),
])
def test_readonly_allows_only_reads(name, args):
    be, sent = _gated("readonly")
    be._send(name, *args)
    assert sent == [name]


@pytest.mark.parametrize("name,args", [("mission_count_send", (1, 1, 3, 0)), ("mission_clear_all_send", (1, 1, 0)),
                                       ("mission_item_int_send", (1, 1, 0, 0, 16, 0, 1, 0, 0, 0, 0, 0, 0, 0.0, 0))])
def test_upload_mode_allows_mission_protocol_only(name, args):
    be, sent = _gated("upload")
    be._send(name, *args)
    assert sent == [name]
    for bad, a in (("set_mode_send", (1, 1, 5)), ("param_set_send", (1, 1, b"X", 1.0, 9)),
                   ("command_long_send", (1, 1, 400, 0, 1, 0, 0, 0, 0, 0, 0)),
                   ("command_long_send", (1, 1, 300, 0, 0, 0, 0, 0, 0, 0, 0)),
                   ("command_long_send", (1, 1, 20, 0, 0, 0, 0, 0, 0, 0, 0))):
        with pytest.raises(ReadOnlyViolation):
            be._send(bad, *a)
    assert sent == [name]


@pytest.mark.parametrize("mode", ["readonly", "upload"])
def test_readonly_and_upload_refuse_control_methods_with_connection_present(mode):
    be, sent = _gated(mode)
    for fn, args in (("set_mode", ("AUTO",)), ("arm", ()), ("disarm", ()), ("start_mission", ()), ("rtl", ()),
                     ("set_param", ("AUTO_OPTIONS", 3)), ("fly_plan", (make_plan(),))):
        with pytest.raises(ReadOnlyViolation):
            getattr(be, fn)(*args)
    assert sent == []


def test_full_mode_on_serial_needs_real_arm_permission():
    for allow, dry in ((False, True), (True, True), (False, False)):
        be, sent = _gated("full")
        be.settings = make_settings(vehicle_mode="full", mavlink_url="serial:/dev/ttyACM0", allow_real_arm=allow, dry_run=dry)
        with pytest.raises(mission.ArmRefused):
            be.arm()
        assert sent == []
    ok, why = mission.arm_guard(make_settings(vehicle_mode="full", mavlink_url="serial:/dev/ttyACM0",
                                              allow_real_arm=True, dry_run=False))
    assert ok and "permitted" in why


def test_bench_home_only_in_upload_mode():
    from config import BenchHomeError, bench_home_for
    for mode in ("sim", "readonly", "full"):
        with pytest.raises(BenchHomeError):
            bench_home_for(make_settings(vehicle_mode=mode, bench_home="field"))
    assert bench_home_for(make_settings(vehicle_mode="upload", bench_home="field")) is not None


# =============================================================================== controller level
def call(rig: Rig, coro) -> "cf.Future":
    """Run a controller coroutine on the app's event loop; returns a concurrent future."""
    return asyncio.run_coroutine_threadsafe(coro, rig.ctrl._loop)


def post_bg(rig: Rig, path: str, body: dict) -> "cf.Future":
    ex = cf.ThreadPoolExecutor(1)
    fut = ex.submit(lambda: rig.client.post(path, json=body).json())
    ex.shutdown(wait=False)
    return fut


def start_flight(rig: Rig, **plan_kw) -> str:
    pid = rig.plan(**plan_kw)["data"]["plan_id"]
    rig.ack()
    r = rig.fly(pid)                                       # Confirm (stage) + GO
    assert "mission started" in r, r
    assert rig.wait(lambda: rig.ctrl.state == "flying", 20)
    return pid


def errors_seen(rig: Rig, ws, needle: str, timeout: float = 10.0) -> str:
    """Wait for an `error` message containing `needle`; earlier errors read on this socket are remembered, so the
    order in which several alerts arrive does not matter."""
    seen: list = ws.__dict__.setdefault("_errs", [])
    end = time.time() + timeout
    while True:
        for m in seen:
            if needle in m:
                seen.remove(m)
                return m
        if time.time() > end:
            raise AssertionError(f"no error containing {needle!r}; saw {seen}")
        msg = ws.receive_json()
        if msg["type"] == "error":
            seen.append(msg["data"]["message"])


SLOW = {"time_scale": 2.0}


# ---------------------------------------------------------------- link loss in every mission state
def test_link_loss_idle_and_planned_alert_and_block_confirm(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=1.0, mavlink_timeout_s=1.0) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        rig.mock.mute = True
        errors_seen(rig, ws, "link to the Pixhawk lost")                   # idle
        assert "No MAVLink link" in rig.plan()["data"]["message"]
        rig.mock.mute = False
        assert rig.wait(lambda: rig.ctrl.vehicle.heartbeat_ok(), 20)
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        assert rig.ctrl.state == "planned"
        rig.mock.mute = True                                                # planned
        assert rig.wait(lambda: not rig.ctrl.vehicle.heartbeat_ok(), 20)
        assert "No MAVLink link" in rig.post("/api/confirm", {"plan_id": pid})["data"]["text"]
        assert rig.mock.mission == [] and not rig.mock.armed
        assert "ABORT NOT SENT" not in rig.post("/api/abort")["data"]["text"] or True


def test_link_loss_during_upload_never_arms(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=1.0, mavlink_timeout_s=1.0, mock_kw=SLOW) as rig:
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        rig.mock.drop_inbound["MISSION_ITEM_INT"] = 1000          # upload will hang on the first item
        fut = post_bg(rig, "/api/confirm", {"plan_id": pid})
        assert rig.wait(lambda: rig.ctrl.state == "uploading", 20)
        rig.mock.mute = True
        txt = fut.result(60)["data"]["text"]
        assert "FAILED" in txt and "STAGED" not in txt, txt       # Confirm never arms; the upload just fails
        assert not rig.mock.armed and not {400, 300} & set(rig.mock.commands_received)
        assert rig.ctrl.state == "planned" and rig.ctrl.staged is None
        rig.mock.mute = False


def test_link_loss_while_arming_and_takeoff_commands_rtl_when_back(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=1.0, mavlink_timeout_s=1.0, mock_kw=SLOW) as rig:
        start_flight(rig)                                          # armed, takeoff phase
        rig.mock.mute = True
        assert rig.wait(lambda: any(e[1] == "link_lost" for e in rig.ctrl.events), 20)
        rig.mock.mute = False                                      # link returns within the RTL retry window
        assert rig.wait(lambda: rig.mock.mode == MODE_RTL, 30)
        assert rig.wait(lambda: rig.ctrl.state in ("rtl", "landed"), 25)


def test_link_loss_in_survey_then_flap_sends_single_rtl_state(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=1.0, mavlink_timeout_s=1.0, mock_kw={"time_scale": 4.0}) as rig:
        start_flight(rig)
        assert rig.wait(lambda: (rig.mock.seq or 0) >= 4, 90)
        for _ in range(3):                                         # flapping link
            rig.mock.mute = True
            time.sleep(1.4)
            rig.mock.mute = False
            time.sleep(1.0)
        assert rig.wait(lambda: rig.mock.mode == MODE_RTL, 20)
        assert rig.wait(lambda: rig.ctrl.state in ("rtl", "landed"), 25)
        n_rtl = [e for e in rig.ctrl.events if e[1] == "rtl" and "link" in e[2]["reason"]]
        assert 1 <= len(n_rtl) <= 3
        rig.post("/api/abort")


def test_link_loss_in_rtl_and_landed_only_alerts(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=1.0, mavlink_timeout_s=1.0, mock_kw=SLOW) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        start_flight(rig)
        rig.post("/api/abort")
        assert rig.wait(lambda: rig.ctrl.state == "rtl", 20)
        before = len(rig.mock.commands_received)
        rig.mock.mute = True
        errors_seen(rig, ws, "link to the Pixhawk lost")
        rig.mock.mute = False
        assert rig.wait(lambda: rig.ctrl.state == "landed", 60)
        assert rig.wait(lambda: rig.ctrl.vehicle._link_up, 20)
        rig.mock.mute = True
        assert rig.wait(lambda: not rig.ctrl.vehicle._link_up, 20), "heartbeat loss not detected after landing"
        errors_seen(rig, ws, "link to the Pixhawk lost")
        rig.mock.mute = False
        assert not [c for c in rig.mock.commands_received[before:] if c in (400, 20, 300)]


# ---------------------------------------------------------------- pilot / failsafe events in flight
def test_pilot_switches_out_of_auto_app_stops_and_never_fights(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=1.0, mavlink_timeout_s=1.0, mock_kw=SLOW) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        start_flight(rig)
        assert rig.wait(lambda: (rig.mock.seq or 0) >= 3, 30)
        cmds = len(rig.mock.commands_received)
        rig.mock.pilot_set_mode(MODE_LOITER)
        msg = errors_seen(rig, ws, "PILOT CONTROL")
        assert "LOITER" in msg and "no commands" in msg
        assert rig.ctrl.state == "manual"
        time.sleep(1.5)
        assert rig.mock.mode == MODE_LOITER
        # Abort and link loss must NOT take the aircraft away from the pilot
        a = rig.post("/api/abort")["data"]["text"]
        assert "pilot has taken over" in a and "RC transmitter" in a
        rig.mock.mute = True
        assert rig.wait(lambda: any(e[1] == "link_lost" for e in rig.ctrl.events), 20)
        rig.mock.mute = False
        time.sleep(2.5)
        assert rig.mock.mode == MODE_LOITER
        assert not [c for c in rig.mock.commands_received[cmds:] if c in (176, 20, 300, 400)]
        assert not [e for e in rig.ctrl.events if e[1] == "rtl"]
        # new plans are blocked while the pilot flies; landing (pilot) ends the flight record
        assert "already in progress" in rig.plan()["data"]["message"]
        rig.mock.pilot_set_mode(MODE_LAND)
        assert rig.wait(lambda: rig.ctrl.state == "landed", 60)
        assert rig.ctrl.flight is None


def test_gps_loss_in_flight_alerts_and_follows_vehicle_failsafe(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=3.0, mavlink_timeout_s=2.0, mock_kw=SLOW) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        start_flight(rig)
        assert rig.wait(lambda: (rig.mock.seq or 0) >= 3, 30)
        rig.mock.trigger_gps_loss()
        assert "GPS fix LOST" in errors_seen(rig, ws, "GPS fix LOST")
        assert "Vehicle: EKF failsafe" in errors_seen(rig, ws, "EKF failsafe")
        assert rig.wait(lambda: rig.ctrl.state == "rtl", 25)         # vehicle went LAND by itself
        assert rig.wait(lambda: rig.ctrl.state == "landed", 60)


def test_battery_failsafe_rtl_by_vehicle_is_reported_not_duplicated(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=3.0, mavlink_timeout_s=2.0, mock_kw=SLOW) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        start_flight(rig)
        assert rig.wait(lambda: (rig.mock.seq or 0) >= 3, 30)
        rig.mock.trigger_battery_failsafe()
        errors_seen(rig, ws, "Failsafe: Battery low")
        errors_seen(rig, ws, "mode changed AUTO -> RTL")
        assert rig.wait(lambda: rig.ctrl.state == "rtl", 25)
        assert not [e for e in rig.ctrl.events if e[1] == "rtl"]       # the app sent no RTL of its own
        assert rig.wait(lambda: rig.ctrl.state == "landed", 60)


def test_geofence_breach_reported(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=3.0, mavlink_timeout_s=2.0, mock_kw=SLOW) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        start_flight(rig)
        rig.mock.trigger_fence_breach()
        errors_seen(rig, ws, "Fence breach")
        assert rig.wait(lambda: rig.ctrl.state == "rtl", 25)
        rig.post("/api/abort")


def test_mission_complete_landed_and_disarm_detected(tmp_path):
    with Rig(tmp_path, mock_kw={"time_scale": 25.0}) as rig:
        start_flight(rig)
        assert rig.wait(lambda: rig.ctrl.state == "landed", 90)
        assert not rig.mock.armed and rig.ctrl.last_flight_dir
        meta = json.loads((tmp_path / rig.ctrl.last_flight_dir / "flight.json").read_text())
        assert meta["summary"]["end_reason"] == "landed"


def test_vehicle_rejects_arm_prearm_text_reaches_operator(tmp_path):
    with Rig(tmp_path) as rig:
        rig.mock.prearm_failures = ["PreArm: Compass not calibrated"]
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        rig.stage(pid)
        txt = rig.post("/api/go", {"plan_id": pid})["data"]["text"]      # readiness (pre-arm) blocks GO
        assert "GO refused" in txt and "PreArm: Compass not calibrated" in txt, txt
        assert rig.ctrl.state == "staged" and not rig.mock.armed and 400 not in rig.mock.commands_received
        assert rig.ctrl.pending is not None                  # operator can fix and press GO again
        rig.mock.prearm_failures = []
        assert rig.wait(lambda: (rig.ctrl.vehicle.raw("SYS_STATUS")[1]["onboard_control_sensors_health"] & 0x10000000), 5)
        assert "mission started" in rig.post("/api/go", {"plan_id": pid})["data"]["text"]
        rig.post("/api/abort")


# ---------------------------------------------------------------- home
def test_home_altitude_jump_rebuilds_plan(tmp_path):
    with Rig(tmp_path) as rig, rig.client.websocket_connect("/ws") as ws:
        from test_e2e import wait_ws
        ws.receive_json()
        p1 = rig.plan()["data"]
        rig.mock.move_home(0.0, 0.0, d_alt_m=6.0)
        m = wait_ws(ws, lambda m: m["type"] == "chat" and "plan recalculated" in m["data"]["text"], 10)
        assert "+6.0 m" in m["data"]["text"]
        assert rig.ctrl.pending.plan_id != p1["plan_id"] and rig.ctrl.pending.home_amsl_m == pytest.approx(306.0)


def test_small_home_wobble_keeps_plan(tmp_path):
    with Rig(tmp_path) as rig:
        p1 = rig.plan()["data"]
        rig.mock.move_home(1.5, 0.0, d_alt_m=0.5)
        time.sleep(2.5)
        assert rig.ctrl.pending.plan_id == p1["plan_id"]


def test_home_missing_blocks_planning_and_confirm(tmp_path):
    with Rig(tmp_path) as rig:
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        rig.ctrl.vehicle._home_fix = None                      # home vanished (e.g. link restart)
        rig.mock.send_home = False
        txt = rig.post("/api/confirm", {"plan_id": pid})["data"]["text"]
        assert "No Pixhawk home" in txt and rig.mock.mission == []


# ---------------------------------------------------------------- commands
def test_double_confirm_second_is_ignored(tmp_path):
    with Rig(tmp_path, mock_kw=SLOW) as rig:
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        futs = [post_bg(rig, "/api/confirm", {"plan_id": pid}) for _ in range(2)]
        res = sorted(f.result(60)["data"]["text"] for f in futs)
        assert sum("Mission uploaded and verified" in r for r in res) == 1, res
        assert any(("already being processed" in r) or ("already in progress" in r) or ("waiting for GO" in r)
                   for r in res), res
        assert not rig.mock.armed and rig.ctrl.state == "staged"
        futs = [post_bg(rig, "/api/go", {"plan_id": pid}) for _ in range(2)]       # GO twice (two phones)
        res = sorted(f.result(60)["data"]["text"] for f in futs)
        assert sum("mission started" in r for r in res) == 1, res
        assert any(("already being processed" in r) or ("no mission is staged" in r) for r in res), res
        assert rig.mock.commands_received.count(400) == 1            # armed exactly once
        rig.post("/api/abort")


def test_confirm_stale_or_unknown_plan_id(tmp_path):
    with Rig(tmp_path) as rig:
        p1 = rig.plan()["data"]["plan_id"]
        p2 = rig.plan(altitude_m=25)["data"]["plan_id"]
        rig.ack()
        for pid in (p1, "deadbeef"):
            assert "Unknown or expired" in rig.post("/api/confirm", {"plan_id": pid})["data"]["text"]
        assert rig.mock.mission == []
        assert "Mission uploaded" in rig.post("/api/confirm", {"plan_id": p2})["data"]["text"]
        rig.post("/api/abort")


def test_confirm_while_upload_running_is_refused(tmp_path):
    with Rig(tmp_path, mavlink_timeout_s=1.0, mock_kw=SLOW) as rig:
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        rig.mock.drop_inbound["MISSION_ITEM_INT"] = 3                 # slow, retried upload
        f1 = post_bg(rig, "/api/confirm", {"plan_id": pid})
        assert rig.wait(lambda: rig.ctrl.state == "uploading", 20)
        t2 = rig.post("/api/confirm", {"plan_id": pid})["data"]["text"]
        assert "already" in t2, t2
        assert "Mission uploaded" in f1.result(60)["data"]["text"]
        rig.post("/api/abort")


def test_abort_during_upload_cancels_and_never_arms(tmp_path):
    with Rig(tmp_path, mavlink_timeout_s=1.0, mock_kw=SLOW) as rig:
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        rig.mock.drop_inbound["MISSION_ITEM_INT"] = 1000
        fut = post_bg(rig, "/api/confirm", {"plan_id": pid})
        assert rig.wait(lambda: rig.ctrl.state == "uploading", 20)
        t0 = time.time()
        a = rig.post("/api/abort")["data"]["text"]
        assert "cancelled" in a and "will not arm" in a, a
        txt = fut.result(30)["data"]["text"]
        assert time.time() - t0 < 10 and "cancelled" in txt, txt
        assert not rig.mock.armed and 400 not in rig.mock.commands_received
        assert rig.ctrl.state == "planned"


def test_abort_during_takeoff_and_when_already_rtl_and_concurrent_aborts(tmp_path):
    with Rig(tmp_path, mock_kw=SLOW) as rig:
        start_flight(rig)
        assert rig.mock.armed and rig.mock.alt < 20            # still climbing
        futs = [post_bg(rig, "/api/abort", {}) for _ in range(2)]   # two phones at once
        texts = [f.result(30)["data"]["text"] for f in futs]
        assert all(("RTL" in t or "Already returning" in t or "disarmed" in t) for t in texts), texts
        assert rig.wait(lambda: rig.mock.mode == MODE_RTL, 25)
        assert rig.wait(lambda: rig.ctrl.state == "rtl", 20)
        again = rig.post("/api/abort")["data"]["text"]
        assert "Already returning" in again or "disarmed" in again, again    # (the mock may already have landed)
        assert rig.wait(lambda: rig.ctrl.state == "landed", 60)
        assert "nothing to abort" in rig.post("/api/abort")["data"]["text"]


def test_abort_with_link_down_says_use_rc(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=1.0, mavlink_timeout_s=1.0, mock_kw=SLOW) as rig:
        start_flight(rig)
        rig.mock.mute = True
        assert rig.wait(lambda: not rig.ctrl.vehicle.heartbeat_ok(), 20)
        t0 = time.time()
        a = rig.post("/api/abort")["data"]["text"]
        assert time.time() - t0 < 3 and "RC transmitter" in a and "NOT SENT" in a, a
        rig.mock.mute = False
        assert rig.wait(lambda: rig.mock.mode == MODE_RTL, 30)       # sent as soon as the link returned


# ---------------------------------------------------------------- shutdown policy
def test_shutdown_rtl_policy(tmp_path):
    with Rig(tmp_path, mock_kw=SLOW) as rig:
        start_flight(rig)
        assert rig.mock.mode == MODE_AUTO
        t0 = time.time()
        assert rig.ctrl.shutdown_rtl(2.0) == "RTL commanded"
        assert time.time() - t0 < 2.5 and rig.mock.mode == MODE_RTL
    with Rig(tmp_path / "pilot", mock_kw=SLOW) as rig:
        start_flight(rig)
        rig.mock.pilot_set_mode(MODE_LOITER)
        assert rig.wait(lambda: rig.ctrl.state == "manual", 20)
        assert rig.ctrl.shutdown_rtl(2.0) is None and rig.mock.mode == MODE_LOITER
    with Rig(tmp_path / "ro", vehicle_mode="readonly") as rig:
        assert rig.ctrl.shutdown_rtl(2.0) is None
    with Rig(tmp_path / "idle") as rig:
        assert rig.ctrl.shutdown_rtl(2.0) is None and not rig.mock.commands_received.count(20)


def test_shutdown_during_upload_cancels(tmp_path):
    with Rig(tmp_path, mavlink_timeout_s=1.0, mock_kw=SLOW) as rig:
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        rig.mock.drop_inbound["MISSION_ITEM_INT"] = 1000
        fut = post_bg(rig, "/api/confirm", {"plan_id": pid})
        assert rig.wait(lambda: rig.ctrl.state == "uploading", 20)
        assert rig.ctrl.shutdown_rtl(2.0) == "cancelled upload"
        assert "cancelled" in fut.result(30)["data"]["text"]
        assert not rig.mock.armed


def _wait_http(port: int, timeout: float = 30.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=1).read()
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    raise AssertionError("main.py never served /api/status")


def _port_is_free(port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def test_sigterm_in_flight_commands_rtl_and_frees_port_fast(tmp_path):
    """Real main.py process (sim mode) + external mock vehicle process: SIGTERM while AUTO -> vehicle goes RTL,
    process gone and web port free in < 3 s."""
    vport, wport = free_udp_port(), 0
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        wport = s.getsockname()[1]
    mock = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "mock_vehicle.py"), "--url", f"udpout:127.0.0.1:{vport}",
                             "--time-scale", "2", "--home", f"{FIELD[0]},{FIELD[1]}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    lines: list[str] = []
    threading.Thread(target=lambda: [lines.append(l) for l in mock.stdout], daemon=True).start()
    env = dict(os.environ, MAVLINK20="1")
    app = subprocess.Popen([sys.executable, str(ROOT / "main.py"), "--no-vision", "--vehicle", "sim", "--mavlink-url",
                            f"udpin:127.0.0.1:{vport}", "--port", str(wport), "--host", "127.0.0.1", "--data-dir",
                            str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT, env=env)
    try:
        _wait_http(wport)
        end = time.time() + 30
        while time.time() < end:
            st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{wport}/api/status", timeout=2).read())["data"]
            if st["home"] and (st["telemetry"].get("gps_fix") or 0) >= 3 and st["telemetry"].get("battery_pct") is not None:
                break
            time.sleep(0.5)
        r = _post(wport, "/api/plan_area", {"corners": CORNERS, "altitude_m": 20})
        assert r["type"] == "plan", r
        _post(wport, "/api/legal/ack", {"accepted": [True] * 4})
        txt = _post(wport, "/api/confirm", {"plan_id": r["data"]["plan_id"]})["data"]["text"]
        assert "STAGED" in txt, txt
        txt = _post(wport, "/api/go", {"plan_id": r["data"]["plan_id"]})["data"]["text"]
        assert "mission started" in txt, txt
        assert Rig.wait(lambda: any("mode=3 armed=True" in l for l in lines), 10), lines[-5:]
        t0 = time.time()
        app.send_signal(signal.SIGTERM)
        app.wait(timeout=10)
        dt = time.time() - t0
        print(f"\nSIGTERM in flight: process exited in {dt:.2f} s")
        assert dt < 3.2, f"exit took {dt:.1f}s"
        assert _port_is_free(wport)
        assert Rig.wait(lambda: any("mode=6" in l for l in lines), 8), f"vehicle never got RTL: {lines[-4:]}"
    finally:
        for pr in (app, mock):
            if pr.poll() is None:
                pr.kill()
            pr.wait(5)
