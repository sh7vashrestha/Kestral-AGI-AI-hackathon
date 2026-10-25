"""End-to-end against real ArduPilot SITL (ArduCopter): phone WS -> plan -> legal ack -> confirm (upload/verify ->
staged) -> GO (readiness -> AUTO -> arm -> start) -> mission -> vision alerts -> mission RTL -> landed -> files.
Also: upload-only mode against SITL (no arming).

Skipped unless ~/ardupilot/build/sitl/bin/arducopter exists. Run:  .venv/bin/pytest -m sitl -s tests/test_e2e_sitl.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from conftest import make_settings  # noqa: E402
from plan_files import parse_wpl, validate_geojson  # noqa: E402
from test_e2e import ASSETS, CORNERS, FIELD, FakeDetector, FakeLLM, FakeMatch, free_udp_port, wait_ws  # noqa: E402

ARDUPILOT = Path(os.environ.get("ARDUPILOT_DIR", Path.home() / "ardupilot"))
BIN = ARDUPILOT / "build" / "sitl" / "bin" / "arducopter"
SPEEDUP = int(os.environ.get("SITL_SPEEDUP", "5"))

pytestmark = [pytest.mark.sitl, pytest.mark.skipif(not BIN.exists(), reason=f"SITL binary not built ({BIN})")]


@pytest.fixture
def sitl(tmp_path):
    port = free_udp_port()
    wd = tmp_path / "sitl"
    wd.mkdir()
    cmd = [str(BIN), "--model", "quad", "--speedup", str(SPEEDUP), "--home", f"{FIELD[0]},{FIELD[1]},300,0",
           "-I3", "--wipe", "--defaults",
           f"{ARDUPILOT}/Tools/autotest/default_params/copter.parm,{ROOT}/scripts/sitl_dronebrain.parm",
           "--serial0", f"udpclient:127.0.0.1:{port}"]
    log = open(wd / "sitl.log", "w")
    proc = subprocess.Popen(cmd, cwd=wd, stdout=log, stderr=subprocess.STDOUT)
    try:
        yield port, proc
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def _rig(tmp_path, port, vision, **over):
    import main
    from pipeline import VisionPipeline
    base = dict(data_dir=tmp_path, mavlink_url=f"udpin:127.0.0.1:{port}", camera_source=f"file:{ASSETS}",
                thermal_source="none", capture_fps=2.0, yolo_conf_threshold=0.5, min_frames=2, match_threshold=0.7)
    base.update(over)
    s = make_settings(**base)
    holder = {}
    pipeline = None
    if vision:
        det = FakeDetector()
        pipeline = VisionPipeline(s, on_alert=lambda a: holder["c"].push_alert(a),
                                  telemetry_fn=lambda: holder["c"].vehicle.telemetry(),
                                  detector_factory=lambda _s: det, backend_factory=lambda _s: FakeMatch())
    app, ctrl, _ = main.build_app(s, vision=vision, pipeline=pipeline, llm=FakeLLM())
    holder["c"] = ctrl
    return s, app, ctrl


def _wait_ready(ctrl, timeout=180.0) -> float:
    """3D fix + home + EKF good enough that RUN_PREARM_CHECKS passes."""
    t0 = time.time()
    v = ctrl.vehicle
    while time.time() - t0 < timeout:
        tel = v.telemetry()
        if tel.connected and (tel.gps_fix or 0) >= 3 and v.home_position() is not None:
            if v.access == "control":
                if v.run_prearm_checks(wait_s=1.0)["ack"] == "ACCEPTED":
                    return time.time() - t0
            else:
                return time.time() - t0
        time.sleep(1.0)
    raise AssertionError(f"SITL not ready after {timeout}s: {v.telemetry()} {[d['text'] for d in v.statustexts()][-5:]}")


def test_sitl_full_mission(tmp_path, sitl):
    port, _proc = sitl
    t = {}
    t0 = time.time()
    s, app, ctrl = _rig(tmp_path, port, vision=True)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        t["ready_s"] = _wait_ready(ctrl)
        ws.receive_json()
        ws.send_json({"type": "plan_area", "corners": CORNERS, "altitude_m": 20, "target_text": "person in a red jacket"})
        plan = wait_ws(ws, lambda m: m["type"] in ("plan", "error"), 60)
        assert plan["type"] == "plan", plan
        p = plan["data"]
        assert p["safety"]["ok"], (p["safety"], p["preflight"])
        assert p["plan"]["home"]["lat"] == pytest.approx(FIELD[0], abs=1e-5)
        t["plan_s"] = time.time() - t0
        assert client.post("/api/legal/ack", json={"accepted": [True] * 4}).status_code == 200
        ws.send_json({"type": "confirm", "plan_id": p["plan_id"]})
        m = wait_ws(ws, lambda m: m["type"] == "chat" and m["data"]["role"] == "system", 90)
        assert "Mission uploaded and verified" in m["data"]["text"] and "STAGED" in m["data"]["text"], m
        ws.send_json({"type": "go", "plan_id": p["plan_id"]})
        m = wait_ws(ws, lambda m: m["type"] == "chat" and m["data"]["text"].startswith("GO"), 90)
        assert "mission started" in m["data"]["text"], m
        t["armed_started_s"] = time.time() - t0
        levels = set()
        wait_ws(ws, lambda m: m["type"] == "alert" and levels.add(m["data"]["level"]) is None
                and {"person", "target_match"} <= levels, 60)
        t["alerts_s"] = time.time() - t0
        max_alt = 0.0
        max_seq = 0

        def done(m):
            nonlocal max_alt, max_seq
            if m["type"] == "telemetry":
                max_alt = max(max_alt, m["data"].get("alt_rel_m") or 0.0)
                max_seq = max(max_seq, m["data"].get("mission_seq") or 0)
            return m["type"] == "status" and m["data"]["mission_state"] == "landed"
        wait_ws(ws, done, 600)
        t["landed_s"] = time.time() - t0
        assert max_alt > 15, max_alt
        assert max_seq >= len(p["plan"]["waypoints"]), max_seq
        flight_dir = tmp_path / ctrl.last_flight_dir
    pdir = tmp_path / "plans" / p["plan_id"]
    assert parse_wpl((pdir / "mission.waypoints").read_text())[-1]["command"] == 20
    validate_geojson(json.loads((pdir / "mission.geojson").read_text()))
    for k in ("telemetry", "detections", "alerts"):
        assert (flight_dir / f"{k}.jsonl").read_text().strip(), k
    meta = json.loads((flight_dir / "flight.json").read_text())
    assert meta["summary"]["end_reason"] == "landed" and meta["summary"]["distance_m"] > 100
    t["max_alt_m"] = max_alt
    print(f"\nSITL (speedup {SPEEDUP}) timings (s, wall clock):", {k: round(v, 1) for k, v in t.items()})
    (tmp_path.parent / "sitl_timings.json").write_text(json.dumps(t))


def test_sitl_upload_only(tmp_path, sitl):
    port, _proc = sitl
    s, app, ctrl = _rig(tmp_path, port, vision=False, vehicle_mode="upload", upload_fence_polygon=True)
    with TestClient(app) as client:
        _wait_ready(ctrl)
        r = client.post("/api/plan_area", json={"corners": CORNERS, "altitude_m": 20}).json()
        assert r["type"] == "plan", r
        p = r["data"]
        assert client.post("/api/legal/ack", json={"accepted": [True] * 4}).status_code == 200
        txt = client.post("/api/confirm", json={"plan_id": p["plan_id"]}).json()["data"]["text"]
        assert "read-back verify OK" in txt and "Fence polygon: 4 vertices, verify OK" in txt, txt
        tel = ctrl.vehicle.telemetry()
        assert not tel.armed and tel.mode == "STABILIZE"
        rb = parse_wpl((tmp_path / "plans" / p["plan_id"] / "readback.waypoints").read_text())
        assert len(rb) == len(p["plan"]["waypoints"]) + 4
        assert "Mission cleared" in client.post("/api/vehicle/clear_mission").json()["data"]["text"]
        rep = client.get("/api/vehicle/check").json()
        print("\nSITL vehicle check:", rep["verdict"], rep["reasons"], rep["firmware"])
        assert rep["firmware"]["available"] and rep["gps"]["fix_type"] >= 3


# =============================================================================================== failure scenarios
# Hardening C: Abort mid-survey, pilot mode change, battery failsafe, GPS loss - against real ArduCopter SITL.
# Parameter changes below are sent ONLY to the SITL process this fixture started (second MAVLink link on a private TCP port).
import contextlib  # noqa: E402

from test_e2e import wait_ws as _wait_ws  # noqa: E402


@pytest.fixture
def sitl2(tmp_path):
    """Like `sitl`, plus a second MAVLink endpoint (SERIAL1 on a private TCP port) used as 'the pilot / test hand'."""
    import socket
    port = free_udp_port()
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        tcp = so.getsockname()[1]
    wd = tmp_path / "sitl"
    wd.mkdir()
    cmd = [str(BIN), "--model", "quad", "--speedup", str(SPEEDUP), "--home", f"{FIELD[0]},{FIELD[1]},300,0",
           "-I4", "--wipe", "--defaults",
           f"{ARDUPILOT}/Tools/autotest/default_params/copter.parm,{ROOT}/scripts/sitl_dronebrain.parm",
           "--serial0", f"udpclient:127.0.0.1:{port}", "--serial1", f"tcp:{tcp}"]
    log = open(wd / "sitl.log", "w")
    proc = subprocess.Popen(cmd, cwd=wd, stdout=log, stderr=subprocess.STDOUT)
    try:
        yield port, tcp, proc
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


class Hand:
    """Second pymavlink connection to the SITL ONLY (never to real hardware): flips modes, sets SITL parameters."""

    def __init__(self, tcp_port: int) -> None:
        from pymavlink import mavutil
        self.c = mavutil.mavlink_connection(f"tcp:127.0.0.1:{tcp_port}", source_system=250, source_component=190)
        assert self.c.wait_heartbeat(timeout=40), "no heartbeat on the SITL's second port"

    def set_param(self, name: str, value: float) -> None:
        self.c.mav.param_set_send(self.c.target_system, self.c.target_component, name.encode(), float(value), 9)
        m = self.c.recv_match(type="PARAM_VALUE", blocking=True, timeout=5)
        end = time.time() + 5
        while m is not None and m.param_id.rstrip("\x00") != name and time.time() < end:
            m = self.c.recv_match(type="PARAM_VALUE", blocking=True, timeout=1)
        assert m is not None and abs(m.param_value - value) < 1e-3, f"SITL did not accept {name}={value}: {m}"

    def set_mode(self, name: str) -> None:
        self.c.set_mode(self.c.mode_mapping()[name])

    def shift_mission_item(self, seq: int, dlat_deg: float) -> int:
        """'Mission Planner edit': download the SITL's mission on this second link, move item `seq` north, upload it."""
        c = self.c
        ts, tc = c.target_system, c.target_component
        while c.recv_match(blocking=False) is not None:      # drop stale MISSION_ACKs etc. from earlier transfers
            pass
        c.mav.mission_request_list_send(ts, tc, 0)
        m = c.recv_match(type="MISSION_COUNT", blocking=True, timeout=10)
        assert m is not None, "no MISSION_COUNT from the SITL"
        n = m.count
        items = []
        for i in range(n):
            got = None
            for _ in range(4):
                c.mav.mission_request_int_send(ts, tc, i, 0)
                end = time.time() + 3
                while time.time() < end:
                    got = c.recv_match(type="MISSION_ITEM_INT", blocking=True, timeout=1)
                    if got is not None and got.seq == i:
                        break
                    got = None
                if got is not None:
                    break
            assert got is not None, f"item {i} not received"
            items.append(got)
        c.mav.mission_ack_send(ts, tc, 0, 0)
        c.mav.mission_count_send(ts, tc, n, 0)
        end = time.time() + 30
        sent: set[int] = set()
        while time.time() < end:
            r = c.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"], blocking=True, timeout=5)
            assert r is not None, "SITL stopped requesting mission items"
            if r.get_type() == "MISSION_ACK":
                if len(sent) < n or getattr(r, "mission_type", 0) != 0:
                    continue                                  # not the ACK of this upload
                assert r.type == 0, f"mission upload refused: {r}"
                return n
            sent.add(r.seq)
            it = items[r.seq]
            x = it.x + (int(round(dlat_deg * 1e7)) if r.seq == seq else 0)
            c.mav.mission_item_int_send(ts, tc, it.seq, it.frame, it.command, 0, it.autocontinue, it.param1, it.param2,
                                        it.param3, it.param4, x, it.y, it.z, 0)
        raise AssertionError("mission upload on the second link did not finish")

    def close(self) -> None:
        self.c.close()


@contextlib.contextmanager
def _flying(tmp_path, port, timings: dict):
    """Plan + Confirm against SITL; yields (client, ws, ctrl) with the mission running (>=12 m altitude)."""
    t0 = time.time()
    s, app, ctrl = _rig(tmp_path, port, vision=False)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        timings["ready_s"] = _wait_ready(ctrl)
        ws.receive_json()
        ws.send_json({"type": "plan_area", "corners": CORNERS, "altitude_m": 20, "target_text": "person"})
        plan = _wait_ws(ws, lambda m: m["type"] in ("plan", "error"), 60)
        assert plan["type"] == "plan" and plan["data"]["safety"]["ok"], plan
        assert client.post("/api/legal/ack", json={"accepted": [True] * 4}).status_code == 200
        t1 = time.time()
        ws.send_json({"type": "confirm", "plan_id": plan["data"]["plan_id"]})
        m = _wait_ws(ws, lambda m: m["type"] == "chat" and m["data"]["role"] == "system", 90)
        assert "STAGED" in m["data"]["text"], m
        ws.send_json({"type": "go", "plan_id": plan["data"]["plan_id"]})
        m = _wait_ws(ws, lambda m: m["type"] == "chat" and m["data"]["text"].startswith("GO"), 90)
        assert "mission started" in m["data"]["text"], m
        timings["confirm_to_started_s"] = time.time() - t1
        _wait_ws(ws, lambda m: m["type"] == "telemetry" and (m["data"].get("alt_rel_m") or 0) > 12, 120)
        timings["airborne_s"] = time.time() - t1
        yield client, ws, ctrl
    timings["total_s"] = time.time() - t0


def _wait_err(ws, needle: str, timeout: float = 60.0) -> str:
    return _wait_ws(ws, lambda m: m["type"] == "error" and needle in m["data"]["message"], timeout)["data"]["message"]


def _wait_state(ws, state: str, timeout: float) -> float:
    t = time.time()
    _wait_ws(ws, lambda m: m["type"] == "status" and m["data"]["mission_state"] == state, timeout)
    return time.time() - t


def _record(tmp_path, name: str, t: dict) -> None:
    print(f"\nSITL {name} (speedup {SPEEDUP}, wall s):", {k: round(v, 1) for k, v in t.items()})
    try:
        p = tmp_path.parent / "sitl_timings.json"
        d = json.loads(p.read_text()) if p.exists() else {}
        d[name] = t
        p.write_text(json.dumps(d))
    except OSError:
        pass


def test_sitl_abort_mid_survey(tmp_path, sitl2):
    port, _tcp, _proc = sitl2
    t: dict = {}
    with _flying(tmp_path, port, t) as (client, ws, ctrl):
        _wait_ws(ws, lambda m: m["type"] == "telemetry" and (m["data"].get("mission_seq") or 0) >= 5, 180)
        t0 = time.time()
        ws.send_json({"type": "abort"})
        m = _wait_ws(ws, lambda m: m["type"] == "chat" and "ABORT" in m["data"]["text"], 30)
        assert "RTL" in m["data"]["text"], m
        t["abort_ack_s"] = time.time() - t0
        _wait_ws(ws, lambda m: m["type"] == "telemetry" and m["data"].get("mode") == "RTL", 30)
        t["rtl_mode_s"] = time.time() - t0
        t["rtl_to_landed_s"] = _wait_state(ws, "landed", 300)
        assert not ctrl.vehicle.telemetry().armed
        assert any(e[1] == "rtl" and "ABORT" in e[2]["reason"] for e in ctrl.events)
        meta = json.loads((tmp_path / ctrl.last_flight_dir / "flight.json").read_text())
        assert meta["summary"]["end_reason"] == "landed"
    _record(tmp_path, "abort_mid_survey", t)


def test_sitl_pilot_mode_change_app_stops_and_does_not_fight(tmp_path, sitl2):
    port, tcp, _proc = sitl2
    t: dict = {}
    hand = Hand(tcp)
    try:
        with _flying(tmp_path, port, t) as (client, ws, ctrl):
            _wait_ws(ws, lambda m: m["type"] == "telemetry" and (m["data"].get("mission_seq") or 0) >= 4, 180)
            t0 = time.time()
            hand.set_mode("LOITER")                           # the RC mode switch
            msg = _wait_err(ws, "PILOT CONTROL", 30)
            t["detect_s"] = time.time() - t0
            assert "LOITER" in msg and ctrl.state == "manual"
            n_events = len(ctrl.events)
            # Abort first: with no RC throttle the SITL sinks in LOITER and lands + disarms within ~20 s sim
            assert "pilot has taken over" in client.post("/api/abort").json()["data"]["text"]
            time.sleep(1)                                     # 1 s wall = 5 s sim at speedup 5: app must stay silent
            assert ctrl.vehicle.telemetry().mode == "LOITER", "app fought the pilot"
            assert not [e for e in list(ctrl.events)[n_events:] if e[1] == "rtl"]
            hand.set_mode("LAND")                             # pilot lands it
            t["land_s"] = _wait_state(ws, "landed", 300)
            assert ctrl.flight is None
    finally:
        hand.close()
    _record(tmp_path, "pilot_mode_change", t)


def test_sitl_battery_failsafe(tmp_path, sitl2):
    port, tcp, _proc = sitl2
    t: dict = {}
    hand = Hand(tcp)
    try:
        with _flying(tmp_path, port, t) as (client, ws, ctrl):
            hand.set_param("BATT_LOW_VOLT", 11.0)             # SITL only
            hand.set_param("BATT_FS_LOW_ACT", 2)              # RTL
            t0 = time.time()
            hand.set_param("SIM_BATT_VOLTAGE", 9.0)           # SITL only
            msg = _wait_err(ws, "attery", 120)
            t["failsafe_msg_s"] = time.time() - t0
            print("SITL battery message:", msg)
            _wait_ws(ws, lambda m: m["type"] == "telemetry" and m["data"].get("mode") == "RTL", 60)
            t["rtl_by_vehicle_s"] = time.time() - t0
            assert not [e for e in ctrl.events if e[1] == "rtl"], "the app must not duplicate the vehicle's own RTL"
            t["landed_s"] = _wait_state(ws, "landed", 300)
    finally:
        hand.close()
    _record(tmp_path, "battery_failsafe", t)


def test_sitl_gps_loss(tmp_path, sitl2):
    port, tcp, _proc = sitl2
    t: dict = {}
    hand = Hand(tcp)
    try:
        with _flying(tmp_path, port, t) as (client, ws, ctrl):
            t0 = time.time()
            hand.set_param("SIM_GPS1_ENABLE", 0)              # SITL only: GPS disappears
            msg = _wait_err(ws, "GPS fix LOST", 60)
            t["gps_alert_s"] = time.time() - t0
            assert "RC transmitter" in msg
            _wait_ws(ws, lambda m: m["type"] == "telemetry" and m["data"].get("mode") in ("LAND", "RTL", "ALT_HOLD", "STABILIZE",
                                                                                           "LOITER"), 90)
            t["vehicle_failsafe_s"] = time.time() - t0
            st = ctrl.vehicle.telemetry()
            print("SITL after GPS loss: mode", st.mode, "armed", st.armed, "state", ctrl.state)
            hand.set_param("SIM_GPS1_ENABLE", 1)              # let it recover so the run can finish
            end = time.time() + 240
            while time.time() < end and ctrl.state not in ("landed",):
                if ctrl.vehicle.telemetry().armed is False:
                    break
                time.sleep(1)
            t["end_state_s"] = time.time() - t0
            print("SITL GPS loss end:", ctrl.state, ctrl.vehicle.telemetry().mode)
    finally:
        hand.close()
    _record(tmp_path, "gps_loss", t)


def test_sitl_staged_go_flow(tmp_path, sitl2):
    """notes/go-flow.md against ArduCopter SITL: Confirm only stages (no mode change, not armed); a Mission Planner-style
    edit on a second link and a disabled fence each block GO; Abort while staged sends nothing; re-Confirm + GO flies;
    no informational chat in flight; Abort -> RTL -> landed -> one flight_summary (template: FakeLLM has no text chat)."""
    port, tcp, _proc = sitl2
    t: dict = {}
    hand = Hand(tcp)
    s, app, ctrl = _rig(tmp_path, port, vision=False, readiness_deep_every_s=3600.0)   # deep runs only on demand here
    try:
        with TestClient(app) as client, client.websocket_connect("/ws") as ws:
            _wait_ready(ctrl)
            ws.receive_json()
            r = client.post("/api/plan_area", json={"corners": CORNERS, "altitude_m": 20}).json()
            assert r["type"] == "plan" and r["data"]["safety"]["ok"], r
            pid = r["data"]["plan_id"]
            assert client.post("/api/legal/ack", json={"accepted": [True] * 4}).status_code == 200
            t0 = time.time()
            txt = client.post("/api/confirm", json={"plan_id": pid}).json()["data"]["text"]
            assert "STAGED" in txt and ctrl.state == "staged", txt
            t["confirm_s"] = time.time() - t0
            tel = ctrl.vehicle.telemetry()
            assert not tel.armed and tel.mode == "STABILIZE"
            t0 = time.time()
            rd = client.post("/api/readiness/refresh").json()["readiness"]
            t["readiness_deep_s"] = time.time() - t0
            assert rd["go_allowed"], rd
            # 1. the mission is edited on the vehicle (second link) -> GO refused, nothing armed
            hand.shift_mission_item(4, 3e-5)
            txt = client.post("/api/go", json={"plan_id": pid}).json()["message"]
            assert "GO refused" in txt and "Mission on vehicle" in txt, txt
            assert not ctrl.vehicle.telemetry().armed and ctrl.state == "staged"
            # 2. Abort while staged: nothing sent, back to planned; Confirm again re-uploads
            assert "discarded" in client.post("/api/abort").json()["data"]["text"]
            assert ctrl.state == "planned" and ctrl.vehicle.telemetry().mode == "STABILIZE"
            assert "STAGED" in client.post("/api/confirm", json={"plan_id": pid}).json()["data"]["text"]
            # 3. fence disabled on the vehicle -> GO refused; re-enabled -> GO flies
            hand.set_param("FENCE_ENABLE", 0)
            txt = client.post("/api/go", json={"plan_id": pid}).json()["message"]
            assert "GO refused" in txt and "Geofence" in txt, txt
            hand.set_param("FENCE_ENABLE", 1)
            t0 = time.time()
            ws.send_json({"type": "go", "plan_id": pid})      # (the refused GO replies above were broadcast to this WS too)
            m = _wait_ws(ws, lambda m: m["type"] == "chat" and m["data"]["text"].startswith("GO")
                         and "refused" not in m["data"]["text"], 90)
            assert "mission started" in m["data"]["text"], m
            t["go_to_started_s"] = time.time() - t0
            seen: list = []
            _wait_ws(ws, lambda m: seen.append(m) is None and m["type"] == "telemetry"
                     and (m["data"].get("mission_seq") or 0) >= 4, 180)
            assert not [m for m in seen if m["type"] == "chat"], [m for m in seen if m["type"] == "chat"]
            ws.send_json({"type": "abort"})
            m = _wait_ws(ws, lambda m: m["type"] == "chat" and "ABORT" in m["data"]["text"], 30)
            assert "RTL" in m["data"]["text"], m
            t["rtl_to_landed_s"] = _wait_state(ws, "landed", 300)
            fs = _wait_ws(ws, lambda m: m["type"] == "flight_summary", 60)["data"]
            assert fs["source"] == "template" and "aborted by the operator" in fs["text"], fs
            assert (tmp_path / fs["flight_dir"] / "summary.md").is_file()
            assert client.get("/api/flights/last/summary").json()["text"] == fs["text"]
    finally:
        hand.close()
    _record(tmp_path, "staged_go_flow", t)
