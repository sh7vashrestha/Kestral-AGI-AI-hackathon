"""Upload -> review + readiness -> GO (notes/go-flow.md), against the pymavlink mock vehicle (loopback UDP only),
plus unit tests of the readiness builder and flight_summary. Nothing here talks to a real Pixhawk."""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import flight_summary  # noqa: E402
import models  # noqa: E402
import vehicle_check  # noqa: E402
from conftest import make_settings  # noqa: E402
from mission import CommandRejected, HomeFix  # noqa: E402
from mock_vehicle import MODE_RTL  # noqa: E402
from test_e2e import CORNERS, Rig, wait_ws  # noqa: E402

FLY_CMDS = {176, 300, 400, 20}            # set mode, mission start, arm, RTL


def post_bg(rig: Rig, path: str, body: dict) -> "cf.Future":
    ex = cf.ThreadPoolExecutor(1)
    fut = ex.submit(lambda: rig.client.post(path, json=body).json())
    ex.shutdown(wait=False)
    return fut


def staged(rig: Rig, **plan_kw) -> str:
    pid = rig.plan(**plan_kw)["data"]["plan_id"]
    rig.ack()
    rig.stage(pid)
    return pid


def go(rig: Rig, pid: str) -> str:
    return rig.post("/api/go", {"plan_id": pid})["data"]["text"]


def ready(rig: Rig) -> dict:
    return rig.post("/api/readiness/refresh")["readiness"]


def item(rd: dict, iid: str) -> dict:
    return next(i for i in rd["items"] if i["id"] == iid)


# =============================================================================== stage / GO basics
def test_confirm_only_stages_then_go_flies(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        assert not rig.mock.armed and not FLY_CMDS & set(rig.mock.commands_received)
        assert len(rig.mock.mission) > 4
        snap = rig.client.get("/api/status").json()["data"]
        assert snap["mission_state"] == "staged" and snap["staged"]["plan_id"] == pid
        assert snap["staged"]["expires_ts"] > time.time() + 800
        rd = ready(rig)
        assert rd["go_allowed"] and rd["go_reason"] == "" and not rd["running"], rd
        ids = [i["id"] for i in rd["items"]]
        assert ids[:2] == ["link", "disarmed"] and {"gps", "ekf", "prearm", "battery", "fence", "failsafe", "home",
                                                    "mission", "geofence_alt", "arm_guard", "vision", "legal"} <= set(ids)
        assert all(i["status"] in ("ok", "warn") for i in rd["items"]), rd
        assert item(rd, "vision")["blocking"] is False and item(rd, "mission")["status"] == "ok"
        assert all(set(i) == {"id", "label", "status", "detail", "blocking"} for i in rd["items"])
        r = rig.client.post("/api/go", json={"plan_id": pid}).json()
        assert r["type"] == "chat" and "mission started" in r["message"] and r["message"] == r["data"]["text"]
        assert rig.mock.armed and rig.ctrl.state == "flying" and rig.ctrl.staged is None
        assert rig.client.get("/api/status").json()["data"]["readiness"] is None
        rig.post("/api/abort")


def test_go_wrong_unknown_and_replaced_plan_id(tmp_path):
    with Rig(tmp_path) as rig:
        p1 = rig.plan()["data"]["plan_id"]
        rig.ack()
        assert "no mission is staged" in go(rig, p1)                 # planned, not staged
        rig.stage(p1)
        for bad in ("deadbeef", p1[::-1]):
            assert "not the plan uploaded" in go(rig, bad)
        assert rig.ctrl.state == "staged" and not rig.mock.armed
        rig.post("/api/abort")                                       # staged -> planned
        p2 = rig.plan(altitude_m=25)["data"]["plan_id"]
        rig.stage(p2)
        assert "not the plan uploaded" in go(rig, p1)                 # the old (replaced) plan
        assert not rig.mock.armed and 400 not in rig.mock.commands_received


def test_go_in_upload_mode_is_refused(tmp_path):
    with Rig(tmp_path, vehicle_mode="upload") as rig:
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        assert "read-back verify OK" in rig.post("/api/confirm", {"plan_id": pid})["data"]["text"]
        assert rig.ctrl.state == "planned"                           # upload mode never stages
        assert "GO is not available" in go(rig, pid)
        assert not rig.mock.armed and not FLY_CMDS & set(rig.mock.commands_received)


def test_go_twice_concurrently_arms_once(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        futs = [post_bg(rig, "/api/go", {"plan_id": pid}) for _ in range(3)]
        res = [f.result(60)["data"]["text"] for f in futs]
        assert sum("mission started" in r for r in res) == 1, res
        assert all(("mission started" in r) or ("already being processed" in r) or ("no mission is staged" in r)
                   for r in res), res
        assert rig.mock.commands_received.count(400) == 1
        rig.post("/api/abort")


def test_ws_go(tmp_path):
    with Rig(tmp_path) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        pid = staged(rig)
        ws.send_json({"type": "go"})
        assert "plan_id" in wait_ws(ws, lambda m: m["type"] == "error", 10)["data"]["message"]
        ws.send_json({"type": "go", "plan_id": pid})
        m = wait_ws(ws, lambda m: m["type"] == "chat" and m["data"]["text"].startswith("GO"), 30)
        assert "mission started" in m["data"]["text"]
        rig.post("/api/abort")


# =============================================================================== readiness blocks GO
def _mute(rig):
    rig.mock.mute = True
    assert rig.wait(lambda: not rig.ctrl.vehicle.heartbeat_ok(), 20)


def _legal_reset(rig):
    rig.ctrl.legal_ack_ts = None


BLOCKERS = {
    "link": (_mute, "MAVLink link"),
    "gps": (lambda rig: setattr(rig.mock, "gps_fix", 1), "GPS"),
    "prearm": (lambda rig: setattr(rig.mock, "prearm_failures", ["PreArm: Throttle below failsafe"]), "Throttle below failsafe"),
    "battery": (lambda rig: setattr(rig.mock, "battery_pct_override", 20), "Battery"),
    "fence": (lambda rig: rig.mock.params.update(FENCE_ENABLE=0.0), "Geofence"),
    "failsafe": (lambda rig: rig.mock.params.update(FS_THR_ENABLE=0.0), "Failsafes"),
    "geofence_alt": (lambda rig: rig.mock.params.update(FENCE_ALT_MAX=20.0), "Altitude vs ceiling"),
    # "legal": checklist removed (operator request, 2026-10-21)
}


@pytest.mark.parametrize("iid", sorted(BLOCKERS))
def test_each_blocking_readiness_item_refuses_go(tmp_path, iid):
    inject, needle = BLOCKERS[iid]
    with Rig(tmp_path, heartbeat_loss_s=1.0, mavlink_timeout_s=1.0) as rig:
        pid = staged(rig)
        inject(rig)
        time.sleep(0.5)                                     # SYS_STATUS / GPS_RAW_INT at 10 Hz
        txt = go(rig, pid)
        assert txt.startswith("GO refused") and needle in txt, txt
        rd = rig.ctrl.readiness
        assert rd is not None and not rd["go_allowed"] and item(rd, iid)["status"] in ("fail", "pending"), rd
        assert not rig.mock.armed and not FLY_CMDS & set(rig.mock.commands_received)
        assert rig.ctrl.state == "staged"                   # still staged: fix it and press GO again
        rig.mock.mute = False


def test_vision_down_only_warns(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        rd = ready(rig)
        assert item(rd, "vision")["status"] == "warn" and rd["go_allowed"]     # NullPipeline: vision disabled
        assert "mission started" in go(rig, pid)
        rig.post("/api/abort")


def test_low_battery_warn_does_not_block(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        rig.mock.battery_pct_override = 40
        time.sleep(0.5)
        rd = ready(rig)
        assert item(rd, "battery")["status"] == "warn" and "40%" in item(rd, "battery")["detail"] and rd["go_allowed"]
        assert "mission started" in go(rig, pid)
        rig.post("/api/abort")


def test_mission_edited_on_vehicle_after_upload(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        rig.mock.mission[3].lat += 3e-5                     # Mission Planner edit (~3 m)
        txt = go(rig, pid)
        assert "GO refused" in txt and "Mission on vehicle" in txt and "changed since upload" in txt, txt
        snap = rig.client.get("/api/status").json()["data"]
        assert snap["readback_info"]["source"] == "read" and not snap["readback_info"]["diff"]["summary"]["all_ok"]
        assert not rig.mock.armed and 400 not in rig.mock.commands_received
        # Abort, Confirm again (re-upload) and GO works
        rig.post("/api/abort")
        rig.stage(pid)
        assert "mission started" in go(rig, pid)
        rig.post("/api/abort")


def test_mission_changed_between_checks_and_arming_never_arms(tmp_path):
    """start_staged re-verifies itself: an edit after the readiness pass still blocks arming."""
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        v = rig.ctrl.vehicle
        orig = v.start_staged

        def edit_then_start(items):
            rig.mock.mission[4].alt += 3.0
            return orig(items)
        v.start_staged = edit_then_start
        txt = go(rig, pid)
        assert "no longer matches" in txt and "nothing armed" in txt, txt
        assert rig.ctrl.state == "planned" and not rig.mock.armed and 400 not in rig.mock.commands_received


def test_home_moved_after_upload_blocks_and_keeps_plan(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        rig.mock.move_home(12.0, 0.0)
        assert rig.wait(lambda: rig.ctrl.vehicle.home_position().version >= 2, 10)
        time.sleep(1.0)
        assert rig.ctrl.pending.plan_id == pid and rig.ctrl.state == "staged"     # not silently re-planned
        txt = go(rig, pid)
        assert "GO refused" in txt and "home moved 12 m" in txt, txt
        assert not rig.mock.armed
        assert "discarded" in rig.post("/api/abort")["data"]["text"]
        r = rig.post("/api/confirm", {"plan_id": pid})["data"]["text"]
        assert "Home moved" in r and "recalculated" in r, r


def test_vehicle_armed_externally_while_staged(tmp_path):
    with Rig(tmp_path) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        pid = staged(rig)
        n = len(rig.mock.commands_received)
        rig.mock.armed = True                               # RC transmitter / ground station
        m = wait_ws(ws, lambda m: m["type"] == "error" and "ARMED outside drone-brain" in m["data"]["message"], 10)
        assert "GO is cancelled" in m["data"]["message"]
        assert rig.wait(lambda: rig.ctrl.state == "manual", 5)
        snap = rig.client.get("/api/status").json()["data"]
        assert snap["readiness"] is None and snap["staged"] is None and rig.ctrl.staged is None
        assert "no mission is staged" in go(rig, pid)
        assert not FLY_CMDS & set(rig.mock.commands_received[n:])          # never fought the pilot
        rig.mock.armed = False
        assert rig.wait(lambda: rig.ctrl.state == "landed", 10)


def test_link_lost_while_staged_then_back(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=1.0, mavlink_timeout_s=1.0) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        pid = staged(rig)
        rig.mock.mute = True
        m = wait_ws(ws, lambda m: m["type"] == "error" and "link to the Pixhawk lost" in m["data"]["message"], 15)
        assert "GO is blocked" in m["data"]["message"]
        assert rig.ctrl.state == "staged" and not FLY_CMDS & set(rig.mock.commands_received)
        txt = go(rig, pid)
        assert "GO refused" in txt and "MAVLink link" in txt, txt
        rig.mock.mute = False
        assert rig.wait(lambda: rig.ctrl.vehicle.heartbeat_ok(), 20)
        time.sleep(0.5)
        assert "mission started" in go(rig, pid)
        rig.post("/api/abort")


def test_link_lost_during_arming_never_arms(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=1.0, mavlink_timeout_s=1.0) as rig:
        pid = staged(rig)
        v = rig.ctrl.vehicle
        orig = v.start_staged

        def mute_then_start(items):
            rig.mock.mute = True
            rig.wait(lambda: not v.heartbeat_ok(), 10)
            return orig(items)
        v.start_staged = mute_then_start
        txt = go(rig, pid)
        assert "NOT armed" in txt or "nothing armed" in txt, txt
        assert not rig.mock.armed and 400 not in rig.mock.commands_received
        assert rig.ctrl.state in ("staged", "planned")
        assert rig.wait(lambda: any(e[1] == "link_lost" for e in rig.ctrl.events), 10)
        rig.mock.mute = False
        assert rig.wait(lambda: v.heartbeat_ok(), 20)
        time.sleep(1.0)
        assert not rig.mock.armed and 400 not in rig.mock.commands_received


def test_arm_failure_at_go_keeps_plan_staged(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        v = rig.ctrl.vehicle
        orig_arm = v.arm

        def refuse():
            raise CommandRejected("arm: FAILED [PreArm: Motors: Check BRD_SAFETY_DEFLT]", 4)
        v.arm = refuse
        txt = go(rig, pid)
        assert "NOT armed" in txt and "still staged" in txt and "BRD_SAFETY" in txt, txt
        assert rig.ctrl.state == "staged" and not rig.mock.armed
        v.arm = orig_arm
        assert "mission started" in go(rig, pid)
        rig.post("/api/abort")


# =============================================================================== abort in the new states
def test_abort_while_staged_sends_nothing(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        n = len(rig.mock.commands_received)
        a = rig.post("/api/abort")["data"]["text"]
        assert "staged plan was discarded" in a and "nothing was sent" in a, a
        assert rig.ctrl.state == "planned" and rig.ctrl.pending.plan_id == pid and rig.ctrl.staged is None
        assert not FLY_CMDS & set(rig.mock.commands_received[n:]) and not rig.mock.armed
        assert len(rig.mock.mission) > 4                    # vehicle mission left as is
        assert "no mission is staged" in go(rig, pid)


def test_abort_during_arming_before_arm(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        v = rig.ctrl.vehicle
        orig = v.start_staged
        gate = threading.Event()

        def slow_start(items):
            gate.wait(20)                                   # GO accepted, checks done; Abort arrives now
            return orig(items)
        v.start_staged = slow_start
        fut = post_bg(rig, "/api/go", {"plan_id": pid})
        assert rig.wait(lambda: rig.ctrl.state == "arming", 20)
        a = rig.post("/api/abort")["data"]["text"]
        gate.set()
        assert "cancelled before arming" in a and "will not arm" in a, a
        txt = fut.result(30)["data"]["text"]
        assert "Not flying" in txt and "cancelled" in txt, txt
        assert rig.ctrl.state == "planned" and not rig.mock.armed and 400 not in rig.mock.commands_received


def test_abort_during_arming_after_arm_rtl(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        v = rig.ctrl.vehicle
        orig_arm = v.arm
        armed_evt, go_on = threading.Event(), threading.Event()

        def arm_then_wait():
            orig_arm()
            armed_evt.set()
            go_on.wait(20)
        v.arm = arm_then_wait
        fut = post_bg(rig, "/api/go", {"plan_id": pid})
        assert armed_evt.wait(30) and rig.wait(lambda: rig.ctrl.vehicle.telemetry().armed, 10)
        assert rig.ctrl.state == "arming"
        a = rig.post("/api/abort")["data"]["text"]
        go_on.set()
        assert "RTL" in a, a
        txt = fut.result(30)["data"]["text"]
        assert "RTL commanded" in txt, txt
        assert rig.mock.mode == MODE_RTL and 300 not in rig.mock.commands_received    # never started the mission
        assert rig.wait(lambda: rig.ctrl.state in ("rtl", "landed"), 20)


# =============================================================================== expiry / chat while staged
def test_staged_expiry_on_go_and_in_background(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        rig.ctrl.staged["uploaded_ts"] -= 10_000
        assert "expired" in go(rig, pid)
        assert rig.ctrl.state == "planned" and not rig.mock.armed
    with Rig(tmp_path / "bg", readiness_interval_s=0.3) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        pid = staged(rig)
        rig.ctrl.staged["uploaded_ts"] -= 10_000
        m = wait_ws(ws, lambda m: m["type"] == "error" and "expired" in m["data"]["message"], 10)
        assert "Confirm" in m["data"]["message"] and rig.ctrl.state == "planned"


def test_background_readiness_runs_while_staged(tmp_path):
    with Rig(tmp_path, readiness_interval_s=0.3, readiness_deep_every_s=1.0) as rig:
        staged(rig)
        assert rig.wait(lambda: sum(e[1] == "readiness" for e in rig.ctrl.events) >= 4, 15)
        assert any(e[1] == "readiness" and e[2]["deep"] for e in rig.ctrl.events)
        assert any(e[1] == "readiness" and not e[2]["deep"] for e in rig.ctrl.events)
        rig.mock.mission[3].lat += 3e-5                     # picked up by the periodic deep run
        assert rig.wait(lambda: rig.ctrl.readiness and item(rig.ctrl.readiness, "mission")["status"] == "fail", 15)
        rig.post("/api/abort")
        n = sum(e[1] == "readiness" for e in rig.ctrl.events)
        time.sleep(1.0)
        assert sum(e[1] == "readiness" for e in rig.ctrl.events) == n          # stops when not staged


def test_new_plan_while_staged_needs_abort(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        r = rig.post("/api/plan_area", {"corners": CORNERS, "altitude_m": 25})
        assert r["type"] == "error" and "waiting for GO" in r["data"]["message"] and "Abort" in r["data"]["message"]
        assert rig.ctrl.pending.plan_id == pid and rig.ctrl.state == "staged"
        assert "already uploaded and waiting for GO" in rig.post("/api/confirm", {"plan_id": pid})["data"]["text"]
        assert "staged" in rig.post("/api/vehicle/clear_mission")["data"]["text"]
        assert len(rig.mock.mission) > 4
        rig.post("/api/abort")
        assert rig.plan(altitude_m=25)["type"] == "plan"


# =============================================================================== in-flight text, summary
def test_in_flight_system_text_goes_to_flight_record(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        sent: list = []
        orig = rig.ctrl._broadcast

        async def spy(msg):
            sent.append(msg)
            await orig(msg)
        rig.ctrl._broadcast = spy
        asyncio.run_coroutine_threadsafe(rig.ctrl._say("staged note"), rig.ctrl._loop).result(5)
        assert any(m.type == "chat" and m.data["text"] == "staged note" for m in sent)
        assert "mission started" in go(rig, pid)
        asyncio.run_coroutine_threadsafe(rig.ctrl._say("MAVLink link restored."), rig.ctrl._loop).result(5)
        assert not any(m.type == "chat" and m.data.get("text") == "MAVLink link restored." for m in sent)
        recs = [json.loads(x) for x in (rig.ctrl.flight.root / "telemetry.jsonl").read_text().splitlines()]
        assert any(r.get("event") == "note" and r["text"] == "MAVLink link restored." for r in recs)
        rig.post("/api/abort")


class TextLLM:
    """achat_json for the target parser + achat for the summary (records what it was sent)."""

    def __init__(self, reply: str = "Searched the field for 1 minute. One person found.", delay: float = 0.0,
                 fail: bool = False) -> None:
        self.reply, self.delay, self.fail = reply, delay, fail
        self.sent: list = []

    async def achat_json(self, messages, model_cls, **kw):
        return model_cls(ok=True, target_description="person in a red jacket", target_attributes=["red jacket"])

    async def achat(self, messages, **kw):
        self.sent.append((messages, kw))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            from llm_client import LLMUnavailable
            raise LLMUnavailable("cannot reach llama-server")
        return self.reply


def test_summary_after_aborted_flight_by_gemma_and_persisted(tmp_path):
    with Rig(tmp_path, vision=True, mock_kw={"time_scale": 10.0}) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        llm = TextLLM()
        rig.ctrl.llm = llm
        pid = staged(rig)
        assert "mission started" in go(rig, pid)
        wait_ws(ws, lambda m: m["type"] == "alert", 30)
        assert rig.wait(lambda: (rig.mock.seq or 0) >= 3, 30)
        assert "RTL" in rig.post("/api/abort")["data"]["text"]
        fs = wait_ws(ws, lambda m: m["type"] == "flight_summary", 90)["data"]
        assert fs["source"] == "gemma" and fs["text"] == llm.reply and fs["images"], fs
        assert set(fs) >= {"text", "flight_dir", "source", "ts", "images"}
        assert set(fs["images"][0]) == {"url", "caption", "conf", "lat", "lon", "t"}
        msgs, kw = llm.sent[0]
        assert kw["extra"] == {"chat_template_kwargs": {"enable_thinking": False}}
        assert all(isinstance(m["content"], str) for m in msgs)               # text only: never pixels
        user = msgs[-1]["content"]
        assert "operator pressed ABORT" in user and "Alerts:" in user and "/data/" not in user
        flight_dir = tmp_path / fs["flight_dir"]
        assert (flight_dir / "summary.md").read_text().strip() == llm.reply
        assert rig.client.get("/api/flights/last/summary").json()["ts"] == fs["ts"]
        assert rig.client.get("/api/status").json()["data"]["last_flight_summary"]["text"] == llm.reply
    # restart: the phone reconnects after landing and still gets it
    with Rig(tmp_path) as rig2:
        assert rig2.client.get("/api/status").json()["data"]["last_flight_summary"]["text"] == llm.reply
        assert rig2.client.get("/api/flights/last/summary").json()["flight_dir"] == fs["flight_dir"]


def test_summary_llm_down_uses_template_and_never_delays_rtl(tmp_path):
    with Rig(tmp_path, mock_kw={"time_scale": 10.0}) as rig:
        rig.ctrl.llm = TextLLM(delay=30.0)
        rig.ctrl.s.flight_summary_timeout_s = 0.5
        pid = staged(rig)
        assert "mission started" in go(rig, pid)
        t0 = time.time()
        assert "RTL" in rig.post("/api/abort")["data"]["text"]
        assert time.time() - t0 < 5
        assert rig.wait(lambda: rig.ctrl.last_flight_summary is not None
                        and rig.ctrl.last_flight_summary["flight_dir"] == rig.ctrl.last_flight_dir, 60)
        fs = rig.ctrl.last_flight_summary
        assert fs["source"] == "template" and "aborted by the operator" in fs["text"] and fs["images"] == []


def test_failed_go_before_arming_writes_no_summary(tmp_path):
    with Rig(tmp_path) as rig:
        pid = staged(rig)
        rig.ctrl.vehicle.arm = lambda: (_ for _ in ()).throw(CommandRejected("arm: FAILED", 4))
        assert "NOT armed" in go(rig, pid)
        time.sleep(0.5)
        assert rig.ctrl.last_flight_summary is None
        assert not list((tmp_path / "flights").glob("*/summary.md"))


# =============================================================================== units: readiness builder
class FakeBackend:
    def __init__(self, **tel) -> None:
        base = dict(connected=True, mode="STABILIZE", armed=False, gps_fix=3, satellites=12, hdop=0.8, lat=47.0,
                    lon=8.0, battery_pct=90.0, voltage_v=16.2, heartbeat_age_s=0.3)
        base.update(tel)
        self.tel = models.Telemetry(**base)
        bits = 0x1 | 0x2 | 0x4 | 0x8 | 0x20 | 0x10000 | 0x10000000
        self.rawd = {"HEARTBEAT": {"type": 2},
                     "SYS_STATUS": {"onboard_control_sensors_present": bits, "onboard_control_sensors_enabled": bits,
                                    "onboard_control_sensors_health": bits},
                     "EKF_STATUS_REPORT": {"flags": 0x3F, "velocity_variance": 0.1, "pos_horiz_variance": 0.1,
                                           "pos_vert_variance": 0.1, "compass_variance": 0.1, "terrain_alt_variance": 0.0},
                     "RC_CHANNELS": {"chancount": 8, "rssi": 255}}
        self.params = {"RTL_ALT": 2000.0, "FENCE_ENABLE": 1.0, "FENCE_TYPE": 3.0, "FENCE_ALT_MAX": 60.0,
                       "FENCE_RADIUS": 300.0, "FENCE_ACTION": 1.0, "BATT_FS_LOW_ACT": 2.0, "FS_THR_ENABLE": 1.0,
                       "WPNAV_SPEED": 500.0}
        self.prearm: list[str] = []

    def telemetry(self):
        return self.tel

    def raw(self, t):
        return (time.time(), self.rawd[t]) if t in self.rawd else None

    def home_position(self):
        return HomeFix(47.0, 8.0, 500.0, time.time(), 1)

    def get_param(self, name, timeout=None):
        return self.params.get(name)

    def run_prearm_checks(self, wait_s=0.0):
        return {"ack": "FAILED" if self.prearm else "ACCEPTED", "messages": [{"text": t, "severity": 2} for t in self.prearm]}

    def statustexts(self, since=0.0):
        return []


def _rd(be, settings, extra=None):
    return vehicle_check.build_readiness(vehicle_check.run_vehicle_check(be, settings, read_frame=False), extra or {})


def _sys(be, health_clear):
    be.rawd["SYS_STATUS"]["onboard_control_sensors_health"] &= ~health_clear


UNIT_CASES = {
    "gps": lambda be: setattr(be, "tel", be.tel.model_copy(update={"gps_fix": 2})),
    "ekf": lambda be: be.rawd["EKF_STATUS_REPORT"].update(pos_horiz_variance=0.9),
    "prearm": lambda be: setattr(be, "prearm", ["PreArm: Compass not calibrated"]),
    "rc": lambda be: be.rawd.pop("RC_CHANNELS"),
    "battery": lambda be: setattr(be, "tel", be.tel.model_copy(update={"battery_pct": 25.0})),
    "fence": lambda be: be.params.update(FENCE_RADIUS=900.0),
    "failsafe": lambda be: be.params.update(BATT_FS_LOW_ACT=0.0),
    "geofence_alt": lambda be: be.params.update(FENCE_ALT_MAX=100.0),
}


def test_unit_all_ok(settings):
    rd = _rd(FakeBackend(), settings)
    assert rd["go_allowed"], rd
    assert [i["id"] for i in rd["items"]] == [x[0] for x in vehicle_check.READINESS_ITEMS]
    assert "16.2 V" in item(rd, "battery")["detail"] and "12 satellites" in item(rd, "gps")["detail"]


@pytest.mark.parametrize("iid", sorted(UNIT_CASES))
def test_unit_each_item_fails_alone(settings, iid):
    be = FakeBackend()
    UNIT_CASES[iid](be)
    rd = _rd(be, settings)
    failing = [i["id"] for i in rd["items"] if i["status"] == "fail"]
    assert failing == [iid], rd["items"]
    assert not rd["go_allowed"] and rd["go_reason"].startswith(item(rd, iid)["label"])


def test_unit_no_link_makes_vehicle_items_pending(settings):
    rd = _rd(FakeBackend(connected=False), settings, {"legal": ("ok", "x")})
    assert item(rd, "link")["status"] == "fail" and item(rd, "gps")["status"] == "pending"
    assert item(rd, "mission")["detail"] == "no MAVLink link" and rd["go_reason"].startswith("MAVLink link")


def test_unit_extra_items_and_non_blocking(settings):
    rd = _rd(FakeBackend(), settings, {"mission": ("pending", "not checked yet"), "vision": ("fail", "camera dead")})
    assert item(rd, "vision")["status"] == "warn" and not rd["go_allowed"] and "not checked" in rd["go_reason"]
    rd = _rd(FakeBackend(armed=True), settings, {"disarmed": ("fail", "vehicle is ARMED")})
    assert item(rd, "disarmed")["status"] == "fail"


def test_unit_light_run_reuses_params_and_skips_prearm(settings):
    be = FakeBackend()
    full = vehicle_check.run_vehicle_check(be, settings, read_frame=False)
    calls = []
    be.get_param = lambda n, timeout=None: calls.append(n)
    be.run_prearm_checks = lambda wait_s=0: calls.append("401")
    pre = models.PreflightResult.model_validate(full["preflight"])
    light = vehicle_check.run_vehicle_check(be, settings, run_prearm=False, read_frame=False, preflight=pre)
    assert calls == [] and light["verdict"] == "READY" and light["prearm"]["ack"] == "skipped"


def test_unit_vehicle_check_report_unchanged_for_callers(settings):
    be = FakeBackend()
    be.prearm = ["PreArm: Compass not calibrated"]
    rep = vehicle_check.run_vehicle_check(be, settings, read_frame=False)
    assert "pre-arm: 1 failure message(s)" in rep["reasons"]                 # same reason text as before
    assert rep["findings"]["prearm"]["fail"] == ["PreArm: Compass not calibrated"]
    json.dumps(rep)                                                           # still JSON-serialisable


class StubVehicle:
    """Never opens a port: RealController with this stub only answers the readiness extras."""
    on_link_lost = on_link_restored = None

    def __init__(self, plan_home) -> None:
        self.h = plan_home

    def telemetry(self):
        return models.Telemetry(connected=True, armed=False, lat=self.h.lat, lon=self.h.lon)

    def home_position(self):
        return HomeFix(self.h.lat, self.h.lon, 0.0, time.time(), 1)


def test_arm_guard_item_on_non_simulated_url(tmp_path):
    from controller_real import RealController
    from webapp.controller import fake_plan
    p = fake_plan()
    for url, ok in (("serial:/dev/ttyACM0", False), ("udpin:192.168.1.5:14550", False), ("udpin:127.0.0.1:14550", True)):
        s = make_settings(data_dir=tmp_path, mavlink_url=url, vehicle_mode="full")
        assert not s.real_arm_permitted
        ctrl = RealController(s, StubVehicle(p.plan.home), llm=object(), manage_connection=False)
        ex = ctrl._readiness_extra({"plan_id": p.plan_id, "items": [], "uploaded_ts": time.time(), "home_version": 1}, p)
        st, detail = ex["arm_guard"]
        assert (st == "ok") == ok, (url, ex["arm_guard"])
        if not ok:
            assert "a human must set ALLOW_REAL_ARM=true in .env" in detail
        assert ex["legal"][0] == "ok" and ex["home"][0] == "ok" and ex["mission"][0] == "pending"


# =============================================================================== units: flight_summary
def _flight(tmp_path: Path, n_tracks: int = 8, crops: int = 7) -> Path:
    d = tmp_path / "flights" / "20261020T120000Z"
    (d / "crops").mkdir(parents=True)
    start = 1_000_000.0
    meta = {"start_ts": start, "end_ts": start + 300, "plan": {"waypoints": [{}] * 10, "altitude_m": 20, "speed_ms": 5,
                                                                "path_length_m": 900},
            "targets": [{"description": "person in a red jacket"}],
            "summary": {"start_ts": start, "duration_s": 300.0, "distance_m": 850.0, "max_alt_m": 20.4, "alerts": 2 * n_tracks,
                        "link_losses": 1, "end_reason": "landed", "aborted": True, "rtl_reasons": ["operator ABORT"]}}
    (d / "flight.json").write_text(json.dumps(meta))
    lines = []
    for t in range(n_tracks):
        for k in range(2):
            url = f"/data/flights/{d.name}/crops/t{t}_f{k}.jpg"
            if t < crops:
                (d / "crops" / f"t{t}_f{k}.jpg").write_bytes(b"\xff\xd8\xff")
            lines.append(json.dumps({"level": "target_match" if t == 0 else "person", "track_id": t, "ts": start + 10 * t + k,
                                     "c_yolo": 0.8, "c_avg": 0.5 + 0.05 * t + 0.01 * k, "image_url": url,
                                     "lat": 33.9 + t * 1e-4, "lon": -84.5, "target_description": "person in a red jacket",
                                     "reason": "red jacket visible"}))
    (d / "alerts.jsonl").write_text("\n".join(lines) + "\n")
    tel = [{"mode": m, "armed": True} for m in ("AUTO", "AUTO", "RTL", "RTL", "LAND")]
    tel.insert(1, {"event": "link_lost", "ts": start + 50})
    (d / "telemetry.jsonl").write_text("\n".join(json.dumps(x) for x in tel) + "\n")
    return d


def test_summary_images_best_per_track_and_captions(tmp_path):
    d = _flight(tmp_path)
    res = asyncio.run(flight_summary.summarize(d, None, data_dir=tmp_path))
    imgs = res["images"]
    assert len(imgs) == 6 and res["source"] == "template" and res["flight_dir"] == f"flights/{d.name}"
    confs = [i["conf"] for i in imgs]
    assert confs == sorted(confs, reverse=True)
    assert len({i["url"].split("/")[-1].split("_")[0] for i in imgs}) == 6               # one per track
    assert "t7_" not in json.dumps(imgs)                                                 # missing crop skipped
    assert imgs[0]["url"].endswith("t6_f1.jpg") and imgs[0]["caption"].startswith("Person 81%")
    assert "33.90060, -84.50000" in imgs[0]["caption"] and "T+1:01" in imgs[0]["caption"] and imgs[0]["t"] == 61.0
    assert (d / "summary.md").is_file() and json.loads((d / "summary.json").read_text())["images"] == imgs
    assert flight_summary.load_last(tmp_path)["text"] == res["text"]


def test_summary_heat_ranks_last():
    alerts = [{"level": "heat", "track_id": 1_000_001, "c_final": 1.0}, {"level": "person", "track_id": 1, "c_avg": 0.5},
              {"level": "target_match", "track_id": 2, "c_final": 0.6}]
    assert [a["track_id"] for a in flight_summary.best_per_track(alerts)] == [2, 1, 1_000_001]


def test_summary_template_and_digest_text(tmp_path):
    d = _flight(tmp_path)
    dg = flight_summary.digest(d)
    txt = flight_summary.template_text(dg)
    assert "person in a red jacket" in txt and "5:00" in txt and "aborted by the operator" in txt
    assert "AUTO -> RTL -> LAND" in txt and "link lost 1" in txt
    gem = flight_summary.digest_text(dg)
    assert "operator pressed ABORT" in gem and "Target match (track 0)" in gem and "/data/" not in gem
    assert gem.count("\n- ") == 8


@pytest.mark.parametrize("llm", [TextLLM(fail=True), TextLLM(delay=5.0), TextLLM(reply="   "), object()])
def test_summary_llm_failures_fall_back_to_template(tmp_path, llm):
    d = _flight(tmp_path)
    t0 = time.time()
    res = asyncio.run(flight_summary.summarize(d, llm, data_dir=tmp_path, timeout_s=0.3))
    assert res["source"] == "template" and time.time() - t0 < 3
    assert res["text"] == flight_summary.template_text(flight_summary.digest(d))


def test_summary_gemma_is_clipped_and_sync_client_ok(tmp_path):
    d = _flight(tmp_path)

    class SyncLLM:
        def chat(self, messages, **kw):
            assert kw["extra"] == flight_summary.NO_THINK
            return "**Report** " + "word " * 400

    res = asyncio.run(flight_summary.summarize(d, SyncLLM(), data_dir=tmp_path))
    assert res["source"] == "gemma" and "**" not in res["text"] and len(res["text"].split()) <= flight_summary.MAX_WORDS + 1


def test_load_last_none_and_md_only(tmp_path):
    assert flight_summary.load_last(tmp_path) is None
    d = tmp_path / "flights" / "20261020T000000Z"
    d.mkdir(parents=True)
    (d / "summary.md").write_text("old text\n")
    assert flight_summary.load_last(tmp_path)["text"] == "old text"
