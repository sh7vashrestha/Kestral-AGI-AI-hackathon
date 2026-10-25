"""End-to-end: phone WebSocket -> RealController -> mock ArduCopter (pymavlink, loopback UDP) -> files.

Vision uses the real Camera with a `file:` source (tests/assets/persons) and the real VisionPipeline/Matcher/
Fuser; only the YOLO detector and the match backend are injected fakes (no GPU in unit tests).
Run with `-s` to see the timings.
"""
from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from conftest import make_settings  # noqa: E402
from mock_vehicle import MODE_RTL, MockVehicle  # noqa: E402
from models import BBox, Detection, MatchResult  # noqa: E402
from plan_files import parse_wpl, validate_geojson  # noqa: E402

FIELD = (33.9371904, -84.5271293)
CORNERS = [{"lat": 33.93745, "lon": -84.52740}, {"lat": 33.93745, "lon": -84.52690},
           {"lat": 33.93790, "lon": -84.52690}, {"lat": 33.93790, "lon": -84.52740}]
ASSETS = ROOT / "tests" / "assets" / "persons"


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeLLM:
    """Planner/target LLM stand-in (planner.aupdate_target_from_message uses achat_json)."""

    def __init__(self) -> None:
        self.calls = 0

    async def achat_json(self, messages, model_cls, **kw):
        self.calls += 1
        if model_cls.__name__ == "LLMTarget":
            text = messages[-1]["content"] if isinstance(messages[-1]["content"], str) else ""
            desc = "person in a blue jacket" if "blue" in text else "person in a red jacket"
            return model_cls(ok=True, target_description=desc, target_attributes=[desc.split(" in a ")[-1]])
        raise AssertionError(f"unexpected LLM schema {model_cls.__name__}")

    def chat_json(self, messages, model_cls, **kw):
        raise AssertionError("sync LLM not expected")


class FakeDetector:
    """Always one person box in the middle of the frame."""

    def __init__(self) -> None:
        self.n = 0

    def detect(self, frame, ts=None, frame_id=0, telemetry=None, conf=None):
        self.n += 1
        h, w = frame.shape[:2]
        return [Detection(frame_id=frame_id, ts=ts or time.time(), c_yolo=0.9, telemetry=telemetry,
                          bbox=BBox(x1=w * 0.25, y1=h * 0.1, x2=w * 0.75, y2=h * 0.95))]

    def detect_roi(self, *a, **kw):
        return []


class FakeMatch:
    def prepare(self, target):
        pass

    def match(self, crop, target):
        return MatchResult(is_human=True, matches=True, match_confidence=0.9, reason="fake: red jacket")


class Rig:
    def __init__(self, tmp: Path, *, vision: bool = False, mock_kw: dict | None = None, **over: Any) -> None:
        import main
        from pipeline import VisionPipeline

        self.port = free_udp_port()
        base = dict(data_dir=tmp, mavlink_url=f"udpin:127.0.0.1:{self.port}", vehicle_mode="sim",
                    camera_source=f"file:{ASSETS}", thermal_source="none", capture_fps=4.0,
                    yolo_conf_threshold=0.5, min_frames=2, match_threshold=0.7, alert_min_interval_s=0.5,
                    heartbeat_loss_s=3.0)
        base.update(over)
        self.s = make_settings(**base)
        mk = dict(home=FIELD, home_amsl_m=300.0, time_scale=10.0)
        mk.update(mock_kw or {})
        self.mock = MockVehicle(f"udpout:127.0.0.1:{self.port}", **mk).start()
        self.llm = FakeLLM()
        self.detector = FakeDetector()
        holder: dict[str, Any] = {}
        pipeline = None
        if vision:
            pipeline = VisionPipeline(self.s, on_alert=lambda a: holder["c"].push_alert(a),
                                      telemetry_fn=lambda: holder["c"].vehicle.telemetry(),
                                      detector_factory=lambda s: self.detector, backend_factory=lambda s: FakeMatch())
        self.app, self.ctrl, self.hub = main.build_app(self.s, vision=vision, pipeline=pipeline, llm=self.llm)
        holder["c"] = self.ctrl
        self.client = TestClient(self.app)

    def __enter__(self) -> "Rig":
        self.client.__enter__()
        v = self.ctrl.vehicle
        assert self.wait(lambda: v.telemetry().gps_fix is not None and v.telemetry().battery_pct is not None
                         and (v.home_position() is not None or not self.mock.send_home), 15), "no vehicle"
        return self

    def __exit__(self, *exc) -> None:
        self.client.__exit__(*exc)
        self.mock.stop()

    @staticmethod
    def wait(pred: Callable[[], bool], timeout: float, poll: float = 0.05) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if pred():
                return True
            time.sleep(poll)
        return pred()

    def post(self, path: str, body: dict | None = None) -> dict:
        r = self.client.post(path, json=body or {})
        assert r.status_code == 200, r.text
        return r.json()

    def plan(self, **kw) -> dict:
        m = self.post("/api/plan_area", {"corners": CORNERS, "altitude_m": 20, **kw})
        return m

    def ack(self) -> None:
        self.post("/api/legal/ack", {"accepted": [True] * 4})

    def stage(self, pid: str) -> str:
        """Confirm = upload + read-back verify only (notes/go-flow.md); returns the reply text."""
        txt = self.post("/api/confirm", {"plan_id": pid})["data"]["text"]
        assert "STAGED" in txt and self.ctrl.state == "staged", txt
        return txt

    def fly(self, pid: str) -> str:
        """Confirm, then GO (arm + start). Returns the GO reply text."""
        self.stage(pid)
        for _ in range(3):      # some rigs use HEARTBEAT_LOSS_S = 1 s with a 1 Hz mock heartbeat: GO refuses on a flap
            self.wait(lambda: self.ctrl.vehicle.heartbeat_ok(), 10)
            txt = self.post("/api/go", {"plan_id": pid})["data"]["text"]
            if not txt.startswith("GO refused: MAVLink link"):
                break
        return txt


def wait_ws(ws, pred: Callable[[dict], bool], timeout: float, seen: list | None = None) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        m = ws.receive_json()
        if seen is not None:
            seen.append(m)
        if pred(m):
            return m
    raise AssertionError("timed out waiting for WS message")


# =============================================================================
def test_e2e_mock_full_flow(tmp_path):
    t = {}
    t0 = time.time()
    with Rig(tmp_path, vision=True) as rig, rig.client.websocket_connect("/ws") as ws:
        t["connected_s"] = time.time() - t0
        first = ws.receive_json()
        assert first["type"] == "status" and first["data"]["home"]["lat"] == pytest.approx(FIELD[0])
        seen: list = []
        # 1. plan the field area from the phone
        ws.send_json({"type": "plan_area", "corners": CORNERS, "altitude_m": 20, "target_text": "person in a red jacket"})
        plan = wait_ws(ws, lambda m: m["type"] in ("plan", "error"), 20, seen)
        assert plan["type"] == "plan", plan
        p = plan["data"]
        assert p["safety"]["ok"], p["safety"]
        assert p["preflight"]["ok"] and p["limits"]["effective_max_alt_m"] == 60
        assert p["plan"]["home"]["lat"] == pytest.approx(FIELD[0])
        assert p["target"]["description"] == "person in a red jacket"
        pid = p["plan_id"]
        t["plan_s"] = time.time() - t0
        # 2. (legal checklist removed at the operator's request: no ack step)
        assert rig.mock.mission == []
        rig.ack()
        # 3. confirm -> upload/verify -> staged (nothing armed); readiness checks; GO -> AUTO -> arm -> start
        ws.send_json({"type": "confirm", "plan_id": pid})
        m = wait_ws(ws, lambda m: m["type"] == "chat" and m["data"]["role"] == "system", 30, seen)
        assert "Mission uploaded and verified" in m["data"]["text"] and "STAGED" in m["data"]["text"], m
        assert not rig.mock.armed and len(rig.mock.mission) == len(p["plan"]["waypoints"]) + 4
        st = wait_ws(ws, lambda m: m["type"] == "status" and (m["data"].get("readiness") or {}).get("ts"), 30, seen)
        assert st["data"]["readiness"]["go_allowed"], st["data"]["readiness"]
        ws.send_json({"type": "go", "plan_id": pid})
        m = wait_ws(ws, lambda m: m["type"] == "chat" and m["data"]["text"].startswith("GO"), 30, seen)
        assert "mission started" in m["data"]["text"], m
        t["started_s"] = time.time() - t0
        assert rig.mock.armed
        # 4. mission progresses + vision alerts arrive over the WS
        levels = set()
        wait_ws(ws, lambda m: m["type"] == "alert" and levels.add(m["data"]["level"]) is None
                and {"person", "target_match"} <= levels, 30, seen)
        t["alerts_s"] = time.time() - t0
        assert rig.wait(lambda: (rig.mock.seq or 0) >= 4, 30), "mission did not progress"
        # 5. abort -> RTL -> landed
        ws.send_json({"type": "abort"})
        m = wait_ws(ws, lambda m: m["type"] == "chat" and "ABORT" in m["data"]["text"], 30, seen)
        assert "RTL" in m["data"]["text"]
        assert rig.mock.mode == MODE_RTL
        wait_ws(ws, lambda m: m["type"] == "status" and m["data"]["mission_state"] == "landed", 60, seen)
        t["landed_s"] = time.time() - t0
        flight_dir = tmp_path / rig.ctrl.last_flight_dir
        states = [e[2]["new"] for e in rig.ctrl.events if e[1] == "state"]
        assert states[:7] == ["planned", "uploading", "staged", "arming", "flying", "rtl", "landed"], states
        # 7. flight summary (FakeLLM has no text chat -> template), pushed once, kept in the snapshot
        fsm = wait_ws(ws, lambda m: m["type"] == "flight_summary", 30, seen)["data"]
        assert fsm["source"] == "template" and "aborted" in fsm["text"] and fsm["images"], fsm
        assert all(i["url"].startswith("/data/") and "%" in i["caption"] for i in fsm["images"])
        assert rig.client.get(fsm["images"][0]["url"]).status_code == 200
        assert rig.client.get("/api/flights/last/summary").json()["text"] == fsm["text"]
        assert (flight_dir / "summary.md").read_text().strip() == fsm["text"]
        assert {m["type"] for m in seen} >= {"telemetry", "status", "plan", "alert", "chat"}

    # 6. files
    pdir = tmp_path / "plans" / pid
    rows = parse_wpl((pdir / "mission.waypoints").read_text())
    assert [r["command"] for r in rows[:3]] == [16, 22, 178] and rows[-1]["command"] == 20
    assert rows[0]["alt"] == 300.0
    validate_geojson(json.loads((pdir / "mission.geojson").read_text()))
    assert json.loads((pdir / "preflight.json").read_text())["ok"]
    assert json.loads((pdir / "plan.json").read_text())["plan_id"] == pid
    for k in ("telemetry", "detections", "alerts", "matches"):
        lines = (flight_dir / f"{k}.jsonl").read_text().splitlines()
        assert lines, f"{k}.jsonl empty"
        json.loads(lines[-1])
    assert list((flight_dir / "frames").glob("*.jpg")) and list((flight_dir / "crops").glob("*.jpg"))
    parse_wpl((flight_dir / "mission.waypoints").read_text())
    meta = json.loads((flight_dir / "flight.json").read_text())
    assert meta["end_ts"] and meta["summary"]["end_reason"] == "landed" and meta["summary"]["plan_id"] == pid
    assert meta["summary"]["alerts"] >= 2 and meta["summary"]["distance_m"] > 10
    print("\nE2E mock timings (s):", {k: round(v, 2) for k, v in t.items()})


def test_download_endpoints(tmp_path):
    with Rig(tmp_path) as rig:
        p = rig.plan()["data"]
        pid = p["plan_id"]
        assert any(x["plan_id"] == pid for x in rig.client.get("/api/plans").json()["plans"])
        r = rig.client.get(f"/api/plans/{pid}/mission.waypoints")
        assert r.status_code == 200 and r.text.startswith("QGC WPL 110")
        assert "attachment" in r.headers["content-disposition"]
        assert rig.client.get(f"/api/plans/{pid}/mission.geojson").json()["type"] == "FeatureCollection"
        assert rig.client.get(f"/api/plans/{pid}").json()["plan_id"] == pid
        assert rig.client.get("/api/plans/../../etc/passwd").status_code == 404
        assert rig.client.get(f"/api/plans/{pid}/secret.txt").status_code == 404
        assert rig.client.get("/api/preflight").json()["ok"]
        leg = rig.client.get("/api/legal").json()
        assert leg["items"] == [] and not leg["acked"]
        rig.ack()
        assert rig.client.get("/api/legal").json()["acked"]
        lim = rig.client.get("/api/limits").json()
        assert lim["min_alt_m"] == 15 and lim["ceiling_source"] == "MAX_ALT_M"


def test_home_move_recalculates_plan(tmp_path):
    with Rig(tmp_path) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        p1 = rig.plan()["data"]
        rig.mock.move_home(12.0, 0.0)
        m = wait_ws(ws, lambda m: m["type"] == "chat" and "plan recalculated" in m["data"]["text"], 10)
        assert "Home moved 12 m" in m["data"]["text"]
        p2 = rig.ctrl.pending
        assert p2 is not None and p2.plan_id != p1["plan_id"]
        assert abs(p2.plan.home.lat - p1["plan"]["home"]["lat"] - 12 / 111_320) < 2e-6
        assert rig.ctrl.snapshot()["home"]["version"] == 2
        rig.ack()
        r = rig.post("/api/confirm", {"plan_id": p1["plan_id"]})
        assert "expired" in r["data"]["text"]
        # the drone sits 12 m from the new home: ArduPilot would reset home at arming -> refuse
        r = rig.post("/api/confirm", {"plan_id": p2.plan_id})
        assert "from its Pixhawk home" in r["data"]["text"]
        assert rig.mock.mission == [] and not rig.mock.armed


def test_confirm_rechecks_home_altitude(tmp_path):
    with Rig(tmp_path) as rig:
        rig.plan()
        rig.ack()
        p = rig.ctrl.pending
        p.home_amsl_m = (p.home_amsl_m or 0) - 5.0      # as if computed against an older home
        r = rig.post("/api/confirm", {"plan_id": p.plan_id})
        assert "Home moved" in r["data"]["text"] and "recalculated" in r["data"]["text"]
        assert rig.mock.mission == []
        assert rig.wait(lambda: rig.ctrl.pending is not None and rig.ctrl.pending.plan_id != p.plan_id, 10)


def test_home_change_in_flight_is_logged_and_alerted(tmp_path):
    with Rig(tmp_path) as rig, rig.client.websocket_connect("/ws") as ws:
        ws.receive_json()
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        assert "mission started" in rig.fly(pid)
        rig.ctrl._flight_meta["start_ts"] -= 60          # past the arming window
        rig.mock.move_home(0.0, 30.0)
        m = wait_ws(ws, lambda m: m["type"] == "error" and "home changed mid-flight" in m["data"]["message"], 10)
        assert "30 m" in m["data"]["message"]
        recs = [json.loads(x) for x in (rig.ctrl.flight.root / "telemetry.jsonl").read_text().splitlines()]
        assert any(r.get("event") == "home_changed" and r["shift_m"] > 29 for r in recs)
        rig.post("/api/abort")


def test_planning_refused_without_home(tmp_path):
    with Rig(tmp_path, mock_kw={"gps_fix": 1}) as rig:
        r = rig.plan()
        assert r["type"] == "error" and "3D GPS fix" in r["data"]["message"]
    with Rig(tmp_path / "b") as rig:
        pass
    port_rig = Rig.__new__(Rig)  # noqa: F841 (documentation: the 2nd case below uses a mock that never sends home)
    rig = Rig(tmp_path / "c")
    rig.mock.send_home = False
    rig.ctrl.vehicle._home_fix = None
    with rig:
        rig.ctrl.vehicle._home_fix = None
        r = rig.plan()
        assert r["type"] == "error" and "Waiting for Pixhawk home position" in r["data"]["message"]


def test_readonly_controller_never_uploads(tmp_path):
    with Rig(tmp_path, vehicle_mode="readonly") as rig:
        p = rig.plan()["data"]
        assert p["safety"]["ok"] and any("Read-only" in w for w in p["safety"]["warnings"])
        rig.ack()
        r = rig.post("/api/confirm", {"plan_id": p["plan_id"]})["data"]["text"]
        assert r.startswith("Read-only mode: plan saved, not uploaded")
        a = rig.post("/api/abort")["data"]["text"]
        assert "RC transmitter" in a
        assert rig.mock.mission == [] and not rig.mock.armed and rig.mock.mode == 0
        assert (tmp_path / "plans" / p["plan_id"] / "mission.waypoints").is_file()
        bad = {"MISSION_COUNT", "MISSION_ITEM_INT", "PARAM_SET", "SET_MODE"}
        assert not bad & set(rig.mock.received_types)


def test_link_loss_in_flight_commands_rtl(tmp_path):
    with Rig(tmp_path, heartbeat_loss_s=1.0, mavlink_timeout_s=1.0, mock_kw={"time_scale": 2.0}) as rig:
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        assert "mission started" in rig.fly(pid)
        rig.mock.mute = True
        assert rig.wait(lambda: any(e[1] == "link_lost" for e in rig.ctrl.events), 10)
        assert rig.wait(lambda: rig.mock.mode == MODE_RTL, 15), "RTL not received by the vehicle"
        rig.mock.mute = False
        assert rig.wait(lambda: any(e[1] == "link_restored" for e in rig.ctrl.events), 10)
        assert any(e[1] == "rtl" and "link" in e[2]["reason"] for e in rig.ctrl.events)


def test_update_target_mid_flight(tmp_path):
    with Rig(tmp_path, vision=True) as rig:
        pid = rig.plan(target_text="red jacket")["data"]["plan_id"]
        rig.ack()
        rig.fly(pid)
        r = rig.post("/api/target", {"text": "now look for someone in blue"})
        assert r["data"]["text"] == "Target updated: person in a blue jacket"
        assert rig.ctrl.target.description == "person in a blue jacket"
        assert rig.wait(lambda: rig.ctrl.pipeline.matcher is not None, 10)
        assert rig.ctrl.pipeline.matcher._target.description == "person in a blue jacket"
        rig.post("/api/abort")


def test_critical_task_failure_rtl(tmp_path):
    with Rig(tmp_path) as rig:
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        rig.fly(pid)
        orig = rig.ctrl._update_state
        calls = {"n": 0}

        def boom(tel):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("injected")
            return orig(tel)
        rig.ctrl._update_state = boom
        assert rig.wait(lambda: rig.mock.mode == MODE_RTL, 10)
        assert rig.ctrl.task_failures.get("telemetry") == 1
        assert rig.wait(lambda: calls["n"] > 3, 10), "telemetry task not restarted"


def test_upload_mode_controller(tmp_path):
    with Rig(tmp_path, vehicle_mode="upload", upload_fence_polygon=True) as rig:
        p = rig.plan()["data"]
        assert p["safety"]["ok"] and any("Upload mode" in w for w in p["safety"]["warnings"])
        rig.ack()
        r = rig.post("/api/confirm", {"plan_id": p["plan_id"]})["data"]["text"]
        assert r.startswith(f"Uploaded {len(p['plan']['waypoints']) + 4} mission items to the Pixhawk, read-back verify OK")
        assert "Fence polygon: 4 vertices, verify OK" in r and "will NOT take off" in r and "props off" in r
        assert len(rig.mock.mission) == len(p["plan"]["waypoints"]) + 4 and len(rig.mock.fence) == 4
        assert not rig.mock.armed and rig.mock.mode == 0 and rig.ctrl.state == "planned"
        assert not {176, 300, 400, 20} & set(rig.mock.commands_received)
        pdir = tmp_path / "plans" / p["plan_id"]
        assert json.loads((pdir / "upload.json").read_text())["verified"]
        assert len(parse_wpl((pdir / "readback.waypoints").read_text())) == len(rig.mock.mission)
        assert rig.client.get(f"/api/plans/{p['plan_id']}/readback.waypoints").status_code == 200
        assert "Upload-only mode" in rig.post("/api/abort")["data"]["text"]
        # armed -> no upload, no clear
        rig.mock.armed = True
        assert rig.wait(lambda: rig.ctrl.vehicle.telemetry().armed, 5)
        assert "ARMED" in rig.post("/api/confirm", {"plan_id": p["plan_id"]})["data"]["text"]
        assert "ARMED" in rig.post("/api/vehicle/clear_mission")["data"]["text"]
        rig.mock.armed = False
        assert rig.wait(lambda: not rig.ctrl.vehicle.telemetry().armed, 5)
        assert "Mission cleared" in rig.post("/api/vehicle/clear_mission")["data"]["text"]
        assert rig.mock.mission == []


def test_vehicle_check_endpoint(tmp_path):
    with Rig(tmp_path, vehicle_mode="readonly") as rig:
        time.sleep(1.5)
        rep = rig.client.get("/api/vehicle/check").json()
        assert rep["verdict"] == "READY", rep["reasons"]
        assert (tmp_path / "vehicle_check.json").is_file()
        assert rig.client.get("/api/vehicle/check?cached=1").json()["ts"] == rep["ts"]
        st = rig.client.get("/api/status").json()["data"]
        assert st["telemetry"]["landed_state"] == "ON_GROUND" and isinstance(st["statustext"], list)
