"""System check, camera open/backoff, STATUSTEXT dedupe, client_log, vehicle-hint/watchdog logic. No hardware."""
from __future__ import annotations

import time

import numpy as np
import pytest

import camera as camera_mod
import syscheck
from camera import Camera
from mission import StatusTextLog


# ---------------- camera ----------------
class FakeCap:
    def __init__(self, frames_after: int | None):
        self.n, self.frames_after = 0, frames_after
        self.released = False

    def isOpened(self):          # GStreamer with a broken argus: opened, but silent
        return True

    def read(self):
        self.n += 1
        if self.frames_after is not None and self.n > self.frames_after:
            return True, np.full((8, 8, 3), 100, np.uint8)
        return False, None

    def release(self):
        self.released = True


def fast(monkeypatch):
    monkeypatch.setattr(camera_mod, "OPEN_TIMEOUT_S", 0.3)
    monkeypatch.setattr(camera_mod, "MIN_BACKOFF_S", 0.05)
    monkeypatch.setattr(camera_mod, "MAX_BACKOFF_S", 0.2)


def test_camera_not_opened_without_frame_then_backoff_and_recovery(monkeypatch, settings_factory, caplog):
    fast(monkeypatch)
    caps = []

    def fake_open(self):
        cap = FakeCap(None if len(caps) < 3 else 0)       # first 3 attempts: no frames; then healthy
        caps.append(cap)
        return cap

    monkeypatch.setattr(Camera, "_open", fake_open)
    caplog.set_level("INFO", logger="camera")
    cam = Camera(settings_factory(camera_source="csi:0")).start()
    time.sleep(0.5)
    assert cam.state == "unavailable" and cam.latest()[0] is None
    assert "camera opened" not in "".join(r.message for r in caplog.records)      # never claimed "opened"
    assert "nvargus-daemon" in cam.hint()
    assert cam.wait_first(5.0)
    cam.stop()
    assert cam.state == "ok" and cam.reconnects >= 3
    errs = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errs) == 1                                                  # failure logged once, not every retry
    assert any("recovered" in r.message for r in caplog.records)
    assert all(c.released for c in caps[:3])


def test_camera_summary_every_interval(monkeypatch, settings_factory, caplog):
    fast(monkeypatch)
    monkeypatch.setattr(camera_mod, "SUMMARY_EVERY_S", 0.4)
    monkeypatch.setattr(Camera, "_open", lambda self: FakeCap(None))
    caplog.set_level("DEBUG", logger="camera")
    cam = Camera(settings_factory(camera_source="csi:0")).start()
    time.sleep(1.6)
    cam.stop()
    warns = [r for r in caplog.records if r.levelname == "WARNING" and "still unavailable" in r.message]
    assert 1 <= len(warns) <= 4 and "nvargus-daemon" in warns[0].message
    assert len([r for r in caplog.records if r.levelname == "ERROR"]) == 1


def test_pipeline_does_not_restart_when_camera_unavailable(settings_factory):
    from pipeline import VisionPipeline

    class Cam:
        def status(self):
            return {"state": "unavailable", "error": "x", "hint": "restart nvargus-daemon"}

    vp = VisionPipeline(settings_factory(), on_alert=lambda a: None, telemetry_fn=lambda: None)
    vp.state = "ready"
    vp.camera = Cam()
    vp._thread = type("T", (), {"is_alive": lambda self: True})()
    vp._last_frame_mono = time.monotonic() - 100
    assert vp.alive() is True
    assert "camera unavailable" in vp.status()["camera_problem"]


# ---------------- STATUSTEXT ----------------
def test_statustext_dedupe_counter_and_summary():
    L = StatusTextLog(summary_s=600)
    assert L.add("PreArm: RC not found", 4, 0.0) == "PreArm: RC not found"
    for t in (30, 60, 90):
        assert L.add("PreArm: RC not found", 4, t) is None
    assert L.add("PreArm: GPS", 4, 100.0) == "PreArm: GPS"
    e = L.entries()
    assert [x["text"] for x in e] == ["PreArm: RC not found", "PreArm: GPS"] and e[0]["count"] == 4
    assert e[0]["first_ts"] == 0.0 and e[0]["ts"] == 90.0
    line = L.add("PreArm: RC not found", 4, 700.0)
    assert line and "repeated 4x" in line
    assert L.entries()[-1]["text"] == "PreArm: RC not found"                 # repeat moves to the end (UI sees it as new)
    assert L.add("PreArm: RC not found", 4, 710.0) is None
    assert L.entries(since=705.0)[0]["ts"] == 710.0


# ---------------- client_log ----------------
def test_client_log_endpoint_and_rate_limit(tmp_path, settings_factory, caplog):
    from fastapi.testclient import TestClient
    from webapp.controller import FakeController
    from webapp.server import create_app
    s = settings_factory(data_dir=tmp_path)
    app = create_app(FakeController(tmp_path, background=False, settings=s), s)
    caplog.set_level("INFO")
    with TestClient(app) as c:
        r = c.post("/api/client_log", json={"level": "error", "message": "boom", "source": "app.js:1:2", "stack": "a\nb"})
        assert r.status_code == 200 and "boom" in caplog.text
        assert c.post("/api/client_log", json={"message": "x" * 5000}).status_code == 422
        codes = [c.post("/api/client_log", json={"message": f"m{i}"}).status_code for i in range(40)]
        assert 429 in codes and codes.count(200) <= 29
        assert c.get("/static/js/clientlog.js").status_code == 200


# ---------------- system check ----------------
def test_quick_check_runs_offline_and_renders(tmp_path, settings_factory, monkeypatch):
    s = settings_factory(data_dir=tmp_path, llm_base_url="http://127.0.0.1:9/v1")      # nothing listens on port 9
    monkeypatch.setattr(syscheck, "check_vehicle", lambda c: c.add("vehicle", "Pixhawk", "skip", "fake"))
    monkeypatch.setattr(syscheck, "check_host", lambda c: c.add("host", "RAM", "ok", "fake"))
    rep = syscheck.run_all(s, quick=True, deadline_s=30)
    by = {(c["group"], c["name"]): c for c in rep["checks"]}
    assert by[("llm", "llama-server /health")]["status"] == "fail" and rep["ready"] is False
    assert by[("camera", "RGB (IMX477)")]["status"] == "skip"
    assert by[("web", "web app serves")]["status"] == "ok"
    assert by[("files", "ALLOW_REAL_ARM")]["status"] == "ok"
    assert rep["summary"]["fail"] >= 1
    text = syscheck.render(rep, colour=True)
    assert "PROBLEMS FOUND" in text and "What to do" in text and "❌" in text
    assert syscheck.save_report(rep, s).exists()


def test_only_and_unknown_group(tmp_path, settings_factory):
    s = settings_factory(data_dir=tmp_path)
    rep = syscheck.run_all(s, only=["files"])
    assert {c["group"] for c in rep["checks"]} == {"files"}
    with pytest.raises(ValueError):
        syscheck.run_all(s, only=["nope"])


def test_vehicle_lines_from_report(settings_factory):
    c = syscheck.Ctx(settings_factory())
    rep = {"link": {"url": "x", "heartbeat_age_s": 0.1}, "vehicle": {"type": "QUADROTOR", "mode": "STABILIZE", "armed": False},
           "firmware": {"version": "4.7.2", "type": "official"}, "frame": {"class": "quad", "type": "X"},
           "gps": {"fix_type": 1, "fix": "NO FIX", "satellites": 0, "home": None}, "ekf": {"level": "OK", "available": True, "variances": {}},
           "sensors": {"available": True, "unhealthy": []}, "prearm": {"messages": [{"text": "PreArm: RC not found"}], "ack": "ACCEPTED"},
           "battery": {"voltage_v": 16.0, "pct": 80}, "rc": {"received": True, "chancount": 8, "rssi": 90},
           "preflight": {"checks": [{"param": "FENCE_ENABLE", "ok": False, "severity": "error", "value": 0, "message": "off", "requirement": "on"}]},
           "verdict": "NOT READY", "reasons": ["GPS"], "warnings": []}
    syscheck.vehicle_lines(c, rep, via="running app")
    st = {x["name"]: x["status"] for x in c.out}
    assert st["GPS"] == "fail" and st["Pre-arm checks"] == "fail" and st["param FENCE_ENABLE"] == "fail"
    assert st["Pixhawk heartbeat"] == "ok" and st["Vehicle verdict"] == "fail"
    assert all(x["via"] == "running app" for x in c.out)


def test_running_app_path_marks_devices(tmp_path, settings_factory):
    s = settings_factory(data_dir=tmp_path, camera_source="csi:0", thermal_source="v4l2:/dev/null")
    snap = {"vision": {"state": "ready", "camera_status": {"state": "unavailable", "error": "CameraError: no frame", "hint": "h"}, "frames": 0}}
    rep = syscheck.run_all(s, only=["camera", "thermal", "yolo", "clip"], app={"snapshot": snap, "models_info": {"detector": "Detector", "weights": "y.engine", "clip": "tensorrt", "backend": "ClipBackend"},
                                                                           "thermal_info": {"fps": 8.6, "min_c": 20, "median_c": 23, "max_c": 31, "stuck_px": 1}})
    st = {c["group"]: c for c in rep["checks"]}
    assert st["camera"]["status"] == "fail" and "nvargus-daemon" in st["camera"]["fix"]
    assert all(c["via"] == "running app" for c in rep["checks"] if c["group"] in ("camera", "thermal", "yolo", "clip"))
    assert st["yolo"]["status"] == "ok" and st["clip"]["status"] == "ok" and st["thermal"]["status"] == "ok"
    assert rep["mode"] == "via running app"


def test_busy_pids_finds_open_file(tmp_path):
    p = tmp_path / "dev"
    p.write_text("x")
    # our own pid is excluded by design; a child holding the file is found
    import subprocess
    ch = subprocess.Popen(["sleep", "2"], stdin=open(p))
    try:
        time.sleep(0.2)
        assert ch.pid in syscheck.busy_pids(str(p))
    finally:
        ch.kill()


def test_system_check_endpoint_in_app(tmp_path, settings_factory):
    import main
    from fastapi.testclient import TestClient
    from mission import create_backend
    s = settings_factory(data_dir=tmp_path, llm_base_url="http://127.0.0.1:9/v1")
    app, ctrl, _ = main.build_app(s, vision=False, vehicle=_FakeVehicle(), llm=object())
    with TestClient(app) as c:
        r = c.get("/api/system/check", params={"quick": "true", "only": "files,web"})
        assert r.status_code == 200 and r.json()["mode"] == "via running app"
        assert {x["group"] for x in r.json()["checks"]} == {"files", "web"}
        assert c.get("/api/system/check", params={"only": "bogus"}).status_code == 422


class _FakeVehicle:
    on_link_lost = on_link_restored = None
    def telemetry(self):
        import models
        return models.Telemetry()
    def home_position(self): return None
    def statustexts(self, since=0.0): return []
    def connect(self, timeout=10.0): raise RuntimeError("no vehicle")
    def close(self): pass
