"""Camera retry on every failure path, bounded shutdown on SIGTERM, live orientation API."""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import camera as camera_mod
from camera import Camera, CameraError
from conftest import make_settings
from webapp import stream

ROOT = Path(__file__).resolve().parents[1]


class FakeCap:
    def __init__(self, mode, n_good=3):
        self.mode, self.n, self.n_good, self.released = mode, 0, n_good, False

    def isOpened(self):  # noqa: N802
        return True

    def read(self):
        if self.mode == "nothing":                       # opened but never a frame ("opened csi:0 but no frame")
            time.sleep(0.01)
            return False, None
        self.n += 1
        if self.mode == "good_then_raise" and self.n > self.n_good:
            raise RuntimeError("boom")
        if self.mode == "good_then_stall" and self.n > self.n_good:
            time.sleep(0.01)
            return False, None
        time.sleep(0.005)
        return True, np.full((24, 32, 3), self.n % 255, np.uint8)

    def release(self):
        self.released = True


def test_retry_covers_every_failure_path(monkeypatch, settings_factory):
    monkeypatch.setattr(camera_mod, "MIN_BACKOFF_S", 0.02)
    monkeypatch.setattr(camera_mod, "MAX_BACKOFF_S", 0.05)
    monkeypatch.setattr(camera_mod, "OPEN_TIMEOUT_S", 0.3)
    monkeypatch.setattr(camera_mod, "READ_STALL_S", 0.3)
    caps = []
    script = ["open_fail", "nothing", "good_then_raise", "good_then_stall", "good"]

    def fake_open(self):
        mode = script[min(len(caps), len(script) - 1)]
        if mode == "open_fail":
            caps.append(None)
            raise CameraError("cannot open csi:0")
        c = FakeCap(mode)
        caps.append(c)
        return c

    monkeypatch.setattr(Camera, "_open", fake_open)
    cam = Camera(settings_factory(camera_source="csi:0")).start()
    try:
        end = time.monotonic() + 15
        while time.monotonic() < end and not (len(caps) >= 5 and cam.state == "ok" and cam.latest()[2] > 8):
            time.sleep(0.05)
        st = cam.status()
        assert len(caps) >= 5, "did not retry through all failure paths"
        assert st["state"] == "ok" and cam.latest()[0] is not None
        assert st["reconnects"] == 4 and st["attempts"] >= 5
        assert all(c.released for c in caps[1:4]), "failed captures must be released"
        assert st["error"] is None
    finally:
        cam.stop()


def test_stop_returns_quickly_while_waiting_to_retry(monkeypatch, settings_factory):
    monkeypatch.setattr(Camera, "_open", lambda self: (_ for _ in ()).throw(CameraError("nope")))
    cam = Camera(settings_factory(camera_source="csi:0")).start()
    time.sleep(0.3)
    t0 = time.monotonic()
    cam.stop()
    assert time.monotonic() - t0 < 1.0


def test_file_source_orientation_applies_at_capture(tmp_path, settings_factory):
    img = np.zeros((48, 64, 3), np.uint8)
    img[:20, :30] = (255, 0, 0)                                    # blue block top-left
    cv2.imwrite(str(tmp_path / "a.png"), img)
    for rot, flip in ((0, "none"), (90, "none"), (180, "none"), (0, "horizontal")):
        from orient import apply
        cam = Camera(settings_factory(camera_source=f"file:{tmp_path}", camera_rotate=rot, camera_flip=flip, capture_fps=20.0)).start(wait_s=5)
        try:
            f, _, _ = cam.latest()
            want = apply(img, rot, flip)
            assert f.shape == want.shape and np.abs(f.astype(int) - want.astype(int)).mean() < 2
        finally:
            cam.stop()


def test_live_set_orient_non_csi_applies_next_frame(tmp_path, settings_factory):
    img = np.zeros((48, 64, 3), np.uint8)
    img[:20, :30] = 255
    cv2.imwrite(str(tmp_path / "a.png"), img)
    cam = Camera(settings_factory(camera_source=f"file:{tmp_path}", camera_rotate=0, capture_fps=20.0)).start(wait_s=5)
    try:
        cam.set_orient(90, "none")
        end = time.monotonic() + 3
        while time.monotonic() < end and (cam.latest()[0] is None or cam.latest()[0].shape != (64, 48, 3)):
            time.sleep(0.05)
        assert cam.latest()[0].shape == (64, 48, 3)
    finally:
        cam.stop()


# ---- SIGTERM ------------------------------------------------------------------------------------------------
def _free_port():
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def _wait_port(port, proc, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise AssertionError(f"live_view exited early ({proc.returncode})")
        try:
            socket.create_connection(("127.0.0.1", port), 0.3).close()
            return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("live_view did not start")


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_live_view_exits_under_3s_on_signal_with_open_stream(tmp_path, sig):
    rgb = tmp_path / "rgb"; rgb.mkdir()
    cv2.imwrite(str(rgb / "a.png"), np.full((48, 64, 3), 90, np.uint8))
    th = tmp_path / "th"; th.mkdir()
    np.save(th / "a.npy", np.full((120, 160), 21.0, np.float32))
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "live_view.py"), "--port", str(port), "--rgb", f"file:{rgb}",
                             "--thermal", f"file:{th}", "--host", "127.0.0.1"], cwd=ROOT, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, env={**os.environ, "DATA_DIR": str(tmp_path / "data")})
    try:
        _wait_port(port, proc)
        # hold an endless MJPEG stream open: this is what used to block uvicorn's graceful shutdown
        for _ in range(60):                       # the file source needs a moment for its first frame
            so = socket.create_connection(("127.0.0.1", port), 3)
            so.sendall(b"GET /stream/rgb.mjpg?fps=10 HTTP/1.1\r\nHost: x\r\n\r\n")
            so.settimeout(5)
            if b" 200 " in so.recv(200):
                break
            so.close()
            time.sleep(0.2)
        else:
            raise AssertionError("no 200 from the stream")
        t0 = time.monotonic()
        proc.send_signal(sig)
        rc = proc.wait(timeout=6)
        dt = time.monotonic() - t0
        assert dt < 3.0, f"took {dt:.1f}s to exit"
        assert rc in (0, 130)
    finally:
        if proc.poll() is None:
            proc.kill()


# ---- orientation API ----------------------------------------------------------------------------------------
def make_app(tmp_path, **kw):
    s = make_settings(data_dir=tmp_path, **kw)
    src = stream.SyntheticSource(size=(640, 360))
    app = FastAPI()
    env = tmp_path / ".env"
    env.write_text("CAMERA_ROTATE=0\nCAMERA_FLIP=none\nTHERMAL_ROTATE=0\nTHERMAL_FLIP=none\n")
    r = stream.make_router(src, s, default_fps=10, data_dir=tmp_path, env_path=env)
    app.include_router(r)
    return TestClient(app), s, env, src


def test_orientation_preview_and_stats(tmp_path):
    c, s, env, src = make_app(tmp_path)
    o = c.get("/api/live/stats").json()["orientation"]
    assert o["editable"] and o["camera"]["rotate"] == 0 and not o["camera"]["dirty"]
    o = c.post("/api/live/orientation", json={"target": "camera", "rotate": 90}).json()
    assert o["camera"]["rotate"] == 90 and o["camera"]["dirty"] and o["camera"]["out_size"] == [1080, 1920]
    assert s.camera_rotate == 90 and s.eff_hfov_deg == s.camera_vfov_deg          # optics follow the preview
    f, _ = src.rgb()
    assert f.shape[:2] == (640, 360)                                                 # rotated frame is served
    assert c.post("/api/live/orientation", json={"target": "camera", "rotate": 45}).status_code == 400
    assert c.post("/api/live/orientation", json={"target": "lens", "rotate": 0}).status_code == 400
    st = c.get("/api/live/stats").json()
    assert st["rgb"]["w"] == 360 and st["rgb"]["h"] == 640


def test_overlay_follows_preview_orientation(tmp_path):
    c, s, env, src = make_app(tmp_path)
    before = src.rgb()[0].shape
    c.post("/api/live/orientation", json={"target": "camera", "rotate": 180})
    r = c.get("/api/live/stats").json()
    assert r["orientation"]["camera"]["rotate"] == 180 and src.rgb()[0].shape == before


def test_save_writes_only_orientation_keys(tmp_path):
    c, s, env, src = make_app(tmp_path)
    original = "# secret comment\nFOO=bar\nCAMERA_ROTATE=0\nALLOW_REAL_ARM=false\nWEB_PORT=8000"   # THERMAL_* keys absent: appended
    env.write_text(original)
    c.post("/api/live/orientation", json={"target": "camera", "rotate": 270})
    c.post("/api/live/orientation", json={"target": "camera", "flip": "horizontal"})
    c.post("/api/live/orientation", json={"target": "thermal", "rotate": 90})
    d = c.post("/api/live/orientation/save").json()
    assert d["restart_required"] is True
    txt = env.read_text().splitlines()
    assert txt[:2] == ["# secret comment", "FOO=bar"] and "ALLOW_REAL_ARM=false" in txt and "WEB_PORT=8000" in txt
    vals = dict(l.split("=", 1) for l in txt if "=" in l and not l.startswith("#"))
    assert vals["CAMERA_ROTATE"] == "270" and vals["CAMERA_FLIP"] == "horizontal"
    assert vals["THERMAL_ROTATE"] == "90" and vals["THERMAL_FLIP"] == "none"
    assert txt.count("CAMERA_ROTATE=270") == 1
    o = c.get("/api/live/stats").json()["orientation"]
    assert not o["camera"]["dirty"] and not o["thermal"]["dirty"]
    from config import Settings
    s2 = Settings(_env_file=env)
    assert (s2.camera_rotate, s2.camera_flip, s2.thermal_rotate) == (270, "horizontal", 90)


def test_pipeline_mode_orientation_is_read_only(tmp_path):
    s = make_settings(data_dir=tmp_path)
    class Src(stream.FrameSource):
        pass
    app = FastAPI()
    app.include_router(stream.make_router(Src(), s, data_dir=tmp_path, env_path=tmp_path / ".env"))
    c = TestClient(app)
    assert c.post("/api/live/orientation", json={"target": "camera", "rotate": 90}).status_code == 409
    assert c.post("/api/live/orientation/save").status_code == 409
    assert c.get("/api/live/stats").json()["orientation"]["editable"] is False
