"""webapp/stream.py: MJPEG format, encode-only-with-clients, fps limit, thermal maths, snapshot, unavailable feeds."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conftest import make_settings
from webapp import stream


class Fake(stream.FrameSource):
    """Frames that change on every call; counts how often they are fetched."""

    def __init__(self, rgb=True, thermal=True, states=None):
        self.has_rgb, self.has_th = rgb, thermal
        self.calls = 0
        self.states = states
        self.n = 0
        self.temp = np.full((120, 160), 20.0, np.float32)
        self.temp[30, 40] = 35.0                    # hottest, x=40 y=30

    def rgb(self):
        if not self.has_rgb:
            return None, 0.0
        self.calls += 1
        self.n += 1
        return np.full((540, 960, 3), self.n % 250, np.uint8), time.time() + self.n * 1e-6

    def thermal(self):
        if not self.has_th:
            return None, 0.0
        self.n += 1
        return self.temp, time.time() + self.n * 1e-6

    def feed_status(self):
        if self.states:
            return self.states
        return {"rgb": {"state": "ok" if self.has_rgb else "unavailable", "error": None if self.has_rgb else "No cameras available",
                        "hint": "" if self.has_rgb else "sudo systemctl restart nvargus-daemon"},
                "thermal": {"state": "ok" if self.has_th else "disabled", "error": None if self.has_th else "THERMAL_SOURCE is none"}}


def make(src, tmp_path, fps=20):
    s = make_settings(data_dir=tmp_path, camera_width=960, camera_height=540)
    app = FastAPI()
    r = stream.make_router(src, s, default_fps=fps, data_dir=tmp_path)
    app.include_router(r)
    return app, r.live_service


class LiveServer:
    """Real uvicorn in a thread (TestClient buffers whole responses, so it cannot read an endless MJPEG)."""

    def __init__(self, app):
        import socket
        import uvicorn
        so = socket.socket(); so.bind(("127.0.0.1", 0)); self.port = so.getsockname()[1]; so.close()
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        self.base = f"http://127.0.0.1:{self.port}"

    def close(self):
        self.server.should_exit = True
        self.thread.join(5)


@pytest.fixture
def live():
    servers = []

    def mk(app):
        srv = LiveServer(app)
        servers.append(srv)
        return srv
    yield mk
    for srv in servers:
        srv.close()


def read_stream(srv, url, n_parts, timeout=8.0):
    import httpx
    parts, buf = [], b""
    t0 = time.time()
    with httpx.Client(base_url=srv.base, timeout=10).stream("GET", url) as resp:
        assert resp.status_code == 200
        ctype = resp.headers["content-type"]
        for chunk in resp.iter_bytes():
            buf += chunk
            while True:
                i = buf.find(b"--frame\r\n")
                j = buf.find(b"--frame\r\n", i + 1) if i >= 0 else -1
                if i < 0 or j < 0:
                    break
                parts.append(buf[i:j])
                buf = buf[j:]
            if len(parts) >= n_parts or time.time() - t0 > timeout:
                break
    return ctype, parts


def test_mjpeg_boundary_format(tmp_path, live):
    app, svc = make(Fake(), tmp_path)
    ctype, parts = read_stream(live(app), "/stream/rgb.mjpg?w=320&fps=20", 3)
    assert ctype == "multipart/x-mixed-replace; boundary=frame"
    assert len(parts) >= 3
    for p in parts:
        head, _, body = p.partition(b"\r\n\r\n")
        assert head.startswith(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
        n = int(head.split(b"Content-Length: ")[1])
        assert body.endswith(b"\r\n") and len(body) - 2 == n
        img = cv2.imdecode(np.frombuffer(body[:n], np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (180, 320, 3)               # ?w= downscale keeps aspect


def test_thermal_and_overlay_streams(tmp_path, live):
    app, svc = make(Fake(), tmp_path)
    srv = live(app)
    _, parts = read_stream(srv, "/stream/thermal.mjpg?palette=ironbow&min=10&max=40", 2)
    body = parts[0].partition(b"\r\n\r\n")[2]
    assert cv2.imdecode(np.frombuffer(body, np.uint8), 1).shape == (480, 640, 3)    # x4 upscale
    _, parts = read_stream(srv, "/stream/overlay.mjpg?alpha=0.7", 2)
    assert len(parts) >= 2


def test_encoder_runs_only_with_clients(tmp_path, live):
    src = Fake()
    app, svc = make(src, tmp_path)
    time.sleep(0.4)
    assert src.calls == 0                                 # nothing fetched/encoded without a client
    read_stream(live(app), "/stream/rgb.mjpg?fps=20", 2)
    time.sleep(stream.IDLE_GRACE_S + 0.8)
    calls = src.calls
    time.sleep(0.5)
    assert src.calls == calls and calls > 0               # stopped again after the last client left
    assert not any(e.running for e in svc._encoders.values())


def test_one_shared_encoder_for_two_clients(tmp_path):
    app, svc = make(Fake(), tmp_path)
    enc = svc.encoder("rgb", svc.norm_params("rgb", {"fps": "20", "w": "320"}))
    assert svc.encoder("rgb", svc.norm_params("rgb", {"fps": "20", "w": "320"})) is enc
    enc.acquire(); enc.acquire()
    try:
        assert enc.clients == 2
        threads = [t for t in threading.enumerate() if t.name == "enc-rgb"]
        assert len(threads) == 1
    finally:
        enc.release(); enc.release()


def test_fps_limiting(tmp_path):
    app, svc = make(Fake(), tmp_path)
    enc = svc.encoder("rgb", svc.norm_params("rgb", {"fps": "5", "w": "160"}))
    enc.acquire()
    try:
        time.sleep(2.2)
        n = enc.frames_encoded
    finally:
        enc.release()
    assert 7 <= n <= 13                                   # ~5 fps for ~2 s, nowhere near the source's call rate


def test_param_clamping(tmp_path):
    _, svc = make(Fake(), tmp_path)
    p = svc.norm_params("overlay", {"fps": "500", "w": "99999", "alpha": "9", "palette": "bogus", "min": "5", "max": "1"})
    assert p["fps"] == 30 and p["w"] == 1920 and p["alpha"] == 1.0 and p["palette"] == "inferno" and "min" not in p


def test_thermal_point_maths():
    t = np.arange(12, dtype=np.float32).reshape(3, 4)     # h=3 w=4
    assert stream.temp_at(t, 0.0, 0.0) == (0.0, 0, 0)
    assert stream.temp_at(t, 0.99, 0.99) == (11.0, 3, 2)
    assert stream.temp_at(t, 1.0, 1.0)[1:] == (3, 2)      # clamped, no IndexError
    assert stream.temp_at(t, 0.5, 0.5) == (6.0, 2, 1)
    assert stream.hottest(t) == (11.0, 3, 2)


def test_thermal_point_endpoint_and_stats(tmp_path):
    app, svc = make(Fake(), tmp_path)
    c = TestClient(app)
    d = c.get("/api/live/thermal_point", params={"x": 40.5 / 160, "y": 30.5 / 120}).json()
    assert d["px"] == [40, 30] and d["c"] == 35.0
    st = c.get("/api/live/stats").json()
    assert st["thermal"]["hottest"]["c"] == 35.0 and st["thermal"]["hottest"]["px"] == [40, 30]
    assert st["thermal"]["min_c"] == 20.0 and st["rgb"]["w"] == 960 and "brightness" in st["rgb"]
    assert "cpu_pct" in st["system"] or "ram_used_mb" in st["system"]
    assert abs(st["thermal"]["hottest"]["x"] - 40.5 / 160) < 1e-3


def test_snapshot_files(tmp_path):
    app, svc = make(Fake(), tmp_path)
    r = TestClient(app).post("/api/live/snapshot")
    assert r.status_code == 200
    files = r.json()["files"]
    assert set(files) == {"rgb", "thermal_npy", "thermal_png", "overlay"}
    for p in files.values():
        assert (tmp_path / "snapshots") in __import__("pathlib").Path(p).parents
    assert np.load(files["thermal_npy"]).shape == (120, 160) and np.load(files["thermal_npy"])[30, 40] == 35.0
    assert cv2.imread(files["rgb"]).shape == (540, 960, 3)


def test_unavailable_camera_states(tmp_path):
    app, svc = make(Fake(rgb=False, thermal=False), tmp_path)
    c = TestClient(app)
    r = c.get("/stream/rgb.mjpg")
    assert r.status_code == 503 and "nvargus" in r.json()["hint"]
    assert c.get("/stream/thermal.mjpg").status_code == 503
    assert c.get("/stream/overlay.mjpg").status_code == 503
    assert c.get("/api/live/thermal_point", params={"x": .5, "y": .5}).status_code == 503
    assert c.post("/api/live/snapshot").status_code == 503
    st = c.get("/api/live/stats").json()                  # the page keeps working
    assert st["rgb"]["state"] == "unavailable" and st["thermal"]["state"] == "disabled"
    assert c.get("/live").status_code == 200


def test_rgb_only_still_streams(tmp_path, live):
    app, svc = make(Fake(thermal=False), tmp_path)
    c = TestClient(app)
    assert c.get("/stream/thermal.mjpg").status_code == 503
    _, parts = read_stream(live(app), "/stream/rgb.mjpg", 2)
    assert len(parts) >= 2
    r = c.post("/api/live/snapshot").json()["files"]
    assert set(r) == {"rgb"}


def test_pipeline_source_disabled_and_assets(tmp_path):
    from pipeline import NullPipeline
    s = make_settings(data_dir=tmp_path)
    src = stream.PipelineSource(NullPipeline(), s)
    fs = src.feed_status()
    assert fs["rgb"]["state"] == "disabled" and "live_view.py" in fs["rgb"]["error"]
    app = FastAPI()
    app.include_router(stream.make_router(src, s, data_dir=tmp_path))
    c = TestClient(app)
    assert c.get("/live-assets/live.js").status_code == 200 and c.get("/live-assets/secret").status_code == 404
    assert c.get("/api/live/stats").json()["mission_url"] == "/"


def test_overlay_blend_and_palettes():
    rgb = np.full((90, 160, 3), 100, np.uint8)
    th = np.full((12, 16, 3), 255, np.uint8)
    H = np.diag([5.0, 5.0, 1.0]) @ np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    H[0, 2] = 40                                          # thermal covers x 40..120 only
    out = stream.blend_overlay(rgb, th, H, 0.5)
    assert out[45, 5, 0] == 100 and abs(int(out[45, 80, 0]) - 177) <= 2
    for p in stream.PALETTES:
        assert stream.colorize_thermal(np.linspace(0, 1, 12).reshape(3, 4).astype(np.float32), 0, 1, p).shape == (12, 16, 3)
