"""Hardening E: web edge cases. WebSocket abuse, Abort under load, REST bad input, traversal, ranges."""
from __future__ import annotations

import asyncio
import json
import os
import socket

import struct
import threading
import time

import pytest
from fastapi.testclient import TestClient

import models
from config import Settings
from webapp.controller import FakeController, _HOME_LAT, _HOME_LON
from webapp.server import Hub, MAX_WS_MESSAGE_BYTES, TokenBucket, create_app


def recv_until(ws, typ, limit=60):
    for _ in range(limit):
        m = ws.receive_json()
        if m["type"] == typ:
            return m
    raise AssertionError(f"no {typ} message received")


def make_settings(data, **kw):
    return Settings(_env_file=None, data_dir=data, camera_hfov_deg=90.0, camera_vfov_deg=60.0, sensor_width_mm=4.8,
                    focal_length_mm=3.67, image_width_px=1920, **kw)


def make_pmtiles(path, size=4096):
    h = bytearray(127)
    h[:7] = b"PMTiles"
    h[7] = 3
    h[99] = 1
    h[100], h[101] = 0, 17
    struct.pack_into("<4i", h, 102, int(-84.54e7), int(33.92e7), int(-84.51e7), int(33.95e7))
    path.write_bytes(bytes(h) + bytes(i % 251 for i in range(size - 127)))


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "data"
    (data / "alerts").mkdir(parents=True)
    (data / "alerts" / "a.jpg").write_bytes(b"\xff\xd8\xff fake jpeg")
    (data / "maps").mkdir()
    make_pmtiles(data / "maps" / "area.pmtiles")
    (tmp_path / "outside.jpg").write_bytes(b"outside")
    (tmp_path / "outdir").mkdir()
    (tmp_path / "outdir" / "x.jpg").write_bytes(b"outside dir")
    os.symlink(tmp_path / "outside.jpg", data / "alerts" / "link.jpg")
    os.symlink(tmp_path / "outdir", data / "linkdir")
    os.symlink(tmp_path / "outside.jpg", data / "maps" / "evil.pmtiles")
    settings = make_settings(data)
    ctrl = FakeController(data, background=False, settings=settings)
    hub = Hub()
    app = create_app(ctrl, settings, hub=hub)
    with TestClient(app) as client:
        yield client, ctrl, hub


# ------------------------------------------------------------------ WebSocket abuse
def test_ws_survives_garbage(env):
    client, _, _ = env
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        garbage = ['', '   ', 'null', '1', '"str"', '{"type": null}', '{"type": 5}', '{"type":"chat","text":5}',
                   '{"type":"chat","text":null}', '{"type":"target","text":""}', '{"type":"confirm","plan_id":7}',
                   '{"type":"plan_area"}', '{"type":"plan_area","corners":"x"}',
                   '{"type":"plan_area","corners":[{"lat":"a","lon":1},{"lat":1,"lon":1},{"lat":2,"lon":2}]}',
                   '{"type":"plan_area","corners":[{"lat":NaN,"lon":1},{"lat":1,"lon":1},{"lat":2,"lon":2}]}',
                   '{"type":"chat","text":"' + "x" * 2500 + '"}', '{' * 10000, '\x00\x01', '{"type":"CHAT"}']
        for g in garbage:
            ws.send_text(g)
            assert recv_until(ws, "error")["data"]["message"], g
        ws.send_json({"type": "ping"})
        assert recv_until(ws, "telemetry")


def test_ws_binary_frame_rejected_connection_survives(env):
    client, _, _ = env
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        ws.send_bytes(b"\x00\x01\x02 binary")
        assert "binary" in recv_until(ws, "error")["data"]["message"]
        ws.send_json({"type": "ping"})
        assert recv_until(ws, "telemetry")


def test_ws_huge_message_rejected_connection_survives(env):
    client, _, _ = env
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        ws.send_text('{"type":"chat","text":"' + "a" * (MAX_WS_MESSAGE_BYTES + 10) + '"}')
        assert "too large" in recv_until(ws, "error")["data"]["message"]
        ws.send_json({"type": "ping"})
        assert recv_until(ws, "telemetry")


def test_ws_flood_is_limited_but_abort_never(env):
    client, ctrl, _ = env
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        for i in range(300):
            ws.send_json({"type": "ping"})
        ws.send_json({"type": "abort"})
        # all pings beyond the bucket are dropped, abort gets through
        deadline = time.time() + 5
        while ("abort", None) not in ctrl.calls and time.time() < deadline:
            time.sleep(0.02)
        assert ("abort", None) in ctrl.calls
        # one "slow down" notice at most per second
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] == "chat" and "ABORT" in m["data"]["text"]:
                break
        n_notes = sum(1 for m in msgs if m["type"] == "error" and "slow down" in m["data"]["message"])
        n_tel = sum(1 for m in msgs if m["type"] == "telemetry")
        assert 1 <= n_notes <= 3
        assert n_tel < 100                      # 300 pings were NOT all answered


def test_abort_during_rate_limited_flood_from_other_phone(env):
    client, ctrl, _ = env
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        recv_until(a, "status")
        recv_until(b, "status")
        for _ in range(200):
            a.send_json({"type": "chat", "text": "hello"})
        b.send_json({"type": "abort"})
        m = recv_until(b, "chat", limit=2000)
        while "ABORT" not in m["data"]["text"]:
            m = recv_until(b, "chat", limit=2000)
        assert ctrl.state == "rtl"


def test_token_bucket_refills():
    b = TokenBucket(rate=10, burst=3)
    b.t = 0.0
    assert [b.take(0.0) for _ in range(4)] == [True, True, True, False]
    assert b.take(0.15) is True and b.take(0.15) is False


def test_ws_too_many_inflight(env):
    client, ctrl, _ = env
    gate = threading.Event()

    async def slow(text):
        while not gate.is_set():
            await asyncio.sleep(0.02)
        return "ok"

    ctrl.handle_chat = slow  # type: ignore[assignment]
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        for _ in range(12):
            ws.send_json({"type": "chat", "text": "x"})
        err = recv_until(ws, "error")
        assert "too many" in err["data"]["message"]
        ws.send_json({"type": "abort"})            # still goes through at the cap
        while True:
            m = ws.receive_json()
            if m["type"] == "chat" and "ABORT" in m["data"]["text"]:
                break
        gate.set()


def test_client_disconnect_mid_request_completes_abort(env):
    client, ctrl, hub = env
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        ws.send_json({"type": "abort"})
    deadline = time.time() + 3
    while ("abort", None) not in ctrl.calls and time.time() < deadline:
        time.sleep(0.02)
    assert ("abort", None) in ctrl.calls and ctrl.state == "rtl"
    deadline = time.time() + 2
    while hub.client_count and time.time() < deadline:
        time.sleep(0.02)
    assert hub.client_count == 0


def test_client_disconnect_mid_slow_chat_no_crash(env):
    client, ctrl, hub = env

    async def slow(text):
        await asyncio.sleep(0.3)
        return "late reply"

    ctrl.handle_chat = slow  # type: ignore[assignment]
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        ws.send_json({"type": "chat", "text": "x"})
    time.sleep(0.6)
    with client.websocket_connect("/ws") as ws2:     # server still fine
        assert recv_until(ws2, "status")


def test_ten_phones_all_get_broadcasts(env):
    client, _, hub = env
    conns = [client.websocket_connect("/ws") for _ in range(10)]
    wss = [c.__enter__() for c in conns]
    try:
        for w in wss:
            recv_until(w, "status")
        assert hub.client_count == 10
        alert = models.Alert(level="person", track_id=3, ts=time.time(), c_yolo=0.9, reason="x")
        client.portal.call(hub.broadcast, models.WsMessage(type="alert", data=alert.model_dump()))
        for w in wss:
            assert recv_until(w, "alert")["data"]["track_id"] == 3
    finally:
        for c in conns:
            c.__exit__(None, None, None)


def test_reconnect_storm_leaves_no_clients(env):
    client, _, hub = env
    for _ in range(40):
        with client.websocket_connect("/ws") as ws:
            assert recv_until(ws, "status")["data"]["mission_state"] in ("idle", "planned", "rtl", "flying")
    deadline = time.time() + 3
    while hub.client_count and time.time() < deadline:
        time.sleep(0.05)
    assert hub.client_count == 0


class StuckWS:
    """A phone whose TCP buffer is full: send_text never returns."""
    def __init__(self): self.sent = 0
    async def accept(self): pass
    async def send_text(self, t):
        await asyncio.sleep(3600)


class GoodWS:
    def __init__(self): self.got = []
    async def send_text(self, t): self.got.append(t)


def test_slow_client_does_not_block_others_and_is_dropped(monkeypatch):
    import webapp.server as srv
    monkeypatch.setattr(srv, "SEND_TIMEOUT_S", 0.3)

    async def run():
        hub = Hub()
        slow, goods = StuckWS(), [GoodWS() for _ in range(5)]
        hub._clients[slow] = asyncio.Lock()
        for g in goods:
            hub._clients[g] = asyncio.Lock()
        t0 = time.monotonic()
        # fast clients receive immediately, even though the slow one is stuck
        task = asyncio.ensure_future(hub.broadcast(models.WsMessage(type="status", data={})))
        await asyncio.sleep(0.05)
        assert all(len(g.got) == 1 for g in goods)
        await task
        assert time.monotonic() - t0 < 1.0
        assert slow not in hub._clients and len(hub._clients) == 5
    asyncio.run(run())


# ------------------------------------------------------------------ Abort under load (latency)
def test_abort_latency_under_load_ws(env):
    """Abort must reach the controller < 300 ms while chat is stuck on a slow LLM, a flood runs and
    other phones are connected."""
    client, ctrl, _ = env
    abort_at = {}
    orig_abort = ctrl.abort

    async def timed_abort():
        abort_at["t"] = time.monotonic()
        return await orig_abort()

    ctrl.abort = timed_abort  # type: ignore[assignment]

    async def slow(text):
        await asyncio.sleep(30)

    ctrl.handle_chat = slow  # type: ignore[assignment]
    ctrl.plan_area = slow    # type: ignore[assignment]
    lat = []
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b, client.websocket_connect("/ws") as c:
        for w in (a, b, c):
            recv_until(w, "status")
        for _ in range(4):
            a.send_json({"type": "chat", "text": "stuck on llm"})
        for _ in range(150):
            c.send_json({"type": "ping"})
        for _ in range(3):
            abort_at.clear()
            t0 = time.monotonic()
            b.send_json({"type": "abort"})
            while "t" not in abort_at and time.monotonic() - t0 < 2:
                time.sleep(0.001)
            lat.append(abort_at.get("t", t0 + 99) - t0)
            ctrl.state = "flying"
    print("abort latency (ws, under load) ms:", [round(x * 1000, 1) for x in lat])
    assert max(lat) < 0.3


def test_abort_rest_fallback_works_without_ws(env):
    client, ctrl, _ = env
    r = client.post("/api/abort")
    assert r.status_code == 200 and ctrl.state == "rtl"


def test_abort_rest_while_chat_inflight(env):
    client, ctrl, _ = env
    gate = threading.Event()

    async def slow(text):
        while not gate.is_set():
            await asyncio.sleep(0.02)
        return "ok"

    ctrl.handle_chat = slow  # type: ignore[assignment]
    th = threading.Thread(target=lambda: client.post("/api/chat", json={"text": "hi"}))
    th.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    assert client.post("/api/abort").status_code == 200
    dt = time.monotonic() - t0
    gate.set()
    th.join(5)
    assert ctrl.state == "rtl" and dt < 0.3, dt


def test_two_phones_abort_at_once(env):
    client, ctrl, _ = env
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        recv_until(a, "status")
        recv_until(b, "status")
        a.send_json({"type": "abort"})
        b.send_json({"type": "abort"})
        for w in (a, b):
            seen = 0
            while seen < 2:
                m = recv_until(w, "chat")
                seen += "ABORT" in m["data"]["text"]
    assert [c for c in ctrl.calls if c[0] == "abort"].__len__() == 2 and ctrl.state == "rtl"


def test_abort_failure_reports_use_rc(env):
    client, ctrl, _ = env

    async def boom():
        raise RuntimeError("link dead")

    ctrl.abort = boom  # type: ignore[assignment]
    m = client.post("/api/abort").json()
    assert m["type"] == "error" and "RC transmitter" in m["data"]["message"]


def test_abort_hung_controller_times_out_with_rc_message(env, monkeypatch):
    import webapp.server as srv
    monkeypatch.setattr(srv, "COMMAND_TIMEOUT_S", 0.3)
    client, ctrl, _ = env

    async def hang():
        await asyncio.sleep(60)

    ctrl.abort = hang  # type: ignore[assignment]
    t0 = time.monotonic()
    m = client.post("/api/abort").json()
    assert m["type"] == "error" and "RC transmitter" in m["data"]["message"] and time.monotonic() - t0 < 2


# ------------------------------------------------------------------ REST bad input
@pytest.mark.parametrize("path,body", [
    ("/api/chat", {"text": 5}), ("/api/chat", {"text": None}), ("/api/chat", {"text": ["a"]}), ("/api/chat", []),
    ("/api/chat", {"text": "x" * 2001}),
    ("/api/target", {}), ("/api/target", {"text": ""}),
    ("/api/confirm", {"plan_id": ""}), ("/api/confirm", {"plan_id": 5}), ("/api/confirm", {"plan_id": "x" * 101}),
    ("/api/parse_coords", {}), ("/api/parse_coords", {"text": ""}), ("/api/parse_coords", {"text": "x" * 4001}),
    ("/api/plan_area", {}), ("/api/plan_area", {"corners": []}),
    ("/api/client_log", {}), ("/api/client_log", {"message": "x" * 1001}),
])
def test_rest_bad_bodies_are_422(env, path, body):
    client, _, _ = env
    assert client.post(path, json=body).status_code == 422


def test_rest_non_json_and_wrong_method(env):
    client, _, _ = env
    assert client.post("/api/chat", content=b"\xff\xfe not json", headers={"content-type": "application/json"}).status_code in (400, 422)
    assert client.post("/api/chat", content=b"text=1", headers={"content-type": "text/plain"}).status_code == 422
    assert client.get("/api/chat").status_code == 405
    assert client.post("/api/status").status_code == 405
    assert client.get("/api/nope").status_code == 404


def _sq(alt=None, **extra):
    from models import GeoPoint
    from survey import local_to_geo
    home = GeoPoint(lat=_HOME_LAT, lon=_HOME_LON)
    pts = [(40, -40), (40, 40), (120, 40), (120, -40)]
    d = {"corners": [{"lat": a, "lon": b} for a, b in (local_to_geo(n, e, home) for n, e in pts)]}
    if alt is not None:
        d["altitude_m"] = alt
    d.update(extra)
    return d


@pytest.mark.parametrize("extra", [
    {"altitude_m": -5}, {"altitude_m": 0}, {"altitude_m": "high"}, {"speed_ms": -1}, {"speed_ms": 0},
    {"overlap": 0.95}, {"overlap": -0.1}, {"target_text": "x" * 2001},
])
def test_plan_area_bad_numbers_422(env, extra):
    client, ctrl, _ = env
    assert client.post("/api/plan_area", json=_sq(**extra)).status_code == 422
    assert not [c for c in ctrl.calls if c[0] == "plan_area"]


@pytest.mark.parametrize("raw", [
    '{"altitude_m": NaN}', '{"altitude_m": Infinity}', '{"speed_ms": Infinity}', '{"speed_ms": NaN}',
    '{"overlap": NaN}',
])
def test_plan_area_nan_inf_never_plans_a_flight(env, raw):
    client, ctrl, _ = env
    body = json.loads(json.dumps(_sq()))
    txt = json.dumps(body)[:-1] + ", " + raw[1:]
    r = client.post("/api/plan_area", content=txt, headers={"content-type": "application/json"})
    if r.status_code == 200:
        d = r.json()
        # if it got through validation it must not be a flyable plan
        assert d["type"] == "error" or not d["data"]["safety"]["ok"], d
    else:
        assert r.status_code == 422


def test_plan_area_corner_nan_and_out_of_range(env):
    client, _, _ = env
    for c in ('{"lat": NaN, "lon": 1}', '{"lat": 91, "lon": 1}', '{"lat": 1, "lon": 181}', '{"lat": 1}', '{"lat": null, "lon": 2}'):
        txt = '{"corners": [%s, {"lat":1,"lon":1}, {"lat":2,"lon":2}, {"lat":3,"lon":1}]}' % c
        assert client.post("/api/plan_area", content=txt, headers={"content-type": "application/json"}).status_code == 422


def test_parse_coords_uses_home_for_swap_hint(env):
    client, _, _ = env
    swapped = f"{_HOME_LON}, {_HOME_LAT}\n{_HOME_LON + 0.0003}, {_HOME_LAT}\n{_HOME_LON + 0.0003}, {_HOME_LAT + 0.0003}"
    r = client.post("/api/parse_coords", json={"text": swapped})
    assert r.status_code == 422 and "swap" in r.json()["detail"].lower()
    ok = f"{_HOME_LAT}, {_HOME_LON}\n{_HOME_LAT + 0.0003}, {_HOME_LON}\n{_HOME_LAT + 0.0003}, {_HOME_LON + 0.0003}"
    r = client.post("/api/parse_coords", json={"text": ok})
    assert r.status_code == 200 and len(r.json()["corners"]) == 3


def test_parse_coords_no_home_still_works(env):
    client, ctrl, _ = env
    ctrl.snapshot = lambda: {"home": None}   # type: ignore[assignment]
    ok = "33.9, -84.5\n33.91, -84.5\n33.91, -84.49"
    assert client.post("/api/parse_coords", json={"text": ok}).status_code == 200
    ctrl.snapshot = lambda: (_ for _ in ()).throw(RuntimeError("boom"))   # type: ignore[assignment]
    assert client.post("/api/parse_coords", json={"text": ok}).status_code == 200


def test_parse_coords_garbage_is_422_not_500(env):
    client, _, _ = env
    for t in ["abc", "1,2,3,4,5", "NaN, NaN\nNaN, NaN\nNaN, NaN", "999, 999\n1,1\n2,2", "\x00\x01", "💥 " * 100]:
        r = client.post("/api/parse_coords", json={"text": t})
        assert r.status_code == 422, (t, r.status_code)


# ------------------------------------------------------------------ path traversal
EVIL = ["../outside.jpg", "%2e%2e/outside.jpg", "..%2foutside.jpg", "%2e%2e%2foutside.jpg", "alerts/../../outside.jpg",
        "/etc/passwd", "//etc/passwd", "alerts/%2e%2e/%2e%2e/outside.jpg", "....//outside.jpg", "..\\outside.jpg",
        "alerts/link.jpg", "linkdir/x.jpg", "%00", "alerts/a.jpg%00.png", "%252e%252e/outside.jpg", ".hidden.jpg"]


@pytest.mark.parametrize("evil", EVIL)
def test_data_traversal_and_symlinks(env, evil):
    client, _, _ = env
    r = client.get("/data/" + evil)
    assert r.status_code in (400, 404, 422), (evil, r.status_code)
    assert b"outside" not in r.content


@pytest.mark.parametrize("evil", ["../area.pmtiles", "%2e%2e/area.pmtiles", "evil.pmtiles", "area.pmtiles/..",
                                  "/etc/passwd", "..%2farea.pmtiles", "a/b.pmtiles", ".hidden.pmtiles", "area.pmtiles%00"])
def test_maps_traversal_and_symlinks(env, evil):
    client, _, _ = env
    r = client.get("/maps/" + evil)
    assert r.status_code in (400, 404), (evil, r.status_code)
    assert b"outside" not in r.content


@pytest.mark.parametrize("path", ["/tiles/sat/../../../etc/passwd", "/tiles/sat/-1/0/0.jpg", "/tiles/sat/99/0/0.jpg",
                                  "/tiles/sat/1/5/0.jpg", "/tiles/sat/a/b/c.jpg", "/tiles/sat/1/0/0.exe",
                                  "/tiles/sat/1/0/0.jpg", "/tiles/sat/1000000/0/0.png"])
def test_tiles_bad_requests_without_satellite(env, path):
    client, _, _ = env
    assert client.get(path).status_code in (404, 422)


# ------------------------------------------------------------------ range requests, HEAD
def get(client, rng=None, method="GET"):
    h = {"Range": rng} if rng is not None else {}
    return client.request(method, "/maps/area.pmtiles", headers=h)


def test_range_variants(env):
    client, _, _ = env
    size = 4096
    full = get(client).content
    assert len(full) == size
    r = get(client, "bytes=0-126")
    assert r.status_code == 206 and r.content == full[:127] and r.headers["content-range"] == f"bytes 0-126/{size}"
    r = get(client, "bytes=100-")                       # open ended
    assert r.status_code == 206 and r.content == full[100:]
    r = get(client, "bytes=-50")                        # suffix
    assert r.status_code == 206 and r.content == full[-50:]
    r = get(client, "bytes=-99999")                     # suffix larger than file -> whole file
    assert r.status_code == 206 and r.content == full
    r = get(client, f"bytes={size - 1}-{size + 500}")   # end beyond EOF clamps
    assert r.status_code == 206 and len(r.content) == 1
    for bad in [f"bytes={size}-", f"bytes={size + 10}-{size + 20}", "bytes=-0", "bytes=-", "bytes=5-2", "bytes=abc",
                "items=0-5", "bytes=0-5,10-20", "bytes=--5", "bytes=1e3-2e3", "", "bytes=-5-"]:
        r = get(client, bad)
        assert r.status_code in (416, 200), (bad, r.status_code)
        if r.status_code == 416:
            assert r.headers["content-range"] == f"bytes */{size}"


def test_multi_range_is_refused_with_416(env):
    client, _, _ = env
    assert get(client, "bytes=0-5,10-20").status_code == 416


def test_head_requests(env):
    client, _, _ = env
    r = get(client, method="HEAD")
    assert r.status_code == 200 and r.headers["content-length"] == "4096" and r.content == b""
    r = get(client, "bytes=0-9", method="HEAD")
    assert r.status_code == 206 and r.headers["content-length"] == "10" and r.content == b""
    assert client.head("/maps/nope.pmtiles").status_code == 404
    assert client.head("/").status_code in (200, 405)
    assert client.head("/data/alerts/a.jpg").status_code in (200, 405)


# ------------------------------------------------------------------ big history, missing map/calibration
def test_huge_alert_history_snapshot_is_bounded(env):
    client, ctrl, _ = env
    ctrl.alerts = [ctrl.make_alert().model_dump() for _ in range(10_000)]
    snap = client.get("/api/status").json()
    assert len(snap["data"]["alerts"]) <= 20
    assert len(json.dumps(snap)) < 100_000
    with client.websocket_connect("/ws") as ws:
        m = ws.receive_json()
        assert len(m["data"]["alerts"]) <= 20


def test_real_controller_alert_buffer_is_bounded():
    import inspect
    import controller_real
    src = inspect.getsource(controller_real)
    assert "deque(maxlen=50)" in src and "list(self.alerts)[-20:]" in src


def test_missing_offline_map_is_graceful(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    s = make_settings(data)
    app = create_app(FakeController(data, background=False, settings=s), s, hub=Hub())
    with TestClient(app) as c:
        d = c.get("/api/map/info").json()
        assert d["has_map"] is False and d["layers"] == []
        assert c.get("/maps/area.pmtiles").status_code == 404
        assert c.get("/tiles/sat/1/0/0.jpg").status_code == 404
        assert c.get("/api/limits").status_code == 200


def test_missing_data_dir_does_not_break_pages(tmp_path):
    s = make_settings(tmp_path / "does" / "not" / "exist")
    app = create_app(FakeController(tmp_path / "x", background=False, settings=s), s, hub=Hub())
    with TestClient(app) as c:
        assert c.get("/").status_code == 200
        assert c.get("/api/map/info").json()["has_map"] is False
        assert c.get("/data/a.jpg").status_code == 404
        assert c.get("/api/status").status_code == 200


def test_limits_without_calibration_numbers(tmp_path):
    """Settings with the default (uncalibrated) camera block must still give finite numbers."""
    import math
    data = tmp_path / "data"
    data.mkdir()
    s = Settings(_env_file=None, data_dir=data)
    app = create_app(FakeController(data, background=False, settings=s), s, hub=Hub())
    with TestClient(app) as c:
        d = c.get("/api/limits").json()
        for k, v in d.items():
            if isinstance(v, (int, float)):
                assert math.isfinite(v), k


def test_client_log_flood_is_429(env):
    client, _, _ = env
    codes = [client.post("/api/client_log", json={"message": f"boom {i}"}).status_code for i in range(80)]
    assert codes.count(200) == 30 and codes.count(429) == 50


def test_client_log_injection_is_single_line(env, caplog):
    client, _, _ = env
    import logging
    with caplog.at_level(logging.INFO):
        client.post("/api/client_log", json={"message": "a\nFAKE LOG LINE\nb", "stack": "x\ny", "level": "weird"})
    assert all("\n" not in r.getMessage() for r in caplog.records)


def test_static_pages_served(env):
    client, _, _ = env
    r = client.get("/")
    assert r.status_code == 200 and "no-cache" in r.headers["cache-control"]
    for f in ("app.js", "app.css"):
        assert client.get(f"/static/{f}").status_code == 200
    assert client.get("/static/../server.py").status_code in (404, 400)
    assert client.get("/static/%2e%2e/server.py").status_code in (404, 400)


# ------------------------------------------------------------------ files router (plans / flights)
@pytest.fixture
def files_env(tmp_path):
    from webapp.files import make_router
    data = tmp_path / "data"
    (data / "plans" / "abc123").mkdir(parents=True)
    (data / "plans" / "abc123" / "plan.json").write_text('{"plan_id": "abc123"}')
    (data / "plans" / "abc123" / "mission.waypoints").write_text("QGC WPL 110\n")
    (data / "plans" / "bad").mkdir()
    (data / "plans" / "bad" / "plan.json").write_text("{not json")
    (data / "plans" / "sneaky").mkdir()
    (tmp_path / "secret.json").write_text('{"secret": 1}')
    os.symlink(tmp_path / "secret.json", data / "plans" / "sneaky" / "plan.json")
    os.symlink(tmp_path / "secret.json", data / "plans" / "sneaky" / "preflight.json")
    (data / "flights" / "f1").mkdir(parents=True)
    (data / "flights" / "f1" / "flight.json").write_text("{}")
    os.symlink(tmp_path / "secret.json", data / "flights" / "f1" / "mission.geojson")
    s = make_settings(data)
    ctrl = FakeController(data, background=False, settings=s)
    app = create_app(ctrl, s, hub=Hub())
    app.include_router(make_router(s, ctrl))
    with TestClient(app) as c:
        yield c


def test_files_happy_and_missing(files_env):
    c = files_env
    assert c.get("/api/plans/abc123").json() == {"plan_id": "abc123"}
    assert c.get("/api/plans/abc123/mission.waypoints").status_code == 200
    assert c.get("/api/plans/abc123/preflight.json").status_code == 404
    assert c.get("/api/plans/nope").status_code == 404
    assert c.get("/api/flights/f1/flight.json").status_code == 200
    assert c.get("/api/plans?limit=-5").status_code == 200 and c.get("/api/plans?limit=abc").status_code == 422
    assert c.get("/api/flights?limit=999999").status_code == 200


def test_files_corrupt_plan_is_not_a_crash(files_env):
    assert files_env.get("/api/plans/bad").status_code == 500


@pytest.mark.parametrize("path", [
    "/api/plans/..", "/api/plans/%2e%2e/plan.json", "/api/plans/..%2f..%2fsecret.json", "/api/plans/abc123/..%2fbad%2fplan.json",
    "/api/plans/abc123/%2e%2e/bad/plan.json", "/api/plans//etc/passwd", "/api/plans/%2fetc%2fpasswd",
    "/api/plans/abc123/plan.json%00", "/api/plans/sneaky", "/api/plans/sneaky/plan.json", "/api/plans/sneaky/preflight.json",
    "/api/flights/f1/mission.geojson", "/api/flights/..%2f..%2fsecret.json/flight.json", "/api/flights/f1/../../../secret.json",
    "/api/flights/f1/..%2fflight.json", "/api/flights/f1/notallowed.txt", "/api/plans/a%20b", "/api/plans/" + "a" * 200,
])
def test_files_traversal_blocked(files_env, path):
    r = files_env.get(path)
    assert r.status_code in (400, 404, 422), (path, r.status_code)
    assert b"secret" not in r.content


def test_legal_and_vehicle_endpoints_on_fake_controller(files_env):
    c = files_env
    assert c.get("/api/legal").status_code == 200
    assert c.post("/api/legal/ack", json={"accepted": [True]}).status_code == 404
    assert c.post("/api/legal/ack", json={"accepted": "yes"}).status_code == 422
    assert c.get("/api/preflight").status_code == 404
    assert c.post("/api/vehicle/clear_mission").status_code == 404
    assert c.post("/api/vehicle/read_mission").status_code == 404


# ------------------------------------------------------------------ real uvicorn + websockets client
class LiveServer:
    def __init__(self, tmp, port=None):
        import uvicorn
        self.data = tmp / "data"
        self.data.mkdir(exist_ok=True)
        self.settings = make_settings(self.data)
        self.ctrl = FakeController(self.data, background=False, settings=self.settings)
        self.app = create_app(self.ctrl, self.settings, hub=Hub())
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = port or s.getsockname()[1]
        self.cfg = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error", ws_ping_interval=None)
        self.server = uvicorn.Server(self.cfg)
        self.th = threading.Thread(target=self.server.run, daemon=True)

    def start(self):
        self.th.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("server did not start")

    def stop(self):
        self.server.should_exit = True
        self.th.join(10)


async def _recv_type(ws, typ, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        m = json.loads(await asyncio.wait_for(ws.recv(), max(0.01, end - time.monotonic())))
        if m["type"] == typ:
            return m
    raise AssertionError(typ)


def test_live_server_big_frame_binary_and_latency(tmp_path):
    import websockets
    srv = LiveServer(tmp_path).start()
    try:
        async def run():
            url = f"ws://127.0.0.1:{srv.port}/ws"
            async with websockets.connect(url, max_size=None) as ws, websockets.connect(url) as ws2:
                await _recv_type(ws, "status")
                await _recv_type(ws2, "status")
                await ws.send("x" * 5_000_000)                       # 5 MB text frame
                assert "too large" in (await _recv_type(ws, "error", 10))["data"]["message"]
                await ws.send(b"\x00" * 100)
                assert "binary" in (await _recv_type(ws, "error"))["data"]["message"]
                await ws.send(json.dumps({"type": "ping"}))
                await _recv_type(ws, "telemetry")
                # abort latency while a flood + stuck LLM chat run
                async def slow(text):
                    await asyncio.sleep(30)
                srv.ctrl.handle_chat = slow
                for _ in range(3):
                    await ws.send(json.dumps({"type": "chat", "text": "stuck"}))
                for _ in range(300):
                    await ws.send(json.dumps({"type": "ping"}))
                lat = []
                for _ in range(5):
                    srv.ctrl.calls.clear()
                    t0 = time.monotonic()
                    await ws2.send(json.dumps({"type": "abort"}))
                    while not [c for c in srv.ctrl.calls if c[0] == "abort"]:
                        await asyncio.sleep(0.001)
                        assert time.monotonic() - t0 < 2
                    lat.append(time.monotonic() - t0)
                print("abort latency (real uvicorn, under load) ms:", [round(x * 1000, 1) for x in lat])
                assert max(lat) < 0.3
        asyncio.run(run())
    finally:
        srv.stop()


def test_server_restart_phone_reconnects_and_resyncs(tmp_path):
    import websockets
    srv = LiveServer(tmp_path).start()
    port = srv.port
    states = []

    async def phone():
        """What app.js does: reconnect with backoff, and take the status the server sends on connect."""
        url = f"ws://127.0.0.1:{port}/ws"
        for attempt in range(60):
            try:
                async with websockets.connect(url) as ws:
                    m = await _recv_type(ws, "status")
                    states.append(m["data"]["mission_state"])
                    if len(states) >= 2:
                        return
                    await ws.recv()      # blocks until the server dies
            except Exception:
                await asyncio.sleep(0.1)
        raise AssertionError("never reconnected")

    async def run():
        t = asyncio.ensure_future(phone())
        await asyncio.sleep(0.7)
        srv.ctrl.state = "rtl"
        srv.stop()
        await asyncio.sleep(0.3)
        # restart on the same port, with the persisted state
        srv2 = LiveServer(tmp_path, port=port)
        srv2.ctrl.state = "rtl"
        srv2.start()
        try:
            await asyncio.wait_for(t, 15)
        finally:
            srv2.stop()
    asyncio.run(run())
    assert states == ["rtl", "rtl"] or states[-1] == "rtl"


def test_each_phone_has_its_own_chat_session(env):
    import assistant
    client, ctrl, _ = env
    seen = []

    async def spy(text):
        seen.append((text, assistant.CHAT_SESSION.get()))
        return "ok"

    ctrl.handle_chat = spy  # type: ignore[assignment]
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        recv_until(a, "status")
        recv_until(b, "status")
        a.send_json({"type": "chat", "text": "from a 1"})
        b.send_json({"type": "chat", "text": "from b 1"})
        a.send_json({"type": "chat", "text": "from a 2"})
        deadline = time.time() + 3
        while len(seen) < 3 and time.time() < deadline:
            time.sleep(0.02)
    sess = {t: s for t, s in seen}
    assert sess["from a 1"] == sess["from a 2"] != sess["from b 1"]
    assert sess["from a 1"].startswith("ws-") and sess["from a 1"] != "default"
    client.post("/api/chat", json={"text": "rest default"})
    client.post("/api/chat", json={"text": "rest p1", "session": "phone-1"})
    sess = {t: s for t, s in seen}
    assert sess["rest default"] == "default" and sess["rest p1"] == "phone-1"


def test_chat_session_survives_a_reconnect(env):
    """The phone sends its own chat id: a new connection (screen lock, WiFi blip, reload) keeps the same draft."""
    import assistant
    client, ctrl, _ = env
    seen = []

    async def spy(text):
        seen.append((text, assistant.CHAT_SESSION.get()))
        return "ok"

    ctrl.handle_chat = spy  # type: ignore[assignment]
    for n, sid in ((1, "c-abc123def456"), (2, "c-abc123def456"), (3, "bad id!")):
        with client.websocket_connect("/ws") as ws:
            recv_until(ws, "status")
            ws.send_json({"type": "chat", "text": f"msg {n}", "session": sid})
            deadline = time.time() + 3
            while len(seen) < n and time.time() < deadline:
                time.sleep(0.02)
    sess = dict(seen)
    assert sess["msg 1"] == sess["msg 2"] == "c-abc123def456"
    assert sess["msg 3"].startswith("ws-")          # an invalid id falls back to the connection's own session
