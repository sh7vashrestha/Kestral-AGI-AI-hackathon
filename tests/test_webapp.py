"""Web app tests: REST, static, path traversal, WebSocket flows, Hub broadcasts."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

import models
from config import Settings
from webapp.controller import Controller, FakeController
from webapp.server import Hub, create_app


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "data"
    (data / "alerts").mkdir(parents=True)
    (data / "alerts" / "a.jpg").write_bytes(b"\xff\xd8\xff fake jpeg")
    (data / "secret.txt").write_text("nope")
    (tmp_path / "outside.jpg").write_bytes(b"outside")
    settings = Settings(data_dir=data)
    ctrl = FakeController(data, background=False)
    hub = Hub()
    app = create_app(ctrl, settings, hub=hub)
    with TestClient(app) as client:
        yield client, ctrl, hub


def recv_until(ws, typ, limit=10):
    for _ in range(limit):
        m = ws.receive_json()
        if m["type"] == typ:
            return m
    raise AssertionError(f"no {typ} message received")


def test_fake_controller_satisfies_protocol(tmp_path):
    assert isinstance(FakeController(tmp_path, background=False), Controller)


def test_index_and_static(env):
    client, _, _ = env
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "abort-btn" in r.text
    assert "http://" not in r.text and "https://" not in r.text   # fully offline
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200
    js = client.get("/static/app.js").text
    assert "cdn" not in js.lower()


def test_status_rest(env):
    client, _, _ = env
    m = client.get("/api/status").json()
    assert m["type"] == "status"
    assert m["data"]["mission_state"] == "idle"
    assert "telemetry" in m["data"]


def test_rest_chat_confirm_abort_target(env):
    client, ctrl, _ = env
    m = client.post("/api/chat", json={"text": "hello"}).json()
    assert m["type"] == "chat" and m["data"]["role"] == "assistant"

    m = client.post("/api/chat", json={"text": "search the field"}).json()
    assert m["type"] == "plan"
    plan = models.PlanProposal.model_validate(m["data"])
    assert plan.safety.ok and len(plan.plan.waypoints) >= 4

    m = client.post("/api/confirm", json={"plan_id": plan.plan_id}).json()
    assert m["type"] == "chat" and "confirmed" in m["data"]["text"].lower()
    assert ctrl.state == "staged"                      # notes/go-flow.md: Confirm = upload only
    st = client.get("/api/status").json()["data"]
    assert st["readiness"]["go_allowed"] and st["staged"]["plan_id"] == plan.plan_id
    m = client.post("/api/go", json={"plan_id": plan.plan_id}).json()
    assert m["type"] == "chat" and "GO" in m["message"]
    assert ctrl.state == "flying"

    m = client.post("/api/target", json={"text": "man in blue"}).json()
    assert "man in blue" in m["data"]["text"]
    assert client.get("/api/status").json()["data"]["target"]["description"] == "man in blue"

    m = client.post("/api/abort").json()
    assert "ABORT" in m["data"]["text"]
    assert ctrl.state == "rtl"


def test_rest_validation(env):
    client, _, _ = env
    assert client.post("/api/chat", json={"text": ""}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422
    assert client.post("/api/chat", json={"text": "x" * 5000}).status_code == 422
    assert client.post("/api/confirm", json={}).status_code == 422


def test_unsafe_plan_cannot_be_confirmed(env):
    client, ctrl, _ = env
    m = client.post("/api/chat", json={"text": "unsafe search"}).json()
    plan = models.PlanProposal.model_validate(m["data"])
    assert not plan.safety.ok and plan.safety.errors
    m = client.post("/api/confirm", json={"plan_id": plan.plan_id}).json()
    assert "safety" in m["data"]["text"].lower()
    assert ctrl.state != "flying"


def test_controller_exception_becomes_error(env):
    client, ctrl, _ = env

    async def boom(text):
        raise RuntimeError("llm down")

    ctrl.handle_chat = boom  # type: ignore[assignment]
    m = client.post("/api/chat", json={"text": "hi"}).json()
    assert m["type"] == "error" and "llm down" in m["data"]["message"]


def test_data_serving_and_traversal(env):
    client, _, _ = env
    r = client.get("/data/alerts/a.jpg")
    assert r.status_code == 200 and r.content.startswith(b"\xff\xd8")
    assert client.get("/data/secret.txt").status_code == 404          # non-image blocked
    assert client.get("/data/nope.jpg").status_code == 404
    for evil in ["../outside.jpg", "%2e%2e/outside.jpg", "..%2foutside.jpg", "alerts/../../outside.jpg",
                 "%2e%2e%2f%2e%2e%2fetc%2fpasswd", "/etc/passwd", "alerts/%2e%2e/%2e%2e/outside.jpg"]:
        assert client.get("/data/" + evil).status_code in (404, 400), evil
    assert client.get("/data/alerts").status_code == 404               # directory


def test_ws_sends_status_on_connect(env):
    client, _, _ = env
    with client.websocket_connect("/ws") as ws:
        m = ws.receive_json()
        assert m["type"] == "status" and m["data"]["mission_state"] == "idle"


def test_ws_chat_to_plan_confirm_abort_target(env):
    client, ctrl, hub = env
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        ws.send_json({"type": "chat", "text": "search the area"})
        user = recv_until(ws, "chat")
        assert user["data"]["role"] == "user"
        plan_msg = recv_until(ws, "plan")
        plan = models.PlanProposal.model_validate(plan_msg["data"])
        st = recv_until(ws, "status")
        assert st["data"]["mission_state"] == "planned"
        assert st["data"]["plan"]["plan_id"] == plan.plan_id

        ws.send_json({"type": "confirm", "plan_id": plan.plan_id})
        m = recv_until(ws, "chat")
        assert "confirmed" in m["data"]["text"].lower()
        assert recv_until(ws, "status")["data"]["mission_state"] == "staged"
        ws.send_json({"type": "go", "plan_id": plan.plan_id})
        assert "GO" in recv_until(ws, "chat")["data"]["text"]
        assert recv_until(ws, "status")["data"]["mission_state"] == "flying"

        ws.send_json({"type": "target", "text": "child with a dog"})
        m = recv_until(ws, "chat")
        assert "child with a dog" in m["data"]["text"]
        assert recv_until(ws, "status")["data"]["target"]["description"] == "child with a dog"

        ws.send_json({"type": "abort"})
        m = recv_until(ws, "chat")
        assert "ABORT" in m["data"]["text"]
        assert recv_until(ws, "status")["data"]["mission_state"] == "rtl"
    assert ("abort", None) in ctrl.calls


def test_ws_chat_reply_str(env):
    client, _, _ = env
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        ws.send_json({"type": "chat", "text": "hi there"})
        recv_until(ws, "chat")  # user echo
        m = recv_until(ws, "chat")
        assert m["data"]["role"] == "assistant" and "hi there" in m["data"]["text"]


def test_ws_bad_messages(env):
    client, _, _ = env
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        for bad in ['not json', '[1,2]', '{"type":"nope"}', '{"type":"chat"}', '{"type":"chat","text":"  "}',
                    '{"type":"confirm"}']:
            ws.send_text(bad)
            assert recv_until(ws, "error")["data"]["message"]
        ws.send_json({"type": "ping"})
        assert recv_until(ws, "telemetry")


def test_abort_not_blocked_by_slow_chat(env):
    client, ctrl, _ = env
    started = threading.Event()

    async def slow(text):
        started.set()
        await asyncio.sleep(3)
        return "late"

    ctrl.handle_chat = slow  # type: ignore[assignment]
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        ws.send_json({"type": "chat", "text": "slow one"})
        recv_until(ws, "chat")  # user echo
        assert started.wait(2)
        t0 = time.time()
        ws.send_json({"type": "abort"})
        m = recv_until(ws, "chat")
        assert "ABORT" in m["data"]["text"]
        assert time.time() - t0 < 2.0
        assert ctrl.state == "rtl"


def test_hub_broadcast_reaches_clients(env):
    client, _, hub = env
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        recv_until(a, "status")
        recv_until(b, "status")
        assert hub.client_count == 2
        alert = models.Alert(level="target_match", track_id=7, ts=time.time(), c_yolo=0.9, c_match=0.8,
                             reason="red jacket", image_url="/data/alerts/a.jpg")
        msg = models.WsMessage(type="alert", data=alert.model_dump())
        # Run broadcast on the app's own loop (the TestClient portal), as main.py would.
        client.portal.call(hub.broadcast, msg)
        for ws in (a, b):
            got = recv_until(ws, "alert")
            assert got["data"]["track_id"] == 7 and got["data"]["level"] == "target_match"


def test_hub_broadcast_threadsafe(env):
    client, _, hub = env
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        out = {}

        def worker():
            out["fut"] = hub.broadcast_threadsafe(models.WsMessage(type="telemetry", data={"mode": "AUTO"}))

        th = threading.Thread(target=worker)
        th.start()
        th.join()
        out["fut"].result(timeout=3)
        got = recv_until(ws, "telemetry")
        assert got["data"]["mode"] == "AUTO"


def test_threadsafe_broadcast_without_loop_is_noop():
    assert Hub().broadcast_threadsafe(models.WsMessage(type="status", data={})) is None


def test_dead_client_is_dropped(env):
    client, _, hub = env
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
    # after close the hub must not keep it
    deadline = time.time() + 2
    while hub.client_count and time.time() < deadline:
        time.sleep(0.05)
    assert hub.client_count == 0
    client.portal.call(hub.broadcast, models.WsMessage(type="status", data={}))   # no error


def test_fake_alert_has_servable_image(tmp_path):
    ctrl = FakeController(tmp_path / "d", background=False)
    alert = ctrl.make_alert()
    assert alert.image_url and (tmp_path / "d" / alert.image_url.removeprefix("/data/")).is_file()


def test_captive_portal_redirects_foreign_hosts(settings_factory):
    from webapp.controller import FakeController
    from webapp.server import create_app
    from fastapi.testclient import TestClient
    s = settings_factory(captive_portal=True)
    with TestClient(create_app(FakeController(s.data_dir), s), base_url="http://connectivitycheck.gstatic.com") as c:
        r = c.get("/generate_204", follow_redirects=False)
        assert r.status_code == 302 and r.headers["location"].startswith("http://") and r.headers["location"].endswith("/")
        assert ":8000" not in r.headers["location"]
    with TestClient(create_app(FakeController(s.data_dir), s), base_url="http://10.42.0.1:8000") as c:
        assert c.get("/", follow_redirects=False).status_code == 200          # the app itself is served normally
    off = settings_factory()
    with TestClient(create_app(FakeController(off.data_dir), off), base_url="http://captive.apple.com") as c:
        assert c.get("/", follow_redirects=False).status_code == 200          # off by default


def test_https_mode_redirects_http_and_serves_the_ca(settings_factory, tmp_path):
    from webapp.controller import FakeController
    from webapp.server import create_app
    from fastapi.testclient import TestClient
    s = settings_factory(https=True, captive_portal=True, data_dir=tmp_path)   # never the real data/tls
    (s.data_dir / "tls").mkdir(parents=True, exist_ok=True)
    (s.data_dir / "tls" / "kestrel-ca.crt").write_text("-----BEGIN CERTIFICATE-----\n")
    app = create_app(FakeController(s.data_dir), s)
    with TestClient(app, base_url="http://10.42.0.1:8000") as c:
        r = c.get("/live?x=1", follow_redirects=False)
        assert r.status_code == 302 and r.headers["location"] == "https://10.42.0.1:8443/live?x=1"
        assert c.post("/api/chat", json={"text": "hi"}).status_code == 403   # commands only over HTTPS
        with pytest.raises(Exception):
            with c.websocket_connect("/ws") as ws:
                ws.receive_json()
        r = c.get("/kestrel-ca.crt")                                    # the CA download stays on plain HTTP
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-x509-ca-cert")
    with TestClient(app, base_url="http://captive.apple.com") as c:   # phone probe -> app address -> HTTPS
        r = c.get("/hotspot-detect.html", follow_redirects=False)
        assert r.status_code == 302 and r.headers["location"].startswith("http://")
    with TestClient(app, base_url="https://10.42.0.1:8443") as c:
        assert c.get("/", follow_redirects=False).status_code == 200   # the app itself over HTTPS (wss: checked on hardware)
