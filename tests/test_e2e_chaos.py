"""Hardening E: seeded chaos during a simulated mission (mock ArduCopter, real RealController + web app).

Random WS drops / page refreshes, link mute, LLM failures and Abort at random moments. Asserts: the vehicle ends in
RTL/landed, no unhandled exceptions in the server log, every alert reaches a phone that stayed connected, files written.
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time

import pytest

from test_e2e import CORNERS, Rig
from mock_vehicle import MODE_RTL

# log messages of exceptions the server handles on purpose (the injected failures)
EXPECTED_ERRORS = ("handle_chat failed", "update_target failed", "plan_area failed", "chat failed")


class Phone:
    """A phone: a WS connection drained by a thread; records everything it receives."""

    def __init__(self, client) -> None:
        self.cm = client.websocket_connect("/ws")
        self.ws = self.cm.__enter__()
        self.msgs: list[dict] = []
        self.closed = False
        self.th = threading.Thread(target=self._drain, daemon=True)
        self.th.start()

    def _drain(self) -> None:
        while not self.closed:
            try:
                self.msgs.append(self.ws.receive_json())
            except Exception:
                return

    def send(self, d: dict) -> None:
        try:
            self.ws.send_json(d)
        except Exception:
            pass

    def close(self) -> None:
        self.closed = True
        try:
            self.cm.__exit__(None, None, None)
        except Exception:
            pass
        self.th.join(2)

    def alerts(self) -> list[dict]:
        return [m["data"] for m in list(self.msgs) if m["type"] == "alert"]


@pytest.mark.parametrize("seed", [1, 7])
def test_chaos_mission(tmp_path, caplog, seed):
    rnd = random.Random(seed)
    caplog.set_level(logging.WARNING)
    unraisable: list = []
    old_hook = threading.excepthook
    threading.excepthook = lambda args: unraisable.append(args)
    try:
        with Rig(tmp_path, vision=True, heartbeat_loss_s=2.0, mavlink_timeout_s=2.0, mock_kw={"time_scale": 8.0}) as rig:
            steady = Phone(rig.client)          # stays connected the whole time
            roaming = Phone(rig.client)
            pid = rig.plan(target_text="person in a red jacket")["data"]["plan_id"]
            rig.ack()
            assert "mission started" in rig.fly(pid)            # Confirm (stage) + GO
            real_llm = rig.llm.achat_json
            aborted = False
            events = []
            end = time.time() + 14
            while time.time() < end and not aborted:
                time.sleep(rnd.uniform(0.2, 0.9))
                ev = rnd.choice(["ws_drop", "refresh", "mute", "llm_fail", "chat", "abort", "flood", "garbage", "target"])
                events.append(ev)
                if ev == "ws_drop":
                    roaming.close()
                    time.sleep(rnd.uniform(0, 0.4))
                    roaming = Phone(rig.client)
                elif ev == "refresh":
                    roaming.close()
                    roaming = Phone(rig.client)
                    assert rig.client.get("/api/status").status_code == 200
                    assert rig.client.get("/api/limits").status_code == 200
                elif ev == "mute":
                    rig.mock.mute = True
                    time.sleep(rnd.uniform(0.3, 1.0))      # short: below the heartbeat-loss threshold
                    rig.mock.mute = False
                elif ev == "llm_fail":
                    async def boom(*a, **kw):
                        raise RuntimeError("injected LLM failure")
                    rig.llm.achat_json = boom
                    r = rig.client.post("/api/target", json={"text": "someone in green"})
                    assert r.status_code == 200
                    rig.llm.achat_json = real_llm
                elif ev == "chat":
                    roaming.send({"type": "chat", "text": "status?"})
                elif ev == "target":
                    rig.client.post("/api/target", json={"text": "person in blue"})
                elif ev == "flood":
                    for _ in range(80):
                        roaming.send({"type": "ping"})
                elif ev == "garbage":
                    roaming.send({"type": "plan_area", "corners": "x"})
                    try:
                        roaming.ws.send_bytes(b"\x00\x01")
                    except Exception:
                        pass
                elif ev == "abort":
                    if rnd.random() < 0.5 or time.time() > end - 5:
                        (roaming if rnd.random() < 0.5 else steady).send({"type": "abort"})
                        aborted = True
            if not aborted:
                steady.send({"type": "abort"})
            # vehicle ends RTL / landed
            assert rig.wait(lambda: rig.mock.mode == MODE_RTL or rig.ctrl.state in ("rtl", "landed"), 20), events
            assert rig.wait(lambda: rig.ctrl.state == "landed", 90), (rig.ctrl.state, events)
            time.sleep(1.0)
            # every alert the flight produced reached the phone that stayed connected
            flight_dir = tmp_path / rig.ctrl.last_flight_dir
            logged = [json.loads(l) for l in (flight_dir / "alerts.jsonl").read_text().splitlines() if l.strip()]
            got = steady.alerts()
            assert logged, "no alerts at all; the test did not exercise vision"
            assert {round(a["ts"], 3) for a in logged} <= {round(a["ts"], 3) for a in got}, (len(logged), len(got), events)
            # the roaming phone reconnected successfully and was resynced
            assert roaming.msgs and roaming.msgs[0]["type"] == "status"
            steady.close()
            roaming.close()
            # files written
            for k in ("telemetry", "alerts", "detections"):
                assert (flight_dir / f"{k}.jsonl").read_text().strip(), k
            meta = json.loads((flight_dir / "flight.json").read_text())
            assert meta["end_ts"] and meta["summary"]["end_reason"] in ("landed", "aborted", "rtl")
            assert (tmp_path / "plans" / pid / "plan.json").is_file()
    finally:
        threading.excepthook = old_hook
    bad = [r for r in caplog.records if r.levelno >= logging.ERROR and not any(x in r.getMessage() for x in EXPECTED_ERRORS)]
    assert not bad, [(r.name, r.getMessage()[:200]) for r in bad]
    assert not [u for u in unraisable if "injected" not in str(u.exc_value)], unraisable


def test_abort_latency_real_controller_under_load(tmp_path):
    """Abort reaches the controller < 300 ms while an upload (confirm), a vehicle check and chat run."""
    with Rig(tmp_path, vision=True, mock_kw={"time_scale": 4.0}) as rig:
        phone = Phone(rig.client)
        pid = rig.plan()["data"]["plan_id"]
        rig.ack()
        hit = {}
        orig = rig.ctrl.abort

        async def timed():
            hit.setdefault("t", time.monotonic())
            return await orig()

        rig.ctrl.abort = timed
        threads = [threading.Thread(target=lambda: rig.client.post("/api/confirm", json={"plan_id": pid})),
                   threading.Thread(target=lambda: rig.client.get("/api/vehicle/check")),
                   threading.Thread(target=lambda: rig.client.get("/api/preflight")),
                   threading.Thread(target=lambda: rig.client.post("/api/chat", json={"text": "hello"}))]
        for t in threads:
            t.start()
        time.sleep(0.4)                       # upload / checks in flight
        t0 = time.monotonic()
        phone.send({"type": "abort"})
        assert rig.wait(lambda: "t" in hit, 2.0), "abort never reached the controller"
        lat = hit["t"] - t0
        print("abort latency (RealController, upload+check+chat running) ms:", round(lat * 1000, 1))
        for t in threads:
            t.join(60)
        assert lat < 0.3
        phone.send({"type": "abort"})         # make sure the mock ends RTL/landed
        try:                                  # Confirm only stages (go-flow): an aborted upload ends planned/idle
            assert rig.wait(lambda: rig.ctrl.state in ("rtl", "landed", "idle", "planned") and not rig.mock.armed
                            or rig.mock.mode == MODE_RTL, 30), rig.ctrl.state
            assert not rig.mock.armed and 400 not in rig.mock.commands_received
        finally:
            phone.close()
