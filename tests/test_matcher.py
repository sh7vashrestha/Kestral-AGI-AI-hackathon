from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from llm_client import LLMUnavailable
from matcher import (ClipBackend, Job, JobQueue, Matcher, VlmBackend, build_messages, make_backend, make_crop,
                     score)
from models import ClipPrompts
from models import Alert, BBox, Detection, MatchResult, Target
from storage import FlightStorage

FRAME = np.full((480, 640, 3), 128, np.uint8)


class FakeClient:
    def __init__(self, results=None, latency=0.0, exc=None):
        self.results = results if results is not None else [MatchResult(is_human=True, matches=True, match_confidence=0.95, reason="ok")]
        self.latency = latency
        self.exc = exc
        self.calls = []
        self.lock = threading.Lock()

    def chat_json(self, messages, model_cls, *, max_tokens=None, **kw):
        with self.lock:
            i = len(self.calls)
            self.calls.append(messages)
        time.sleep(self.latency)
        if self.exc:
            raise self.exc
        r = self.results[min(i, len(self.results) - 1)]
        return r(messages) if callable(r) else r


def det(x1=100, y1=100, x2=160, y2=220, fid=0, ts=0.0, c=0.9):
    return Detection(frame_id=fid, ts=ts, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), c_yolo=c)


def mr(human=True, match=True, conf=0.9, reason="r"):
    return MatchResult(is_human=human, matches=match, match_confidence=conf, reason=reason)


class Harness:
    def __init__(self, tmp_path, settings_factory, client, **over):
        self.s = settings_factory(data_dir=tmp_path, **over)
        self.storage = FlightStorage(self.s)
        self.alerts: list[Alert] = []
        self.now = 1000.0
        self.m = Matcher(self.s, client, self.storage, self.alerts.append, clock=lambda: self.now)
        self.m.update_target(Target(description="man in red jacket", attributes=["red jacket"]))

    def wait(self, pred, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            if pred():
                return True
            time.sleep(0.01)
        return False

    def process(self, n_done):
        return self.wait(lambda: self.m.stats()["processed"] >= n_done)


@pytest.fixture
def mk(tmp_path, settings_factory):
    made = []

    def f(client, **over):
        h = Harness(tmp_path, settings_factory, client, **over)
        made.append(h)
        return h
    yield f
    for h in made:
        h.m.stop(1.0)


# ---------- crop ----------
def test_crop_padding_clamp_upscale(settings_factory):
    s = settings_factory(crop_padding=0.2, crop_min_px=224)
    big = np.zeros((1000, 1000, 3), np.uint8)
    c = make_crop(big, BBox(x1=400, y1=400, x2=600, y2=700), s)
    assert c.shape[:2] == (420, 280)                       # 300*1.4 x 200*1.4, no upscale
    c = make_crop(big, BBox(x1=0, y1=0, x2=250, y2=250), s)  # clamped at the top-left
    assert c.shape[:2] == (300, 300)
    c = make_crop(big, BBox(x1=980, y1=980, x2=1100, y2=1100), s)
    assert c.shape[0] >= 224 and c.shape[1] >= 224
    c = make_crop(big, BBox(x1=500, y1=500, x2=530, y2=590), s)   # small: 42x126 -> short 224
    assert min(c.shape[:2]) == 224 and abs(c.shape[0] / c.shape[1] - 126 / 42) < 0.05
    with pytest.raises(ValueError):
        make_crop(big, BBox(x1=5, y1=5, x2=5, y2=9), s)


def test_prompt_contains_target():
    msgs = build_messages(Target(description="woman in blue", attributes=["backpack"]), b"\xff\xd8x")
    txt = msgs[1]["content"][0]["text"]
    assert "woman in blue" in txt and "backpack" in txt and msgs[0]["role"] == "system"


# ---------- scoring ----------
def test_score_rules():
    assert score(0.9, mr(conf=0.8)) == (0.8, pytest.approx(0.72))
    assert score(0.9, mr(human=False, conf=0.9)) == (0.0, 0.0)
    assert score(0.9, mr(match=False, conf=0.9)) == (0.0, 0.0)


def test_person_alert_immediate_and_once(mk):
    h = mk(FakeClient(latency=10))   # VLM never answers in time; not started anyway
    h.m.submit(FRAME, [det(fid=1, ts=1)])
    assert [a.level for a in h.alerts] == ["person"]
    a = h.alerts[0]
    assert a.image_url.startswith("/data/flights/") and a.track_id == 1
    h.m.submit(FRAME, [det(fid=2, ts=2)])
    assert len(h.alerts) == 1                      # same person, no second alert
    h.m.submit(FRAME, [det(x1=400, x2=460, fid=3, ts=3)])
    assert len(h.alerts) == 2 and h.alerts[1].track_id == 2


def test_low_yolo_ignored(mk):
    h = mk(FakeClient())
    h.m.submit(FRAME, [det(c=0.3)])
    assert h.alerts == []


def test_target_match_needs_min_frames_and_avg(mk):
    # c_yolo .9 * .95 = .855 >= .7 ; needs 2 results
    c = FakeClient([mr(conf=0.95)])
    h = mk(c, min_frames=2, match_threshold=0.7)
    h.m.start()
    h.m.submit(FRAME, [det(fid=1, ts=1)])
    assert h.process(1)
    assert [a.level for a in h.alerts] == ["person"]
    h.m.submit(FRAME, [det(fid=2, ts=2)])
    assert h.process(2) and h.wait(lambda: len(h.alerts) == 2)
    t = h.alerts[1]
    assert t.level == "target_match" and t.n_frames == 2
    assert t.c_avg == pytest.approx(0.855) and t.c_final == pytest.approx(0.855)
    assert t.target_description == "man in red jacket"


def test_avg_below_threshold_no_alert(mk):
    c = FakeClient([mr(conf=0.95), mr(match=False, conf=0.9)])   # .855 and 0 -> avg .43
    h = mk(c, min_frames=2)
    h.m.start()
    for i in (1, 2):
        h.m.submit(FRAME, [det(fid=i, ts=i)])
        assert h.process(i)
    assert [a.level for a in h.alerts] == ["person"]


def test_once_per_target_unless_improved_and_rate_limit(mk):
    c = FakeClient([mr(conf=0.8), mr(conf=0.8), mr(conf=0.8), mr(conf=1.0), mr(conf=1.0), mr(conf=1.0)])
    h = mk(c, min_frames=2, match_threshold=0.6, alert_min_interval_s=5.0)
    h.m.start()

    def feed(i):
        h.m.submit(FRAME, [det(fid=i, ts=i)])
        assert h.process(i)
    feed(1); feed(2)                                   # avg .72 -> alert
    n_target = lambda: sum(a.level == "target_match" for a in h.alerts)
    assert n_target() == 1
    feed(3)                                            # same avg, no repeat
    assert n_target() == 1
    feed(4)                                            # avg improves (.9*(.8,.8,.8,1)=.774)< +0.1 -> no
    assert n_target() == 1
    feed(5); feed(6)                                   # avg 0.846 ; improved>=.1 but rate-limited (now unchanged)
    assert n_target() == 1
    h.now += 6
    feed(7)
    assert n_target() == 2


def test_target_update_resets_history(mk):
    c = FakeClient([mr(conf=0.95)])
    h = mk(c, min_frames=2)
    h.m.start()
    for i in (1, 2):
        h.m.submit(FRAME, [det(fid=i, ts=i)])
        assert h.process(i)
    assert sum(a.level == "target_match" for a in h.alerts) == 1
    h.m.update_target(Target(description="child in green"))
    assert all(len(t.match_hist) == 0 for t in h.m.tracker.tracks.values())
    h.m.submit(FRAME, [det(fid=3, ts=3)])
    assert h.process(3)
    assert sum(a.level == "target_match" for a in h.alerts) == 1      # only 1 result under the new target
    assert "child in green" in c.calls[-1][1]["content"][0]["text"]
    h.m.submit(FRAME, [det(fid=4, ts=4)])
    assert h.process(4) and h.wait(lambda: sum(a.level == "target_match" for a in h.alerts) == 2)
    assert h.alerts[-1].target_description == "child in green"


def test_stale_result_discarded_after_target_change(mk):
    c = FakeClient(latency=0.3)
    h = mk(c, min_frames=1)
    h.m.start()
    h.m.submit(FRAME, [det(fid=1, ts=1)])
    assert h.wait(lambda: len(c.calls) == 1)
    h.m.update_target(Target(description="other"))
    assert h.wait(lambda: h.m.stats()["stale"] == 1)
    assert [a.level for a in h.alerts] == ["person"]


# ---------- queue ----------
def test_queue_bounded_best_kept():
    q = JobQueue(2)
    mkj = lambda tid, sc: Job(tid, sc, FRAME[:2, :2], 0.9, 0.0, None, det())
    assert q.offer(mkj(1, 10)) and q.offer(mkj(1, 5)) is False and q.offer(mkj(1, 20))
    q.offer(mkj(2, 1)); q.offer(mkj(3, 1))
    assert len(q) == 2 and q.get(0).track_id == 2          # track 1 (oldest) dropped
    q2 = JobQueue(3)
    q2.offer(mkj(1, 10)); q2.offer(mkj(1, 30)); q2.offer(mkj(1, 20))
    assert q2.get(0).score == 30


def test_submit_never_blocks(mk):
    c = FakeClient(latency=0.5)
    h = mk(c, vlm_queue_size=3)
    h.m.start()
    t0 = time.monotonic()
    for i in range(50):
        # 8 distinct people spread over the frame, growing box over time for track 0
        dets = [det(x1=10 + 70 * k, y1=100, x2=60 + 70 * k, y2=200 + (i if k == 0 else 0), fid=i, ts=i * 0.01) for k in range(8)]
        h.m.submit(FRAME, dets)
        assert h.m.stats()["queue_depth"] <= 3
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, elapsed
    st = h.m.stats()
    assert st["queue_depth"] <= 3 and st["dropped"] > 0


def test_best_crop_kept_for_track(mk):
    c = FakeClient(latency=0.0)
    h = mk(c)                                   # worker not started: jobs stay pending
    for i, (x2, conf) in enumerate([(160, 0.85), (200, 0.95), (170, 0.9)]):
        h.m.submit(FRAME, [det(x2=x2, fid=i, ts=i, c=conf)])
    job = h.m.queue.get(0)
    assert job.det.bbox.x2 == 200 and job.crop_path and "f000001" in job.crop_path


# ---------- resilience ----------
def test_vlm_errors_do_not_crash_worker(mk):
    c = FakeClient(exc=LLMUnavailable("down"))
    h = mk(c)
    h.m.start()
    h.m.submit(FRAME, [det(fid=1, ts=1)])
    assert h.wait(lambda: h.m.stats()["errors"] >= 1)
    c.exc = None
    h.m.submit(FRAME, [det(fid=2, ts=2)])
    assert h.process(1, ) is True
    assert h.m._thread.is_alive()


def test_on_alert_exception_and_garbage_frame(mk):
    h = mk(FakeClient())
    h.m.on_alert = lambda a: 1 / 0
    h.m.submit(FRAME, [det()])                       # must not raise
    h.m.submit(FRAME, [det(x1=2000, x2=2100)])       # outside the image: logged, not raised


def test_stop_joins(mk):
    h = mk(FakeClient())
    h.m.start()
    h.m.stop(2.0)
    assert not h.m._thread.is_alive()


def test_records_written(mk):
    h = mk(FakeClient(), min_frames=1)
    h.m.start()
    h.m.submit(FRAME, [det(fid=1, ts=1)])
    assert h.process(1) and h.wait(lambda: len(h.alerts) == 2)
    root = h.storage.root
    assert len((root / "detections.jsonl").read_text().splitlines()) == 1
    assert len((root / "matches.jsonl").read_text().splitlines()) == 1
    assert len((root / "alerts.jsonl").read_text().splitlines()) == 2
    assert list((root / "frames").glob("*.jpg")) and list((root / "crops").glob("*.jpg"))


# ---------- backends ----------
class FakeEncoder:
    """Image 'embedding' = mean colour direction; texts map by keyword."""
    def __init__(self):
        self.text_calls = 0

    def encode_image(self, images):
        out = []
        for im in images:
            v = im.reshape(-1, 3).mean(0).astype(np.float32) + 1e-6
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    def encode_text(self, texts):
        self.text_calls += 1
        m = {"red": [0, 0, 1], "blue": [1, 0, 0]}
        out = []
        for t in texts:
            v = next((np.array(c, np.float32) for k, c in m.items() if k in t), np.array([0, 1, 0], np.float32))
            out.append(v)
        return np.stack(out)


def test_clip_backend_scoring():
    enc = FakeEncoder()
    b = ClipBackend(enc)
    t = Target(description="man in red jacket")
    b.prepare(t)
    red = np.zeros((50, 50, 3), np.uint8); red[..., 2] = 200
    blue = np.zeros((50, 50, 3), np.uint8); blue[..., 0] = 200
    r = b.match(red, t)
    assert r.is_human and r.matches and r.match_confidence > 0.9 and r.reason.startswith("CLIP:")
    r2 = b.match(blue, t)
    assert not r2.matches and r2.match_confidence < 0.5


def test_clip_custom_prompts_and_lazy_prepare():
    enc = FakeEncoder()
    b = ClipBackend(enc)
    b.set_prompts(ClipPrompts(target_description="x", positive=["red shirt"], negative=["blue shirt"]))
    red = np.zeros((8, 8, 3), np.uint8); red[..., 2] = 255
    assert b.match(red, Target(description="x")).match_confidence == pytest.approx(1.0, abs=1e-3)
    n = enc.text_calls
    b.match(red, Target(description="x"))
    assert enc.text_calls == n            # prompts cached
    b.match(red, Target(description="other red"))
    assert enc.text_calls == n + 1        # re-prepared for a new target


def test_make_backend(settings_factory):
    assert isinstance(make_backend(settings_factory(match_backend="clip"), encoder=FakeEncoder()), ClipBackend)
    assert isinstance(make_backend(settings_factory(match_backend="vlm"), client=FakeClient()), VlmBackend)
    with pytest.raises(ValueError):
        make_backend(settings_factory(match_backend="clip"))


def test_matcher_with_clip_backend_end_to_end(tmp_path, settings_factory):
    s = settings_factory(data_dir=tmp_path, min_frames=2)
    alerts = []
    m = Matcher(s, ClipBackend(FakeEncoder()), FlightStorage(s), alerts.append)
    m.update_target(Target(description="man in red jacket"))
    m.start()
    red = np.zeros((480, 640, 3), np.uint8); red[..., 2] = 200
    try:
        for i in (1, 2):
            m.submit(red, [det(fid=i, ts=i)])
            end = time.time() + 3
            while m.stats()["processed"] < i and time.time() < end:
                time.sleep(0.01)
        assert [a.level for a in alerts] == ["person", "target_match"]
        assert alerts[1].reason.startswith("CLIP")
    finally:
        m.stop(1.0)


def test_vlm_confirm_top(tmp_path, settings_factory):
    s = settings_factory(data_dir=tmp_path, min_frames=1, vlm_confirm_top=True)
    alerts = []
    vlm = FakeClient([mr(conf=0.9, reason="red jacket visible")])
    m = Matcher(s, ClipBackend(FakeEncoder()), FlightStorage(s), alerts.append, confirm_backend=VlmBackend(vlm))
    m.update_target(Target(description="man in red jacket"))
    m.start()
    red = np.zeros((480, 640, 3), np.uint8); red[..., 2] = 200
    try:
        for i in (1, 2):
            m.submit(red, [det(fid=i, ts=i)])
            time.sleep(0.3)
        end = time.time() + 3
        while len(alerts) < 3 and time.time() < end:
            time.sleep(0.01)
        assert [a.level for a in alerts] == ["person", "target_match", "target_match"]
        assert alerts[2].reason == "red jacket visible"
        assert len(vlm.calls) == 1               # once per track/target
    finally:
        m.stop(1.0)
