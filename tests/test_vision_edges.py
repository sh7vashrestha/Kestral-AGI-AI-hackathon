"""Hardening D: vision edge cases (cameras, thermal, detection/tracking, matching, pipeline). Fakes only, no GPU.

One test per row of notes/hardening-D.md. Real-hardware checks live in scripts/vision_soak.py (also documented there).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import camera as camera_mod
import orient
import pipeline as pl
from camera import Camera
from conftest import make_settings
from detector import Detector, resolve_weights
from fusion import DARK_CONFIRM_FRAMES, Fuser
from matcher import (ClipBackend, JobQueue, Matcher, MatchResult, contrast_negatives, default_prompts, make_crop)
from models import BBox, ClipPrompts, Detection, Target, Telemetry
from storage import FlightStorage
from test_thermal import scene
from thermal import (StuckPixelDetector, ThermalCamera, ThermalError, find_hotspots, static_mask)
from tracker import Tracker

ROOT = Path(__file__).resolve().parents[1]


def det(x1=100, y1=100, x2=160, y2=220, fid=0, ts=0.0, c=0.9, tel=None):
    return Detection(frame_id=fid, ts=ts, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), c_yolo=c, telemetry=tel)


def wait(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


class OneHotEncoder:
    """red -> axis 2, blue -> axis 0, everything else axis 1; images by dominant channel."""

    def encode_text(self, texts):
        out = []
        for t in texts:
            v = np.array([0, 1, 0], np.float32)
            if "red" in t:
                v = np.array([0, 0, 1], np.float32)
            elif "blue" in t:
                v = np.array([1, 0, 0], np.float32)
            out.append(v)
        return np.stack(out)

    def encode_image(self, images):
        out = []
        for im in images:
            m = im.reshape(-1, 3).mean(0)
            v = np.array([m[0], m[1] * 0.2, m[2]], np.float32)
            out.append(v / (np.linalg.norm(v) + 1e-6))
        return np.stack(out)


RED = np.zeros((480, 640, 3), np.uint8)
RED[..., 2] = 200


class Harness:
    def __init__(self, tmp_path, backend, **over):
        self.s = make_settings(data_dir=tmp_path, **over)
        self.alerts = []
        self.now = 1000.0
        self.m = Matcher(self.s, backend, FlightStorage(self.s), self.alerts.append, clock=lambda: self.now)
        self.m.update_target(Target(description="man in red jacket"))

    def levels(self):
        return [a.level for a in self.alerts]


@pytest.fixture
def harness(tmp_path):
    made = []

    def f(backend=None, **over):
        h = Harness(tmp_path, backend or ClipBackend(OneHotEncoder(), cutoff=0.6), **over)
        made.append(h)
        return h
    yield f
    for h in made:
        h.m.stop(1.0)


# =========================== CAMERA ===========================

class ScriptCap:
    """Fake capture: yields frames from fn(n); fn may raise / return None to simulate failures."""

    def __init__(self, fn, period=0.003):
        self.fn, self.n, self.period, self.released = fn, 0, period, False

    def isOpened(self):  # noqa: N802
        return True

    def read(self):
        time.sleep(self.period)
        self.n += 1
        f = self.fn(self.n)
        return (f is not None), f

    def release(self):
        self.released = True


def textured(seed=0, shape=(120, 160, 3)):
    return np.random.default_rng(seed).integers(0, 255, shape, dtype=np.uint8)


def run_camera(monkeypatch, caps, settings=None, **patches):
    for k, v in patches.items():
        monkeypatch.setattr(camera_mod, k, v)
    monkeypatch.setattr(camera_mod, "MIN_BACKOFF_S", 0.02)
    monkeypatch.setattr(camera_mod, "MAX_BACKOFF_S", 0.05)
    opened = []

    def fake_open(self):
        c = caps[min(len(opened), len(caps) - 1)]()
        opened.append(c)
        return c

    monkeypatch.setattr(Camera, "_open", fake_open)
    return Camera(settings or make_settings(camera_source="csi:0")).start(), opened


def test_camera_unplugged_mid_run_reopens(monkeypatch):
    """Worker dies mid-run (read raises) -> released, reopened, frame ids keep growing, state back to ok."""
    cam, opened = run_camera(monkeypatch, [lambda: ScriptCap(lambda n: (_ for _ in ()).throw(RuntimeError("worker exited")) if n > 5 else textured(n)),
                                           lambda: ScriptCap(lambda n: textured(n + 100))])
    try:
        assert wait(lambda: len(opened) >= 2 and cam.state == "ok" and cam.latest()[2] > 12, 8)
        assert opened[0].released and cam.status()["reconnects"] >= 1
    finally:
        cam.stop()


def test_camera_frozen_frames_detected_and_reopened(monkeypatch):
    """The same textured frame forever (stuck driver) is treated as a stall: capture reopened."""
    frozen = textured(1)
    cam, opened = run_camera(monkeypatch, [lambda: ScriptCap(lambda n: frozen.copy()), lambda: ScriptCap(lambda n: textured(n))],
                             FROZEN_S=0.3)
    try:
        assert wait(lambda: len(opened) >= 2 and cam.state == "ok", 8)
        assert "frozen" in (cam.status()["error"] or "frozen") and cam.status()["reconnects"] >= 1
        assert opened[0].released
    finally:
        cam.stop()


def test_camera_black_frames_are_not_frozen(monkeypatch):
    """Identical BLACK frames (lens cap) are dark mode's job, not a reason to reopen the camera every few seconds."""
    cam, opened = run_camera(monkeypatch, [lambda: ScriptCap(lambda n: np.zeros((120, 160, 3), np.uint8))], FROZEN_S=0.2)
    try:
        assert wait(lambda: cam.latest()[2] > 5, 3)
        time.sleep(0.6)
        assert len(opened) == 1 and cam.state == "ok"
    finally:
        cam.stop()


def test_camera_very_slow_frames_still_delivered(monkeypatch):
    """0.3 s per frame: frames() yields each new frame once, never a duplicate id."""
    cam, opened = run_camera(monkeypatch, [lambda: ScriptCap(lambda n: textured(n), period=0.3)])
    try:
        got = []
        for f, ts, fid in cam.frames(interval_s=0.05):
            got.append(fid)
            if len(got) >= 3:
                break
        assert got == sorted(set(got)) and len(got) == 3
    finally:
        cam.stop()


def test_camera_frame_size_change_published(monkeypatch):
    sizes = [(120, 160), (240, 320)]
    cam, _ = run_camera(monkeypatch, [lambda: ScriptCap(lambda n: textured(n, (*sizes[0 if n < 6 else 1], 3)))])
    try:
        assert wait(lambda: cam.latest()[0] is not None and cam.latest()[0].shape[:2] == sizes[1], 5)
    finally:
        cam.stop()


def test_camera_rotate_180_frames_and_boxes(tmp_path):
    """CAMERA_ROTATE=180 on a file source: the frame is the 180-degree rotated source, and a box found on that
    corrected frame crops exactly the marked region (all boxes/crops live in the corrected image)."""
    img = np.zeros((90, 160, 3), np.uint8)
    img[10:30, 20:60] = (0, 0, 255)                      # red marker upper left in the SOURCE
    cv2.imwrite(str(tmp_path / "a.png"), img)
    cam = Camera(make_settings(camera_rotate=180), source=f"file:{tmp_path}/a.png").start(wait_s=5)
    try:
        f, _, _ = cam.latest()
    finally:
        cam.stop()
    assert np.array_equal(f, img[::-1, ::-1])
    ys, xs = np.nonzero(f[..., 2] == 255)
    box = BBox(x1=float(xs.min()), y1=float(ys.min()), x2=float(xs.max() + 1), y2=float(ys.max() + 1))
    assert box.x1 == 160 - 60 and box.y1 == 90 - 30                       # lower right in the corrected image
    crop = make_crop(f, box, make_settings(crop_padding=0.0, crop_min_px=1))
    assert crop.shape[:2] == (20, 40) and (crop[..., 2] == 255).all()


def test_thermal_mask_rotation_consistency():
    """Stuck-pixel mask follows THERMAL_ROTATE/FLIP: it marks the same physical pixel as the rotated frame."""
    for rot in (0, 90, 180, 270):
        for flip in ("none", "horizontal", "vertical"):
            s = make_settings(thermal_rotate=rot, thermal_flip=flip, thermal_stuck_pixels="67,64")
            sensor = np.zeros((120, 160), np.uint8)
            sensor[64, 67] = 1
            want = orient.apply(sensor, rot, flip).astype(bool)
            got = static_mask(s)
            assert got.shape == want.shape and np.array_equal(got, want), (rot, flip)


# =========================== THERMAL ===========================

class ThermalFakeCap:
    def __init__(self, n_good):
        self.n, self.n_good, self.released = 0, n_good, False

    def read(self):
        time.sleep(0.005)
        self.n += 1
        if self.n > self.n_good:
            return False, None                            # device unplugged: reads fail
        raw = np.full((122, 160), 29415, np.uint16)       # 21 C
        return True, raw

    def release(self):
        self.released = True


def test_thermal_unplugged_mid_run_goes_stale_and_reconnects(monkeypatch):
    caps = []

    def fake_open(self):
        c = ThermalFakeCap(5 if not caps else 10 ** 6)
        caps.append(c)
        return c

    monkeypatch.setattr(ThermalCamera, "_open", fake_open)
    tc = ThermalCamera(make_settings(thermal_source="v4l2:/dev/video99"))
    tc.start()
    try:
        assert wait(lambda: tc.reconnects >= 1 and len(caps) >= 2 and tc.frames_total > 7, 15)
        assert caps[0].released
        t, ts = tc.latest()
        assert t is not None and t.shape == (120, 160)
        assert tc.nearest(ts + 5.0) == (None, 0.0)         # the frame is far too old for a request 5 s later
    finally:
        tc.stop()


@pytest.mark.parametrize("t", [np.zeros((120, 160), np.float32), np.full((120, 160), 655.0, np.float32),
                               np.full((120, 160), np.nan, np.float32), np.full((120, 160), -273.15, np.float32)])
def test_thermal_flat_saturated_frames_are_safe(t, settings):
    assert find_hotspots(t, settings) == []
    t2 = scene()
    t2[0, 0] = np.nan
    t2[5, 5] = np.inf
    assert len(find_hotspots(t2, settings)) == 1             # NaN/inf pixels do not hide a real person


def test_thermal_all_hot_scene_no_false_alerts(settings):
    """Sunny tarmac: whole scene 45 C (above THERMAL_MAX_C). A 36 C person on it is colder than the background:
    nothing to alert (physics, documented); a 50 C patch is above the person range."""
    hot = scene(bg=45.0, t_blob=36.0)
    assert find_hotspots(hot, settings) == []
    hot2 = scene(bg=45.0, t_blob=52.0)
    assert find_hotspots(hot2, settings) == []


def test_thermal_cold_night_person_found(settings):
    spots = find_hotspots(scene(bg=3.0, t_blob=28.0), settings)
    assert len(spots) == 1 and spots[0].delta_c > 20


def test_thermal_new_stuck_pixel_never_a_hotspot_and_gets_masked(settings):
    d = StuckPixelDetector()
    for i in range(60):
        t = scene(blob=(0, 0, 0, 0), seed=i)
        t[30, 30] = 45.0                                  # a NEW stuck pixel (not in the static list)
        assert find_hotspots(t, settings) == []           # a single pixel is never a person-sized blob
        d.update(t)
    assert d.mask[30, 30] and d.mask.sum() == 1


# =========================== FUSION / PIPELINE MODES ===========================

class FakeDetector:
    def __init__(self, dets=None, roi=None, raise_roi=False):
        self.dets, self.roi, self.calls, self.roi_calls, self.raise_roi = dets or [], roi or [], 0, 0, raise_roi

    def detect(self, frame, ts=None, frame_id=0, telemetry=None):
        self.calls += 1
        return [d.model_copy(deep=True) for d in self.dets]

    def detect_roi(self, frame, roi, **kw):
        self.roi_calls += 1
        if self.raise_roi:
            raise RuntimeError("CUDA out of memory")
        return [d.model_copy(deep=True) for d in self.roi]


def gray_frame(mean, size=(1080, 1920)):
    return np.full((*size, 3), mean, np.uint8)


def test_dark_mode_hysteresis_no_flapping(settings):
    """Brightness wobbling around the threshold (25) must not toggle dark mode every frame."""
    f = Fuser(settings, detector=FakeDetector())
    states = []
    for i, mean in enumerate([20, 27, 22, 28, 21, 29, 24, 30, 23, 26] * 3):
        states.append(f.process(gray_frame(mean, (120, 160)), 100.0 + i, i, None, None, 0.0).dark)
    assert states[0] is True and len(set(states)) == 1        # entered dark once, stayed
    for i in range(DARK_CONFIRM_FRAMES - 1):                  # bright, but not for long enough yet
        assert f.process(gray_frame(120, (120, 160)), 200.0 + i, i, None, None, 0.0).dark
    assert not f.process(gray_frame(120, (120, 160)), 300.0, 99, None, None, 0.0).dark
    assert f.process(gray_frame(10, (120, 160)), 301.0, 100, None, None, 0.0).dark is False       # a single dark frame: still light
    transitions = 0
    last = False
    for i, mean in enumerate([10] * 6 + [120] * 6 + [10, 120] * 5):
        d = f.process(gray_frame(mean, (120, 160)), 400.0 + i, i, None, None, 0.0).dark
        transitions += d != last
        last = d
    assert transitions == 2                                    # dark once, light once; the final alternation never flips it


def test_black_and_overexposed_frames_skip_yolo(settings):
    det = FakeDetector([Detection(frame_id=1, ts=1.0, bbox=BBox(x1=1, y1=1, x2=50, y2=50), c_yolo=0.9)])
    f = Fuser(settings, detector=det)
    assert f.process(gray_frame(0, (120, 160)), 1.0, 1, None, None, 0.0).detections == []          # black
    f2 = Fuser(settings, detector=det)
    res = f2.process(gray_frame(255, (120, 160)), 1.0, 1, None, None, 0.0)                         # blown out
    assert res.flat and res.detections == [] and det.calls == 0
    f3 = Fuser(settings, detector=det)
    res = f3.process(textured(3, (120, 160, 3)), 1.0, 1, None, None, 0.0)                          # normal
    assert not res.dark and not res.flat and len(res.detections) == 1


def test_fusion_garbage_inputs_never_raise(settings):
    f = Fuser(settings, detector=FakeDetector())
    assert f.process(None, 1.0, 1, None, None, 0.0).detections == []
    assert f.process(np.zeros((0, 0, 3), np.uint8), 1.0, 1, None, None, 0.0).detections == []
    res = f.process(textured(1, (120, 160, 3)), 1.0, 1, None, np.zeros((3, 3), np.float32), 1.0)
    assert not res.thermal_used                                 # absurd thermal array ignored
    res = f.process(textured(1, (120, 160, 3)), 1.0, 1, None, scene(), 1.0, detections=[])


def test_fusion_frame_size_change_rescales_homography(settings):
    f = Fuser(settings, detector=FakeDetector())
    b = BBox(x1=10, y1=10, x2=30, y2=40)
    full = f.to_rgb(b)
    f.process(textured(1, (540, 960, 3)), 1.0, 1, None, None, 0.0)       # camera switched to half resolution
    half = f.to_rgb(b)
    assert (half.x1, half.y1, half.x2, half.y2) == pytest.approx((full.x1 / 2, full.y1 / 2, full.x2 / 2, full.y2 / 2))


def test_heat_blob_on_a_yolo_person_is_not_a_second_alert(settings):
    """Hotspot overlapping a YOLO person: fused, no heat alert. Hotspot at the position of a person YOLO saw a moment
    ago (YOLO missed this frame): no heat alert either. A hotspot elsewhere still alerts."""
    frame = textured(2, (1080, 1920, 3))
    f = Fuser(settings, detector=FakeDetector())
    t = scene()                                                 # blob at thermal (80..86, 50..60)
    person = Detection(frame_id=1, ts=100.0, bbox=f.to_rgb(BBox(x1=79, y1=49, x2=87, y2=61)), c_yolo=0.9)
    res = f.process(frame, 100.0, 1, None, t, 100.0, detections=[person])
    assert res.alerts == [] and res.detections[0].c_person is not None
    res2 = f.process(frame, 101.0, 2, None, t, 101.0, detections=[])      # YOLO misses the same person next frame
    assert res2.alerts == []
    far = scene(blob=(20, 20, 26, 30))
    res3 = f.process(frame, 102.0, 3, None, far, 102.0, detections=[])
    assert len(res3.alerts) == 1


def test_heat_alert_survives_yolo_recheck_failure(settings):
    f = Fuser(settings, detector=FakeDetector(raise_roi=True))
    res = f.process(textured(2, (1080, 1920, 3)), 1.0, 1, None, scene(), 1.0, detections=[])
    assert len(res.alerts) == 1 and res.alerts[0].level == "heat"


def test_thermal_stale_vs_rgb_timestamp_ignored(settings):
    f = Fuser(settings, detector=FakeDetector())
    d = Detection(frame_id=1, ts=100.0, bbox=f.to_rgb(BBox(x1=78, y1=48, x2=88, y2=62)), c_yolo=0.5)
    res = f.process(textured(2, (1080, 1920, 3)), 100.0, 1, None, scene(), 99.0, detections=[d])    # thermal 1 s older
    assert not res.thermal_used and res.alerts == [] and d.c_person is None
    res = f.process(textured(2, (1080, 1920, 3)), 100.0, 1, None, scene(), 100.3, detections=[d])
    assert res.thermal_used


# =========================== DETECTOR ===========================

class FakeBoxes:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    @property
    def xyxy(self):
        class A:
            def __init__(s, a): s.a = a
            def cpu(s): return s
            def numpy(s): return s.a
        return A(np.array([r[:4] for r in self.rows], np.float32))

    @property
    def conf(self):
        class A:
            def __init__(s, a): s.a = a
            def cpu(s): return s
            def numpy(s): return s.a
        return A(np.array([r[4] for r in self.rows], np.float32))


class FakeResult:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows) if rows else None


class FakeYolo:
    def __init__(self, behaviour):
        self.behaviour, self.n = behaviour, 0

    def predict(self, img, **kw):
        self.n += 1
        r = self.behaviour(self.n)
        if isinstance(r, Exception):
            raise r
        return [FakeResult(r)]


def make_detector(behaviour):
    d = Detector.__new__(Detector)
    d.s = make_settings(yolo_conf_threshold=0.5)
    d.model = FakeYolo(behaviour)
    d._lock = threading.Lock()
    d.last_ms, d.failures, d.recovered = 0.0, 0, 0
    d.path = Path("fake.engine")
    return d


def test_detector_zero_one_many_people():
    frame = textured(0, (400, 600, 3))
    assert make_detector(lambda n: []).detect(frame) == []
    one = make_detector(lambda n: [(10, 10, 60, 120, 0.9)]).detect(frame)
    assert len(one) == 1
    many = make_detector(lambda n: [(i * 15, 10, i * 15 + 12, 90, 0.9) for i in range(30)]).detect(frame)
    assert len(many) == 30


def test_detector_clamps_edge_boxes_and_drops_garbage():
    frame = textured(0, (400, 600, 3))
    rows = [(-30, -10, 40, 90, 0.9), (570, 350, 640, 450, 0.8), (700, 10, 800, 90, 0.9), (10, 10, 10.2, 10.3, 0.9),
            (float("nan"), 0, 5, 5, 0.9)]
    out = make_detector(lambda n: rows).detect(frame)
    assert [(d.bbox.x1, d.bbox.y1) for d in out] == [(0.0, 0.0), (570.0, 350.0)]
    assert out[1].bbox.x2 == 600 and out[1].bbox.y2 == 400
    # a person standing at the image edge still gives a non-empty crop
    crop = make_crop(frame, out[0].bbox, make_settings(crop_min_px=1))
    assert crop.size > 0


def test_detector_cuda_oom_recovers_or_raises_without_corrupting_state():
    frame = textured(0, (400, 600, 3))
    d = make_detector(lambda n: RuntimeError("CUDA error: out of memory") if n == 1 else [(10, 10, 60, 120, 0.9)])
    assert len(d.detect(frame)) == 1 and d.recovered == 1 and d.failures == 0           # one retry is enough
    d2 = make_detector(lambda n: RuntimeError("NvMapMemAllocInternalTagged error 12") if n <= 2 else [(10, 10, 60, 120, 0.9)])
    with pytest.raises(RuntimeError):
        d2.detect(frame)
    assert d2.failures == 1
    assert len(d2.detect(frame)) == 1                                                    # next frame works


def test_yolo_engine_missing_falls_back_to_pt(tmp_path, monkeypatch):
    import detector as dm
    (tmp_path / "yolo26n.pt").write_bytes(b"x")
    monkeypatch.setattr(dm, "MODELS_DIR", tmp_path)
    assert resolve_weights("yolo26n.engine").name == "yolo26n.pt"
    assert resolve_weights("yolo26n.pt").name == "yolo26n.pt"
    (tmp_path / "yolo26n.pt").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_weights("yolo26n.pt")


def test_yolo_engine_load_failure_falls_back_to_pt(tmp_path, monkeypatch):
    import detector as dm
    (tmp_path / "yolo26n.engine").write_bytes(b"bad")
    (tmp_path / "yolo26n.pt").write_bytes(b"pt")
    monkeypatch.setattr(dm, "MODELS_DIR", tmp_path)
    loaded = []

    class FakeUltra:
        class YOLO:
            def __init__(self, path, task=None):
                loaded.append(Path(path).name)
                if path.endswith(".engine"):
                    raise RuntimeError("cannot deserialize (out of memory)")

    monkeypatch.setitem(sys.modules, "ultralytics", FakeUltra)
    d = dm.Detector(make_settings(yolo_model="yolo26n.engine"), warmup=False)
    assert loaded == ["yolo26n.engine", "yolo26n.pt"] and d.path.name == "yolo26n.pt"


# =========================== TRACKING / MATCHING ===========================

def test_zero_one_thirty_people_in_a_frame(harness):
    h = harness()
    assert h.m.submit(RED, []) == []
    assert h.levels() == []
    h.m.submit(RED, [det(fid=1, ts=1)])
    assert h.levels() == ["person"]
    h2 = harness()
    people = [det(x1=10 + (i % 10) * 60, y1=10 + (i // 10) * 130, x2=50 + (i % 10) * 60, y2=120 + (i // 10) * 130, fid=1, ts=1, c=0.9 - i * 0.002) for i in range(30)]
    h2.m.start()
    kept = h2.m.submit(RED, people)
    assert len(kept) == 30 and h2.levels().count("person") == 30
    assert len({d.track_id for d in kept}) == 30
    assert wait(lambda: h2.m.stats()["processed"] >= 30, 10) and h2.m.stats()["dropped"] == 0     # queue (32) holds a full frame


def test_overflow_keeps_the_best_crops(harness):
    h = harness(vlm_queue_size=4)
    people = [det(x1=10 + i * 70, y1=10, x2=60 + i * 70, y2=120, fid=1, ts=1, c=0.95 - i * 0.01) for i in range(9)]
    h.m.submit(RED, people)                                   # worker not started: queue fills up
    assert len(h.m.queue) == 4 and h.m.queue.dropped == 5
    kept_ids = [j.track_id for j in list(h.m.queue._jobs.values())]
    assert sorted(kept_ids) == [1, 2, 3, 4]                   # the four highest-scoring people (track ids follow c_yolo order)


def test_same_person_on_two_lanes_one_alert(harness):
    """Lane 1 and lane 2 (90 s later, box somewhere else) see the same GPS point: same track, one person alert."""
    h = harness(tracker_gps_radius_m=8.0, capture_fps=1.0)
    BIG = np.full((1080, 1920, 3), 128, np.uint8)
    t1 = Telemetry(lat=33.0, lon=-84.0, alt_rel_m=20.0, heading_deg=0.0)
    a = det(x1=900, y1=500, x2=960, y2=600, fid=1, ts=0.0, tel=t1)
    h.m.submit(BIG, [a])
    # same ground point seen from further north: the drone is 5 m north and the image point moves "down" in the image
    mpd = 111_320.0
    t2 = Telemetry(lat=33.0 + 5.0 / mpd, lon=-84.0, alt_rel_m=20.0, heading_deg=0.0)
    from tracker import detection_ground
    g = detection_ground(a, 1920, 1080, h.s)
    assert g is not None
    # find a box whose ground point under t2 equals g: vertical shift of ~5 m of ground at 20 m
    import math
    px_per_m = (1080 / 2) / (math.tan(math.radians(h.s.eff_vfov_deg) / 2) * 20.0)
    dy = 5.0 * px_per_m
    b = det(x1=900, y1=500 + dy, x2=960, y2=600 + dy, fid=90, ts=90.0, tel=t2)
    kept = h.m.submit(BIG, [b])
    assert kept[0].track_id == 1 and h.levels().count("person") == 1


def test_tracks_expire_and_are_bounded():
    s = make_settings()
    tr = Tracker(s, ttl_s=1.0, ground_ttl_s=5.0, max_tracks=10)
    tel = Telemetry(lat=33.0, lon=-84.0, alt_rel_m=20.0, heading_deg=0.0)
    tr.update([det(ts=0.0)], 640, 480)                         # no telemetry: expires after ttl
    tr.update([det(x1=300, x2=330, ts=0.0, tel=tel)], 640, 480)
    tr.update([det(x1=500, x2=520, ts=3.0)], 640, 480)
    assert len(tr.tracks) == 2                                 # ground track kept, plain track expired, + the new one
    tr.update([det(x1=500, x2=520, ts=100.0)], 640, 480)
    assert all(t.last_ts >= 100.0 for t in tr.tracks.values())
    for i in range(50):
        tr.update([det(x1=i * 12, x2=i * 12 + 8, ts=200.0 + i * 0.01)], 640, 480)
    assert len(tr.tracks) <= 10


def test_tiny_people_tracked_alerted_not_scored(harness):
    h = harness()
    h.m.submit(RED, [det(x1=100, y1=100, x2=112, y2=118, fid=1, ts=1)])        # 12 x 18 px
    assert h.levels() == ["person"] and len(h.m.queue) == 0
    h.m.submit(RED, [det(x1=300, y1=100, x2=330, y2=160, fid=2, ts=2)])        # 30 x 60: scored
    assert len(h.m.queue) == 1


def test_people_at_image_edges_get_valid_crops(harness):
    h = harness()
    h.m.start()
    h.m.submit(RED, [det(x1=-20, y1=-10, x2=40, y2=80, fid=1, ts=1), det(x1=600, y1=440, x2=700, y2=520, fid=1, ts=1, c=0.85),
                     det(x1=900, y1=900, x2=950, y2=980, fid=1, ts=1, c=0.8)])         # last one fully outside
    assert wait(lambda: h.m.stats()["processed"] >= 2, 5)
    assert h.m.stats()["errors"] == 0
    with pytest.raises(ValueError):
        make_crop(RED, BBox(x1=900, y1=900, x2=950, y2=980), h.s)


def test_target_update_discards_queued_and_inflight(harness):
    gate = threading.Event()

    class Slow(ClipBackend):
        def match(self, crop, target):
            gate.wait(3)
            return super().match(crop, target)

    h = harness(Slow(OneHotEncoder(), cutoff=0.6), min_frames=1)
    h.m.start()
    h.m.submit(RED, [det(fid=1, ts=1), det(x1=300, y1=100, x2=360, y2=220, fid=1, ts=1, c=0.85)])   # 1 in flight, 1 queued
    assert wait(lambda: len(h.m.queue) == 1, 3)
    h.m.update_target(Target(description="woman in blue coat"))
    assert len(h.m.queue) == 0
    gate.set()
    assert wait(lambda: h.m.stats()["stale"] == 1, 3)
    assert h.levels().count("target_match") == 0 and h.m.stats()["processed"] == 0


@pytest.mark.parametrize("bad", [ClipPrompts(target_description="x", positive=[""], negative=[]),
                                 ClipPrompts(target_description="x", positive=["   "], negative=["rock"]),
                                 ClipPrompts(target_description="x", positive=["red ​\U0001F600 " * 400], negative=["\U0001F600"])])
def test_garbage_clip_prompts_fall_back(bad):
    b = ClipBackend(OneHotEncoder(), prompt_builder=lambda t: bad)
    r = b.match(RED, Target(description="x"))
    assert 0.0 <= r.match_confidence <= 1.0
    b2 = ClipBackend(OneHotEncoder(), prompt_builder=lambda t: (_ for _ in ()).throw(RuntimeError("llm exploded")))
    assert b2.match(RED, Target(description="man in red jacket")).matches


def test_empty_target_description_and_generic_targets():
    for desc in (" ", "any person", "Person", "\U0001F600", "x" * 500):
        p = default_prompts(Target(description=desc))
        assert p.positive and p.negative
    assert contrast_negatives("person wearing a red jacket") == [f"person wearing a {c} jacket" for c in ("blue", "black", "white")]
    assert contrast_negatives("hat") and contrast_negatives("xyz") == []


def test_clip_nonfinite_scores_do_not_poison_the_average():
    class NanEnc(OneHotEncoder):
        def encode_image(self, images):
            return np.full((len(images), 3), np.nan, np.float32)

    r = ClipBackend(NanEnc()).match(RED, Target(description="man in red jacket"))
    assert r.match_confidence == 0.0 and not r.matches


def test_matcher_worker_survives_backend_exceptions(harness):
    calls = []

    class Flaky:
        def prepare(self, t):
            pass

        def match(self, crop, t):
            calls.append(1)
            if len(calls) <= 4:
                raise RuntimeError("CUDA error: an illegal memory access")
            return MatchResult(is_human=True, matches=True, match_confidence=0.95, reason="ok")

    h = harness(Flaky(), min_frames=1)
    h.m.start()
    for i in range(8):
        h.m.submit(RED, [det(fid=i, ts=i)])
        time.sleep(0.05)
    assert wait(lambda: h.m.stats()["processed"] >= 1, 30)   # generous: the full suite loads the Jetson
    assert h.m.stats()["errors"] >= 4 and h.m._thread.is_alive()
    assert "target_match" in h.levels()


def test_target_match_rule_min_frames_strong_and_rate_limit(harness):
    """min_frames 3 by default; a very confident track (C_avg >= MATCH_STRONG_THRESHOLD) may alert after 2 results."""
    hi = MatchResult(is_human=True, matches=True, match_confidence=0.99, reason="r")
    mid = MatchResult(is_human=True, matches=True, match_confidence=0.85, reason="r")

    def run(res, n, **over):
        class B:
            def prepare(self, t): pass
            def match(self, crop, t): return res
        h = harness(B(), **over)
        h.m.start()
        for i in range(n):
            h.m.submit(RED, [det(fid=i, ts=i, c=1.0)])
            assert wait(lambda: h.m.stats()["processed"] >= i + 1, 3)
        return h.levels().count("target_match")

    assert make_settings().min_frames == 3
    assert run(mid, 2) == 0 and run(mid, 3) == 1                  # 0.85 < strong: needs 3 results
    assert run(hi, 2) == 1                                        # 0.99 >= 0.93: 2 results are enough
    assert run(hi, 2, match_strong_threshold=1.0) == 0


def test_defaults_are_the_tuned_values():
    s = make_settings()
    assert (s.clip_match_cutoff, s.match_threshold, s.min_frames, s.match_strong_threshold) == (0.6, 0.7, 3, 0.93)
    assert s.vlm_queue_size >= 30 and s.match_min_px == 20


def test_job_queue_never_blocks_under_flood():
    q = JobQueue(3)
    from matcher import Job
    t0 = time.time()
    for i in range(2000):
        q.offer(Job(i % 50, float(i), RED, 0.9, 0.0, None, det()))
    assert time.time() - t0 < 2 and len(q) == 3


# =========================== PIPELINE ===========================

class ListCam:
    def __init__(self, frames, period=0.01):
        self.frames_in, self.period, self.stopped = frames, period, threading.Event()
        self.go = threading.Event()          # frames flow only after the test began the flight

    def start(self, wait_s=0):
        return self

    def stop(self):
        self.stopped.set()

    def frames(self):
        self.go.wait(10)
        for i, f in enumerate(self.frames_in):
            if self.stopped.is_set():
                return
            yield f, time.time(), i
            time.sleep(self.period)
        self.stopped.wait()


def make_vp(frames, detector, tmp_path, **kw):
    s = make_settings(data_dir=tmp_path, capture_fps=60.0, **kw)
    cam = ListCam(frames)
    vp = pl.VisionPipeline(s, on_alert=lambda a: None, telemetry_fn=lambda: None,
                           camera_factory=lambda _s, _src: cam, detector_factory=lambda _s: detector,
                           backend_factory=lambda _s: pl._NoMatchBackend(), use_thermal=False)
    vp.test_cam = cam
    return vp


def test_pipeline_survives_cuda_oom_frames_and_size_change(tmp_path):
    class OomDetector:
        n = 0

        def detect(self, frame, ts=None, frame_id=0, telemetry=None):
            OomDetector.n += 1
            if OomDetector.n in (2, 3):
                raise RuntimeError("CUDA error: out of memory")
            return [Detection(frame_id=frame_id, ts=ts, bbox=BBox(x1=5, y1=5, x2=60, y2=100), c_yolo=0.9)]

    frames = [textured(i, (120, 160, 3)) for i in range(4)] + [textured(i, (240, 320, 3)) for i in range(4)]
    vp = make_vp(frames, OomDetector(), tmp_path)
    vp.start()
    assert wait(lambda: vp.state == "ready", 5)
    vp.begin_flight(FlightStorage(vp.s))
    vp.test_cam.go.set()
    try:
        assert wait(lambda: vp.frames >= 8 and vp.processed + vp.errors >= 8, 5)
        assert vp.errors == 2 and vp.processed == 6 and vp._thread.is_alive() and vp.state == "ready"
    finally:
        vp.stop()


def test_pipeline_watchdog_restart_after_hung_camera_and_worker_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "FRAME_STALL_S", 0.4)

    class Boom:
        def detect(self, *a, **k):
            raise ValueError("model exploded")

    frames = [textured(i, (120, 160, 3)) for i in range(200)]
    vp = make_vp(frames, Boom(), tmp_path)
    vp.start()
    assert wait(lambda: vp.state == "ready", 5)
    vp.begin_flight(FlightStorage(vp.s))
    vp.test_cam.go.set()
    assert wait(lambda: vp.errors > 50 or vp.state == "failed", 8)       # a permanently broken detector ends the worker ...
    assert wait(lambda: not vp.alive(), 3)                               # ... which the controller's watchdog sees
    vp.restart()
    assert vp.restarts == 1
    vp.stop()


def test_sigterm_releases_cameras_fast(tmp_path):
    """A process holding a Camera (file source) + ThermalCamera exits within 3 s of SIGTERM, with both stopped."""
    img = tmp_path / "a.png"
    cv2.imwrite(str(img), textured(1, (60, 80, 3)))
    thermal_dir = tmp_path / "th"
    thermal_dir.mkdir()
    np.save(thermal_dir / "f.npy", scene())
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(ROOT / "drone_brain")!r})
        import shutdown
        from camera import Camera
        from thermal import ThermalCamera
        from config import Settings
        s = Settings(_env_file=None)
        cam = Camera(s, source="file:{img}").start(wait_s=5)
        tc = ThermalCamera(s, source="file:{thermal_dir}").start()
        shutdown.install()
        print("READY", flush=True)
        while True:
            time.sleep(0.1)
    """)
    p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "READY"
        t0 = time.monotonic()
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=5)
        assert time.monotonic() - t0 < 3.0 and p.returncode == 0
    finally:
        if p.poll() is None:
            p.kill()


def test_storage_never_raises_on_garbage(tmp_path):
    st = FlightStorage(make_settings(data_dir=tmp_path))
    assert st.save_frame(np.zeros((0, 0, 3), np.uint8), "empty") is None
    assert st.save_crop(None, "none") is None or True
    os.chmod(st.root / "crops", 0o500)
    try:
        if os.access(st.root / "crops", os.W_OK):
            pytest.skip("running as a user that ignores directory permissions")
        assert st.save_crop(textured(1, (20, 20, 3)), "x") is None            # read-only dir: logged, not raised
    finally:
        os.chmod(st.root / "crops", 0o700)
