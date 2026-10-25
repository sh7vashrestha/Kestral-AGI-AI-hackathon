from __future__ import annotations

import numpy as np
import pytest

from fusion import Fuser, default_homography, fuse_score, load_homography, project_bbox, save_homography
from models import BBox, Detection, Telemetry
from test_thermal import scene


class FakeDetector:
    def __init__(self, roi_result=None):
        self.roi_calls = []
        self.roi_result = roi_result or []

    def detect(self, frame, ts=None, frame_id=0, telemetry=None):
        return []

    def detect_roi(self, frame, roi, ts=None, frame_id=0, telemetry=None, conf=None):
        self.roi_calls.append((roi, conf))
        return [d.model_copy(deep=True) for d in self.roi_result]


@pytest.fixture
def frame():
    return np.full((1080, 1920, 3), 110, np.uint8)


def det_over(fuser, tbox, c=0.5):
    return Detection(frame_id=1, ts=100.0, bbox=fuser.to_rgb(BBox(x1=tbox[0], y1=tbox[1], x2=tbox[2], y2=tbox[3])), c_yolo=c)


def test_homography_round_trip(settings):
    f = Fuser(settings)
    b = BBox(x1=10, y1=20, x2=50, y2=70)
    back = f.to_thermal(f.to_rgb(b))
    assert (back.x1, back.y1, back.x2, back.y2) == pytest.approx((10, 20, 50, 70), abs=1e-6)
    H = default_homography(settings)
    c = project_bbox(H, BBox(x1=80, y1=60, x2=80, y2=60))
    assert (c.x1, c.y1) == pytest.approx((960, 540))


def test_boosted_with_heat(settings, frame):
    f = Fuser(settings, detector=FakeDetector())
    d = det_over(f, (78, 48, 88, 62), 0.5)
    res = f.process(frame, 100.0, 1, None, scene(), 100.0, detections=[d])
    assert res.thermal_used and d.c_yolo == 0.5
    assert d.c_heat == 1.0 and d.c_person == pytest.approx(fuse_score(0.5, 1.0, 0.5)) == pytest.approx(0.75)
    assert res.alerts == []


def test_no_heat_unchanged(settings, frame):
    f = Fuser(settings, detector=FakeDetector())
    d = det_over(f, (10, 10, 25, 30), 0.6)
    t = scene(blob=(0, 0, 0, 0))
    res = f.process(frame, 100.0, 1, None, t, 100.0, detections=[d])
    assert d.c_heat == 0.0 and d.c_person == pytest.approx(0.6)


def test_stale_thermal_ignored(settings, frame):
    f = Fuser(settings, detector=FakeDetector())
    d = det_over(f, (78, 48, 88, 62), 0.5)
    res = f.process(frame, 100.0, 1, None, scene(), 98.0, detections=[d])
    assert not res.thermal_used and d.c_person is None and d.c_heat is None and res.alerts == []


def test_heat_only_rechecks_then_alerts(settings, frame):
    det = FakeDetector()
    f = Fuser(settings, detector=det)
    tel = Telemetry(lat=47.0, lon=8.0, alt_rel_m=5.0, heading_deg=0.0)
    res = f.process(frame, 100.0, 1, tel, scene(), 100.0, detections=[])
    assert len(det.roi_calls) == 1 and det.roi_calls[0][1] == settings.thermal_recheck_conf
    assert len(res.alerts) == 1
    a = res.alerts[0]
    assert a.level == "heat" and a.lat is not None and a.c_final == 1.0
    # rate limited at the same ground position
    res2 = f.process(frame, 101.0, 2, tel, scene(), 101.0, detections=[])
    assert res2.alerts == []


def test_heat_only_roi_finds_person(settings, frame):
    found = Detection(frame_id=1, ts=100.0, bbox=BBox(x1=900, y1=500, x2=960, y2=600), c_yolo=0.4)
    f = Fuser(settings, detector=FakeDetector([found]))
    res = f.process(frame, 100.0, 1, None, scene(), 100.0, detections=[])
    assert res.alerts == [] and len(res.detections) == 1
    d = res.detections[0]
    assert d.c_yolo == 0.4 and d.c_person == pytest.approx(fuse_score(0.4, 1.0, 0.5))


def test_dark_mode_skips_yolo_but_alerts(settings):
    dark = np.full((1080, 1920, 3), 3, np.uint8)
    det = FakeDetector()
    f = Fuser(settings, detector=det)
    d = Detection(frame_id=1, ts=100.0, bbox=BBox(x1=0, y1=0, x2=10, y2=10), c_yolo=0.9)
    res = f.process(dark, 100.0, 1, None, scene(), 100.0, detections=[d])
    assert res.dark and res.detections == [] and det.roi_calls == [] and len(res.alerts) == 1


def test_stuck_pixel_no_heat_alert(settings, frame):
    f = Fuser(settings, detector=FakeDetector())
    t = scene(blob=(0, 0, 0, 0))
    t[64, 67] = 40.0
    res = f.process(frame, 100.0, 1, None, t, 100.0, detections=[])
    assert res.alerts == [] and res.hotspots == []


def test_calibration_roundtrip(settings, tmp_path):
    s = settings.model_copy(update={"data_dir": tmp_path})
    H = default_homography(s) * 1.0
    save_homography(s, H, 1.5, 6)
    assert np.allclose(load_homography(s), H)
    assert np.allclose(load_homography(settings.model_copy(update={"data_dir": tmp_path / "x"})), default_homography(settings))
