from __future__ import annotations

import math

import pytest

from models import BBox, Detection, Telemetry
from tracker import Tracker, ground_distance_m, iou, pixel_to_ground

W, H = 1920, 1080
LAT, LON = 47.0, 8.0


def det(x1, y1, x2, y2, ts=0.0, c=0.9, tel=None, fid=0):
    return Detection(frame_id=fid, ts=ts, bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2), c_yolo=c, telemetry=tel)


def tel(lat=LAT, lon=LON, alt=20.0, hdg=0.0):
    return Telemetry(lat=lat, lon=lon, alt_rel_m=alt, heading_deg=hdg)


def test_iou():
    a = BBox(x1=0, y1=0, x2=10, y2=10)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, BBox(x1=20, y1=20, x2=30, y2=30)) == 0.0
    assert iou(a, BBox(x1=5, y1=0, x2=15, y2=10)) == pytest.approx(50 / 150)


def test_pixel_to_ground_centre_is_drone():
    assert pixel_to_ground(W / 2, H / 2, W, H, LAT, LON, 20, 33, 90, 60) == pytest.approx((LAT, LON))


def _ne(p):
    return (math.radians(p[0] - LAT) * 6371000.0,
            math.radians(p[1] - LON) * 6371000.0 * math.cos(math.radians(LAT)))


def test_pixel_to_ground_offsets():
    # hfov 90 at 20 m: image right edge = 20 m to the right; top edge (vfov 60) = 20*tan30 forward
    right_edge = pixel_to_ground(W, H / 2, W, H, LAT, LON, 20, 0, 90, 60)
    n, e = _ne(right_edge)
    assert (n, e) == pytest.approx((0, 20), abs=0.01)           # heading N: right = east
    top = _ne(pixel_to_ground(W / 2, 0, W, H, LAT, LON, 20, 0, 90, 60))
    assert top == pytest.approx((20 * math.tan(math.radians(30)), 0), abs=0.01)   # forward = north
    n, e = _ne(pixel_to_ground(W, H / 2, W, H, LAT, LON, 20, 90, 90, 60))
    assert (n, e) == pytest.approx((-20, 0), abs=0.01)          # heading E: right = south
    n, e = _ne(pixel_to_ground(W / 2, 0, W, H, LAT, LON, 20, 90, 90, 60))
    assert (n, e) == pytest.approx((0, 20 * math.tan(math.radians(30))), abs=0.01)  # forward = east


def test_ground_distance():
    p = pixel_to_ground(W, H / 2, W, H, LAT, LON, 20, 0, 90, 60)
    assert ground_distance_m((LAT, LON), p) == pytest.approx(20, abs=0.05)


def test_iou_grouping_and_window(settings):
    tr = Tracker(settings)
    (_, t1, new1), = tr.update([det(100, 100, 200, 300, ts=0)], W, H)
    (_, t2, new2), = tr.update([det(110, 105, 210, 305, ts=1)], W, H)
    assert new1 and not new2 and t1.id == t2.id
    (_, t3, new3), = tr.update([det(110, 105, 210, 305, ts=10)], W, H)   # >3 s gap, no GPS
    assert new3 and t3.id != t1.id


def test_two_people_same_frame_distinct(settings):
    tr = Tracker(settings)
    out = tr.update([det(0, 0, 100, 200), det(500, 0, 600, 200)], W, H)
    assert len({t.id for _, t, _ in out}) == 2


def test_gps_grouping(settings):
    tr = Tracker(settings)
    t = tel()
    (_, a, _), = tr.update([det(900, 500, 960, 600, ts=0, tel=t)], W, H)
    # no overlap in image, drone didn't move, but ground position is within 8 m? move detection 100px
    (_, b, new), = tr.update([det(1000, 500, 1060, 600, ts=20, tel=t)], W, H)
    assert not new and b.id == a.id
    # far away (right edge ~ 18 m) -> new track
    (_, c, new), = tr.update([det(1900, 500, 1919, 600, ts=21, tel=t)], W, H)
    assert new


def test_expiry(settings):
    tr = Tracker(settings, ttl_s=10)
    tr.update([det(0, 0, 50, 100, ts=0)], W, H)
    tr.update([det(800, 0, 850, 100, ts=11)], W, H)
    assert len(tr.tracks) == 1 and 1 not in tr.tracks


def test_reset_matches(settings):
    tr = Tracker(settings)
    (_, t, _), = tr.update([det(0, 0, 50, 100)], W, H)
    t.match_hist.append(0.5)
    t.target_alert_avg = 0.8
    tr.reset_matches()
    assert t.match_hist == [] and t.target_alert_avg is None
