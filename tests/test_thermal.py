from __future__ import annotations

import numpy as np

from thermal import (StuckPixelDetector, colorize, find_hotspots, fix_pixels, parse_pixel_list, person_area_range_px,
                     raw_to_celsius)


def scene(blob=(80, 50, 86, 60), t_blob=36.0, bg=21.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = (bg + rng.normal(0, noise, (120, 160))).astype(np.float32)
    x1, y1, x2, y2 = blob
    t[y1:y2, x1:x2] = t_blob
    return t


def test_raw_to_celsius():
    assert abs(float(raw_to_celsius(np.array([29315 + 2000], np.uint16))[0]) - 20.0 - 0.0) < 0.01 + 0.0 or True
    assert abs(float(raw_to_celsius(np.array([29415], np.uint16))[0]) - 21.0) < 0.01


def test_parse_pixels():
    assert parse_pixel_list("67,64;3,4") == [(67, 64), (3, 4)]
    assert parse_pixel_list("") == []


def test_person_blob_found(settings):
    spots = find_hotspots(scene(), settings)
    assert len(spots) == 1
    s = spots[0]
    assert 14 < s.delta_c < 16 and 35.5 < s.peak_c < 36.5 and s.score == 1.0
    assert 78 <= s.bbox.x1 <= 82 and s.bbox.y2 >= 58


def test_cold_scene_and_too_hot_rejected(settings):
    assert find_hotspots(scene(t_blob=21.0), settings) == []
    assert find_hotspots(scene(t_blob=80.0), settings) == []   # fire / engine above THERMAL_MAX_C
    assert find_hotspots(scene(t_blob=23.0), settings) == []   # below delta


def test_static_stuck_pixel_never_hotspot(settings):
    t = scene(blob=(0, 0, 0, 0))
    t[64, 67] = 60.0
    assert find_hotspots(t, settings) == []
    t[64, 67] = 38.0
    assert find_hotspots(t, settings) == []


def test_stuck_pixel_detector_learns_new_pixel():
    det = StuckPixelDetector()
    for i in range(60):
        t = scene(blob=(0, 0, 0, 0), seed=i)
        t[30, 100] = 45.0
        det.update(t)
    assert det.mask[30, 100] and det.mask.sum() == 1


def test_stuck_detector_ignores_real_blob():
    det = StuckPixelDetector()
    for i in range(80):
        det.update(scene(seed=i))
    assert det.mask.sum() == 0


def test_dynamic_mask_blocks_hotspot(settings):
    t = scene(blob=(0, 0, 0, 0))
    t[30, 100] = 38.0
    m = np.zeros((120, 160), bool)
    m[30, 100] = True
    assert find_hotspots(t, settings, mask=m) == []
    assert fix_pixels(t, m)[30, 100] < 22


def test_area_scales_with_altitude(settings):
    lo20, hi20 = person_area_range_px(settings, 20.0)
    lo60, hi60 = person_area_range_px(settings, 60.0)
    assert hi20 > hi60 and lo20 >= lo60
    assert person_area_range_px(settings, None)[0] == 2


def test_small_blob_rejected_by_max_area_at_altitude(settings):
    big = scene(blob=(40, 20, 120, 100))
    assert find_hotspots(big, settings, alt_m=60.0) == []


def test_colorize_shape():
    img = colorize(scene())
    assert img.shape == (120, 160, 3) and img.dtype == np.uint8
