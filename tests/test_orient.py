"""Rotation / flip of the cameras: orientation maths, nvvidconv mapping, effective optics, plan numbers."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

import orient
from fusion import default_homography, load_homography, save_homography
from models import GeoPoint, Mission
from survey import build_survey, footprint, gsd
from thermal import ThermalCamera, find_hotspots, static_mask

ALL = [(r, f) for r in orient.ROTATIONS for f in orient.FLIPS]


def asym(h=6, w=10):
    return np.arange(h * w, dtype=np.uint8).reshape(h, w)


def test_rotate_is_clockwise_and_flip_after_rotation():
    a = asym()
    assert np.array_equal(orient.apply(a, 90), np.rot90(a, -1))          # clockwise
    assert np.array_equal(orient.apply(a, 180), a[::-1, ::-1])
    assert np.array_equal(orient.apply(a, 270), np.rot90(a, 1))
    assert np.array_equal(orient.apply(a, 90, "horizontal"), np.rot90(a, -1)[:, ::-1])
    assert np.array_equal(orient.apply(a, 90, "vertical"), np.rot90(a, -1)[::-1])
    assert orient.apply(a, 90).shape == (10, 6) and orient.apply(a, 180).shape == (6, 10)


def test_check_rejects_garbage():
    with pytest.raises(ValueError):
        orient.check(45, "none")
    with pytest.raises(ValueError):
        orient.check(0, "diagonal")


EXPECTED_FLIP_METHOD = {(0, "none"): 0, (90, "none"): 3, (180, "none"): 2, (270, "none"): 1,
                        (0, "horizontal"): 4, (0, "vertical"): 6, (180, "horizontal"): 6, (180, "vertical"): 4,
                        (90, "vertical"): 5, (270, "horizontal"): 5, (90, "horizontal"): 7, (270, "vertical"): 7}


@pytest.mark.parametrize("rot,flip", ALL)
def test_gst_flip_method_table(rot, flip):
    assert orient.gst_flip_method(rot, flip) == EXPECTED_FLIP_METHOD[(rot, flip)]


def _nvvidconv_frame(method, img):
    p = ("appsrc name=src ! videoconvert ! video/x-raw,format=RGBA ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
         f"nvvidconv flip-method={method} ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
         "appsink drop=true max-buffers=1 sync=false")
    return p


def test_nvvidconv_flip_methods_match_cv2_on_hardware(tmp_path):
    """Pushes an asymmetric PNG through the real `nvvidconv flip-method=N` and checks it equals orient.apply."""
    ref = np.full((144, 256, 3), 40, np.uint8)
    cv2.rectangle(ref, (0, 0), (90, 50), (255, 0, 0), -1)
    cv2.rectangle(ref, (230, 120), (255, 143), (0, 255, 0), -1)
    cv2.circle(ref, (200, 25), 14, (0, 0, 255), -1)
    png = tmp_path / "ref.png"
    cv2.imwrite(str(png), ref)
    checked = 0
    for (rot, flip), m in EXPECTED_FLIP_METHOD.items():
        pipe = (f"multifilesrc location={png} loop=true caps=image/png ! pngdec ! videoconvert ! video/x-raw,format=RGBA ! "
                f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvvidconv flip-method={m} ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false")
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        got = None
        for _ in range(8):
            ok, got = cap.read()
            if ok:
                break
        cap.release()
        if got is None:
            pytest.skip("nvvidconv / GStreamer not available")
        want = orient.apply(ref, rot, flip)
        assert got.shape == want.shape
        assert np.abs(got.astype(int) - want.astype(int)).mean() < 3.0, (rot, flip, m)
        checked += 1
    assert checked == len(EXPECTED_FLIP_METHOD)


# ---- effective optics ---------------------------------------------------------------------------------------
def real(settings_factory, **kw):
    """The real IMX477 + 6 mm values (not the conftest test lens)."""
    return settings_factory(camera_hfov_deg=52.8, camera_vfov_deg=31.2, sensor_width_mm=5.952, sensor_height_mm=3.348,
                            focal_length_mm=6.0, image_width_px=1920, camera_width=1920, camera_height=1080, **kw)


def test_eff_values_unrotated_and_180(settings_factory):
    for rot in (0, 180):
        s = real(settings_factory, camera_rotate=rot)
        assert (s.eff_hfov_deg, s.eff_vfov_deg) == (52.8, 31.2)
        assert (s.eff_camera_width, s.eff_camera_height, s.eff_image_width_px, s.eff_sensor_width_mm) == (1920, 1080, 1920, 5.952)


def test_eff_values_swap_at_90_and_270(settings_factory):
    for rot in (90, 270):
        s = real(settings_factory, camera_rotate=rot)
        assert (s.eff_hfov_deg, s.eff_vfov_deg) == (31.2, 52.8)
        assert (s.eff_camera_width, s.eff_camera_height) == (1080, 1920)
        assert s.eff_image_width_px == 1080 and s.eff_sensor_width_mm == pytest.approx(3.348)
    # the flip never changes the optics
    assert real(settings_factory, camera_flip="horizontal").eff_hfov_deg == 52.8


def test_rotation_swaps_footprint_and_gsd_consistently(settings_factory):
    s0, s90 = real(settings_factory, camera_rotate=0), real(settings_factory, camera_rotate=90)
    p0 = build_survey(Mission(length_m=100, width_m=60, altitude_m=20), GeoPoint(lat=47, lon=8, alt_m=0), s0)
    p90 = build_survey(Mission(length_m=100, width_m=60, altitude_m=20), GeoPoint(lat=47, lon=8, alt_m=0), s90)
    assert p90.footprint_w_m == pytest.approx(p0.footprint_l_m) and p90.footprint_l_m == pytest.approx(p0.footprint_w_m)
    assert p90.line_spacing_m == pytest.approx(p0.footprint_l_m * 0.75)
    assert p90.v_max_ms == pytest.approx(p0.footprint_w_m * 0.75 / s0.frame_interval_s)
    # GSD along the (now) horizontal axis: both give the same m/px because pixels are square
    assert p90.gsd_m == pytest.approx(p0.gsd_m, rel=0.01)
    assert p90.gsd_m == pytest.approx(gsd(20, 3.348, 6.0, 1080))
    # footprint W / pixels across = GSD (sanity of the pair)
    assert p90.footprint_w_m / 1080 == pytest.approx(p90.gsd_m, rel=0.02)
    assert p0.footprint_w_m / 1920 == pytest.approx(p0.gsd_m, rel=0.02)


def test_default_plan_numbers_for_20m(settings_factory):
    """Documented numbers (notes/phase5.md): 100x60 m area at 20 m, real IMX477 + 6 mm lens, 25 % overlap, 1 fps."""
    out = {}
    for rot in (0, 90, 180):
        s = real(settings_factory, camera_rotate=rot)
        p = build_survey(Mission(length_m=100, width_m=60, altitude_m=20), GeoPoint(lat=47, lon=8, alt_m=0), s)
        out[rot] = (round(p.footprint_w_m, 1), round(p.footprint_l_m, 1), round(p.line_spacing_m, 1), round(p.v_max_ms, 1),
                    round(p.gsd_m * 100, 2), round(p.person_px), p.lane_count)
    assert out[0] == (19.9, 11.2, 14.9, 8.4, 1.03, 165, 5)
    assert out[180] == out[0]                                    # 180 does not change the optics
    assert out[90][:2] == (11.2, 19.9) and out[90][2:4] == (8.4, 14.9)   # W and L swapped; lanes 8.4 m apart, v_max 14.9
    assert out[90][6] == 8                                # more, narrower lanes


def test_tracker_pixel_to_ground_uses_effective_fov(settings_factory):
    from tracker import pixel_to_ground
    a = pixel_to_ground(1080 * 0.75, 960, 1080, 1920, 47.0, 8.0, 20.0, 0.0, 31.2, 52.8)
    s = real(settings_factory, camera_rotate=90)
    import tracker as t
    assert (s.eff_hfov_deg, s.eff_vfov_deg) == (31.2, 52.8) and a is not None
    # the tracker module reads s.eff_* (grep guard)
    assert "eff_hfov_deg" in open(t.__file__).read()


# ---- fusion homography / calibration ------------------------------------------------------------------------
def test_default_homography_uses_effective_sizes_and_fovs(settings_factory):
    s = real(settings_factory, camera_rotate=90, thermal_rotate=90)
    H = default_homography(s)
    # RGB 1080x1920 (hfov 31.2, vfov 52.8), thermal 120x160 (hfov 44, vfov 57): thermal centre -> RGB centre
    c = H @ np.array([60.0, 80.0, 1.0])
    assert (c[0] / c[2], c[1] / c[2]) == pytest.approx((540.0, 960.0))
    fx_r = 540 / math.tan(math.radians(31.2 / 2)); fx_t = 60 / math.tan(math.radians(44 / 2))
    assert H[0, 0] == pytest.approx(fx_r / fx_t)
    H0 = default_homography(real(settings_factory))
    assert H0[0, 0] != pytest.approx(H[0, 0])


def test_calibration_is_ignored_for_other_orientation(tmp_path, settings_factory, caplog):
    s0 = real(settings_factory, data_dir=tmp_path)
    Hc = np.array([[3.0, 0, 5], [0, 3.0, 7], [0, 0, 1]])
    save_homography(s0, Hc, 1.0, 8)
    assert np.allclose(load_homography(s0), Hc)
    s180 = real(settings_factory, data_dir=tmp_path, camera_rotate=180)
    assert np.allclose(load_homography(s180), default_homography(s180))     # made unrotated: not valid now


# ---- thermal -------------------------------------------------------------------------------------------------
def test_stuck_pixel_mask_follows_rotation(settings_factory):
    for rot, flip in ALL:
        s = settings_factory(thermal_stuck_pixels="67,64;3,5", thermal_rotate=rot, thermal_flip=flip)
        sensor = np.zeros((120, 160), bool)
        sensor[64, 67] = sensor[5, 3] = True
        want = orient.apply(sensor.view(np.uint8), rot, flip).astype(bool)
        got = static_mask(s)
        assert got.shape == want.shape and np.array_equal(got, want)
        assert got.sum() == 2
        assert np.array_equal(static_mask(s, want.shape), want)


def test_thermal_camera_rotates_frames_and_mask(tmp_path, settings_factory):
    t = np.full((120, 160), 20.0, np.float32)
    t[10:20, 100:110] = 35.0                                             # warm patch, upper right of the sensor
    np.save(tmp_path / "a.npy", t)
    s = settings_factory(thermal_rotate=90, thermal_stuck_pixels="67,64")
    tc = ThermalCamera(s, source=f"file:{tmp_path}").start()
    try:
        import time
        end = time.time() + 5
        while tc.latest()[0] is None and time.time() < end:
            time.sleep(0.05)
        f, _ = tc.latest()
        assert f.shape == (160, 120)
        assert np.array_equal(f > 30, orient.apply((t > 30).view(np.uint8), 90).astype(bool))
        assert tc.mask.shape == (160, 120) and tc.mask.sum() >= 1
        tc.set_orient(180, "none")
        time.sleep(0.5)
        f2, _ = tc.latest()
        assert f2 is not None and f2.shape == (120, 160) and f2[120 - 20:120 - 10, 160 - 110:160 - 100].mean() > 30
    finally:
        tc.stop()


def test_find_hotspots_works_on_rotated_frame(settings_factory):
    s = settings_factory(thermal_rotate=90, thermal_stuck_pixels="")
    t = np.full((160, 120), 22.0, np.float32)
    t[40:48, 50:58] = 33.0
    spots = find_hotspots(t, s, alt_m=10)
    assert len(spots) == 1 and 50 <= spots[0].bbox.x1 <= 52
