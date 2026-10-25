from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from camera import Camera, CameraError, gst_csi_pipeline


def test_pipeline_string():
    p = gst_csi_pipeline(0, 1920, 1080)
    assert "sensor-id=0" in p and "appsink drop=true max-buffers=1 sync=false" in p


def test_file_folder_latest_and_frames(tmp_path, settings_factory):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.full((48, 64, 3), i * 50, np.uint8))
    cam = Camera(settings_factory(camera_source=f"file:{tmp_path}", capture_fps=20.0)).start(wait_s=5)
    try:
        f, ts, fid = cam.latest()
        assert f is not None and f.shape == (48, 64, 3) and fid >= 1
        got = []
        for fr, t, i in cam.frames():
            got.append(i)
            if len(got) == 3:
                break
        assert got == sorted(set(got))
    finally:
        cam.stop()


def test_bad_source_reports_error_and_retries(settings_factory):
    cam = Camera(settings_factory(camera_source="file:/nonexistent/xyz")).start()
    time.sleep(1.0)
    cam.stop()
    assert cam.last_error and "not found" in cam.last_error and cam.latest()[0] is None


def test_bad_scheme(settings_factory):
    cam = Camera(settings_factory(camera_source="foo:1"))
    with pytest.raises(CameraError):
        cam._open()
