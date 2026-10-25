from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from config import ROOT

ENGINE = ROOT / "models" / "yolo26n.engine"
IMG = ROOT / "tests" / "assets" / "persons" / "p01_red_jacket_ski.jpg"

pytestmark = pytest.mark.skipif(not ENGINE.exists(), reason="TensorRT engine not exported (scripts/export_models.py yolo)")


@pytest.fixture(scope="module")
def detector():
    from config import Settings
    from detector import Detector
    return Detector(Settings(_env_file=None, yolo_conf_threshold=0.5, yolo_model="yolo26n.engine"))


def test_detects_person(detector):
    img = cv2.imread(str(IMG))
    dets = detector.detect(img, ts=1.0, frame_id=3)
    assert dets and all(0 <= d.c_yolo <= 1 and d.frame_id == 3 for d in dets)


def test_roi_maps_back(detector):
    img = cv2.copyMakeBorder(cv2.imread(str(IMG)), 300, 300, 300, 300, cv2.BORDER_CONSTANT)
    h, w = cv2.imread(str(IMG)).shape[:2]
    from models import BBox
    dets = detector.detect_roi(img, BBox(x1=300, y1=300, x2=300 + w, y2=300 + h), conf=0.3)
    assert dets and dets[0].bbox.x1 >= 250 and dets[0].bbox.x2 <= 300 + w + 50


# ---- CLIP preprocessing (CPU only, no model) ----
def test_clip_preprocess_pads_to_square():
    import numpy as np
    from clip_encoder import RES, preprocess, preprocess_u8
    tall = np.zeros((300, 100, 3), np.uint8)
    out = preprocess_u8(tall)
    assert out.shape == (RES, RES, 3) and out[:, 0].mean() > 100          # grey padding left/right
    x = preprocess(tall, np.zeros(3, np.float32), np.ones(3, np.float32))
    assert x.shape == (3, RES, RES) and x.dtype == np.float32
