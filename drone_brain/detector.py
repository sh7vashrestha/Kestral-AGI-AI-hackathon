"""YOLO26n person detector (Ultralytics; TensorRT FP16 engine preferred, .pt fallback).

    det = Detector(settings)                                   # loads + warms up
    dets = det.detect(frame_bgr, ts=..., frame_id=..., telemetry=...)      # conf >= yolo_conf_threshold
    dets = det.detect_roi(frame_bgr, roi_bbox, ts=..., frame_id=..., conf=settings.thermal_recheck_conf)
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import ROOT, Settings
from models import BBox, Detection, Telemetry

log = logging.getLogger(__name__)
PERSON_CLASS = 0
MODELS_DIR = ROOT / "models"


def resolve_weights(name: str) -> Path:
    """Prefer models/<stem>.engine, then the configured path, then models/<name>.pt."""
    p = Path(name)
    cands = [MODELS_DIR / (p.stem + ".engine"), p if p.is_absolute() else MODELS_DIR / p.name, ROOT / p]
    if p.suffix == ".engine":
        cands = [c for c in cands if c.suffix == ".engine"] + cands
    cands.append(MODELS_DIR / (p.stem + ".pt"))        # engine missing / not exported yet: the PyTorch weights still work (slower)
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"no weights for {name!r}; run scripts/export_models.py yolo (looked in {MODELS_DIR})")


class Detector:
    def __init__(self, settings: Settings, weights: Optional[str] = None, warmup: bool = True) -> None:
        from ultralytics import YOLO

        self.s = settings
        self.path = resolve_weights(weights or settings.yolo_model)
        if self.path.suffix != ".engine":
            log.warning("using %s (no TensorRT engine; run scripts/export_models.py yolo) - slower", self.path.name)
        self._lock = threading.Lock()
        self.last_ms = 0.0
        self.failures = 0              # inference calls that failed even after the retry (pipeline counts the frame as an error)
        self.recovered = 0             # inference calls that failed once and worked on the retry
        try:
            self.model = YOLO(str(self.path), task="detect")
            if warmup:
                self.warmup()
        except Exception as exc:  # noqa: BLE001 - engine built with another TensorRT / GPU memory: fall back to the .pt
            pt = self._fallback_pt()
            if self.path.suffix != ".engine" or pt is None:
                raise
            log.error("YOLO engine %s failed to load (%s); falling back to %s", self.path.name, str(exc)[:120], pt.name)
            self._free_cuda()
            self.path = pt
            self.model = YOLO(str(pt), task="detect")
            if warmup:
                self.warmup()

    def _fallback_pt(self) -> Optional[Path]:
        for c in (MODELS_DIR / (self.path.stem + ".pt"), MODELS_DIR / Path(self.s.yolo_model).with_suffix(".pt").name):
            if c.exists():
                return c
        return None

    @staticmethod
    def _free_cuda() -> None:
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def warmup(self, n: int = 3) -> None:
        img = np.zeros((self.s.eff_camera_height, self.s.eff_camera_width, 3), np.uint8)
        for _ in range(n):
            self._predict(img, 0.5)

    def _predict(self, img: np.ndarray, conf: float) -> list[tuple[float, float, float, float, float]]:
        t0 = time.perf_counter()
        with self._lock:
            try:
                res = self.model.predict(img, imgsz=self.s.yolo_imgsz, conf=conf, classes=[PERSON_CLASS], verbose=False)
            except Exception as exc:  # noqa: BLE001 - CUDA OOM / NvMap error 12 under memory pressure: free, wait, retry once
                log.warning("YOLO inference failed (%s); freeing GPU cache and retrying once", str(exc)[:120])
                self._free_cuda()
                time.sleep(0.3)
                try:
                    res = self.model.predict(img, imgsz=self.s.yolo_imgsz, conf=conf, classes=[PERSON_CLASS], verbose=False)
                    self.recovered += 1
                except Exception:
                    self.failures += 1
                    raise
        self.last_ms = (time.perf_counter() - t0) * 1000
        b = res[0].boxes
        if b is None or len(b) == 0:
            return []
        xyxy, cf = b.xyxy.cpu().numpy(), b.conf.cpu().numpy()
        h, w = img.shape[:2]
        out = []
        for x, c in zip(xyxy, cf):
            x1, y1, x2, y2 = (float(v) for v in x)
            if not all(np.isfinite([x1, y1, x2, y2, c])):
                continue
            x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))      # boxes at the image edge: clamp, drop empty ones
            y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
            if x2 - x1 >= 1.0 and y2 - y1 >= 1.0:
                out.append((x1, y1, x2, y2, float(c)))
        return out

    def detect(self, frame: np.ndarray, ts: Optional[float] = None, frame_id: int = 0,
               telemetry: Optional[Telemetry] = None, conf: Optional[float] = None) -> list[Detection]:
        ts = time.time() if ts is None else ts
        c = self.s.yolo_conf_threshold if conf is None else conf
        if frame is None or frame.ndim != 3 or min(frame.shape[:2]) < 8:
            return []
        return [Detection(frame_id=frame_id, ts=ts, bbox=BBox(x1=a, y1=b, x2=c2, y2=d), c_yolo=cf, telemetry=telemetry)
                for a, b, c2, d, cf in self._predict(frame, c)]

    def detect_roi(self, frame: np.ndarray, roi: BBox, ts: Optional[float] = None, frame_id: int = 0,
                   telemetry: Optional[Telemetry] = None, conf: Optional[float] = None,
                   pad: float = 1.0) -> list[Detection]:
        """Re-check a region: crop (padded by `pad` x its size per side, min 64 px), upscale to ~imgsz, detect at
        a lower threshold, map boxes back to full-frame coordinates."""
        ts = time.time() if ts is None else ts
        c = self.s.thermal_recheck_conf if conf is None else conf
        h, w = frame.shape[:2]
        rw, rh = roi.x2 - roi.x1, roi.y2 - roi.y1
        cx, cy = (roi.x1 + roi.x2) / 2, (roi.y1 + roi.y2) / 2
        half = max(rw, rh) * (0.5 + pad)
        half = max(half, 32.0)
        x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
        x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return []
        crop = frame[y1:y2, x1:x2]
        scale = self.s.yolo_imgsz / max(crop.shape[:2])
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            scale = 1.0
        out = []
        for a, b, c2, d, cf in self._predict(crop, c):
            out.append(Detection(frame_id=frame_id, ts=ts, telemetry=telemetry, c_yolo=cf,
                                 bbox=BBox(x1=x1 + a / scale, y1=y1 + b / scale, x2=x1 + c2 / scale, y2=y1 + d / scale)))
        return out
