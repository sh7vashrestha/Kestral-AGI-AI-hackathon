#!/usr/bin/env python
"""Capture N frames from both cameras; save RGB jpg, thermal colour png and a thermal-on-RGB overlay to
data/camera_test/ (gitignored). Prints fps, brightness, thermal min/median/max.

    .venv/bin/python scripts/camera_test.py [-n 20] [--rgb csi:0] [--thermal v4l2:/dev/...] [--alpha 0.5]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "drone_brain")]

from camera import Camera  # noqa: E402
from config import get_settings  # noqa: E402
from fusion import load_homography  # noqa: E402
from thermal import ThermalCamera, colorize, find_hotspots  # noqa: E402


def overlay(rgb: np.ndarray, temp: np.ndarray, H: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = rgb.shape[:2]
    col = cv2.warpPerspective(colorize(temp), H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    valid = cv2.warpPerspective(np.full(temp.shape, 255, np.uint8), H, (w, h)) > 0
    out = rgb.copy()
    out[valid] = cv2.addWeighted(rgb, 1 - alpha, col, alpha, 0)[valid]
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--rgb")
    ap.add_argument("--thermal")
    ap.add_argument("--alpha", type=float, default=0.5)
    a = ap.parse_args()
    s = get_settings()
    out = Path(s.data_dir) / "camera_test"
    out.mkdir(parents=True, exist_ok=True)
    cam = Camera(s, a.rgb).start()
    tc = ThermalCamera(s, a.thermal).start()
    if not cam.wait_first(20):
        print("RGB: no frame:", cam.last_error)
        return 1
    H = load_homography(s)
    t0, ids, bright, tmin, tmed, tmax = time.time(), set(), [], [], [], []
    last_rgb, last_t = None, None
    while len(ids) < a.n and time.time() - t0 < 60:
        f, ts, fid = cam.latest()
        if f is not None and fid not in ids:
            ids.add(fid)
            bright.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
            last_rgb = f
        t, tts = tc.latest()
        if t is not None:
            tmin.append(float(t.min())); tmed.append(float(np.median(t))); tmax.append(float(t.max()))
            last_t = t
        time.sleep(0.03)
    dt = time.time() - t0
    print(f"RGB: {len(ids)} frames in {dt:.1f}s = {len(ids)/dt:.1f} fps (unique), shape {None if last_rgb is None else last_rgb.shape}, "
          f"mean brightness {np.mean(bright):.1f}")
    if tc.enabled:
        print(f"Thermal: {tc.fps():.1f} fps, min {np.mean(tmin) if tmin else float('nan'):.1f} / median "
              f"{np.mean(tmed) if tmed else float('nan'):.1f} / max {np.mean(tmax) if tmax else float('nan'):.1f} C, "
              f"stuck px {np.argwhere(tc.mask).tolist()}, err {tc.last_error}")
    if last_rgb is not None:
        cv2.imwrite(str(out / "rgb.jpg"), last_rgb)
    if last_t is not None:
        cv2.imwrite(str(out / "thermal.png"), colorize(last_t, scale=4))
        print("hotspots:", [(round(h.peak_c, 1), round(h.delta_c, 1), h.area_px) for h in find_hotspots(last_t, s)])
        if last_rgb is not None:
            cv2.imwrite(str(out / "overlay.jpg"), overlay(last_rgb, last_t, H, a.alpha))
    print("saved to", out)
    cam.stop(); tc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
