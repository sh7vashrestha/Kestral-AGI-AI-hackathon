#!/usr/bin/env python
"""Fit the thermal->RGB homography (data/calibration/thermal_to_rgb.json).

AUTO (no interaction beyond moving a warm object): use a mug of hot water or a hand held against a PLAIN,
cool background, with nothing else warm/moving in view of either camera.
    .venv/bin/python scripts/calibrate_thermal.py --auto [--points 8] [--delay 6]
 1. 3 s with the object OUT of view -> RGB reference frame.
 2. For each of N positions (spread over the whole image, incl. corners; the script prints a countdown):
    move the object, hold still; both frames are captured; thermal = centroid of the hottest blob,
    RGB = centroid of the largest region that differs from the reference (frame differencing).
 3. cv2.findHomography(RANSAC), reprojection error printed (RGB px), file saved. Needs >= 4 good points.
MANUAL: --pairs pairs.json   with {"pairs": [{"thermal": [x, y], "rgb": [x, y]}, ...]}  (thermal in 160x120 px)
Other: --reset deletes the calibration (back to the FOV-ratio default); --show prints the current matrix.
Saved debug images (data/calibration/): points.jpg.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "drone_brain")]

from camera import Camera  # noqa: E402
from config import Settings, get_settings  # noqa: E402
from fusion import calibration_path, default_homography, load_homography, project_points, save_homography  # noqa: E402
from thermal import ThermalCamera, delta_map  # noqa: E402


def thermal_point(temp: np.ndarray, mask: np.ndarray, min_delta: float = 4.0) -> Optional[tuple[float, float]]:
    t, bg, d = delta_map(temp, mask)
    peak = d.max()
    if peak < min_delta:
        return None
    sel = (d >= max(min_delta, 0.5 * peak)).astype(np.uint8)
    n, lab, stats, cen = cv2.connectedComponentsWithStats(sel, connectivity=8)
    if n < 2:
        return None
    best = 1 + int(np.argmax([(d * (lab == i)).sum() for i in range(1, n)]))
    return float(cen[best][0]), float(cen[best][1])


def rgb_point(frame: np.ndarray, ref: np.ndarray, min_frac: float = 0.0005, max_frac: float = 0.25) -> Optional[tuple[float, float]]:
    a = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    diff = cv2.absdiff(a, b)
    _, m = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if diff.max() < 25:
        return None
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    frac = stats[i, cv2.CC_STAT_AREA] / m.size
    if not (min_frac <= frac <= max_frac):
        return None
    return float(cen[i][0]), float(cen[i][1])


def fit(pairs: list[tuple[tuple[float, float], tuple[float, float]]], s: Settings, thresh_px: float = 12.0):
    src = np.array([p[0] for p in pairs], np.float64)
    dst = np.array([p[1] for p in pairs], np.float64)
    if len(pairs) < 4:
        raise SystemExit(f"need >= 4 point pairs, have {len(pairs)}")
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, thresh_px)
    if H is None:
        raise SystemExit("homography fit failed (points degenerate?)")
    inl = inl.ravel().astype(bool)
    err = np.linalg.norm(project_points(H, src) - dst, axis=1)
    print("per-point reprojection error (RGB px):", np.round(err, 1).tolist(), "inliers", int(inl.sum()), "/", len(pairs))
    rms = float(np.sqrt((err[inl] ** 2).mean())) if inl.any() else float("inf")
    if inl.sum() >= 4:
        H, _ = cv2.findHomography(src[inl], dst[inl], 0)
        err = np.linalg.norm(project_points(H, src[inl]) - dst[inl], axis=1)
        rms = float(np.sqrt((err ** 2).mean()))
    print(f"RMS reprojection error: {rms:.2f} RGB px (default FOV-ratio H would give "
          f"{np.sqrt((np.linalg.norm(project_points(default_homography(s), src) - dst, axis=1) ** 2).mean()):.1f} px)")
    return H, rms, int(inl.sum())


def auto(s: Settings, n_points: int, delay: float) -> list:
    cam = Camera(s).start()
    tc = ThermalCamera(s).start()
    if not cam.wait_first(20):
        raise SystemExit(f"RGB camera: {cam.last_error}")
    print("Keep the object OUT of view of both cameras for 3 s ...")
    time.sleep(3)
    ref, _, _ = cam.latest()
    pairs, vis = [], ref.copy()
    for i in range(n_points):
        for c in range(int(delay), 0, -1):
            print(f"  point {i+1}/{n_points}: move the warm object to a NEW position (spread over the image) - capture in {c}s", end="\r")
            time.sleep(1)
        frame, _, _ = cam.latest()
        temp, _ = tc.latest()
        if frame is None or temp is None:
            print("\n  no frames, skipped")
            continue
        tp, rp = thermal_point(temp, tc.mask), rgb_point(frame, ref)
        if tp is None or rp is None:
            print(f"\n  point {i+1}: object not found (thermal={tp}, rgb={rp}), skipped")
            continue
        print(f"\n  point {i+1}: thermal ({tp[0]:.1f},{tp[1]:.1f}) <-> rgb ({rp[0]:.0f},{rp[1]:.0f})")
        pairs.append((tp, rp))
        cv2.circle(vis, (int(rp[0]), int(rp[1])), 14, (0, 255, 0), 3)
    cam.stop(); tc.stop()
    out = calibration_path(s).parent
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "points.jpg"), vis)
    return pairs


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--delay", type=float, default=6.0)
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    if a.reset:
        calibration_path(s).unlink(missing_ok=True)
        print("calibration removed; default FOV-ratio mapping in use")
        return 0
    if a.show:
        print(np.round(load_homography(s), 5))
        return 0
    if a.pairs:
        d = json.loads(a.pairs.read_text())
        pairs = [(tuple(p["thermal"]), tuple(p["rgb"])) for p in d["pairs"]]
    elif a.auto:
        pairs = auto(s, a.points, a.delay)
    else:
        ap.print_help()
        return 2
    H, rms, n = fit(pairs, s)
    if not a.no_save:
        print("saved", save_homography(s, H, rms, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
