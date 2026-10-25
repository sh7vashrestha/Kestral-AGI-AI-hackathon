"""Image orientation (rotate clockwise, then flip) for the RGB and thermal cameras.

    apply(img, rotate, flip)        -> numpy image  (cv2.rotate + cv2.flip; works for 2-D and 3-D arrays)
    gst_flip_method(rotate, flip)   -> nvvidconv `flip-method` that produces the same image on the Jetson
    swaps_axes(rotate)              -> True for 90/270 (output W x H becomes H x W, FOVs swap)

Definition: the output is `flip(rotate_clockwise(img, rotate), flip)` where flip 'horizontal' mirrors
left<->right and 'vertical' mirrors top<->bottom of the ROTATED image. Pure functions, no I/O.
"""
from __future__ import annotations

import cv2
import numpy as np

ROTATIONS = (0, 90, 180, 270)
FLIPS = ("none", "horizontal", "vertical")
_CV_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def check(rotate: int, flip: str) -> tuple[int, str]:
    r = int(rotate) % 360
    if r not in ROTATIONS:
        raise ValueError(f"rotate must be 0, 90, 180 or 270 (got {rotate!r})")
    f = str(flip).strip().lower()
    if f not in FLIPS:
        raise ValueError(f"flip must be one of {FLIPS} (got {flip!r})")
    return r, f


def swaps_axes(rotate: int) -> bool:
    return int(rotate) % 180 == 90


def apply(img: np.ndarray, rotate: int = 0, flip: str = "none") -> np.ndarray:
    r, f = check(rotate, flip)
    out = cv2.rotate(img, _CV_ROT[r]) if r else img
    if f == "horizontal":
        out = cv2.flip(out, 1)
    elif f == "vertical":
        out = cv2.flip(out, 0)
    return out


def out_size(w: int, h: int, rotate: int) -> tuple[int, int]:
    """(width, height) of a w x h image after rotation."""
    return (h, w) if swaps_axes(rotate) else (w, h)


# nvvidconv flip-method -> equivalent (rotate, flip). Verified on the Orin Nano (JetPack 5, L4T 35.x) by pushing
# an asymmetric test image through `nvvidconv flip-method=N` and comparing with cv2 (tests/test_orient.py repeats it).
# NOTE the numbering differs from GStreamer's videoflip: 4 = horizontal, 5 = upper-right diagonal, 6 = vertical,
# 7 = upper-left diagonal.
_NV_METHOD = {0: (0, "none"), 1: (270, "none"), 2: (180, "none"), 3: (90, "none"),
              4: (0, "horizontal"), 5: (90, "vertical"), 6: (0, "vertical"), 7: (90, "horizontal")}


def gst_flip_method(rotate: int = 0, flip: str = "none") -> int:
    """The nvvidconv flip-method whose result equals apply(img, rotate, flip) (all 8 combinations exist)."""
    probe = np.arange(15, dtype=np.uint8).reshape(3, 5)
    want = apply(probe, rotate, flip)
    for m, (r, f) in _NV_METHOD.items():
        got = apply(probe, r, f)
        if got.shape == want.shape and np.array_equal(got, want):
            return m
    raise ValueError(f"no flip-method for rotate={rotate} flip={flip}")  # unreachable


def label(rotate: int, flip: str) -> str:
    r, f = check(rotate, flip)
    return f"{r} deg" + ("" if f == "none" else f", flip {f}")
