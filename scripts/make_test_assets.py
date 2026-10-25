"""Rebuild tests/assets/{persons,nonperson}/ from source images (COCO val2017 + ultralytics assets).

Usage: .venv/bin/python scripts/make_test_assets.py <dir with source jpgs> <dir with yolo crops 'cand2'>
The crops were produced by YOLO11n (conf>=0.6, person boxes >=130 px tall, +10% padding) and
hand-selected; INDEX maps the output name to the candidate file. Labels live in tests/assets/match/labels.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent / "tests" / "assets"
INDEX = {  # name -> candidate crop file
    "p01_red_jacket_ski": "000000000785_0", "p02_white_cap_runner": "000000000872_0",
    "p03_white_polo_cap": "000000000885_0", "p04_white_tee_cap": "000000001000_1",
    "p05_orange_cap_kid": "000000001000_6", "p06_white_top_woman": "000000001296_0",
    "p07_colour_jacket_skier": "000000002473_0", "p08_green_top_cap_skier": "000000002532_0",
    "p09_black_tee_crouch": "000000003156_0", "p10_white_shirt_tie": "000000004134_0",
    "p11_yellow_top_surfer": "000000004395_0", "p12_beige_coat_helmet_child": "000000005001_1",
    "p13_green_polo_sitting": "000000005060_0", "p14_striped_brown_boy": "000000005193_3",
    "p15_black_beanie": "000000005193_4", "p16_cream_fleece": "bus_1",
    "p17_navy_coat_glasses": "bus_2", "p18_black_suit": "zidane_0",
}
AERIAL = {"a01_red_jacket_ski": "p01_red_jacket_ski", "a02_white_cap_runner": "p02_white_cap_runner",
          "a03_green_top_cap_skier": "p08_green_top_cap_skier", "a04_green_polo_sitting": "p13_green_polo_sitting",
          "a05_cream_fleece": "p16_cream_fleece", "a06_black_suit": "p18_black_suit"}
# non-person: (source image stem, x1, y1, x2, y2) in source pixels
NONPERSON = {"n01_bus_body": ("bus", 0, 120, 480, 600), "n02_pavement": ("bus", 0, 900, 300, 1080),
             "n03_zidane_crowd_bg": ("zidane", 0, 0, 300, 400)}


def _fit(im, max_side=400):
    s = max_side / max(im.shape[:2])
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im


def aerial_like(im, h=90, min_side=224):
    """Downscale to ~h px tall then upscale so the short side >= min_side (what matcher does)."""
    s = h / im.shape[0]
    small = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    up = max(1.0, min_side / min(small.shape[:2]))
    return cv2.resize(small, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)


def main(src: Path, cand: Path) -> None:
    (ROOT / "persons").mkdir(parents=True, exist_ok=True)
    (ROOT / "nonperson").mkdir(parents=True, exist_ok=True)
    for name, c in INDEX.items():
        cv2.imwrite(str(ROOT / "persons" / f"{name}.jpg"), _fit(cv2.imread(str(cand / f"{c}.jpg"))), [cv2.IMWRITE_JPEG_QUALITY, 90])
    for name, base in AERIAL.items():
        im = cv2.imread(str(ROOT / "persons" / f"{base}.jpg"))
        cv2.imwrite(str(ROOT / "persons" / f"{name}.jpg"), aerial_like(im), [cv2.IMWRITE_JPEG_QUALITY, 90])
    for name, (stem, x1, y1, x2, y2) in NONPERSON.items():
        im = cv2.imread(str(src / f"{stem}.jpg"))[y1:y2, x1:x2]
        cv2.imwrite(str(ROOT / "nonperson" / f"{name}.jpg"), _fit(im), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
