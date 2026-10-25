#!/usr/bin/env python
"""Export models to TensorRT FP16 ON THIS JETSON (engines are device/TRT-version specific, gitignored).

    .venv/bin/python scripts/export_models.py yolo        # models/yolo26n.pt -> models/yolo26n.engine
    .venv/bin/python scripts/export_models.py clip        # CLIP image encoder -> ONNX -> models/clip_image.engine
    .venv/bin/python scripts/export_models.py all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "drone_brain")]

from config import get_settings  # noqa: E402

log = logging.getLogger("export")
MODELS = ROOT / "models"


def export_yolo(imgsz: int, workspace_gb: float, weights: str = "yolo26n.pt") -> Path:
    from ultralytics import YOLO

    MODELS.mkdir(exist_ok=True)
    pt = MODELS / Path(weights).name
    if not pt.exists():
        # ultralytics downloads into cwd; do it from models/ so the repo stays clean
        import os
        cwd = os.getcwd()
        os.chdir(MODELS)
        try:
            YOLO(pt.name)
        finally:
            os.chdir(cwd)
    model = YOLO(str(pt))
    out = model.export(format="engine", half=True, imgsz=imgsz, workspace=workspace_gb, device=0, batch=1,
                       simplify=True, verbose=False)
    log.info("yolo engine: %s", out)
    return Path(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["yolo", "clip", "all"])
    ap.add_argument("--imgsz", type=int, default=s.yolo_imgsz)
    ap.add_argument("--workspace-gb", type=float, default=1.0)
    a = ap.parse_args()
    if a.what in ("yolo", "all"):
        export_yolo(a.imgsz, a.workspace_gb)
    if a.what in ("clip", "all"):
        from clip_encoder import export_image_engine
        export_image_engine(workspace_gb=a.workspace_gb)


if __name__ == "__main__":
    main()
