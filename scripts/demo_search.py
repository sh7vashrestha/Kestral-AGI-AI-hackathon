#!/usr/bin/env python3
"""Live search demo (no flying, no vehicle, no Gemma): RGB + thermal cameras, YOLO finds people, CLIP ranks them
against a typed description ("person in a red jacket"), the best match is boxed red on the live view.

    .venv/bin/python scripts/demo_search.py [--port 8090] [--rgb csi:0] [--thermal auto|none] [--fps 10]
    open http://<jetson-ip>:8090/live        (hotspot: http://10.42.0.1:8090/live)

Same page as scripts/live_view.py plus a search bar. Opens both cameras itself, so main.py / the drone-brain
service must NOT be running (sudo systemctl stop drone-brain). Each person box shows its CLIP match and, when the
thermal camera covers it, the warmest temperature inside (a real person reads ~30-37 C; a photo or poster does not).
Thermal->RGB alignment uses data/calibration/thermal_to_rgb.json when present, else the FOV-ratio default.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "drone_brain")]

from config import get_settings  # noqa: E402
from scripts.live_view import lan_addresses, resolve_thermal  # noqa: E402

log = logging.getLogger("demo_search")
RED, ORANGE, GREEN = (40, 40, 230), (30, 150, 245), (90, 200, 90)   # BGR
WARM_C = 29.0           # warmest pixel in a person box at/above this = "warm" (skin/clothes of a live person indoors)
EMA = 0.5               # smoothing of a person's score across frames (matched by box overlap)
MAX_HZ = 5.0            # search loop rate cap: YOLO + CLIP for a handful of people fits easily


class Query(BaseModel):
    text: str = ""


def demo_prompts(target):
    """The app's deterministic prompts without the aerial wording: the demo camera sees people at eye level."""
    from matcher import ClipPrompts, default_prompts
    p = default_prompts(target)
    pos = [x for x in p.positive if "from above" not in x] or p.positive
    return ClipPrompts(target_description=p.target_description, positive=pos, negative=p.negative)


def iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


class Searcher:
    """Background loop: latest RGB frame -> YOLO people -> CLIP score per crop (smoothed) + warmest thermal pixel."""

    def __init__(self, s: Any, cam: Any, th: Any) -> None:
        from clip_encoder import ClipEncoder
        from detector import Detector
        from fusion import load_homography
        from matcher import ClipBackend
        self.s, self.cam, self.th = s, cam, th
        self.cutoff = s.clip_match_cutoff
        self.det = Detector(s)
        self.backend = ClipBackend(ClipEncoder(s), prompt_builder=demo_prompts, cutoff=self.cutoff)
        self.H_r2t = np.linalg.inv(load_homography(s))       # RGB px -> thermal px
        self._lock = threading.Lock()
        self.target = None
        self.people: list[dict] = []
        self.best_jpg: Optional[bytes] = None
        self.best: Optional[dict] = None
        self.loop_ms = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="search", daemon=True)

    def start(self) -> "Searcher":
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def set_target(self, text: str) -> None:
        from models import Target
        text = " ".join((text or "").split())[:120]
        t = Target(description=text) if text else None
        if t is not None:
            self.backend.prepare(t)                     # encode the prompts once, off the loop
        with self._lock:
            self.target, self.people, self.best, self.best_jpg = t, [], None, None
        log.info("target: %s", text or "(none: people only)")

    def state(self) -> dict:
        with self._lock:
            return {"target": self.target.description if self.target else None, "cutoff": self.cutoff,
                    "people": [{k: p[k] for k in ("box", "score", "temp_c", "warm", "best", "match")} for p in self.people],
                    "best": self.best, "loop_ms": round(self.loop_ms), "has_best_image": self.best_jpg is not None}

    def boxes(self) -> list[tuple]:
        with self._lock:
            out = []
            for p in self.people:
                heat = f" {p['temp_c']:.0f}C" if p["temp_c"] is not None else ""
                if p["score"] is None:
                    style = (f"person{heat}", GREEN, 2)
                elif p["match"]:
                    style = (f"MATCH {p['score']:.0%}{heat}", RED, 4)
                elif p["best"]:
                    style = (f"closest {p['score']:.0%}{heat}", ORANGE, 3)
                else:
                    style = (f"{p['score']:.0%}{heat}", GREEN, 2)
                out.append((*p["box"], p["score"], style))
            return out

    # ------------------------------------------------------------------ loop
    def _warmest(self, box) -> Optional[float]:
        if self.th is None or not getattr(self.th, "enabled", True):
            return None
        temp, tts = self.th.latest()
        if temp is None or time.time() - tts > 2.0:
            return None
        from fusion import project_points
        x1, y1, x2, y2 = box
        q = project_points(self.H_r2t, np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]]))
        h, w = temp.shape[:2]
        a, b = np.clip(q.min(0), 0, [w - 1, h - 1]).astype(int), np.clip(q.max(0), 0, [w - 1, h - 1]).astype(int)
        if b[0] <= a[0] or b[1] <= a[1]:
            return None
        return float(np.percentile(temp[a[1]:b[1] + 1, a[0]:b[0] + 1], 98))

    def _run(self) -> None:
        last_ts = 0.0
        while not self._stop.is_set():
            t0 = time.time()
            frame, ts, _ = self.cam.latest()
            if frame is None or ts == last_ts:
                time.sleep(0.05)
                continue
            last_ts = ts
            try:
                self._step(frame, ts)
            except Exception as exc:  # noqa: BLE001 - keep the demo alive (e.g. a transient CUDA allocation failure)
                log.warning("search step failed: %s", exc)
                time.sleep(0.5)
            self.loop_ms = (time.time() - t0) * 1000
            time.sleep(max(0.0, 1.0 / MAX_HZ - (time.time() - t0)))

    def _step(self, frame: np.ndarray, ts: float) -> None:
        with self._lock:
            target, prev = self.target, list(self.people)
        dets = self.det.detect(frame, ts=ts)
        h, w = frame.shape[:2]
        people = []
        for d in dets:
            box = (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)
            x1, y1, x2, y2 = (int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2])), int(min(h, box[3])))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = frame[y1:y2, x1:x2]
            score = None
            if target is not None:
                score = float(self.backend.match(crop, target).match_confidence)
                old = max(prev, key=lambda p: iou(p["box"], box), default=None)
                if old is not None and old["score"] is not None and iou(old["box"], box) > 0.3:
                    score = EMA * old["score"] + (1 - EMA) * score
            temp = self._warmest(box)
            people.append({"box": box, "score": score, "temp_c": None if temp is None else round(temp, 1),
                           "warm": temp is not None and temp >= WARM_C, "best": False, "match": False,
                           "c_yolo": d.c_yolo, "_crop": crop})
        best = None
        if target is not None and people:
            best = max(people, key=lambda p: p["score"])
            best["best"] = True
            best["match"] = best["score"] >= self.cutoff
        jpg = None
        if best is not None and best["match"]:
            ok, buf = cv2.imencode(".jpg", best["_crop"], [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg = buf.tobytes() if ok else None
        for p in people:
            p.pop("_crop")
        with self._lock:
            if target is not self.target:
                return                                    # the target changed while this frame was scored
            self.people = people
            if jpg is not None and (self.best is None or best["score"] >= self.best["score"] - 0.05
                                    or time.time() - self.best["ts"] > 5):
                self.best_jpg = jpg
                self.best = {"score": round(best["score"], 3), "temp_c": best["temp_c"], "warm": best["warm"], "ts": time.time()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--rgb", default=None, help="camera source (default CAMERA_SOURCE, csi:0)")
    ap.add_argument("--thermal", default="auto", help="auto | v4l2:/dev/... | none")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import contextlib

    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import RedirectResponse, Response

    import shutdown
    from camera import Camera
    from thermal import ThermalCamera
    from webapp import stream

    s = get_settings()
    rgb_src = a.rgb or s.camera_source
    th_src = resolve_thermal(a.thermal, s.thermal_source)
    cam = Camera(s, source=rgb_src).start()
    th = ThermalCamera(s, source=th_src).start()
    print(f"RGB: {rgb_src}   thermal: {th_src}   (loading YOLO + CLIP ...)", flush=True)
    searcher = Searcher(s, cam, th).start()

    class SearchSource(stream.CamerasSource):
        has_boxes = True

        def boxes(self):
            return {"rgb": searcher.boxes(), "thermal": []}

    dev = th_src.split(":", 1)[1] if th_src.startswith("v4l2:") else ""
    src = SearchSource(cam, th, s, thermal_dev=dev)
    app = FastAPI(title="drone-brain search demo")
    app.include_router(stream.make_router(src, s, default_fps=a.fps))

    @app.get("/")
    def root():
        return RedirectResponse("/live")

    @app.get("/api/search")
    def get_search():
        return searcher.state()

    @app.post("/api/search")
    def post_search(q: Query):
        searcher.set_target(q.text)
        return searcher.state()

    @app.get("/api/search/best.jpg")
    def best_jpg():
        if searcher.best_jpg is None:
            raise HTTPException(404, "no match yet")
        return Response(searcher.best_jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    print("\nSearch demo:")
    for ip in lan_addresses() or ["<jetson-ip>"]:
        print(f"  http://{ip}:{a.port}/live")
    print("If RGB says 'No cameras available': sudo systemctl restart nvargus-daemon\n", flush=True)

    class _Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):
            yield

    server = _Server(uvicorn.Config(app, host=a.host, port=a.port, log_level="warning", timeout_graceful_shutdown=1))
    shutdown.install(on_signal=lambda: setattr(server, "should_exit", True), hard_exit_s=2.8)
    try:
        server.run()
    finally:
        searcher.stop()
        for c in (cam, th):
            c.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
