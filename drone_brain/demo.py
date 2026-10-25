"""Ground demo (main.py --demo / ./demo.sh): the whole app runs as usual (Gemma chat plan -> Confirm -> GO -> alerts
-> flight summary), but the vehicle is scripts/mock_vehicle.py on loopback, hovering at home: nothing can fly. GO starts
the live camera search on the real RGB + thermal cameras.

What is demo-only:
  * eye_level(): CLIP prompts without the aerial wording (people in a hall are seen at eye level, not from above)
  * DemoView: per-person CLIP score (the matcher's own track averages) + warmest thermal pixel, best match boxed red on
    /live, and GET/POST /api/search for the search panel of webapp/static/live.html (same JSON as scripts/demo_search.py)
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np
from pydantic import BaseModel

from models import ClipPrompts, Target

log = logging.getLogger("demo")
RED, ORANGE, GREEN = (40, 40, 230), (30, 150, 245), (90, 200, 90)   # BGR
WARM_C = 29.0           # warmest pixel in a person box at/above this = "warm" (a live person indoors reads ~30-37 C)
STALE_S = 1.5           # detections older than this are not drawn
_AERIAL = [(re.compile(p, re.I), r) for p, r in (
    (r"\ba photo from above of\b", "a photo of"),
    (r"\b(an? )?(aerial|overhead|top-down|bird'?s-eye|drone)( view| photo| shot)?( of)?\b", ""),
    (r"\b(seen |viewed |taken )?from (above|the air|a drone|high up)\b", ""),
    (r"\bseen from a distance\b", ""),
)]


def _eye(text: str) -> str:
    for rx, rep in _AERIAL:
        text = rx.sub(rep, text)
    text = " ".join(text.split()).strip(" ,")
    return text if not text.lower().startswith(("of ", "a of ")) else "a photo " + text.split(" ", 1)[1]


def eye_level(p: ClipPrompts) -> ClipPrompts:
    def fix(xs: list[str]) -> list[str]:
        out, seen = [], set()
        for x in map(_eye, xs):
            if len(x.split()) >= 3 and x.lower() not in seen:
                seen.add(x.lower())
                out.append(x)
        return out or xs
    pos = fix(p.positive)
    keys = {x.lower() for x in pos}
    return p.model_copy(update={"positive": pos, "negative": [n for n in fix(p.negative) if n.lower() not in keys]})


class EyeLevelBuilder:
    """Wraps a prompt builder (Gemma's LlmPromptBuilder or default_prompts) and drops the aerial wording."""

    def __init__(self, inner: Callable[[Target], ClipPrompts]) -> None:
        self.inner = inner

    def __call__(self, target: Target) -> ClipPrompts:
        p = eye_level(self.inner(target))
        log.info("demo CLIP prompts for %r: + %s | - %s", target.description, p.positive, p.negative)
        return p


class Query(BaseModel):
    text: str = ""


class DemoView:
    """Reads the running pipeline (latest detections, matcher tracks, cameras); never touches the GPU."""

    def __init__(self, s: Any, pipeline: Any, ctrl: Any) -> None:
        self.s, self.pipeline, self.ctrl = s, pipeline, ctrl
        try:
            from fusion import load_homography
            self.H_r2t = np.linalg.inv(load_homography(s))      # RGB px -> thermal px
        except Exception:  # noqa: BLE001
            self.H_r2t = None
        self._lock = threading.Lock()
        self._cache: tuple[float, list[dict]] = (0.0, [])
        self.best: Optional[dict] = None
        self.best_jpg: Optional[bytes] = None
        self._best_key: Any = None

    # ------------------------------------------------------------------ people
    def _warmest(self, box) -> Optional[float]:
        th = getattr(self.pipeline, "thermal", None)
        if th is None or self.H_r2t is None:
            return None
        temp, tts = th.latest()
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

    def _searching(self) -> bool:
        return getattr(self.pipeline, "matcher", None) is not None and self.ctrl.state in ("arming", "flying")

    def people(self) -> list[dict]:
        """Current people with their smoothed CLIP score; cached ~0.2 s (the stream and every phone poll call this)."""
        with self._lock:
            ts, ppl = self._cache
            if time.time() - ts < 0.2:
                return ppl
        res = getattr(self.pipeline, "last_dets", None)
        ppl = []
        if res and time.time() - res[1] <= STALE_S and self._searching():
            from matcher import AVG_WINDOW
            m = self.pipeline.matcher
            s = self.s
            for d in res[0]:
                tr = m.tracker.get(d.track_id) if d.track_id is not None else None
                hist = list(tr.match_hist[-AVG_WINDOW:]) if tr is not None else []
                score = sum(hist) / len(hist) if hist else None
                box = (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)
                temp = self._warmest(box)
                ppl.append({"box": box, "track": d.track_id, "score": score, "n": len(hist),
                            "temp_c": None if temp is None else round(temp, 1),
                            "warm": temp is not None and temp >= WARM_C, "best": False,
                            "match": score is not None and score >= s.match_threshold
                            and (len(hist) >= s.min_frames or score >= s.match_strong_threshold),
                            "crop": tr.best_crop_path if tr is not None else None})
            scored = [p for p in ppl if p["score"] is not None]
            if scored:
                best = max(scored, key=lambda p: p["score"])
                best["best"] = True
                if best["match"]:
                    self._keep_best(best)
        with self._lock:
            self._cache = (time.time(), ppl)
        return ppl

    def _keep_best(self, p: dict) -> None:
        """Photo of the best match: a fresh crop of the live frame, kept after the person leaves the view."""
        old = self.best
        if old is not None and p["score"] < old["score"] - 0.05 and time.time() - old["ts"] < 5:
            return
        cam = getattr(self.pipeline, "camera", None)
        frame = cam.latest()[0] if cam is not None else None
        jpg = None
        if frame is not None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = p["box"]
            px, py = 0.15 * (x2 - x1), 0.08 * (y2 - y1)
            x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
            x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                ok, buf = cv2.imencode(".jpg", frame[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpg = buf.tobytes() if ok else None
        if jpg is None and p.get("crop"):
            try:
                jpg = open(p["crop"], "rb").read()
            except OSError:
                pass
        if jpg is not None:
            self.best_jpg = jpg
            self.best = {"score": round(p["score"], 3), "temp_c": p["temp_c"], "warm": p["warm"], "track": p["track"],
                         "ts": time.time()}

    def reset_best(self) -> None:
        self.best, self.best_jpg = None, None

    # ------------------------------------------------------------------ views
    def state(self) -> dict:
        st, t = self.ctrl.state, self.ctrl.target
        ppl = self.people()
        hint = {"flying": None, "arming": "Starting the search ..."}.get(st, (
            "Press GO to start the live search." if st == "staged" else
            "Search ended: the flight summary is in the app." if st in ("landed", "rtl") and self.best else
            "Plan in the app chat (who to find), Confirm, then GO. Nothing flies."))
        if self.best is not None and t is not None and self.best.get("target") not in (None, t.description):
            self.reset_best()
        if self.best is not None and t is not None:
            self.best["target"] = t.description
        return {"demo": True, "target": t.description if t else None, "cutoff": self.s.match_threshold, "hint": hint,
                "searching": self._searching(), "state": st,
                "people": [{k: p[k] for k in ("box", "score", "temp_c", "warm", "best", "match")} for p in ppl],
                "best": self.best, "loop_ms": round(getattr(self.pipeline, "last_loop_ms", 0.0) or 0),
                "has_best_image": self.best_jpg is not None}

    def boxes(self) -> Optional[list[tuple]]:
        if not self._searching():
            return None
        out = []
        for p in self.people():
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


def make_source(pipeline: Any, s: Any, view: DemoView) -> Any:
    from webapp import stream

    class DemoSource(stream.PipelineSource):
        def boxes(self):
            b = view.boxes()
            return None if b is None else {"rgb": b, "thermal": None}

    return DemoSource(pipeline, s)


def make_router(view: DemoView) -> Any:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import Response

    r = APIRouter()

    @r.get("/api/search")
    def get_search():
        return view.state()

    @r.post("/api/search")
    async def post_search(q: Query):
        """The live page's search bar: same as changing the target in the app (Gemma parses it, CLIP re-prompts)."""
        text = " ".join((q.text or "").split())[:120]
        if text:
            view.reset_best()
            await view.ctrl.update_target(text)
        return view.state()

    @r.get("/api/search/best.jpg")
    def best_jpg():
        if view.best_jpg is None:
            raise HTTPException(404, "no match yet")
        return Response(view.best_jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    return r
