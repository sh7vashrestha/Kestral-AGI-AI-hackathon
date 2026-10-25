"""Live video: RGB + thermal (+ overlay) as MJPEG, plus stats / thermal-point / snapshot APIs.

One router (`make_router`) serves the same page from two places:
  * scripts/live_view.py  - opens both cameras itself (CameraSource)
  * main.py               - reuses the vision pipeline's cameras (PipelineSource, no second open)

Design:
  * FrameSource: tiny abstraction (rgb() / thermal() / feed_status() / boxes()); the cameras stay owned by the caller.
  * SharedEncoder: ONE encoder thread per distinct stream parameter set, started by the first client and stopped
    ~1 s after the last one leaves. Clients only ever receive the newest JPEG (frames are dropped, never queued).
  * Endpoints: /live, /stream/{rgb,thermal,overlay}.mjpg, /api/live/{stats,thermal_point,snapshot,palette.png}.
"""
from __future__ import annotations

import asyncio
import collections
import glob
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

import orient
from config import ROOT, Settings

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
ASSETS = {"live.js": "text/javascript", "live.css": "text/css"}
BOUNDARY = "frame"
MIME = f"multipart/x-mixed-replace; boundary={BOUNDARY}"
IDLE_GRACE_S = 1.0
STALL_S = 3.0               # no new frame for this long => feed "stalled"
PALETTES = ("inferno", "ironbow", "turbo", "gray")
THERMAL_SCALE = 4
KINDS = ("rgb", "thermal", "overlay")


# ------------------------------------------------------------------ palettes / rendering
def _build_luts() -> dict[str, np.ndarray]:
    luts: dict[str, np.ndarray] = {}
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    luts["inferno"] = cv2.applyColorMap(ramp, cv2.COLORMAP_INFERNO)
    luts["turbo"] = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)
    luts["gray"] = np.repeat(ramp, 3, axis=1).reshape(256, 1, 3)
    stops = [(0.00, (0, 0, 0)), (0.18, (30, 0, 110)), (0.38, (140, 0, 150)), (0.55, (225, 40, 60)),
             (0.72, (255, 140, 0)), (0.88, (255, 225, 60)), (1.00, (255, 255, 255))]   # RGB
    xs = np.linspace(0, 1, 256)
    rgb = np.stack([np.interp(xs, [s[0] for s in stops], [s[1][c] for s in stops]) for c in range(3)], axis=1)
    luts["ironbow"] = rgb[:, ::-1].astype(np.uint8).reshape(256, 1, 3)               # -> BGR
    return luts


LUTS = _build_luts()


def palette_png(name: str) -> bytes:
    lut = LUTS.get(name, LUTS["inferno"]).reshape(1, 256, 3)
    ok, buf = cv2.imencode(".png", np.repeat(lut, 4, axis=0))
    return buf.tobytes()


def auto_range(temp: np.ndarray) -> tuple[float, float]:
    lo = float(np.percentile(temp, 1))
    hi = max(float(np.percentile(temp, 99.5)), lo + 10.0)
    return lo, hi


def colorize_thermal(temp: np.ndarray, lo: float, hi: float, palette: str = "inferno",
                     scale: int = THERMAL_SCALE) -> np.ndarray:
    """Temperature (deg C) -> BGR image, nearest-neighbour upscaled by `scale`."""
    g = np.clip((temp - lo) / max(hi - lo, 1e-3) * 255.0, 0, 255).astype(np.uint8)
    img = cv2.LUT(cv2.merge([g, g, g]), LUTS.get(palette, LUTS["inferno"]))
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return img


def temp_at(temp: np.ndarray, x: float, y: float) -> tuple[float, int, int]:
    """deg C at normalised page coordinates (0..1, origin top-left). Out-of-range values are clamped."""
    h, w = temp.shape[:2]
    ix = int(min(max(x, 0.0), 1.0) * w)
    iy = int(min(max(y, 0.0), 1.0) * h)
    ix, iy = min(ix, w - 1), min(iy, h - 1)
    return float(temp[iy, ix]), ix, iy


def hottest(temp: np.ndarray) -> tuple[float, int, int]:
    iy, ix = np.unravel_index(int(np.argmax(temp)), temp.shape)
    return float(temp[iy, ix]), int(ix), int(iy)


def blend_overlay(rgb_small: np.ndarray, thermal_bgr: np.ndarray, H_t2out: np.ndarray, alpha: float) -> np.ndarray:
    """Warp the (small) coloured thermal image into the RGB view and alpha-blend where it has coverage."""
    h, w = rgb_small.shape[:2]
    warped = cv2.warpPerspective(thermal_bgr, H_t2out, (w, h), flags=cv2.INTER_LINEAR)
    cover = cv2.warpPerspective(np.full(thermal_bgr.shape[:2], 255, np.uint8), H_t2out, (w, h), flags=cv2.INTER_NEAREST)
    a = (cover.astype(np.float32) / 255.0 * float(alpha))[..., None]
    return (rgb_small.astype(np.float32) * (1 - a) + warped.astype(np.float32) * a).astype(np.uint8)


def draw_boxes(img: np.ndarray, boxes: list[tuple], sx: float, sy: float, color=(255, 170, 60), label=True) -> None:
    """boxes: (x1, y1, x2, y2, conf[, (text, bgr, thickness)]) - the optional style is used by scripts/demo_search.py."""
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        text, col, th = (b[5] if len(b) > 5 and b[5] else (None, color, 2))
        p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(img, p1, p2, col, th)
        if text is None and label and len(b) > 4 and b[4] is not None:
            text = f"{b[4]:.2f}"
        if text:
            (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            y = max(tht + 6, p1[1])
            cv2.rectangle(img, (p1[0], y - tht - 6), (p1[0] + tw + 6, y), col, -1)
            cv2.putText(img, text, (p1[0] + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ------------------------------------------------------------------ frame sources
class FrameSource:
    """What the router needs. Frames are BGR uint8 (rgb) and float32 deg C (thermal), plus a wall-clock timestamp."""
    mode = "standalone"
    has_boxes = False
    mission_url: Optional[str] = None

    def rgb(self) -> tuple[Optional[np.ndarray], float]:
        return None, 0.0

    def thermal(self) -> tuple[Optional[np.ndarray], float]:
        return None, 0.0

    def feed_status(self) -> dict:
        return {"rgb": {"state": "unavailable", "error": "no source"},
                "thermal": {"state": "unavailable", "error": "no source"}}

    def source_fps(self) -> dict:
        return {"rgb": 0.0, "thermal": 0.0}

    def thermal_mask(self) -> Optional[np.ndarray]:
        return None

    def boxes(self) -> Optional[dict]:
        """{"rgb": [(x1,y1,x2,y2,conf)] in full-res RGB px, "thermal": [(...)] in thermal px} or None."""
        return None

    def rgb_size(self) -> Optional[tuple[int, int]]:
        return None

    # ---- orientation (rotate/flip) of the two cameras: target is "camera" or "thermal"
    can_orient = False

    def get_orient(self, target: str) -> tuple[int, str]:
        return 0, "none"

    def set_orient(self, target: str, rotate: int, flip: str) -> None:
        raise HTTPException(status_code=409, detail="orientation cannot be previewed here - use scripts/live_view.py")


ENV_KEYS = {"camera": ("CAMERA_ROTATE", "CAMERA_FLIP"), "thermal": ("THERMAL_ROTATE", "THERMAL_FLIP")}


def read_env_values(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def write_env_keys(path: Path, values: dict[str, str]) -> None:
    """Set KEY=value for the given keys only; every other line (comments, order, secrets) is preserved byte for byte.
    Atomic: temp file + rename."""
    try:
        lines = path.read_text().splitlines(keepends=True)
    except FileNotFoundError:
        lines = []
    left = dict(values)
    for i, line in enumerate(lines):
        k, sep, _ = line.partition("=")
        if sep and not line.lstrip().startswith("#") and k.strip() in left:
            nl = "\n" if line.endswith("\n") else ""
            lines[i] = f"{k.strip()}={left.pop(k.strip())}{nl}"
    if left:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines += [f"{k}={v}\n" for k, v in left.items()]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(lines))
    try:
        os.chmod(tmp, path.stat().st_mode & 0o777)
    except OSError:
        pass
    os.replace(tmp, path)


def find_holder(dev: str) -> str:
    """'pid NNN (cmdline)' of another process holding `dev` open, '' if none / unknown."""
    try:
        target = os.path.realpath(dev)
    except OSError:
        return ""
    me = os.getpid()
    for p in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            pid = int(p.split("/")[2])
            if pid != me and os.path.realpath(p) == target:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:80]
                return f"held by pid {pid} ({cmd.strip()})"
        except (OSError, ValueError, IndexError):
            continue
    return ""


class CamerasSource(FrameSource):
    """Wraps camera.Camera + thermal.ThermalCamera objects (either may be None / disabled)."""

    def __init__(self, camera: Any, thermal: Any, settings: Settings, thermal_dev: str = "") -> None:
        self._camera, self._thermal, self.s = camera, thermal, settings
        self.thermal_dev = thermal_dev
        self._holder_cache: tuple[float, str] = (0.0, "")

    def _cams(self) -> tuple[Any, Any]:
        return self._camera, self._thermal

    def _disabled_reason(self) -> str:
        return ""

    def rgb(self):
        cam, _ = self._cams()
        if cam is None:
            return None, 0.0
        f, ts, _ = cam.latest()
        return f, ts

    def thermal(self):
        _, th = self._cams()
        if th is None or not getattr(th, "enabled", True):
            return None, 0.0
        return th.latest()

    def thermal_mask(self):
        _, th = self._cams()
        return getattr(th, "mask", None) if th is not None else None

    def rgb_size(self):
        f, _ = self.rgb()
        return (f.shape[1], f.shape[0]) if f is not None else None

    def source_fps(self):
        cam, th = self._cams()
        return {"rgb": round(cam.fps(), 1) if cam is not None and hasattr(cam, "fps") else 0.0,
                "thermal": round(th.fps(), 1) if th is not None and hasattr(th, "fps") else 0.0}

    can_orient = True

    def get_orient(self, target):
        o = self._camera if target == "camera" else self._thermal
        if o is not None and hasattr(o, "rotate"):
            return o.rotate, o.flip
        return (self.s.camera_rotate, self.s.camera_flip) if target == "camera" else (self.s.thermal_rotate, self.s.thermal_flip)

    def set_orient(self, target, rotate, flip):
        if target not in ENV_KEYS:
            raise HTTPException(status_code=400, detail="target must be camera or thermal")
        try:
            r, f = orient.check(rotate, flip)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        o = self._camera if target == "camera" else self._thermal
        # the shared Settings object feeds find_hotspots / the default homography / effective optics
        setattr(self.s, f"{target}_rotate", r)
        setattr(self.s, f"{target}_flip", f)
        if o is not None:
            o.set_orient(r, f)

    def _holder(self) -> str:
        if not self.thermal_dev:
            return ""
        now = time.monotonic()
        if now - self._holder_cache[0] > 10:
            self._holder_cache = (now, find_holder(self.thermal_dev))
        return self._holder_cache[1]

    def feed_status(self) -> dict:
        cam, th = self._cams()
        out: dict = {}
        # ---- RGB
        if cam is None:
            out["rgb"] = {"state": "disabled", "error": self._disabled_reason() or "no RGB camera", "hint": ""}
        else:
            st = cam.status() if hasattr(cam, "status") else {"state": "ok", "error": None, "hint": ""}
            state = st.get("state", "ok")
            f, ts, _ = cam.latest()
            if state == "ok" and f is not None and time.time() - ts > STALL_S:
                state = "stalled"
            out["rgb"] = {"state": state, "source": st.get("source"), "error": st.get("error"),
                          "hint": st.get("hint", ""), "reconnects": st.get("reconnects"), "attempts": st.get("attempts"),
                          "next_retry_s": st.get("next_retry_s")}
        # ---- thermal
        if th is None or not getattr(th, "enabled", True):
            out["thermal"] = {"state": "disabled", "error": self._disabled_reason() or "THERMAL_SOURCE is none",
                              "hint": "set THERMAL_SOURCE in .env or run scripts/live_view.py --thermal auto"}
        else:
            t, ts = th.latest()
            err = getattr(th, "last_error", None)
            if t is None:
                hold = self._holder()
                out["thermal"] = {"state": "unavailable" if err else "starting", "source": getattr(th, "source", ""),
                                  "error": err, "hint": hold, "reconnects": getattr(th, "reconnects", None)}
            elif time.time() - ts > STALL_S:
                out["thermal"] = {"state": "stalled", "source": getattr(th, "source", ""), "error": err, "hint": ""}
            else:
                out["thermal"] = {"state": "ok", "source": getattr(th, "source", ""), "error": None, "hint": "",
                                  "reconnects": getattr(th, "reconnects", None)}
        return out


class PipelineSource(CamerasSource):
    """main.py: reuse the running VisionPipeline's cameras and latest detections. Never opens a device."""
    mode = "pipeline"

    def __init__(self, pipeline: Any, settings: Settings) -> None:
        super().__init__(None, None, settings)
        self.pipeline = pipeline
        self.mission_url = "/"

    @property
    def has_boxes(self) -> bool:  # type: ignore[override]
        return getattr(self.pipeline, "state", "") == "ready"

    def _cams(self):
        return getattr(self.pipeline, "camera", None), getattr(self.pipeline, "thermal", None)

    def _disabled_reason(self) -> str:
        try:
            st = self.pipeline.status()
        except Exception:  # noqa: BLE001
            return "vision pipeline unavailable"
        if st.get("state") == "disabled":
            return f"{st.get('reason', 'vision disabled')} - run scripts/live_view.py"
        return st.get("reason") or ""

    def boxes(self):
        res = getattr(self.pipeline, "last_dets", None)       # (detections, ts) set by VisionPipeline._process
        if not res or time.time() - res[1] > 1.5:
            return None
        rgb = [(d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2,
                d.c_person if getattr(d, "c_person", None) is not None else d.c_yolo) for d in res[0]]
        return {"rgb": rgb, "thermal": None}


class SyntheticSource(FrameSource):
    """Moving gradient + warm blob; for tests, demos and screenshots (no hardware)."""

    def __init__(self, size=(1280, 720), fps=15.0, with_thermal=True) -> None:
        self.size, self.fps, self.with_thermal = size, fps, with_thermal
        self.rgb_calls = 0
        self.t0 = time.time()
        self.o = {"camera": (0, "none"), "thermal": (0, "none")}

    def _tick(self) -> float:
        return time.time()

    def rgb(self):
        self.rgb_calls += 1
        w, h = self.size
        ts = int(time.time() * self.fps) / self.fps
        ph = (ts - self.t0) * 40
        x = (np.arange(w, dtype=np.float32)[None, :] + ph) % w
        img = np.zeros((h, w, 3), np.uint8)
        img[..., 0] = (x / w * 200).astype(np.uint8)
        img[..., 1] = (np.arange(h)[:, None] / h * 160).astype(np.uint8)
        img[..., 2] = 90
        cv2.circle(img, (int(w * 0.5 + 200 * np.sin(ph / 80)), int(h * 0.55)), 60, (220, 220, 235), -1)
        cv2.putText(img, "SYNTHETIC RGB", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
        return np.ascontiguousarray(orient.apply(img, *self.o["camera"])), ts

    def thermal(self):
        if not self.with_thermal:
            return None, 0.0
        ts = int(time.time() * 9) / 9
        ph = (ts - self.t0) * 0.8
        yy, xx = np.mgrid[0:120, 0:160].astype(np.float32)
        t = 21 + 1.5 * np.sin(xx / 25 + ph) + 0.01 * yy
        cx, cy = 80 + 35 * np.sin(ph), 60 + 15 * np.cos(ph)
        t += 15 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 6 ** 2))
        t += 7 * np.exp(-((xx - 30) ** 2 + (yy - 95) ** 2) / (2 * 4 ** 2))
        return np.ascontiguousarray(orient.apply(t.astype(np.float32), *self.o["thermal"])), ts

    can_orient = True

    def get_orient(self, target):
        return self.o[target]

    def set_orient(self, target, rotate, flip):
        if target not in self.o:
            raise HTTPException(status_code=400, detail="target must be camera or thermal")
        try:
            self.o[target] = orient.check(rotate, flip)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    def feed_status(self):
        return {"rgb": {"state": "ok", "error": None, "hint": "", "source": "synthetic"},
                "thermal": ({"state": "ok", "error": None, "hint": "", "source": "synthetic"} if self.with_thermal else
                            {"state": "disabled", "error": "THERMAL_SOURCE is none", "hint": ""})}

    def source_fps(self):
        return {"rgb": self.fps, "thermal": 9.0 if self.with_thermal else 0.0}

    def rgb_size(self):
        return orient.out_size(*self.size, self.o["camera"][0])


# ------------------------------------------------------------------ shared encoder
class SharedEncoder:
    """One encoder thread for one stream parameter set; runs only while it has clients."""

    def __init__(self, kind: str, params: dict, produce: Callable[["SharedEncoder"], Optional[tuple[bytes, float]]]) -> None:
        self.kind, self.params, self._produce = kind, params, produce
        self.period = 1.0 / params["fps"]
        self._lock = threading.Lock()
        self._clients = 0
        self._thread: Optional[threading.Thread] = None
        self._jpeg: Optional[bytes] = None
        self._seq = 0
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._times: collections.deque[float] = collections.deque(maxlen=40)
        self.frames_encoded = 0
        self.encode_ms = 0.0
        self.last_src_ts = 0.0
        self.range: Optional[tuple[float, float]] = None     # thermal: range actually used
        self._idle_since = 0.0
        self.last_used = time.monotonic()
        self.running = False

    # -- clients
    def acquire(self) -> None:
        with self._lock:
            self._clients += 1
            if self._thread is None or not self._thread.is_alive():
                self.running = True
                self._thread = threading.Thread(target=self._run, name=f"enc-{self.kind}", daemon=True)
                self._thread.start()

    def release(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)
            if self._clients == 0:
                self._idle_since = time.monotonic()

    @property
    def clients(self) -> int:
        return self._clients

    def reset(self) -> None:
        """Forget the last JPEG (it has the old orientation / settings); new clients wait for a fresh frame."""
        self._jpeg = None
        self.last_src_ts = 0.0
        self.range = None

    def latest(self) -> tuple[Optional[bytes], int]:
        return self._jpeg, self._seq

    def add_waiter(self, loop, ev) -> None:
        with self._lock:
            self._waiters.append((loop, ev))

    def remove_waiter(self, loop, ev) -> None:
        with self._lock:
            try:
                self._waiters.remove((loop, ev))
            except ValueError:
                pass

    def fps(self) -> float:
        ts = list(self._times)
        now = time.monotonic()
        ts = [t for t in ts if now - t < 4.0]
        return round((len(ts) - 1) / (ts[-1] - ts[0]), 1) if len(ts) >= 2 and ts[-1] > ts[0] else 0.0

    # -- thread
    def _run(self) -> None:
        next_t = time.monotonic()
        try:
            while True:
                with self._lock:
                    if self._clients == 0 and time.monotonic() - self._idle_since > IDLE_GRACE_S:
                        self.running = False
                        self._thread = None
                        return
                t0 = time.monotonic()
                if self._clients > 0:
                    try:
                        out = self._produce(self)
                    except Exception:  # noqa: BLE001
                        log.exception("%s encoder failed", self.kind)
                        out = None
                    if out is not None:
                        jpeg, src_ts = out
                        self.encode_ms = (time.monotonic() - t0) * 1000
                        self.last_src_ts = src_ts
                        self._jpeg, self._seq = jpeg, self._seq + 1
                        self.frames_encoded += 1
                        self._times.append(time.monotonic())
                        with self._lock:
                            waiters = list(self._waiters)
                        for loop, ev in waiters:
                            try:
                                loop.call_soon_threadsafe(ev.set)
                            except RuntimeError:
                                pass
                next_t += self.period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, 0.25) if self._clients == 0 else delay)
                else:
                    next_t = time.monotonic()
        finally:
            self.running = False


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class LiveService:
    """Owns the encoders, the homography cache and the stats counters for one FrameSource."""

    def __init__(self, source: FrameSource, settings: Settings, *, default_fps: int = 10, data_dir: Optional[Path] = None,
                 env_path: Optional[Path] = None):
        self.src, self.s = source, settings
        self.env_path = Path(env_path) if env_path else ROOT / ".env"
        self.default_fps = int(default_fps)
        self.data_dir = Path(data_dir or settings.data_dir)
        self._encoders: dict[tuple, SharedEncoder] = {}
        self._lock = threading.Lock()
        self._H: Optional[np.ndarray] = None
        self._H_t = 0.0
        self._cpu_prev: Optional[tuple[float, float, float]] = None   # (wall, proc_cpu, sys_busy/total)
        self._sys_prev: Optional[tuple[int, int]] = None
        self._range_ema: dict[tuple, tuple[float, float]] = {}
        self.t_start = time.time()

    # ---- params
    def norm_params(self, kind: str, q: dict) -> dict:
        def f(name, default, lo, hi, cast=float):
            try:
                return _clamp(cast(q.get(name, default)), lo, hi)
            except (TypeError, ValueError):
                return default
        p: dict = {"fps": int(f("fps", self.default_fps, 1, 30, int)), "boxes": str(q.get("boxes", "0")) in ("1", "true")}
        if kind in ("rgb", "overlay"):
            p["w"] = int(f("w", 960, 160, 1920, int))
        if kind in ("thermal", "overlay"):
            pal = q.get("palette", "inferno")
            p["palette"] = pal if pal in PALETTES else "inferno"
            mn, mx = q.get("min"), q.get("max")
            try:
                if mn not in (None, "") and mx not in (None, ""):
                    lo, hi = round(float(mn), 1), round(float(mx), 1)
                    if hi > lo:
                        p["min"], p["max"] = lo, hi
            except ValueError:
                pass
        if kind == "overlay":
            p["alpha"] = round(f("alpha", 0.5, 0.0, 1.0), 2)
        return p

    def encoder(self, kind: str, params: dict) -> SharedEncoder:
        key = (kind, tuple(sorted(params.items())))
        with self._lock:
            for k in [k for k, e in self._encoders.items()
                      if e.clients == 0 and not e.running and time.monotonic() - e.last_used > 30]:
                del self._encoders[k]
            enc = self._encoders.get(key)
            if enc is None:
                enc = SharedEncoder(kind, params, getattr(self, f"_make_{kind}"))
                self._encoders[key] = enc
            enc.last_used = time.monotonic()
            return enc

    def encoders_for(self, kind: str) -> list[SharedEncoder]:
        with self._lock:
            return [e for e in self._encoders.values() if e.kind == kind and e.running]

    # ---- orientation (live preview + "Save as default")
    def orientation(self) -> dict:
        env = read_env_values(self.env_path)
        out: dict = {"editable": bool(self.src.can_orient), "env_file": str(self.env_path)}
        for t, (kr, kf) in ENV_KEYS.items():
            r, f = self.src.get_orient(t)
            fields = type(self.s).model_fields
            try:
                sr = orient.check(env.get(kr, fields[kr.lower()].default), "none")[0]
            except ValueError:
                sr = fields[kr.lower()].default
            sf = env.get(kf, fields[kf.lower()].default)
            out[t] = {"rotate": r, "flip": f, "label": orient.label(r, f), "saved_rotate": sr, "saved_flip": sf,
                      "dirty": (r, f) != (sr, sf)}
        out["camera"]["out_size"] = list(orient.out_size(self.s.camera_width, self.s.camera_height, out["camera"]["rotate"]))
        return out

    def set_orientation(self, target: str, rotate: Optional[int], flip: Optional[str]) -> dict:
        if not self.src.can_orient:
            raise HTTPException(status_code=409, detail="read-only here: this page shows the running app's cameras. "
                                "Use scripts/live_view.py to preview, save, then restart main.py")
        if target not in ENV_KEYS:
            raise HTTPException(status_code=400, detail="target must be camera or thermal")
        cr, cf = self.src.get_orient(target)
        try:
            r, f = orient.check(cr if rotate is None else rotate, cf if flip is None else flip)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        self.src.set_orient(target, r, f)
        setattr(self.s, f"{target}_rotate", r)          # effective optics + default homography follow the preview
        setattr(self.s, f"{target}_flip", f)
        self._H = None
        self._range_ema.clear()
        with self._lock:
            for e in self._encoders.values():
                e.reset()
        return self.orientation()

    def save_orientation(self) -> dict:
        if not self.src.can_orient:
            raise HTTPException(status_code=409, detail="nothing to save: orientation is not editable here")
        vals: dict[str, str] = {}
        for t, (kr, kf) in ENV_KEYS.items():
            r, f = self.src.get_orient(t)
            vals[kr], vals[kf] = str(r), f
        try:
            write_env_keys(self.env_path, vals)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"cannot write {self.env_path}: {exc}")
        return {"saved": vals, "env_file": str(self.env_path), "restart_required": True,
                "note": "main.py (the flight pipeline) must be restarted to use these. A thermal-to-RGB calibration made "
                        "with other orientations is ignored until you recalibrate.",
                "orientation": self.orientation()}

    # ---- homography
    def homography(self) -> np.ndarray:
        if self._H is None or time.monotonic() - self._H_t > 5:
            try:
                from fusion import load_homography
                self._H = load_homography(self.s)
            except Exception:  # noqa: BLE001
                log.exception("homography load failed")
                from fusion import default_homography
                self._H = default_homography(self.s)
            self._H_t = time.monotonic()
        return self._H

    def _h_thermal_to_view(self, rgb_full_size: tuple[int, int], out_size: tuple[int, int], th_size: tuple[int, int],
                           th_img_scale: int) -> np.ndarray:
        """thermal-image px (scaled by th_img_scale) -> output (downscaled RGB) px."""
        H = self.homography()
        cw, ch = self.s.eff_camera_width, self.s.eff_camera_height
        fw, fh = rgb_full_size
        ow, oh = out_size
        # H maps thermal px -> RGB px at (cw,ch). Re-target to the real frame, then to the output size.
        S = np.diag([ow / cw, oh / ch, 1.0])
        T = np.diag([1.0 / th_img_scale, 1.0 / th_img_scale, 1.0])
        return S @ H @ T

    # ---- producers (run in encoder threads)
    def _encode(self, img: np.ndarray, q: int) -> bytes:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()

    def _rgb_small(self, frame: np.ndarray, w: int) -> np.ndarray:
        h = max(1, int(round(frame.shape[0] * w / frame.shape[1])))
        return frame if w >= frame.shape[1] else cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    def _thermal_range(self, enc: SharedEncoder, temp: np.ndarray) -> tuple[float, float]:
        p = enc.params
        if "min" in p:
            enc.range = (p["min"], p["max"])
            return enc.range
        lo, hi = auto_range(temp)
        prev = enc.range
        if prev is not None:                      # smooth so the colours do not flicker
            lo, hi = prev[0] + 0.2 * (lo - prev[0]), prev[1] + 0.2 * (hi - prev[1])
        enc.range = (lo, hi)
        return enc.range

    def _make_rgb(self, enc: SharedEncoder):
        frame, ts = self.src.rgb()
        if frame is None or ts == enc.last_src_ts:
            return None
        small = self._rgb_small(frame, enc.params["w"])
        if enc.params["boxes"]:
            b = self.src.boxes()
            if b and b.get("rgb"):
                small = small.copy()
                draw_boxes(small, b["rgb"], small.shape[1] / frame.shape[1], small.shape[0] / frame.shape[0])
        return self._encode(small, 70), ts

    def _thermal_frame(self) -> tuple[Optional[np.ndarray], float]:
        temp, ts = self.src.thermal()
        if temp is None:
            return None, 0.0
        m = self.src.thermal_mask()
        if m is not None and m.shape == temp.shape and m.any():
            from thermal import fix_pixels
            temp = fix_pixels(temp, m)
        return temp, ts

    def _hotspot_boxes(self, temp: np.ndarray) -> list[tuple]:
        from thermal import find_hotspots
        return [(h.bbox.x1, h.bbox.y1, h.bbox.x2, h.bbox.y2, h.peak_c) for h in
                find_hotspots(temp, self.s, mask=self.src.thermal_mask())]

    def _make_thermal(self, enc: SharedEncoder):
        temp, ts = self._thermal_frame()
        if temp is None or ts == enc.last_src_ts:
            return None
        lo, hi = self._thermal_range(enc, temp)
        img = colorize_thermal(temp, lo, hi, enc.params["palette"])
        if enc.params["boxes"] and self.src.has_boxes:
            spots = self._hotspot_boxes(temp)
            draw_boxes(img, [(*s[:4], None) for s in spots], THERMAL_SCALE, THERMAL_SCALE, (60, 220, 90))
            for s in spots:
                cv2.putText(img, f"{s[4]:.1f}C", (int(s[0] * THERMAL_SCALE) + 2, max(12, int(s[1] * THERMAL_SCALE) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 90), 1, cv2.LINE_AA)
        return self._encode(img, 75), ts

    def _make_overlay(self, enc: SharedEncoder):
        frame, rts = self.src.rgb()
        temp, tts = self._thermal_frame()
        if frame is None or temp is None:
            return None
        ts = max(rts, tts)
        if ts == enc.last_src_ts:
            return None
        small = self._rgb_small(frame, enc.params["w"])
        lo, hi = self._thermal_range(enc, temp)
        cimg = colorize_thermal(temp, lo, hi, enc.params["palette"], scale=1)
        Hv = self._h_thermal_to_view((frame.shape[1], frame.shape[0]), (small.shape[1], small.shape[0]),
                                     temp.shape[::-1], 1)
        out = blend_overlay(small, cimg, Hv, enc.params["alpha"])
        return self._encode(out, 70), ts

    # ---- stats
    def _sys_stats(self) -> dict:
        out: dict = {}
        try:
            with open("/proc/stat") as fh:
                v = [int(x) for x in fh.readline().split()[1:]]
            idle, total = v[3] + v[4], sum(v)
            if self._sys_prev:
                di, dt = idle - self._sys_prev[0], total - self._sys_prev[1]
                out["cpu_pct"] = round(100.0 * (1 - di / dt), 1) if dt > 0 else 0.0
            self._sys_prev = (idle, total)
            mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
            out["ram_used_mb"] = round((mem["MemTotal"] - mem["MemAvailable"]) / 1024)
            out["ram_total_mb"] = round(mem["MemTotal"] / 1024)
            now, pc = time.monotonic(), time.process_time()
            if self._cpu_prev:
                out["proc_cpu_pct"] = round(100.0 * (pc - self._cpu_prev[1]) / max(now - self._cpu_prev[0], 1e-3), 1)
            self._cpu_prev = (now, pc, 0.0)
            for l in open("/proc/self/status"):
                if l.startswith("VmRSS"):
                    out["proc_rss_mb"] = round(int(l.split()[1]) / 1024)
        except Exception:  # noqa: BLE001
            pass
        return out

    def stats(self) -> dict:
        feeds = self.src.feed_status()
        sfps = self.src.source_fps()
        now = time.time()
        rgb_f, rgb_ts = self.src.rgb()
        temp, th_ts = self._thermal_frame()
        out: dict = {"mode": self.src.mode, "mission_url": self.src.mission_url, "has_boxes": bool(self.src.has_boxes),
                     "defaults": {"fps": self.default_fps}, "palettes": list(PALETTES), "ts": now,
                     "uptime_s": round(now - self.t_start, 1)}
        rgb: dict = {**feeds.get("rgb", {}), "source_fps": sfps["rgb"], "clients": 0, "fps": 0.0}
        th: dict = {**feeds.get("thermal", {}), "source_fps": sfps["thermal"], "clients": 0, "fps": 0.0}
        ov: dict = {"clients": 0, "fps": 0.0}
        for kind, d in (("rgb", rgb), ("thermal", th), ("overlay", ov)):
            encs = self.encoders_for(kind)
            d["clients"] = sum(e.clients for e in encs)
            d["fps"] = round(sum(e.fps() for e in encs), 1) if encs else 0.0
            if encs and kind != "overlay":
                d["encode_ms"] = round(max(e.encode_ms for e in encs), 1)
        if rgb_f is not None:
            rgb.update(w=int(rgb_f.shape[1]), h=int(rgb_f.shape[0]), age_s=round(now - rgb_ts, 2),
                       brightness=round(float(rgb_f[::16, ::16].mean()), 1))
        if temp is not None:
            hv, hx, hy = hottest(temp)
            hh, hw = temp.shape
            th.update(w=int(hw), h=int(hh), age_s=round(now - th_ts, 2), min_c=round(float(temp.min()), 2),
                      median_c=round(float(np.median(temp)), 2), max_c=round(float(temp.max()), 2),
                      hottest={"x": round((hx + 0.5) / hw, 4), "y": round((hy + 0.5) / hh, 4), "px": [hx, hy],
                               "c": round(hv, 2)})
            encs = self.encoders_for("thermal") or self.encoders_for("overlay")
            rng = next((e.range for e in encs if e.range), None) or auto_range(temp)
            th["range"] = [round(rng[0], 1), round(rng[1], 1)]
            if rgb_f is not None:
                try:
                    v = self._h_thermal_to_view((rgb_f.shape[1], rgb_f.shape[0]), (1, 1), (hw, hh), 1) @ \
                        np.array([hx + 0.5, hy + 0.5, 1.0])
                    th["hottest"]["overlay"] = {"x": round(float(v[0] / v[2]), 4), "y": round(float(v[1] / v[2]), 4)}
                except Exception:  # noqa: BLE001
                    pass
        out.update(rgb=rgb, thermal=th, overlay=ov, system=self._sys_stats(), orientation=self.orientation())
        return out

    # ---- snapshot
    def snapshot(self) -> dict:
        frame, _ = self.src.rgb()
        temp, _ = self._thermal_frame()
        if frame is None and temp is None:
            raise HTTPException(status_code=503, detail="no camera frames available to save")
        d = self.data_dir / "snapshots" / time.strftime("%Y%m%d_%H%M%S")
        n = 1
        while d.exists():
            d = d.with_name(f"{d.name.split('_x')[0]}_x{n}")
            n += 1
        d.mkdir(parents=True)
        saved: dict[str, str] = {}
        if frame is not None:
            cv2.imwrite(str(d / "rgb.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved["rgb"] = str(d / "rgb.jpg")
        if temp is not None:
            np.save(d / "thermal.npy", temp.astype(np.float32))
            lo, hi = auto_range(temp)
            cimg = colorize_thermal(temp, lo, hi, "inferno")
            cv2.imwrite(str(d / "thermal.png"), cimg)
            saved["thermal_npy"], saved["thermal_png"] = str(d / "thermal.npy"), str(d / "thermal.png")
            if frame is not None:
                small = self._rgb_small(frame, 960)
                Hv = self._h_thermal_to_view((frame.shape[1], frame.shape[0]), (small.shape[1], small.shape[0]),
                                             temp.shape[::-1], 1)
                cv2.imwrite(str(d / "overlay.jpg"),
                            blend_overlay(small, colorize_thermal(temp, lo, hi, "inferno", scale=1), Hv, 0.5),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved["overlay"] = str(d / "overlay.jpg")
        meta = {"ts": time.time(), "orientation": {t: list(self.src.get_orient(t)) for t in ENV_KEYS}, "thermal_range_c": [round(lo, 1), round(hi, 1)] if temp is not None else None,
                "feeds": self.src.feed_status()}
        (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        return {"dir": str(d), "files": saved}


# ------------------------------------------------------------------ router
def make_router(source: FrameSource, settings: Settings, *, default_fps: int = 10,
                data_dir: Optional[Path] = None, env_path: Optional[Path] = None) -> APIRouter:
    svc = LiveService(source, settings, default_fps=default_fps, data_dir=data_dir, env_path=env_path)
    r = APIRouter()
    r.live_service = svc          # type: ignore[attr-defined]  (tests / scripts)
    nocache = {"Cache-Control": "no-store"}

    @r.get("/live")
    async def page() -> FileResponse:
        return FileResponse(STATIC_DIR / "live.html", headers={"Cache-Control": "no-cache"})

    @r.get("/live-assets/{name}")
    async def asset(name: str) -> FileResponse:
        if name not in ASSETS:
            raise HTTPException(status_code=404)
        return FileResponse(STATIC_DIR / name, media_type=ASSETS[name], headers={"Cache-Control": "no-cache"})

    async def mjpeg(kind: str, request: Request):
        q = dict(request.query_params)
        feeds = source.feed_status()
        need = ("rgb",) if kind == "rgb" else ("thermal",) if kind == "thermal" else ("rgb", "thermal")
        for f in need:
            fr = source.rgb() if f == "rgb" else source.thermal()
            if fr[0] is None:
                st = feeds.get(f, {})
                return JSONResponse({"error": f"{f} unavailable", "state": st.get("state"), "detail": st.get("error"),
                                     "hint": st.get("hint")}, status_code=503, headers=nocache)
        enc = svc.encoder(kind, svc.norm_params(kind, q))
        loop = asyncio.get_running_loop()
        ev = asyncio.Event()

        async def gen():
            enc.acquire()
            enc.add_waiter(loop, ev)
            last = -1
            try:
                while True:
                    ev.clear()
                    jpeg, seq = enc.latest()
                    if jpeg is not None and seq != last:
                        last = seq
                        yield (f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
                               ).encode() + jpeg + b"\r\n"
                        continue
                    try:
                        await asyncio.wait_for(ev.wait(), 1.0)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            return
            finally:
                enc.remove_waiter(loop, ev)
                enc.release()

        return StreamingResponse(gen(), media_type=MIME, headers=nocache)

    @r.get("/stream/rgb.mjpg")
    async def rgb_stream(request: Request):
        return await mjpeg("rgb", request)

    @r.get("/stream/thermal.mjpg")
    async def thermal_stream(request: Request):
        return await mjpeg("thermal", request)

    @r.get("/stream/overlay.mjpg")
    async def overlay_stream(request: Request):
        return await mjpeg("overlay", request)

    @r.get("/api/live/stats")
    def stats() -> JSONResponse:
        return JSONResponse(svc.stats(), headers=nocache)

    @r.get("/api/live/thermal_point")
    def thermal_point(x: float, y: float) -> dict:
        temp, ts = svc._thermal_frame()
        if temp is None:
            raise HTTPException(status_code=503, detail="thermal unavailable")
        c, ix, iy = temp_at(temp, x, y)
        return {"c": round(c, 2), "px": [ix, iy], "x": x, "y": y, "age_s": round(time.time() - ts, 2)}

    @r.post("/api/live/orientation")
    async def orientation(request: Request) -> dict:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="JSON body expected: {target, rotate?, flip?}")
        return svc.set_orientation(str(body.get("target", "camera")), body.get("rotate"), body.get("flip"))

    @r.post("/api/live/orientation/save")
    def orientation_save() -> dict:
        return svc.save_orientation()

    @r.post("/api/live/snapshot")
    def snapshot() -> dict:
        return svc.snapshot()

    @r.get("/api/live/palette.png")
    def palette(name: str = "inferno") -> Response:
        return Response(palette_png(name if name in PALETTES else "inferno"), media_type="image/png",
                        headers={"Cache-Control": "max-age=86400"})

    return r
