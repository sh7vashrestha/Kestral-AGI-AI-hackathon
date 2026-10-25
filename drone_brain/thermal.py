"""FLIR Lepton 3.x (PureThermal, radiometric Y16) capture + hotspot detection.

Public API:
    tc = ThermalCamera(settings).start()
    temp_c, ts = tc.latest()                 # float32 (120,160) deg C or (None, 0.0)
    temp_c, ts = tc.nearest(t, max_age_s)    # frame closest to t within max_age_s, else (None, 0.0)
    spots = find_hotspots(temp_c, settings, alt_m=telemetry.alt_rel_m)
    img = colorize(temp_c)                   # BGR for saving / alert images

Sources (settings.thermal_source): v4l2:/dev/... | file:<folder of .npy deg C or 16-bit .png centikelvin> | none.
Stuck pixels: static list (settings.thermal_stuck_pixels) + dynamic detector (StuckPixelDetector): isolated
single pixels that deviate from their 3x3 median by > N deg C in most recent frames. Masked pixels are
replaced by the median of their unmasked neighbours and can never form a hotspot.
"""
from __future__ import annotations

import collections
import logging
import math
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import orient
import shutdown
from config import Settings
from models import BBox, HotSpot

log = logging.getLogger(__name__)

T_W, T_H = 160, 120
BG_KSIZE = 31          # local-background median window (px, odd)
OPEN_TIMEOUT_S = 10.0


class ThermalError(RuntimeError):
    pass


# ---------- helpers ------------------------------------------------------

def raw_to_celsius(raw: np.ndarray) -> np.ndarray:
    """Lepton TLinear (centikelvin, uint16) -> deg C float32."""
    return (raw.astype(np.float32) / 100.0 - 273.15).astype(np.float32)


def parse_pixel_list(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in spec.replace("|", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            x, y = item.split(",")
            out.append((int(x), int(y)))
        except ValueError:
            log.warning("bad THERMAL_STUCK_PIXELS item %r ignored", item)
    return out


def static_mask(settings: Settings, shape: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Known bad pixels as a mask in the CORRECTED thermal frame (`shape` = (rows, cols) of that frame).

    THERMAL_STUCK_PIXELS are SENSOR coordinates (160x120); the mask is built there and then rotated/flipped
    with THERMAL_ROTATE/FLIP, so it stays on the same physical pixels whatever the orientation."""
    r, f = settings.thermal_rotate, settings.thermal_flip
    if shape is None:
        shape = (T_W, T_H) if orient.swaps_axes(r) else (T_H, T_W)
    sshape = (shape[1], shape[0]) if orient.swaps_axes(r) else tuple(shape)   # sensor-frame shape
    m = np.zeros(sshape, bool)
    for x, y in parse_pixel_list(settings.thermal_stuck_pixels):
        if 0 <= y < sshape[0] and 0 <= x < sshape[1]:
            m[y, x] = True
    return np.ascontiguousarray(orient.apply(m.view(np.uint8), r, f)).astype(bool)


def fix_pixels(temp_c: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels by the median of their unmasked 3x3 neighbours (returns a copy if mask has any)."""
    if not mask.any():
        return temp_c
    out = temp_c.copy()
    h, w = temp_c.shape
    for y, x in zip(*np.nonzero(mask)):
        y0, y1, x0, x1 = max(0, y - 1), min(h, y + 2), max(0, x - 1), min(w, x + 2)
        patch = temp_c[y0:y1, x0:x1][~mask[y0:y1, x0:x1]]
        if patch.size:
            out[y, x] = np.median(patch)
    return out


class StuckPixelDetector:
    """Flags isolated pixels that deviate from their 3x3 median by > dev_c in most recent frames.

    Uses an EMA of the per-frame indicator (alpha), flags above on_thr after >= min_frames, unflags below off_thr.
    A real target is never isolated to one pixel *and* static in the sensor frame while the drone moves.
    """

    def __init__(self, shape: tuple[int, int] = (T_H, T_W), dev_c: float = 5.0, alpha: float = 0.04,
                 on_thr: float = 0.8, off_thr: float = 0.3, min_frames: int = 40) -> None:
        self.dev_c, self.alpha, self.on_thr, self.off_thr, self.min_frames = dev_c, alpha, on_thr, off_thr, min_frames
        self.ema = np.zeros(shape, np.float32)
        self.mask = np.zeros(shape, bool)
        self.n = 0

    def update(self, temp_c: np.ndarray) -> np.ndarray:
        t = temp_c.astype(np.float32)
        med = cv2.medianBlur(np.clip((t + 50) * 4, 0, 255).astype(np.uint8), 3).astype(np.float32) / 4 - 50
        dev = t - med
        # isolated: no 8-neighbour is itself more than half-way up to the pixel
        k = np.ones((3, 3), np.uint8)
        k[1, 1] = 0
        nb_max = cv2.dilate(t, k)
        isolated = np.where(dev > 0, nb_max < t - 0.5 * dev, False)
        hot = (np.abs(dev) > self.dev_c) & (isolated | (dev < 0))
        self.ema += self.alpha * (hot.astype(np.float32) - self.ema)
        self.n += 1
        if self.n >= self.min_frames:
            self.mask = (self.mask & (self.ema > self.off_thr)) | (self.ema > self.on_thr)
        return self.mask


# ---------- camera -------------------------------------------------------

class ThermalCamera:
    def __init__(self, settings: Settings, source: Optional[str] = None, history: int = 12) -> None:
        self.s = settings
        self.source = source or settings.thermal_source
        self.rotate, self.flip = settings.thermal_rotate, settings.thermal_flip
        self._static = np.zeros((T_H, T_W), bool)
        for x, y in parse_pixel_list(settings.thermal_stuck_pixels):
            if 0 <= y < T_H and 0 <= x < T_W:
                self._static[y, x] = True
        self.stuck = StuckPixelDetector()          # also in sensor coords: runs before the orientation is applied
        self._lock = threading.Lock()
        self._hist: collections.deque[tuple[np.ndarray, float]] = collections.deque(maxlen=history)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None
        self.frames_total = 0
        self._t0 = 0.0
        self.reconnects = 0
        self.attempts = 0

    @property
    def enabled(self) -> bool:
        return self.source not in ("", "none")

    @property
    def mask(self) -> np.ndarray:
        """Stuck-pixel mask in the corrected (oriented) frame."""
        return np.ascontiguousarray(orient.apply((self._static | self.stuck.mask).view(np.uint8), self.rotate, self.flip)).astype(bool)

    def set_orient(self, rotate: int, flip: str) -> None:
        """Change the output orientation live (preview); the next frame is already oriented."""
        self.rotate, self.flip = orient.check(rotate, flip)
        with self._lock:
            self._hist.clear()

    def start(self) -> "ThermalCamera":
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return self
        self._stop.clear()
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, name="thermal", daemon=True)
        self._thread.start()
        shutdown.register(self)
        return self

    def stop(self) -> None:
        """Flag -> join (<= 2.5 s) -> the thread's finally releases the V4L2 device. Daemon thread."""
        self._stop.set()
        t = self._thread
        if t and t is not threading.current_thread():
            t.join(timeout=2.5)
            if t.is_alive():
                log.warning("thermal thread still busy after 2.5 s (blocked read); daemon thread")

    def latest(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            return self._hist[-1] if self._hist else (None, 0.0)

    def nearest(self, ts: float, max_age_s: Optional[float] = None) -> tuple[Optional[np.ndarray], float]:
        max_age = self.s.thermal_max_age_s if max_age_s is None else max_age_s
        with self._lock:
            best = min(self._hist, key=lambda it: abs(it[1] - ts), default=None)
        if best is None or abs(best[1] - ts) > max_age:
            return None, 0.0
        return best

    def fps(self) -> float:
        dt = time.time() - self._t0
        return self.frames_total / dt if dt > 0 and self.frames_total else 0.0

    # ---- thread ----
    def _open(self):
        kind, _, arg = self.source.partition(":")
        if kind == "v4l2":
            cap = cv2.VideoCapture(arg, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise ThermalError(f"cannot open {arg}")
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, T_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, T_H)
            return cap
        if kind == "file":
            return _ThermalFiles(arg)
        raise ThermalError(f"bad THERMAL_SOURCE {self.source!r}")

    @staticmethod
    def _decode(f: np.ndarray) -> np.ndarray:
        raw = np.ascontiguousarray(f).view(np.uint16).reshape(-1)
        rows = raw.size // T_W
        if rows < T_H:
            raise ThermalError(f"unexpected thermal frame size {raw.size}")
        return raw_to_celsius(raw[: T_W * rows].reshape(rows, T_W)[:T_H])

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            cap = None
            try:
                box: dict = {}
                abandoned = threading.Event()

                def opener() -> None:
                    try:
                        c = self._open()
                    except Exception as exc:  # noqa: BLE001
                        box["err"] = exc
                        return
                    box["cap"] = c
                    if abandoned.is_set() and not isinstance(c, _ThermalFiles):   # we gave up waiting: do not leak the device
                        c.release()

                th = threading.Thread(target=opener, daemon=True, name="thermal-open")
                th.start()
                end = time.monotonic() + OPEN_TIMEOUT_S
                while th.is_alive() and time.monotonic() < end and not self._stop.is_set():
                    th.join(0.1)
                if "cap" not in box:
                    abandoned.set()
                    if self._stop.is_set():
                        break
                    raise box.get("err") or ThermalError(f"timeout opening {self.source}")
                cap = box["cap"]
                self.last_error = None
                self.attempts += 1
                log.info("thermal opened: %s", self.source)
                backoff, fails = 0.5, 0
                while not self._stop.is_set():
                    if isinstance(cap, _ThermalFiles):
                        temp = cap.read()
                        ok = True
                    else:
                        ok, f = cap.read()
                        temp = self._decode(f) if ok else None
                    if not ok or temp is None:
                        fails += 1
                        if fails > 30:
                            raise ThermalError("too many failed reads")
                        time.sleep(0.05)
                        continue
                    fails = 0
                    self.stuck.update(temp)                       # sensor coordinates
                    temp = fix_pixels(temp, self._static | self.stuck.mask)
                    temp = np.ascontiguousarray(orient.apply(temp, self.rotate, self.flip))
                    with self._lock:
                        self._hist.append((temp, time.time()))
                    self.frames_total += 1
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.error("thermal error (%s); retry in %.1fs", self.last_error, backoff)
                self.reconnects += 1
                self._stop.wait(backoff)
                backoff = min(10.0, backoff * 2)
            finally:
                if cap is not None and not isinstance(cap, _ThermalFiles):
                    cap.release()


class _ThermalFiles:
    def __init__(self, folder: str, fps: float = 9.0) -> None:
        p = Path(folder).expanduser()
        self.files = sorted(q for q in p.glob("*") if q.suffix.lower() in (".npy", ".png")) if p.is_dir() else []
        if not self.files:
            raise ThermalError(f"no .npy/.png thermal frames in {p}")
        self.i, self.period = 0, 1.0 / fps

    def read(self) -> Optional[np.ndarray]:
        time.sleep(self.period)
        f = self.files[self.i % len(self.files)]
        self.i += 1
        if f.suffix.lower() == ".npy":
            return np.load(f).astype(np.float32)
        raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        return raw_to_celsius(raw) if raw is not None else None


# ---------- hotspots -----------------------------------------------------

def heat_score(delta_c: float) -> float:
    """0..1 evidence from delta over background: clamp((delta-2)/8)."""
    return float(min(1.0, max(0.0, (delta_c - 2.0) / 8.0)))


def person_area_range_px(settings: Settings, alt_m: Optional[float], shape: Optional[tuple[int, int]] = None) -> tuple[int, int]:
    """Expected (min, max) blob area in thermal px from altitude + thermal HFOV (~0.3..1.8 m person)."""
    h, w = shape or (settings.eff_thermal_size[1], settings.eff_thermal_size[0])
    cap = int(0.4 * h * w)
    if alt_m is None or alt_m < 1.0:
        return 2, cap
    footprint_w = 2 * alt_m * math.tan(math.radians(settings.eff_thermal_hfov_deg) / 2)
    gsd = footprint_w / w                                   # m / px
    lo = 0.5 * (settings.thermal_min_person_m / gsd) ** 2   # half a small person footprint
    hi = 0.8 * (settings.thermal_max_person_m / gsd) * max(settings.thermal_min_person_m / gsd, 1.0) * 1.5
    return max(2, int(round(lo))), min(cap, max(4, int(round(hi))))


def delta_map(temp_c: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(temp with stuck pixels fixed, local background, temp - background). Background = 31x31 median in
    0.25 degC steps (range -10..53.75 degC)."""
    temp_c = temp_c.astype(np.float32)
    if not np.isfinite(temp_c).all():                       # corrupt frame: NaN/inf -> median (never reaches uint8 casts)
        ok = np.isfinite(temp_c)
        temp_c = np.where(ok, temp_c, np.float32(np.median(temp_c[ok]) if ok.any() else 0.0)).astype(np.float32)
    t = fix_pixels(temp_c, mask) if mask is not None else temp_c
    q = np.clip((t + 10.0) * 4.0, 0, 255).astype(np.uint8)
    bg = cv2.medianBlur(q, BG_KSIZE).astype(np.float32) / 4.0 - 10.0
    return t, bg, t - bg


def find_hotspots(temp_c: np.ndarray, settings: Settings, alt_m: Optional[float] = None,
                  mask: Optional[np.ndarray] = None) -> list[HotSpot]:
    """Warm blobs: >= thermal_delta_c above the local background, peak within thermal_min_c..thermal_max_c,
    area within the altitude-scaled person range. Stuck pixels (static list + `mask`) are removed first."""
    if temp_c is None or temp_c.ndim != 2:
        return []
    m = static_mask(settings, temp_c.shape)
    if mask is not None:
        m = m | mask
    t, bg, delta = delta_map(temp_c, m)
    cand = ((delta >= settings.thermal_delta_c) & (t >= settings.thermal_min_c - 4.0)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) if cand.sum() > 6 else cand
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    amin, amax = person_area_range_px(settings, alt_m, t.shape)
    spots: list[HotSpot] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < amin or area > amax:
            continue
        sel = lab == i
        vals = t[sel]
        peak = float(vals.max())
        if not (settings.thermal_min_c <= peak <= settings.thermal_max_c):
            continue
        d = float(peak - bg[sel].mean())
        spots.append(HotSpot(bbox=BBox(x1=x, y1=y, x2=x + w, y2=y + h), peak_c=peak, mean_c=float(vals.mean()),
                             delta_c=d, area_px=area, score=heat_score(d)))
    spots.sort(key=lambda s: -s.score)
    return spots


def colorize(temp_c: np.ndarray, lo: Optional[float] = None, hi: Optional[float] = None, scale: int = 1) -> np.ndarray:
    """INFERNO false-colour BGR image. Auto range (1st..99.5th percentile, min span 10 degC) unless given."""
    if lo is None:
        lo = float(np.percentile(temp_c, 1))
    if hi is None:
        hi = max(float(np.percentile(temp_c, 99.5)), lo + 10.0)
    g = np.clip((temp_c - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return img
