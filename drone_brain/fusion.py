"""RGB (YOLO) + thermal fusion and heat-only alerts.

    H = load_homography(settings)                      # thermal -> RGB 3x3 (data/calibration/thermal_to_rgb.json,
                                                       #  else FOV-ratio default)
    fuser = Fuser(settings, H, detector=detector, storage=storage)
    res = fuser.process(frame, ts, frame_id, telemetry, temp_c, temp_ts)   # -> FusionResult
    matcher.submit(frame, res.detections); for a in res.alerts: on_alert(a)

Per YOLO box: c_heat = heat_score(delta of warmest pixels in the box projected into thermal);
c_person = 1 - (1-c_yolo)(1 - w*c_heat). Heat-only hotspots (no YOLO box) trigger detector.detect_roi at the lower
THERMAL_RECHECK_CONF; a person found there becomes a normal fused detection, otherwise a level="heat" Alert.
Stale thermal (|ts - thermal_ts| > THERMAL_MAX_AGE_S) is ignored. Dark RGB (mean < RGB_DARK_THRESHOLD) skips YOLO.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

import cv2
import numpy as np

from config import Settings
from models import Alert, BBox, Detection, HotSpot, Telemetry
from thermal import colorize, delta_map, find_hotspots, heat_score, static_mask
from tracker import ground_distance_m, pixel_to_ground

log = logging.getLogger(__name__)
HEAT_TRACK_BASE = 1_000_000     # heat alerts use track ids >= this (no clash with Tracker ids)
DARK_EXIT_FACTOR = 1.4          # leave dark mode only above rgb_dark_threshold * this (hysteresis band)
DARK_CONFIRM_FRAMES = 3         # ... and only after this many consecutive frames on the other side (no flapping)
FLAT_STD = 3.0                  # gray std below this AND mean >= FLAT_MIN_MEAN = blown-out picture (sun / white wall)
FLAT_MIN_MEAN = 235.0           # (black is dark mode; mid-grey flat frames are synthetic test frames, not a failure)
PERSON_MEMORY_S = 10.0          # a heat blob this close (in time/place) to a YOLO person is not alerted separately


# ---------- homography ----------------------------------------------------

def orientation_key(settings: Settings) -> dict:
    """Camera orientations a calibration is valid for (json-comparable)."""
    return {"camera": [settings.camera_rotate, settings.camera_flip], "thermal": [settings.thermal_rotate, settings.thermal_flip]}


def calibration_path(settings: Settings) -> Path:
    return Path(settings.data_dir) / "calibration" / "thermal_to_rgb.json"


def default_homography(settings: Settings, rgb_size: Optional[tuple[int, int]] = None,
                       thermal_size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Uncalibrated thermal->RGB mapping: pinhole scale about both image centres (FOV ratio).

    Works on the CORRECTED images (after CAMERA_ROTATE/FLIP and THERMAL_ROTATE/FLIP), with the effective FOVs.
    It assumes both corrected images show the scene the same way up; a wrong relative rotation/flip shows up
    in the live overlay and is fixed by changing THERMAL_ROTATE/FLIP or CAMERA_ROTATE/FLIP."""
    rw, rh = rgb_size or (settings.eff_camera_width, settings.eff_camera_height)
    tw, th = thermal_size or settings.eff_thermal_size
    fx_r = (rw / 2) / math.tan(math.radians(settings.eff_hfov_deg) / 2)
    fy_r = (rh / 2) / math.tan(math.radians(settings.eff_vfov_deg) / 2)
    fx_t = (tw / 2) / math.tan(math.radians(settings.eff_thermal_hfov_deg) / 2)
    fy_t = (th / 2) / math.tan(math.radians(settings.eff_thermal_vfov_deg) / 2)
    sx, sy = fx_r / fx_t, fy_r / fy_t
    return np.array([[sx, 0, rw / 2 - sx * tw / 2], [0, sy, rh / 2 - sy * th / 2], [0, 0, 1]], np.float64)


def load_homography(settings: Settings, path: Optional[Path] = None) -> np.ndarray:
    p = path or calibration_path(settings)
    try:
        d = json.loads(p.read_text())
        H = np.array(d["H"], np.float64).reshape(3, 3)
        want = orientation_key(settings)
        have = d.get("orientation") or {"camera": [0, "none"], "thermal": [0, "none"]}   # files from before rotation = unrotated
        if have != want:
            raise ValueError(f"calibration was made with orientation {have}, now {want}: recalibrate")
        rs = d.get("rgb_size")
        cw, ch = settings.eff_camera_width, settings.eff_camera_height
        if rs and tuple(rs) != (cw, ch):
            S = np.diag([cw / rs[0], ch / rs[1], 1.0])
            log.warning("calibration made at RGB %s, now %dx%d: rescaling", rs, cw, ch)
            H = S @ H
        if np.linalg.cond(H) > 1e8:
            raise ValueError("degenerate homography")
        log.info("loaded thermal->RGB calibration %s (rms %.2f px)", p, d.get("rms_px", float("nan")))
        return H
    except FileNotFoundError:
        log.info("no thermal calibration at %s: using FOV-ratio default", p)
    except Exception as exc:  # noqa: BLE001
        log.error("bad calibration %s (%s): using FOV-ratio default", p, exc)
    return default_homography(settings)


def save_homography(settings: Settings, H: np.ndarray, rms_px: float, n_points: int, path: Optional[Path] = None) -> Path:
    p = path or calibration_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"H": np.asarray(H).tolist(), "rgb_size": [settings.eff_camera_width, settings.eff_camera_height],
                             "thermal_size": list(settings.eff_thermal_size), "orientation": orientation_key(settings), "rms_px": rms_px, "n_points": n_points,
                             "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    return p


def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def project_bbox(H: np.ndarray, b: BBox) -> BBox:
    c = project_points(H, [(b.x1, b.y1), (b.x2, b.y1), (b.x2, b.y2), (b.x1, b.y2)])
    return BBox(x1=float(c[:, 0].min()), y1=float(c[:, 1].min()), x2=float(c[:, 0].max()), y2=float(c[:, 1].max()))


def _overlap(a: BBox, b: BBox) -> bool:
    return min(a.x2, b.x2) > max(a.x1, b.x1) and min(a.y2, b.y2) > max(a.y1, b.y1)


# ---------- fusion ----------------------------------------------------------

class RoiDetector(Protocol):
    def detect_roi(self, frame: np.ndarray, roi: BBox, ts: Optional[float] = ..., frame_id: int = ...,
                   telemetry: Optional[Telemetry] = ..., conf: Optional[float] = ...) -> list[Detection]: ...
    def detect(self, frame: np.ndarray, ts: Optional[float] = ..., frame_id: int = ...,
               telemetry: Optional[Telemetry] = ...) -> list[Detection]: ...


@dataclass
class FusionResult:
    detections: list[Detection] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)          # level="heat"
    hotspots: list[HotSpot] = field(default_factory=list)
    thermal_used: bool = False
    dark: bool = False
    flat: bool = False          # washed-out / covered lens: no usable picture (std < FLAT_STD), YOLO skipped too


def fuse_score(c_yolo: float, c_heat: float, w: float) -> float:
    return float(1.0 - (1.0 - c_yolo) * (1.0 - w * c_heat))


class Fuser:
    def __init__(self, settings: Settings, homography: Optional[np.ndarray] = None,
                 detector: Optional[RoiDetector] = None, storage=None,
                 stuck_mask: Optional[Callable[[], np.ndarray]] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.s = settings
        self.H = default_homography(settings) if homography is None else np.asarray(homography, np.float64)
        self.Hinv = np.linalg.inv(self.H)
        self.detector = detector
        self.storage = storage
        self._stuck = stuck_mask
        self.clock = clock
        self._dark: Optional[bool] = None
        self._dark_votes = 0
        self._rgb_size = (settings.eff_camera_width, settings.eff_camera_height)
        self._H0 = self.H.copy()
        self._next_id = HEAT_TRACK_BASE
        self._recent: list[tuple[float, Optional[tuple[float, float]], tuple[float, float]]] = []  # ts, ground, thermal xy

    # ---- helpers ----
    def to_thermal(self, b: BBox) -> BBox:
        return project_bbox(self.Hinv, b)

    def to_rgb(self, b: BBox) -> BBox:
        return project_bbox(self.H, b)

    def c_heat_for(self, delta: np.ndarray, temp: np.ndarray, tb: BBox) -> Optional[float]:
        x1, y1 = max(0, int(math.floor(tb.x1))), max(0, int(math.floor(tb.y1)))
        x2, y2 = min(delta.shape[1], int(math.ceil(tb.x2))), min(delta.shape[0], int(math.ceil(tb.y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        d, t = delta[y1:y2, x1:x2].ravel(), temp[y1:y2, x1:x2].ravel()
        d = d[t <= self.s.thermal_max_c]
        if d.size == 0:
            return 0.0
        k = max(1, min(3, d.size))
        return heat_score(float(np.sort(d)[-k:].mean()))

    def _fit_frame_size(self, frame: np.ndarray) -> None:
        """The camera frame size can differ from the configured one (mode change, other source): rescale H to it."""
        size = (frame.shape[1], frame.shape[0])
        if size != self._rgb_size:
            cw, ch = self._rgb_size
            log.warning("RGB frame size changed %s -> %s: rescaling the thermal homography", self._rgb_size, size)
            self.H = np.diag([size[0] / cw, size[1] / ch, 1.0]) @ self._H0
            self.Hinv = np.linalg.inv(self.H)
            self._rgb_size = size
            self._H0 = self.H.copy()

    def _update_dark(self, gray_mean: float) -> bool:
        """Dark mode with hysteresis: enter below the threshold, leave above threshold*DARK_EXIT_FACTOR, and only after
        DARK_CONFIRM_FRAMES consecutive frames on the other side (a cloud / headlights cannot make it flap)."""
        thr = self.s.rgb_dark_threshold
        if self._dark is None:
            self._dark = gray_mean < thr
            log.warning("RGB %s (mean %.1f)", "DARK" if self._dark else "light OK", gray_mean)
            return self._dark
        other = gray_mean >= thr * DARK_EXIT_FACTOR if self._dark else gray_mean < thr
        self._dark_votes = self._dark_votes + 1 if other else 0
        if self._dark_votes >= DARK_CONFIRM_FRAMES:
            self._dark, self._dark_votes = not self._dark, 0
            log.warning("RGB %s (mean %.1f): %s", "DARK" if self._dark else "light OK", gray_mean,
                        "skipping YOLO/CLIP, heat alerts only" if self._dark else "back to YOLO+thermal")
        return self._dark

    def _note_person(self, now: float, d: Detection, tb: BBox, frame: np.ndarray, tel: Optional[Telemetry]) -> None:
        fh, fw = frame.shape[:2]
        cx, cy = (d.bbox.x1 + d.bbox.x2) / 2, (d.bbox.y1 + d.bbox.y2) / 2
        ground = None
        if (tel and tel.lat is not None and tel.lon is not None and tel.alt_rel_m is not None
                and tel.heading_deg is not None and tel.alt_rel_m > 0.5):
            ground = pixel_to_ground(cx, cy, fw, fh, tel.lat, tel.lon, tel.alt_rel_m, tel.heading_deg,
                                     self.s.eff_hfov_deg, self.s.eff_vfov_deg)
        self._recent.append((now, ground, ((tb.x1 + tb.x2) / 2, (tb.y1 + tb.y2) / 2)))
        del self._recent[:-200]

    def _rate_ok(self, now: float, ground: Optional[tuple[float, float]], txy: tuple[float, float]) -> bool:
        self._recent = [r for r in self._recent if now - r[0] < max(self.s.thermal_alert_interval_s, PERSON_MEMORY_S)]
        for _, g, p in self._recent:
            if ground is not None and g is not None:
                if ground_distance_m(ground, g) <= self.s.tracker_gps_radius_m:
                    return False
            elif math.hypot(txy[0] - p[0], txy[1] - p[1]) <= 15:
                return False
        self._recent.append((now, ground, txy))
        return True

    # ---- main entry ----
    def process(self, frame: np.ndarray, ts: float, frame_id: int, telemetry: Optional[Telemetry],
                temp_c: Optional[np.ndarray], temp_ts: float = 0.0,
                detections: Optional[list[Detection]] = None) -> FusionResult:
        res = FusionResult()
        if frame is None or frame.ndim != 3 or frame.shape[0] < 8 or frame.shape[1] < 8:
            return res                                   # garbage frame: nothing to do (never raises)
        self._fit_frame_size(frame)
        gray = cv2.cvtColor(frame[::4, ::4], cv2.COLOR_BGR2GRAY)
        gray_mean, gray_std = float(gray.mean()), float(gray.std())
        res.dark = self._update_dark(gray_mean)
        res.flat = (not res.dark) and gray_std < FLAT_STD and gray_mean >= FLAT_MIN_MEAN
        if res.flat and not getattr(self, "_flat_logged", False):
            log.warning("RGB frame is flat (mean %.0f, std %.1f): overexposed; skipping YOLO/CLIP", gray_mean, gray_std)
        self._flat_logged = res.flat
        blind = res.dark or res.flat
        if detections is None:
            detections = [] if (blind or self.detector is None) else self.detector.detect(frame, ts, frame_id, telemetry)
        elif blind:
            detections = []
        dets = list(detections)

        thermal_ok = temp_c is not None and abs(ts - temp_ts) <= self.s.thermal_max_age_s
        if thermal_ok:
            temp_c = np.asarray(temp_c, np.float32)
            thermal_ok = temp_c.ndim == 2 and temp_c.shape[0] >= 8 and temp_c.shape[1] >= 8
        if not thermal_ok:
            res.detections = dets
            return res
        assert temp_c is not None
        if not np.isfinite(temp_c).all():                # NaN/inf pixels (corrupt frame): treat as the median
            bad = ~np.isfinite(temp_c)
            temp_c = np.where(bad, np.float32(np.median(temp_c[~bad])) if (~bad).any() else np.float32(0.0), temp_c).astype(np.float32)
        res.thermal_used = True
        alt = telemetry.alt_rel_m if telemetry else None
        mask = static_mask(self.s, temp_c.shape)
        if self._stuck is not None:
            sm = self._stuck()
            if sm is not None and sm.shape == mask.shape:
                mask = mask | sm
        t_fixed, _, delta = delta_map(temp_c, mask)
        res.hotspots = find_hotspots(temp_c, self.s, alt_m=alt, mask=mask)

        boxes_t: list[BBox] = []
        for d in dets:
            tb = self.to_thermal(d.bbox)
            boxes_t.append(tb)
            ch = self.c_heat_for(delta, t_fixed, tb)
            if ch is not None:
                d.c_heat = ch
                d.c_person = fuse_score(d.c_yolo, ch, self.s.thermal_weight)

        now = self.clock()
        for d, tb in zip(dets, boxes_t):                 # YOLO persons: remember where, so a heat blob there is not a 2nd alert
            self._note_person(now, d, tb, frame, telemetry)
        for hs in res.hotspots:
            if any(_overlap(hs.bbox, tb) for tb in boxes_t):
                continue
            roi = self.to_rgb(hs.bbox)
            if self.detector is not None and not blind:
                try:
                    found = self.detector.detect_roi(frame, roi, ts=ts, frame_id=frame_id, telemetry=telemetry,
                                                     conf=self.s.thermal_recheck_conf)
                except Exception as exc:  # noqa: BLE001 - re-check failed (e.g. CUDA OOM): fall through to a heat alert
                    log.warning("YOLO re-check of a hotspot failed (%s); heat alert only", exc)
                    found = []
                if found:
                    for d in found:
                        d.c_heat = hs.score
                        d.c_person = fuse_score(d.c_yolo, hs.score, self.s.thermal_weight)
                        dets.append(d)
                        boxes_t.append(self.to_thermal(d.bbox))
                    continue
            alert = self._heat_alert(frame, t_fixed, hs, roi, ts, telemetry, now)
            if alert is not None:
                res.alerts.append(alert)
        res.detections = dets
        return res

    def _heat_alert(self, frame: np.ndarray, temp: np.ndarray, hs: HotSpot, roi: BBox, ts: float,
                    tel: Optional[Telemetry], now: float) -> Optional[Alert]:
        fh, fw = frame.shape[:2]
        cx, cy = (roi.x1 + roi.x2) / 2, (roi.y1 + roi.y2) / 2
        ground: Optional[tuple[float, float]] = None
        if (tel and tel.lat is not None and tel.lon is not None and tel.alt_rel_m is not None
                and tel.heading_deg is not None and tel.alt_rel_m > 0.5):
            ground = pixel_to_ground(cx, cy, fw, fh, tel.lat, tel.lon, tel.alt_rel_m, tel.heading_deg,
                                     self.s.eff_hfov_deg, self.s.eff_vfov_deg)
        txy = ((hs.bbox.x1 + hs.bbox.x2) / 2, (hs.bbox.y1 + hs.bbox.y2) / 2)
        if not self._rate_ok(now, ground, txy):
            return None
        img = self._side_by_side(frame, temp, hs, roi)
        url = None
        tid = self._next_id
        self._next_id += 1
        if self.storage is not None:
            path = self.storage.save_crop(img, f"heat{tid - HEAT_TRACK_BASE}_f{int(ts)}")
            url = self.storage.url_for(path) if path else None
        return Alert(level="heat", track_id=tid, ts=now, c_yolo=0.0, c_final=hs.score, n_frames=1,
                     reason=f"heat signature: peak {hs.peak_c:.1f}C, +{hs.delta_c:.1f}C above background, "
                            f"no visual person confirmation",
                     image_url=url, lat=ground[0] if ground else (tel.lat if tel else None),
                     lon=ground[1] if ground else (tel.lon if tel else None),
                     alt_rel_m=tel.alt_rel_m if tel else None)

    def _side_by_side(self, frame: np.ndarray, temp: np.ndarray, hs: HotSpot, roi: BBox, size: int = 224) -> np.ndarray:
        fh, fw = frame.shape[:2]
        # RGB crop around the projected hotspot (generous context)
        cx, cy = (roi.x1 + roi.x2) / 2, (roi.y1 + roi.y2) / 2
        half = max(roi.x2 - roi.x1, roi.y2 - roi.y1, 40.0) * 1.5
        x1, y1, x2, y2 = (int(max(0, cx - half)), int(max(0, cy - half)), int(min(fw, cx + half)), int(min(fh, cy + half)))
        rgb = cv2.resize(frame[y1:y2, x1:x2], (size, size)) if x2 > x1 and y2 > y1 else np.zeros((size, size, 3), np.uint8)
        # thermal crop around the blob, same ground region
        b = hs.bbox
        tcx, tcy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
        th = max(b.x2 - b.x1, b.y2 - b.y1, 8.0) * 2
        a1, b1, a2, b2 = (int(max(0, tcx - th)), int(max(0, tcy - th)), int(min(temp.shape[1], tcx + th)), int(min(temp.shape[0], tcy + th)))
        col = colorize(temp)
        cv2.rectangle(col, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (255, 255, 255), 1)
        tcrop = cv2.resize(col[b1:b2, a1:a2], (size, size), interpolation=cv2.INTER_NEAREST)
        cv2.putText(tcrop, f"{hs.peak_c:.1f}C", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return np.hstack([tcrop, rgb])
