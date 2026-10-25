"""Track association so that one person produces one alert.

Detection -> track by (a) IoU with the track's last box when seen <= ``iou_window_s``
ago, or (b) both positions known and ground distance <= settings.tracker_gps_radius_m.
Tracks expire after ``ttl_s`` seconds unseen. Time is taken from ``Detection.ts``.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Optional

from config import Settings
from models import BBox, Detection, MatchResult, Telemetry

EARTH_R = 6_371_000.0


def iou(a: BBox, b: BBox) -> float:
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / ua if ua > 0 else 0.0


def pixel_to_ground(
    px: float, py: float, img_w: int, img_h: int,
    lat: float, lon: float, alt_m: float, heading_deg: float,
    hfov_deg: float, vfov_deg: float,
) -> tuple[float, float]:
    """Project an image point to (lat, lon) on flat ground for a nadir camera.

    Image up = drone forward (heading, deg clockwise from north), image right = drone right.
    The image centre maps to the drone's own position.
    """
    right = (px - img_w / 2) / (img_w / 2) * math.tan(math.radians(hfov_deg) / 2) * alt_m
    fwd = -(py - img_h / 2) / (img_h / 2) * math.tan(math.radians(vfov_deg) / 2) * alt_m
    h = math.radians(heading_deg)
    north = fwd * math.cos(h) - right * math.sin(h)
    east = fwd * math.sin(h) + right * math.cos(h)
    dlat = math.degrees(north / EARTH_R)
    dlon = math.degrees(east / (EARTH_R * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def ground_distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    dn = math.radians(b[0] - a[0]) * EARTH_R
    de = math.radians(b[1] - a[1]) * EARTH_R * math.cos(math.radians((a[0] + b[0]) / 2))
    return math.hypot(dn, de)


def detection_ground(det: Detection, img_w: int, img_h: int, s: Settings) -> Optional[tuple[float, float]]:
    t: Optional[Telemetry] = det.telemetry
    if t is None or t.lat is None or t.lon is None or t.alt_rel_m is None or t.heading_deg is None:
        return None
    if t.alt_rel_m <= 0.5:
        return None
    b = det.bbox
    return pixel_to_ground((b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2, img_w, img_h,
                           t.lat, t.lon, t.alt_rel_m, t.heading_deg,
                           s.eff_hfov_deg, s.eff_vfov_deg)


@dataclass
class Track:
    id: int
    first_ts: float
    last_ts: float
    last_bbox: BBox
    last_ground: Optional[tuple[float, float]] = None
    last_det: Optional[Detection] = None
    n_det: int = 1
    c_yolo_hist: list[float] = field(default_factory=list)
    best_score: float = 0.0
    best_crop_path: Optional[str] = None
    # per-target-description state (cleared by Tracker.reset_matches)
    match_hist: list[float] = field(default_factory=list)      # C_final per VLM result
    results: list[MatchResult] = field(default_factory=list)
    target_alert_avg: Optional[float] = None
    last_target_alert_ts: Optional[float] = None
    vlm_confirmed_target: Optional[str] = None
    # per-track state
    person_alerted: bool = False
    last_person_alert_ts: Optional[float] = None


class Tracker:
    def __init__(self, settings: Settings, *, ttl_s: float = 30.0, iou_window_s: float = 3.0,
                 ground_ttl_s: float = 900.0, max_tracks: int = 400) -> None:
        self.s = settings
        self.ttl_s = ttl_s
        self.ground_ttl_s = ground_ttl_s     # tracks with a GPS position stay matchable much longer (next lane, 1-3 min later)
        self.max_tracks = max_tracks
        self.iou_window_s = iou_window_s
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._lock = threading.RLock()

    @property
    def tracks(self) -> dict[int, Track]:
        return self._tracks

    def get(self, track_id: int) -> Optional[Track]:
        with self._lock:
            return self._tracks.get(track_id)

    def update(self, detections: list[Detection], img_w: int, img_h: int) -> list[tuple[Detection, Track, bool]]:
        """Assign track ids (also written to det.track_id). Returns (det, track, is_new)."""
        out: list[tuple[Detection, Track, bool]] = []
        with self._lock:
            now = max((d.ts for d in detections), default=None)
            if now is not None:
                for tid in [t.id for t in self._tracks.values()
                            if now - t.last_ts > (self.ground_ttl_s if t.last_ground is not None else self.ttl_s)]:
                    del self._tracks[tid]

            used: set[int] = set()
            for det in sorted(detections, key=lambda d: -d.c_yolo):
                ground = detection_ground(det, img_w, img_h, self.s)
                track = self._match(det, ground, used)
                is_new = track is None
                if track is None:
                    track = Track(id=self._next_id, first_ts=det.ts, last_ts=det.ts, last_bbox=det.bbox)
                    self._next_id += 1
                    self._tracks[track.id] = track
                else:
                    track.n_det += 1
                track.last_ts = max(track.last_ts, det.ts)
                track.last_bbox = det.bbox
                track.last_det = det
                if ground is not None:
                    track.last_ground = ground
                track.c_yolo_hist.append(det.c_yolo)
                del track.c_yolo_hist[:-50]
                used.add(track.id)
                det.track_id = track.id
                out.append((det, track, is_new))
            if len(self._tracks) > self.max_tracks:           # bounded memory: drop the longest unseen (never this frame's)
                for t in sorted(self._tracks.values(), key=lambda t: t.last_ts)[:len(self._tracks) - self.max_tracks]:
                    if t.id not in used:
                        del self._tracks[t.id]
        return out

    def _match(self, det: Detection, ground: Optional[tuple[float, float]], used: set[int]) -> Optional[Track]:
        best: Optional[Track] = None
        best_iou = 0.0
        for t in self._tracks.values():
            if t.id in used or det.ts - t.last_ts > self.iou_window_s:
                continue
            v = iou(det.bbox, t.last_bbox)
            if v >= self.s.tracker_iou and v > best_iou:
                best, best_iou = t, v
        if best is not None or ground is None:
            return best
        best_d = self.s.tracker_gps_radius_m
        for t in self._tracks.values():
            if t.id in used or t.last_ground is None:
                continue
            d = ground_distance_m(ground, t.last_ground)
            if d <= best_d:
                best, best_d = t, d
        return best

    def reset_matches(self) -> None:
        """Forget per-target match history (target description changed)."""
        with self._lock:
            for t in self._tracks.values():
                t.match_hist.clear()
                t.results.clear()
                t.target_alert_avg = None
                t.last_target_alert_ts = None
                t.vlm_confirmed_target = None
