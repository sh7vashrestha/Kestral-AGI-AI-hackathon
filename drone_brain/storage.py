"""Per-flight on-disk record: frames, crops and JSONL logs.

Layout (under settings.data_dir):
    flights/<UTC timestamp>/
        frames/ crops/                       JPEGs
        detections.jsonl matches.jsonl alerts.jsonl telemetry.jsonl
        flight.json                          plan, target history, start/end

All methods are thread-safe and never raise: IO errors (disk full, ...) are logged
(rate-limited) and the method returns None / False so the camera loop keeps running.
Paths returned are relative to ``settings.data_dir``; ``url_for`` maps them to the
``/data/...`` route served by the webapp (jpg/png only).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
from pydantic import BaseModel

from config import Settings
from models import SurveyPlan, Target

log = logging.getLogger(__name__)

LOG_KINDS = ("detections", "matches", "alerts", "telemetry")
_ERR_LOG_EVERY_S = 30.0


class FlightStorage:
    def __init__(self, settings: Settings, *, jpeg_quality: int = 85) -> None:
        self._data_dir = Path(settings.data_dir)
        self._quality = jpeg_quality
        self._lock = threading.Lock()
        self._last_err_log = 0.0
        self._start_ts = time.time()
        self._end_ts: Optional[float] = None
        self._plan: Optional[dict[str, Any]] = None
        self._targets: list[dict[str, Any]] = []
        self._summary: dict[str, Any] = {}
        self.ok = True

        stamp = datetime.fromtimestamp(self._start_ts, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        flights = self._data_dir / "flights"
        root = flights / stamp
        n = 0
        while root.exists():
            n += 1
            root = flights / f"{stamp}_{n}"
        self.root = root
        try:
            (root / "frames").mkdir(parents=True)
            (root / "crops").mkdir()
            for k in LOG_KINDS:
                (root / f"{k}.jsonl").touch()
        except OSError as exc:
            self.ok = False
            self._err("cannot create flight directory %s: %s", root, exc)
        self._write_meta()

    # ---- helpers ----
    def _err(self, msg: str, *args: Any) -> None:
        now = time.monotonic()
        if now - self._last_err_log >= _ERR_LOG_EVERY_S or self._last_err_log == 0.0:
            self._last_err_log = now
            log.error("storage: " + msg, *args)

    def rel(self, path: Union[str, Path]) -> str:
        """Path relative to data_dir. Works when DATA_DIR itself is relative ('data'): a crop written to
        'data/flights/x.jpg' must become 'flights/x.jpg', not 'data/flights/x.jpg' (-> /data/data/... 404)."""
        p = Path(path)
        try:
            return p.resolve().relative_to(Path(self._data_dir).resolve()).as_posix()
        except ValueError:
            if p.is_absolute():
                raise
            return p.as_posix()                       # already relative to data_dir

    def url_for(self, path: Union[str, Path]) -> str:
        """'/data/<path relative to data_dir>' (absolute paths are made relative)."""
        try:
            return "/data/" + self.rel(path)
        except ValueError:
            return "/data/" + Path(path).name

    def _write_bytes(self, sub: str, name: str, data: bytes) -> Optional[str]:
        path = self.root / sub / name
        try:
            with self._lock:
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, path)
            return self.rel(path)
        except (OSError, ValueError) as exc:
            self._err("write %s failed: %s", path, exc)
            return None

    def _jpeg(self, img: Union[np.ndarray, bytes]) -> Optional[bytes]:
        if isinstance(img, (bytes, bytearray)):
            return bytes(img)
        try:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        except cv2.error as exc:
            self._err("jpeg encode failed: %s", exc)
            return None
        return buf.tobytes() if ok else None

    # ---- images ----
    def save_frame(self, frame: Union[np.ndarray, bytes], name: str) -> Optional[str]:
        """Save a full frame as frames/<name>.jpg; returns the data_dir-relative path."""
        data = self._jpeg(frame)
        return None if data is None else self._write_bytes("frames", f"{name}.jpg", data)

    def save_crop(self, crop: Union[np.ndarray, bytes], name: str) -> Optional[str]:
        data = self._jpeg(crop)
        return None if data is None else self._write_bytes("crops", f"{name}.jpg", data)

    # ---- records ----
    def append(self, kind: str, record: Union[BaseModel, dict[str, Any]]) -> bool:
        """Append one JSON line to <kind>.jsonl (kind in LOG_KINDS)."""
        if kind not in LOG_KINDS:
            raise ValueError(f"unknown log kind {kind!r}")
        try:
            line = record.model_dump_json() if isinstance(record, BaseModel) else json.dumps(record, default=str)
            with self._lock, open(self.root / f"{kind}.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return True
        except (OSError, TypeError, ValueError) as exc:
            self._err("append %s failed: %s", kind, exc)
            return False

    # ---- flight.json ----
    def set_plan(self, plan: Union[SurveyPlan, dict[str, Any], None]) -> None:
        self._plan = plan.model_dump() if isinstance(plan, BaseModel) else plan
        self._write_meta()

    def record_target(self, target: Target) -> None:
        self._targets.append({"ts": time.time(), **target.model_dump()})
        self._write_meta()

    def update_summary(self, **fields: Any) -> None:
        """Merge fields into flight.json's "summary" (Phase 7: plan_id, state, alerts, distance, ...)."""
        self._summary.update(fields)
        self._write_meta()

    def write_text(self, name: str, text: str) -> Optional[str]:
        """Write an extra file into the flight directory (e.g. mission.waypoints)."""
        if "/" in name or name.startswith("."):
            raise ValueError(f"bad file name {name!r}")
        return self._write_bytes(".", name, text.encode("utf-8"))

    def close(self) -> None:
        self._end_ts = time.time()
        self._write_meta()

    def _write_meta(self) -> None:
        meta = {"start_ts": self._start_ts, "end_ts": self._end_ts,
                "plan": self._plan, "targets": self._targets, "summary": self._summary}
        path = self.root / "flight.json"
        try:
            with self._lock:
                tmp = path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(meta, default=str, indent=1), encoding="utf-8")
                os.replace(tmp, path)
        except OSError as exc:
            self._err("write flight.json failed: %s", exc)


class PlanStore:
    """Per-plan export directory (Phase 7): <data_dir>/plans/<plan_id>/{plan.json, mission.waypoints,
    mission.geojson, preflight.json}. Never raises; returns the directory or None."""

    FILES = ("plan.json", "mission.waypoints", "mission.geojson", "preflight.json")

    def __init__(self, settings: Settings) -> None:
        self.root = Path(settings.data_dir) / "plans"

    def dir_for(self, plan_id: str) -> Path:
        if not plan_id or not all(c.isalnum() or c in "-_" for c in plan_id):
            raise ValueError(f"bad plan id {plan_id!r}")
        return self.root / plan_id

    def save(self, plan_id: str, files: dict[str, Union[str, bytes, dict, list, BaseModel]]) -> Optional[Path]:
        try:
            d = self.dir_for(plan_id)
            d.mkdir(parents=True, exist_ok=True)
            for name, data in files.items():
                if isinstance(data, BaseModel):
                    raw = data.model_dump_json(indent=1).encode()
                elif isinstance(data, (dict, list)):
                    raw = json.dumps(data, default=str, indent=1).encode()
                elif isinstance(data, str):
                    raw = data.encode("utf-8")
                else:
                    raw = bytes(data)
                tmp = d / (name + ".tmp")
                tmp.write_bytes(raw)
                os.replace(tmp, d / name)
            return d
        except (OSError, ValueError, TypeError) as exc:
            log.error("plan store: saving %s failed: %s", plan_id, exc)
            return None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return out
        dirs = sorted((d for d in self.root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
        for d in dirs[:limit]:
            out.append({"plan_id": d.name, "mtime": d.stat().st_mtime,
                        "files": sorted(f.name for f in d.iterdir() if f.is_file() and not f.name.endswith(".tmp"))})
        return out
