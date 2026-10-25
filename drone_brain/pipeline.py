"""Vision pipeline behind one interface (Phase 7): camera -> YOLO (+thermal fusion) -> matcher -> alerts.

    vp = VisionPipeline(settings, on_alert=cb, telemetry_fn=vehicle.telemetry)
    vp.start()                      # returns at once; models load in the worker thread
    vp.update_target(Target(...))
    vp.begin_flight(storage)        # frames are processed + recorded only during a flight
    vp.end_flight(); vp.stop()

Every Phase 5 component (camera.py, detector.py, thermal.py, fusion.py, clip_encoder.py) is imported lazily
and optional: if one is missing or fails to load, the pipeline degrades instead of taking main.py down:

    Load order (GPU memory on the Orin, see notes/phase5.md): match backend (ClipEncoder) -> Detector -> cameras.
    camera / detector missing  -> vision disabled (status()["state"] == "disabled", reason logged)
    no camera frame for > FRAME_STALL_S -> alive() is False -> the controller's watchdog calls restart()
    CLIP encoder fails         -> person / heat alerts only (no target matching). Images never go to the LLM.
    thermal missing / "none"   -> RGB only

`NullPipeline` is the --no-vision stand-in (same interface, does nothing).
Factories can be injected for tests (detector_factory, backend_factory, camera_factory).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

from config import Settings
from models import Alert, MatchResult, Target, Telemetry

log = logging.getLogger(__name__)

FRAME_STALL_S = 5.0     # camera.py has no hard-hang watchdog: no new frame for this long -> restart
CLIP_LOAD_ATTEMPTS = 3


class _NoMatchBackend:
    """Used when neither CLIP nor the VLM is available: person alerts still fire, never a target match."""

    def prepare(self, target: Target) -> None:
        pass

    def match(self, crop_bgr, target: Target) -> MatchResult:
        return MatchResult(is_human=True, matches=False, match_confidence=0.0, reason="no match backend available")


class NullPipeline:
    """--no-vision: same interface as VisionPipeline, does nothing."""

    def __init__(self, reason: str = "vision disabled (--no-vision)") -> None:
        self.reason = reason

    def start(self) -> None:
        log.info("vision pipeline: %s", self.reason)

    def stop(self) -> None:
        pass

    def begin_flight(self, storage: Any) -> None:
        pass

    def end_flight(self) -> None:
        pass

    def update_target(self, target: Optional[Target]) -> None:
        pass

    def alive(self) -> bool:
        return True

    def restart(self) -> None:
        pass

    @property
    def critical(self) -> bool:
        return False

    def status(self) -> dict:
        return {"state": "disabled", "reason": self.reason}


class VisionPipeline:
    def __init__(self, settings: Settings, *, on_alert: Callable[[Alert], None],
                 telemetry_fn: Callable[[], Optional[Telemetry]], camera_source: Optional[str] = None,
                 detector_factory: Optional[Callable[[Settings], Any]] = None,
                 backend_factory: Optional[Callable[[Settings], Any]] = None,
                 camera_factory: Optional[Callable[[Settings, str], Any]] = None,
                 use_thermal: bool = True, save_all_frames: bool = True) -> None:
        self.s = settings
        self.on_alert = on_alert
        self.telemetry_fn = telemetry_fn
        self.camera_source = camera_source or settings.camera_source
        self._detector_factory = detector_factory
        self._backend_factory = backend_factory
        self._camera_factory = camera_factory
        self.use_thermal = use_thermal
        self.save_all_frames = save_all_frames

        self.camera: Any = None
        self.thermal: Any = None
        self.detector: Any = None
        self.fuser: Any = None
        self.backend: Any = None
        self.matcher: Any = None
        self._storage: Any = None
        self._target: Optional[Target] = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.state = "stopped"          # stopped | loading | ready | disabled | failed
        self.reason = ""
        self.frames = 0
        self.processed = 0
        self.detections = 0
        self.errors = 0
        self.last_frame_ts = 0.0
        self._last_frame_mono = 0.0
        self._gen = 0
        self.restarts = 0
        self.last_loop_ms = 0.0
        self.last_dets: Optional[tuple] = None
        self.critical = True            # a crash while flying -> controller RTLs
        self._retrying = False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._spawn()

    def _spawn(self) -> None:
        self._stop.clear()
        self._gen += 1
        self._last_frame_mono = time.monotonic()
        self._thread = threading.Thread(target=self._main, args=(self._gen,), name=f"vision-{self._gen}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.end_flight()
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:  # noqa: BLE001
                log.exception("camera stop failed")
        if self.thermal is not None:
            try:
                self.thermal.stop()
            except Exception:  # noqa: BLE001
                log.exception("thermal stop failed")
        if self._thread:
            self._thread.join(timeout=5)
        self.state = "stopped" if self.state not in ("disabled", "failed") else self.state

    def alive(self) -> bool:
        """False if the worker died or the camera stalled (> FRAME_STALL_S without a frame once ready).
        Disabled / loading count as alive (nothing to restart)."""
        if self.state in ("disabled", "loading", "stopped") or self._stop.is_set():
            return True
        if self._thread is None or not self._thread.is_alive():
            return False
        cs = self._camera_status()
        if cs and cs.get("state") in ("starting", "unavailable"):
            return True       # camera.py is already retrying with backoff and reports once; restarting changes nothing
        return time.monotonic() - self._last_frame_mono <= max(FRAME_STALL_S, 3 * self.s.frame_interval_s)

    def _camera_status(self) -> Optional[dict]:
        fn = getattr(self.camera, "status", None)
        try:
            return fn() if callable(fn) else None
        except Exception:  # noqa: BLE001
            return None

    def restart(self) -> None:
        """Drop the camera(s) and start a fresh worker (models stay loaded). Safe if the old thread is hung."""
        self.restarts += 1
        log.warning("vision pipeline: restarting worker (#%d, state %s, %.1fs since last frame)", self.restarts,
                    self.state, time.monotonic() - self._last_frame_mono)
        cam, self.camera = self.camera, None
        if cam is not None:
            threading.Thread(target=cam.stop, name="camera-stop", daemon=True).start()   # may block on a hung pipeline
        old = self._thread
        if old is not None:
            old.join(timeout=3.0)
        self.state = "restarting"
        self._spawn()

    def camera_info(self) -> dict:
        """Cheap live facts about the RGB stream (for system_check): size, fps, mean gray, age of the last frame."""
        cam = self.camera
        out: dict = {}
        if cam is None or not hasattr(cam, "latest"):
            return out
        f, ts, _ = cam.latest()
        if f is not None:
            out.update(w=int(f.shape[1]), h=int(f.shape[0]), mean_gray=round(float(f[::16, ::16].mean()), 1),
                       age_s=round(time.time() - ts, 2))
        if hasattr(cam, "fps"):
            out["fps"] = round(cam.fps(), 1)
        return out

    def thermal_info(self) -> dict:
        t = self.thermal
        if t is None or not getattr(t, "enabled", False):
            return {}
        out: dict = {"fps": round(t.fps(), 1), "frames": t.frames_total, "error": t.last_error}
        try:
            temp, ts = t.latest()
            if temp is not None:
                import numpy as np
                out.update(min_c=round(float(temp.min()), 1), median_c=round(float(np.median(temp)), 1),
                           max_c=round(float(temp.max()), 1), age_s=round(time.time() - ts, 2),
                           stuck_px=int(t.mask.sum()))
        except Exception:  # noqa: BLE001
            pass
        return out

    def models_info(self) -> dict:
        enc = getattr(getattr(self.backend, "encoder", None), "trt", "n/a")
        return {"detector": type(self.detector).__name__ if self.detector is not None else None,
                "weights": getattr(getattr(self.detector, "path", None), "name", None),
                "clip": None if enc == "n/a" else ("tensorrt" if enc is not None else "torch"),
                "backend": type(self.backend).__name__ if self.backend is not None else None}

    # ------------------------------------------------------------------ flight / target
    def begin_flight(self, storage: Any) -> None:
        with self._lock:
            self._storage = storage
            if self.fuser is not None:
                self.fuser.storage = storage
            if self.state == "ready" and self.backend is not None:
                self._start_matcher(storage)
        self._retry_backend_later()

    def _start_matcher(self, storage: Any) -> None:
        from matcher import Matcher
        if self.matcher is not None:
            self.matcher.storage = storage
            return
        self.matcher = Matcher(self.s, self.backend, storage, self._alert)
        if self._target is not None:
            self.matcher.update_target(self._target)
        self.matcher.start()

    def end_flight(self) -> None:
        with self._lock:
            m, self.matcher = self.matcher, None
            self._storage = None
            if self.fuser is not None:
                self.fuser.storage = None
        if m is not None:
            m.stop()

    def update_target(self, target: Optional[Target]) -> None:
        with self._lock:
            self._target = target
            m = self.matcher
        if m is not None:
            m.update_target(target)
        elif target is not None and self.backend is not None:
            try:
                self.backend.prepare(target)   # e.g. CLIP text prompts, ahead of the flight
            except Exception:  # noqa: BLE001
                log.exception("backend.prepare failed")

    def status(self) -> dict:
        st = {"state": self.state, "reason": self.reason, "camera": self.camera_source, "frames": self.frames,
              "processed": self.processed, "detections": self.detections, "errors": self.errors,
              "recording": self._storage is not None, "last_loop_ms": round(self.last_loop_ms, 1),
              "thermal": bool(self.thermal is not None and getattr(self.thermal, "enabled", False)),
              "backend": type(self.backend).__name__ if self.backend is not None else None}
        cs = self._camera_status()
        if cs is not None:
            st["camera_status"] = cs
            if cs.get("state") == "unavailable":
                st["camera_problem"] = "camera unavailable: " + (cs.get("error") or "") + (
                    f" - {cs['hint']}" if cs.get("hint") else "")
        m = self.matcher
        if m is not None:
            st["matcher"] = m.stats()
        return st

    # ------------------------------------------------------------------ loading
    def _load(self) -> bool:
        self.state = "loading"
        # match backend first (ClipEncoder: large CUDA allocations before the camera / detector, notes/phase5.md)
        if self.backend is None:
            self.backend = self._make_backend()
        # detector (required)
        if self.detector is None:
            try:
                if self._detector_factory is not None:
                    self.detector = self._detector_factory(self.s)
                else:
                    from detector import Detector
                    self.detector = Detector(self.s)
            except Exception as exc:  # noqa: BLE001
                return self._disable(f"person detector failed to load: {exc}")
        # RGB camera (required), started after the models are on the GPU
        if self.camera is None:
            try:
                if self._camera_factory is not None:
                    self.camera = self._camera_factory(self.s, self.camera_source)
                else:
                    from camera import Camera
                    self.camera = Camera(self.s, source=self.camera_source)
                self.camera.start(wait_s=20)
            except Exception as exc:  # noqa: BLE001
                self.camera = None
                return self._disable(f"camera unavailable ({self.camera_source}): {exc}")
        # thermal + fusion (optional)
        if self.use_thermal and self.thermal is None and self.s.thermal_source not in ("", "none"):
            try:
                from thermal import ThermalCamera
                self.thermal = ThermalCamera(self.s).start()
            except Exception as exc:  # noqa: BLE001
                log.warning("vision: thermal camera unavailable (%s); RGB only", exc)
                self.thermal = None
        if self.fuser is None:
            try:
                from fusion import Fuser, load_homography
                self.fuser = Fuser(self.s, load_homography(self.s), detector=self.detector, storage=self._storage,
                                   stuck_mask=(lambda: self.thermal.mask) if self.thermal is not None else None)
            except Exception as exc:  # noqa: BLE001
                log.warning("vision: fusion unavailable (%s); YOLO only", exc)
                self.fuser = None
        self.state = "ready"
        self.reason = ""
        with self._lock:
            if self._target is not None:
                try:
                    self.backend.prepare(self._target)
                except Exception:  # noqa: BLE001
                    log.exception("backend.prepare failed")
            if self._storage is not None:
                self._start_matcher(self._storage)
        log.info("vision pipeline ready: camera=%s detector=%s thermal=%s backend=%s", self.camera_source,
                 type(self.detector).__name__, self.thermal is not None, type(self.backend).__name__)
        return True

    def _make_backend(self) -> Any:
        if self._backend_factory is not None:
            try:
                return self._backend_factory(self.s)
            except Exception as exc:  # noqa: BLE001
                log.error("vision: injected match backend failed: %s", exc)
                return _NoMatchBackend()
        # Policy: the LLM gets text only, never images -> CLIP is the only match backend (no VlmBackend,
        # no confirm backend). Gemma is used only to write the CLIP prompt lists (text).
        if self.s.match_backend != "clip":
            log.warning("MATCH_BACKEND=%s ignored: images never go to the LLM; using CLIP", self.s.match_backend)
        if getattr(self.s, "vlm_confirm_top", False):
            log.warning("VLM_CONFIRM_TOP=true ignored: images never go to the LLM")
        for attempt in range(CLIP_LOAD_ATTEMPTS):
            try:
                return self._clip_backend()
            except Exception as exc:  # noqa: BLE001 - CUDA OOM (NvMap error 12) is common under memory pressure
                log.error("vision: CLIP backend load failed (attempt %d/%d): %s", attempt + 1, CLIP_LOAD_ATTEMPTS, exc)
                self._free_cuda()
                time.sleep(2.0 * (attempt + 1))
        log.error("vision: CLIP unavailable; person/heat alerts only until a later retry succeeds")
        return _NoMatchBackend()

    def _clip_backend(self) -> Any:
        from clip_encoder import ClipEncoder
        from matcher import ClipBackend, default_prompts
        builder: Any = default_prompts
        if getattr(self.s, "clip_prompts_llm", True):
            from clip_prompts import LlmPromptBuilder
            from llm_client import LLMClient
            builder = LlmPromptBuilder(LLMClient(settings=self.s, retries=0))
        if getattr(self.s, "demo_mode", False):
            from demo import EyeLevelBuilder
            builder = EyeLevelBuilder(builder)       # ground demo: people are seen at eye level, not from above
        return ClipBackend(ClipEncoder(self.s), prompt_builder=builder, cutoff=getattr(self.s, "clip_match_cutoff", 0.5))

    @staticmethod
    def _free_cuda() -> None:
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _retry_backend_later(self) -> None:
        """If CLIP failed at start-up, try again in the background (e.g. at the start of each flight)."""
        if self._backend_factory is not None or not isinstance(self.backend, _NoMatchBackend) or self._retrying:
            return
        self._retrying = True

        def run() -> None:
            try:
                self._free_cuda()
                b = self._clip_backend()
            except Exception as exc:  # noqa: BLE001
                log.warning("vision: CLIP retry failed: %s", exc)
                return
            finally:
                self._retrying = False
            with self._lock:
                self.backend = b
                target, m = self._target, self.matcher
            if target is not None:
                try:
                    b.prepare(target)
                except Exception:  # noqa: BLE001
                    log.exception("backend.prepare failed")
            if m is not None:
                m.backend = b
            log.warning("vision: CLIP backend loaded on retry; target matching enabled")

        threading.Thread(target=run, name="clip-retry", daemon=True).start()

    def _disable(self, reason: str) -> bool:
        log.error("vision pipeline DISABLED: %s (system still plans/flies, no vision alerts)", reason)
        self.state, self.reason = "disabled", reason
        return False

    # ------------------------------------------------------------------ worker
    def _alert(self, alert: Alert) -> None:
        try:
            self.on_alert(alert)
        except Exception:  # noqa: BLE001
            log.exception("on_alert failed")

    def _main(self, gen: int) -> None:
        if self.camera is None or self.detector is None or self.state != "ready":
            if not self._load():
                return
        cam = self.camera
        self._last_frame_mono = time.monotonic()
        try:
            for frame, ts, fid in cam.frames():
                if self._stop.is_set() or gen != self._gen:
                    break
                self.frames += 1
                self.last_frame_ts = ts
                self._last_frame_mono = time.monotonic()
                with self._lock:
                    storage, matcher = self._storage, self.matcher
                if storage is None:
                    continue          # not flying: keep the camera warm, spend no GPU
                t0 = time.perf_counter()
                try:
                    self._process(frame, ts, fid, storage, matcher)
                except Exception:  # noqa: BLE001 - one bad frame must not end the search
                    self.errors += 1
                    log.exception("vision: frame %s failed", fid)
                    if self.errors > 50 and self.errors > self.processed:
                        raise
                self.last_loop_ms = (time.perf_counter() - t0) * 1000
            if not self._stop.is_set() and gen == self._gen:
                self.state, self.reason = "failed", "camera frame stream ended"
                log.error("vision: camera frame stream ended unexpectedly")
        except Exception as exc:  # noqa: BLE001
            self.state, self.reason = "failed", f"{exc.__class__.__name__}: {exc}"
            log.exception("vision worker crashed")
            raise

    def _process(self, frame, ts: float, fid: int, storage: Any, matcher: Any) -> None:
        tel = self.telemetry_fn()
        temp, temp_ts = (None, 0.0)
        if self.thermal is not None:
            temp, temp_ts = self.thermal.nearest(ts)
        if self.fuser is not None:
            res = self.fuser.process(frame, ts, fid, tel, temp, temp_ts)
            dets = res.detections
            for a in res.alerts:
                storage.append("alerts", a)
                self._alert(a)
        else:
            dets = self.detector.detect(frame, ts=ts, frame_id=fid, telemetry=tel)
        self.processed += 1
        self.detections += len(dets)
        self.last_dets = (list(dets), time.time())      # read by webapp/stream.py (?boxes=1)
        if temp is not None:
            try:
                from thermal import colorize
                storage.save_frame(colorize(temp, scale=4), f"{fid:06d}_thermal")
            except Exception:  # noqa: BLE001
                log.debug("thermal frame save failed", exc_info=True)
        if matcher is not None:
            kept = matcher.submit(frame, dets)      # saves the frame when it has detections
        else:
            kept = []
            for d in dets:
                storage.append("detections", d)
        if self.save_all_frames and not kept:
            storage.save_frame(frame, f"{fid:06d}")
