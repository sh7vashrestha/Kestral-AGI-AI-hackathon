"""RGB camera capture thread (latest-frame only).

Sources (settings.camera_source):
    csi:<sensor-id>          Jetson CSI via nvarguscamerasrc (GStreamer)
    v4l2:/dev/videoN         USB UVC
    file:<video | folder of images | image>   loops forever (tests / SITL)

Public API:
    cam = Camera(settings); cam.start(); frame, ts, fid = cam.latest(); cam.stop()
    for frame, ts, fid in cam.frames(): ...   # paced at settings.capture_fps (latest frame, never a backlog)
"""
from __future__ import annotations

import logging
import os
import select
import struct
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

import orient
import shutdown
from config import Settings

log = logging.getLogger(__name__)

OPEN_TIMEOUT_S = 15.0        # open + FIRST REAL FRAME must arrive within this (isOpened() alone proves nothing)
MIN_BACKOFF_S = 0.5
MAX_BACKOFF_S = 30.0
READ_STALL_S = 5.0           # an open stream that delivers nothing for this long is treated as failed (reopened)
STOP_JOIN_S = 2.5            # stop(): how long to wait for the capture thread
FROZEN_S = 5.0               # identical textured frames for this long = frozen stream (sensor/driver stuck) -> reopen
WORKER_TERM_S = 2.0          # CSI worker: SIGTERM -> wait this long -> SIGKILL
SUMMARY_EVERY_S = 60.0       # while unavailable: one summary line per this many seconds
ARGUS_HINT = "No cameras available (argus) - try: sudo systemctl restart nvargus-daemon"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class CameraError(RuntimeError):
    pass


def gst_csi_pipeline(sensor_id: int, w: int, h: int, fps: int = 30, flip_method: int = 0) -> str:
    """`flip_method` is nvvidconv's hardware rotate/flip (see orient.gst_flip_method); 90/270 give an h x w frame."""
    return (f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
            f"nvvidconv flip-method={int(flip_method)} ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false")


class _CsiWorker:
    """cv2.VideoCapture-like handle on a camera_worker.py child process (one process per open attempt).

    read() waits at most `timeout_s` for a frame, so the capture thread is never stuck in a blocking read;
    release() = SIGTERM (worker sets the pipeline to NULL) -> SIGKILL only if it ignores that for WORKER_TERM_S."""
    HDR = struct.Struct("<4sIId")

    def __init__(self, pipeline: str, timeout_s: float = 0.5) -> None:
        self.timeout_s = timeout_s
        script = Path(__file__).with_name("camera_worker.py")
        self.proc = subprocess.Popen([sys.executable, "-u", str(script), "--pipeline", pipeline],
                                     stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, bufsize=0, close_fds=True)
        self.fd = self.proc.stdout.fileno()
        try:
            import fcntl
            fcntl.fcntl(self.fd, 1031, 1 << 20)           # F_SETPIPE_SZ: ~1 frame in flight
        except Exception:  # noqa: BLE001
            pass
        self._closed = False

    def isOpened(self) -> bool:  # noqa: N802 (cv2 name)
        return not self._closed and self.proc.poll() is None

    def _read_n(self, buf: memoryview, deadline: float, first_wait: bool) -> bool:
        off = 0
        while off < len(buf):
            left = deadline - time.monotonic()
            if left <= 0:
                if first_wait and off == 0:
                    return False
                raise CameraError("camera worker stalled mid-frame")
            r, _, _ = select.select([self.fd], [], [], min(left, 0.25))
            if not r:
                if self.proc.poll() is not None:
                    raise CameraError(f"camera worker exited (code {self.proc.returncode})")
                continue
            n = os.readv(self.fd, [buf[off:]])
            if n == 0:
                raise CameraError(f"camera worker exited (code {self.proc.wait(1)})")
            off += n
        return True

    def read(self):
        hdr = bytearray(self.HDR.size)
        t_end = time.monotonic() + self.timeout_s
        if not self._read_n(memoryview(hdr), t_end, True):
            return False, None
        magic, w, h, _ts = self.HDR.unpack(hdr)
        if magic != b"FRM1" or not (0 < w <= 8192 and 0 < h <= 8192):
            raise CameraError("camera worker sent a corrupt frame header")
        img = np.empty((h, w, 3), np.uint8)
        self._read_n(memoryview(img).cast("B"), time.monotonic() + 3.0, False)
        return True, img

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        p = self.proc
        try:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(WORKER_TERM_S)
                except subprocess.TimeoutExpired:
                    log.warning("camera worker ignored SIGTERM for %.0f s: SIGKILL", WORKER_TERM_S)
                    p.kill()
                    p.wait(2)
        finally:
            try:
                p.stdout.close()
            except Exception:  # noqa: BLE001
                pass


class _FileSource:
    """Looping video file or image folder, paced at ~native fps (default 10) so it behaves like a camera."""

    def __init__(self, path: str, fps: float = 10.0) -> None:
        p = Path(path).expanduser()
        self.cap: Optional[cv2.VideoCapture] = None
        self.images: list[Path] = []
        self.i = 0
        self.period = 1.0 / fps
        if p.is_dir():
            self.images = sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT)
        elif p.suffix.lower() in IMG_EXT:
            self.images = [p]
        elif p.exists():
            self.cap = cv2.VideoCapture(str(p))
            if not self.cap.isOpened():
                raise CameraError(f"cannot open video {p}")
            f = self.cap.get(cv2.CAP_PROP_FPS)
            if f and f > 1:
                self.period = 1.0 / f
        else:
            raise CameraError(f"file source not found: {p}")
        if self.cap is None and not self.images:
            raise CameraError(f"no images in {p}")

    def read(self) -> Optional[np.ndarray]:
        time.sleep(self.period)
        if self.cap is not None:
            ok, f = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, f = self.cap.read()
            return f if ok else None
        f = cv2.imread(str(self.images[self.i % len(self.images)]))
        self.i += 1
        return f

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


class _FrozenWatch:
    """Detects a stream that keeps delivering the SAME picture (driver/sensor stuck, the last buffer re-sent).
    Real sensors always have noise, so identical subsampled pixels for FROZEN_S seconds means frozen. Flat frames
    (black / blown out) are ignored here: they legitimately look identical and are handled by dark/flat mode."""

    def __init__(self, skip: bool = False) -> None:
        self.skip = skip            # file sources replay the same picture on purpose
        self.sig: Optional[int] = None
        self.since = 0.0
        self.n = 0

    def seen(self, f: np.ndarray) -> bool:
        if self.skip:
            return False
        sub = np.ascontiguousarray(f[::16, ::16])
        sig = zlib.crc32(sub.tobytes())
        now = time.monotonic()
        if sig != self.sig or float(sub.std()) < 3.0:
            self.sig, self.since, self.n = sig, now, 0
            return False
        self.n += 1
        return self.n >= 10 and now - self.since >= FROZEN_S


class Camera:
    def __init__(self, settings: Settings, source: Optional[str] = None) -> None:
        self.s = settings
        self.source = source or settings.camera_source
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts = 0.0
        self._id = 0
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None
        self.reconnects = 0              # failed attempts so far (every failure path: open, no first frame, stall, exception)
        self.attempts = 0                # opens tried
        self.last_attempt_t: Optional[float] = None
        self.next_retry_s: Optional[float] = None
        self.rotate, self.flip = orient.check(settings.camera_rotate, settings.camera_flip)
        self._orient_gen = 0             # bumped by set_orient(); a csi worker is restarted when it changes
        self._cap = None
        self._fps_t0, self._fps_id0 = 0.0, 0
        self.state = "starting"          # starting | ok | unavailable
        self._state_since = time.monotonic()

    # ---- lifecycle ----
    def start(self, wait_s: float = 0.0) -> "Camera":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        shutdown.register(self)
        if wait_s > 0:
            self.wait_first(wait_s)
        return self

    def stop(self) -> None:
        """Stop within ~3 s: flag -> join the thread -> release the device (CSI worker: SIGTERM, then SIGKILL)."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        cap = self._cap
        if isinstance(cap, _CsiWorker):          # kill the worker first: unblocks anything waiting on it
            self._release(cap)
        t = self._thread
        if t and t is not threading.current_thread():
            t.join(timeout=STOP_JOIN_S)
            if t.is_alive():
                log.warning("camera thread still busy after %.1f s (blocked read); it is a daemon thread", STOP_JOIN_S)
        elif cap is not None and not (t and t.is_alive()):
            self._release(cap)

    def set_orient(self, rotate: int, flip: str) -> None:
        """Change rotation/flip live (live preview). csi: the pipeline is restarted with the new flip-method;
        other sources: applied to the very next frame."""
        r, f = orient.check(rotate, flip)
        if (r, f) != (self.rotate, self.flip):
            self.rotate, self.flip = r, f
            self._orient_gen += 1
            with self._cond:
                self._frame = None           # never serve a frame with the previous orientation
            if self.source.startswith("csi") and self.state == "ok":
                self._set_state("starting")  # pipeline restarts (a few seconds)

    def wait_first(self, timeout_s: float) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self._frame is not None, timeout=timeout_s)

    # ---- access ----
    def latest(self) -> tuple[Optional[np.ndarray], float, int]:
        with self._lock:
            return self._frame, self._ts, self._id

    def frames(self, interval_s: Optional[float] = None) -> Iterator[tuple[np.ndarray, float, int]]:
        """Yield the newest frame every interval_s (default settings.frame_interval_s) until stop()."""
        interval = interval_s or self.s.frame_interval_s
        last_id = -1
        nxt = time.monotonic()
        while not self._stop.is_set():
            nxt += interval
            with self._cond:
                self._cond.wait_for(lambda: self._id != last_id or self._stop.is_set(), timeout=2.0)
                f, ts, fid = self._frame, self._ts, self._id
            if f is not None and fid != last_id:
                last_id = fid
                yield f, ts, fid
            delay = nxt - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                nxt = time.monotonic()

    # ---- thread ----
    def _open(self):
        kind, _, arg = self.source.partition(":")
        w, h = self.s.camera_width, self.s.camera_height
        if kind == "csi":
            cap = _CsiWorker(gst_csi_pipeline(int(arg or 0), w, h, flip_method=orient.gst_flip_method(self.rotate, self.flip)))
            self._cap = cap          # so stop() can terminate the worker even while the first frame is awaited
            return cap
        elif kind == "v4l2":
            cap = cv2.VideoCapture(arg, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif kind == "file":
            return _FileSource(arg)
        else:
            raise CameraError(f"bad CAMERA_SOURCE {self.source!r} (csi:N | v4l2:/dev/videoN | file:PATH)")
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"cannot open {self.source}")
        return cap

    def _open_with_timeout(self):
        """Open the source AND read its first real frame, all within OPEN_TIMEOUT_S.

        With a broken nvargus-daemon GStreamer reports isOpened()=True but never delivers a frame, so
        "opened" is only declared once a frame arrived. Returns (cap, first_frame)."""
        box: dict = {}
        abandoned = threading.Event()

        def work() -> None:
            cap = None
            try:
                cap = self._open()
                deadline = time.monotonic() + OPEN_TIMEOUT_S
                while not abandoned.is_set() and time.monotonic() < deadline:
                    ok, f = self._read(cap)
                    if ok and f is not None:
                        box["res"] = (cap, f)
                        return
                    time.sleep(0.05)
                raise CameraError(f"opened {self.source} but no frame within {OPEN_TIMEOUT_S:g}s")
            except Exception as exc:  # noqa: BLE001
                box["err"] = exc
            finally:
                if "res" not in box and cap is not None:
                    self._release(cap)

        t = threading.Thread(target=work, daemon=True, name="camera-open")
        t.start()
        t.join(OPEN_TIMEOUT_S + 2.0)
        if t.is_alive():
            abandoned.set()
            raise CameraError(f"timeout opening {self.source} (no frame after {OPEN_TIMEOUT_S:g}s)")
        if "err" in box:
            raise box["err"]
        return box["res"]

    def _read(self, cap):
        if isinstance(cap, _FileSource):
            f = cap.read()
            ok = f is not None
        else:
            ok, f = cap.read()
        if ok and f is not None and not isinstance(cap, _CsiWorker) and (self.rotate or self.flip != "none"):
            f = np.ascontiguousarray(orient.apply(f, self.rotate, self.flip))   # csi is oriented in hardware (flip-method)
        return ok, f

    @staticmethod
    def _release(cap) -> None:
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            pass

    def hint(self) -> str:
        """Human fix hint for the current error (empty when healthy)."""
        if self.state == "ok" or not self.last_error:
            return ""
        return ARGUS_HINT if self.source.startswith("csi") else ""

    def status(self) -> dict:
        """{"state": starting|ok|unavailable, "source", "error", "hint", "reconnects", "since_s"}"""
        return {"state": self.state, "source": self.source, "error": self.last_error, "hint": self.hint(),
                "reconnects": self.reconnects, "attempts": self.attempts, "next_retry_s": self.next_retry_s,
                "last_attempt_age_s": None if self.last_attempt_t is None else round(time.monotonic() - self.last_attempt_t, 1),
                "rotate": self.rotate, "flip": self.flip, "since_s": round(time.monotonic() - self._state_since, 1)}

    def _set_state(self, st: str) -> None:
        if st != self.state:
            self.state, self._state_since = st, time.monotonic()
            self._fps_t0 = 0.0

    def _run(self) -> None:
        """Open -> read -> (any failure) -> release -> back off -> open again, until stop(). EVERY failure path ends
        in the except below: failed open, no first frame, read failures, a stalled stream, unexpected exceptions."""
        backoff = MIN_BACKOFF_S
        failing_since: Optional[float] = None
        last_summary = 0.0
        while not self._stop.is_set():
            cap = None
            gen = self._orient_gen
            try:
                self.attempts += 1
                self.last_attempt_t = time.monotonic()
                self.next_retry_s = None
                cap, first = self._open_with_timeout()
                self._cap = cap
                if failing_since is not None:
                    log.warning("camera recovered after %.0f s: %s", time.monotonic() - failing_since, self.source)
                else:
                    log.info("camera opened: %s (first frame %dx%d, rotate %d flip %s)", self.source, first.shape[1],
                             first.shape[0], self.rotate, self.flip)
                failing_since = None
                self.last_error = None
                backoff = MIN_BACKOFF_S
                self._set_state("ok")
                self._publish(first)
                last_ok = time.monotonic()
                frozen = _FrozenWatch(skip=isinstance(cap, _FileSource))
                frozen.seen(first)
                while not self._stop.is_set():
                    if self._orient_gen != gen and isinstance(cap, _CsiWorker):
                        break                                   # orientation changed: reopen with the new flip-method
                    ok, f = self._read(cap)
                    if not ok or f is None:
                        if time.monotonic() - last_ok > READ_STALL_S:
                            raise CameraError(f"stream stalled: no frame for {READ_STALL_S:g} s")
                        time.sleep(0.02)
                        continue
                    last_ok = time.monotonic()
                    if frozen.seen(f):
                        raise CameraError(f"stream frozen: identical frames for {FROZEN_S:g} s")
                    self._publish(f)
            except Exception as exc:  # noqa: BLE001
                if self._stop.is_set():
                    break
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.reconnects += 1
                self._set_state("unavailable")
                now = time.monotonic()
                if failing_since is None:           # log the failure once ...
                    failing_since = last_summary = now
                    log.error("camera unavailable: %s%s", self.last_error, f" - {self.hint()}" if self.hint() else "")
                elif now - last_summary >= SUMMARY_EVERY_S:   # ... then a summary every minute
                    last_summary = now
                    log.warning("camera still unavailable for %.0f s (%d attempts): %s%s", now - failing_since,
                                self.reconnects, self.last_error, f" - {self.hint()}" if self.hint() else "")
                else:
                    log.debug("camera retry in %.1fs: %s", backoff, self.last_error)
                self.next_retry_s = backoff
                self._stop.wait(backoff)
                backoff = min(MAX_BACKOFF_S, backoff * 2)
            finally:
                self._cap = None
                if cap is not None:
                    self._release(cap)

    def fps(self) -> float:
        """Average delivered fps since the stream last (re)started."""
        with self._lock:
            n, dt = self._id - self._fps_id0, time.time() - self._fps_t0
        return n / dt if n > 0 and dt > 0 else 0.0

    def _publish(self, f: np.ndarray) -> None:
        with self._cond:
            if self.state == "ok" and self._fps_t0 == 0.0:
                self._fps_t0, self._fps_id0 = time.time(), self._id
            self._frame, self._ts, self._id = f, time.time(), self._id + 1
            self._cond.notify_all()
