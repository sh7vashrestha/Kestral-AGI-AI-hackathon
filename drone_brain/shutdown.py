"""Clean, bounded shutdown for processes that own the cameras (live_view, main.py, system_check).

    shutdown.register(obj)            # obj.stop() releases a device (Camera / ThermalCamera do this in start())
    shutdown.release_all(timeout_s)   # stop every registered object in parallel, never waits longer than timeout_s
    shutdown.install(...)             # SIGINT/SIGTERM -> on_signal() -> release_all -> exit within hard_exit_s

A second signal exits immediately. The hard exit uses os._exit so a blocked GStreamer/V4L2 call or a streaming
HTTP connection can never keep /dev/video0 or the CSI camera open.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
import weakref
from typing import Callable, Optional

log = logging.getLogger(__name__)
_objs: "weakref.WeakSet" = weakref.WeakSet()
_lock = threading.Lock()
_fired = threading.Event()


def register(obj) -> None:
    with _lock:
        _objs.add(obj)


def release_all(timeout_s: float = 2.5) -> int:
    """Call .stop() on every registered object concurrently; returns how many did not finish in time."""
    with _lock:
        objs = list(_objs)
    threads = []
    for o in objs:
        t = threading.Thread(target=lambda o=o: _safe_stop(o), daemon=True, name="release")
        t.start()
        threads.append(t)
    end = time.monotonic() + timeout_s
    for t in threads:
        t.join(max(0.0, end - time.monotonic()))
    stuck = sum(t.is_alive() for t in threads)
    if stuck:
        log.warning("%d device(s) did not release within %.1f s", stuck, timeout_s)
    return stuck


def _safe_stop(o) -> None:
    try:
        o.stop()
    except Exception:  # noqa: BLE001
        log.exception("stop failed for %r", o)


def arm_hard_exit(after_s: float, code: int = 0) -> None:
    """Release every device and os._exit(code) after `after_s` seconds, whatever the rest of the process is doing."""
    def run() -> None:
        time.sleep(after_s)
        release_all(0.4)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)
    threading.Thread(target=run, daemon=True, name="hard-exit").start()


def install(on_signal: Optional[Callable[[], None]] = None, hard_exit_s: float = 2.8, exit_code: int = 0) -> None:
    """Handle SIGINT/SIGTERM in the main thread. Order: on_signal() (e.g. ask uvicorn to stop) -> release devices
    -> exit. If the process is still alive after hard_exit_s it is terminated."""
    def handler(signum, _frame) -> None:
        if _fired.is_set():
            os._exit(130)
        _fired.set()
        log.warning("%s: releasing cameras and exiting", signal.Signals(signum).name)
        arm_hard_exit(hard_exit_s, exit_code)
        if on_signal is not None:
            try:
                on_signal()
            except Exception:  # noqa: BLE001
                log.exception("on_signal failed")
        threading.Thread(target=release_all, args=(hard_exit_s - 0.5,), daemon=True).start()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)
