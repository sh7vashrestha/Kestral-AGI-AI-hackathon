#!/usr/bin/env python3
"""CSI capture child process: runs the GStreamer pipeline and streams raw BGR frames to stdout.

Why a separate process: libargus keeps a connection to nvargus-daemon INSIDE the process. When the daemon is
restarted (or wedges) that connection is dead for the life of the process - "Receive thread is not running" - so
retrying in-process can never recover. A retry therefore means a NEW worker process (fresh argus connection), and a
hung pipeline can be terminated from outside. Used by camera.Camera for `csi:` sources; not meant to be run by hand.

Wire format (little endian): b"FRM1" | u32 width | u32 height | f64 unix_ts | width*height*3 bytes BGR.
SIGTERM/SIGINT: pipeline released (cap.release() = state NULL) within ~1.5 s, then exit.
"""
from __future__ import annotations

import argparse
import os
import signal
import struct
import sys
import threading
import time

HEADER = struct.Struct("<4sIId")
MAGIC = b"FRM1"
RELEASE_TIMEOUT_S = 1.5


def _write_all(fd: int, data) -> None:
    mv = memoryview(data).cast("B")
    while len(mv):
        n = os.write(fd, mv)
        mv = mv[n:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    a = ap.parse_args(argv)
    # libargus / GStreamer print status lines to stdout: keep the frame pipe private and send fd 1 to stderr
    frame_fd = os.dup(1)
    os.dup2(2, 1)
    parent = os.getppid()     # (no PDEATHSIG: it fires when the spawning THREAD exits). Parent death = EPIPE or a changed ppid.
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    import cv2
    holder: dict = {}

    def capture() -> None:
        try:
            cap = cv2.VideoCapture(a.pipeline, cv2.CAP_GSTREAMER)
            holder["cap"] = cap
            if not cap.isOpened():
                holder["err"] = "cannot open the GStreamer pipeline"
                return
            out = frame_fd
            while not stop.is_set():
                ok, f = cap.read()
                if not ok or f is None:
                    time.sleep(0.02)
                    continue
                h, w = f.shape[:2]
                _write_all(out, HEADER.pack(MAGIC, w, h, time.time()))
                _write_all(out, f)
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001
            holder["err"] = f"{type(exc).__name__}: {exc}"
        finally:
            holder["done"] = True
            stop.set()

    t = threading.Thread(target=capture, daemon=True, name="capture")
    t.start()
    while not stop.wait(0.3):
        if os.getppid() != parent:
            stop.set()
    # release the pipeline (state NULL) but never hang on it
    cap = holder.get("cap")
    if cap is not None:
        rel = threading.Thread(target=cap.release, daemon=True)
        rel.start()
        rel.join(RELEASE_TIMEOUT_S)
    if holder.get("err"):
        print(f"camera_worker: {holder['err']}", file=sys.stderr, flush=True)
        os._exit(3)
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
