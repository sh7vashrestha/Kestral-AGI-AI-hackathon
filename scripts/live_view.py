#!/usr/bin/env python3
"""Standalone live camera page: RGB + thermal (+ overlay) over HTTP, no main.py needed.

    .venv/bin/python scripts/live_view.py [--port 8090] [--rgb csi:0] [--thermal auto|v4l2:...|none] [--fps 10]
    open http://<jetson-ip>:8090/live        (hotspot: http://10.42.0.1:8090/live)

Opens both cameras itself, so main.py must NOT hold them (only one process can). To watch while main.py runs, use
http://<jetson-ip>:8000/live instead (shares the pipeline's frames). --demo serves synthetic frames (no hardware).
If a camera is busy or broken the page still works and shows that feed as unavailable with the reason.
"""
from __future__ import annotations

import argparse
import glob
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "drone_brain")]

from config import get_settings  # noqa: E402


def resolve_thermal(spec: str, configured: str) -> str:
    if spec != "auto":
        return spec
    if configured and configured != "none":
        return configured
    devs = sorted(glob.glob("/dev/v4l/by-id/*PureThermal*-video-index0"))
    return f"v4l2:{devs[0]}" if devs else "none"


def lan_addresses() -> list[str]:
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=3).stdout.split()
    except Exception:  # noqa: BLE001
        out = []
    return [a for a in out if "." in a and not a.startswith("172.17.")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--rgb", default=None, help="camera source (default CAMERA_SOURCE, csi:0)")
    ap.add_argument("--thermal", default="auto", help="auto | v4l2:/dev/... | file:<dir> | none")
    ap.add_argument("--fps", type=int, default=10, help="default stream rate (page can change it)")
    ap.add_argument("--demo", action="store_true", help="synthetic frames, no cameras")
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse

    from webapp import stream

    s = get_settings()
    cam = th = None
    if a.demo:
        src = stream.SyntheticSource()
    else:
        from camera import Camera
        from thermal import ThermalCamera
        rgb_src = a.rgb or s.camera_source
        th_src = resolve_thermal(a.thermal, s.thermal_source)
        cam = Camera(s, source=rgb_src).start()
        th = ThermalCamera(s, source=th_src).start()
        dev = th_src.split(":", 1)[1] if th_src.startswith("v4l2:") else ""
        src = stream.CamerasSource(cam, th, s, thermal_dev=dev)
        print(f"RGB: {rgb_src}   thermal: {th_src}")
        print("If RGB says 'No cameras available': sudo systemctl restart nvargus-daemon")

    app = FastAPI(title="drone-brain live view")
    app.include_router(stream.make_router(src, s, default_fps=a.fps))

    @app.get("/")
    def root():
        return RedirectResponse("/live")

    print("\nLive view:")
    for ip in lan_addresses() or ["<jetson-ip>"]:
        print(f"  http://{ip}:{a.port}/live" + ("   (hotspot)" if ip == "10.42.0.1" else ""))
    print(f"  http://10.42.0.1:{a.port}/live   (if on the Jetson hotspot)\n", flush=True)
    import contextlib

    import shutdown

    class _Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):      # we own SIGINT/SIGTERM: uvicorn would wait for the open MJPEG streams
            yield

    server = _Server(uvicorn.Config(app, host=a.host, port=a.port, log_level="warning", timeout_graceful_shutdown=1))
    # SIGTERM/SIGINT: stop uvicorn, release BOTH cameras, exit within ~3 s (hard exit if anything blocks)
    shutdown.install(on_signal=lambda: setattr(server, "should_exit", True), hard_exit_s=2.8)
    try:
        server.run()
    finally:
        for c in (cam, th):
            if c is not None:
                c.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
