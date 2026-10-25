#!/usr/bin/env python3
"""drone-brain orchestrator (Phase 7): web UI + planner + vehicle + vision, one asyncio process.

    python main.py --mock-vehicle                  # full demo, no hardware (fake ArduCopter at the KSU field)
    python main.py --mavlink-url udpin:127.0.0.1:14550 --camera file:tests/assets/persons   # with ArduPilot SITL
    python main.py --check [--quick]               # check EVERYTHING in one go (read-only), then exit
    python main.py --vehicle readonly --mavlink-url serial:/dev/ttyACM0 --baud 115200 --no-vision   # real Pixhawk, read-only
    python main.py --vehicle upload --mavlink-url serial:/dev/ttyACM0 --baud 115200 --no-vision     # + upload for review
    python main.py --vehicle upload ... --bench-home field   # indoors (no GPS fix): plan/upload with the field as home

Vehicle modes (VEHICLE_MODE / --vehicle):
    sim       only a simulator/mock link is accepted (loopback, or wildcard UDP with DRY_RUN=true)
    readonly  any link; telemetry + parameter reads only, every command is refused before any I/O
    upload    readonly + mission upload/verify/clear when the operator taps Confirm (review in Mission Planner/QGC);
              never changes mode, arms, starts or RTLs
    full      any link; arming real hardware still needs ALLOW_REAL_ARM=true and DRY_RUN=false (set by a human)

Tasks: uvicorn (phone UI + WebSocket), controller loops (vehicle link, 2 Hz telemetry/state/home, vision watchdog),
vision worker thread + matcher thread (per flight). SIGINT/SIGTERM: stop camera + matcher, close the flight
record, close the vehicle link, stop the mock.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
for _p in (ROOT, ROOT / "drone_brain"):          # the modules live in drone_brain/ and import each other by plain name
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import FIELD_AMSL_M, FIELD_HOME, BenchHomeError, Settings, bench_home_for, get_settings  # noqa: E402,F401

log = logging.getLogger("main")

# FIELD_HOME / FIELD_AMSL_M (KSU Marietta soccer field) live in config.py: mock / SITL start and `--bench-home field`
MOCK_PORT = 14560


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="drone-brain orchestrator")
    p.add_argument("--vehicle", choices=["sim", "readonly", "upload", "full"], default=None,
                   help="vehicle mode (default: VEHICLE_MODE from .env, else sim)")
    p.add_argument("--bench-home", default=None, metavar="LAT,LON[,AMSL]|field",
                   help="indoor UPLOAD test only (needs --vehicle upload): plan with this home while the Pixhawk has no "
                        "GPS home yet; 'field' = KSU soccer field centre. A real Pixhawk home always wins.")
    p.add_argument("--mock-vehicle", action="store_true",
                   help="run scripts/mock_vehicle.py in-process at the field home (no hardware)")
    p.add_argument("--demo", action="store_true",
                   help="ground demo: hovering mock vehicle (nothing flies), real cameras; GO = live CLIP search")
    p.add_argument("--mock-speed", type=float, default=1.0, help="mock simulation speed-up")
    p.add_argument("--mavlink-url", default=None, help="override MAVLINK_URL (e.g. serial:/dev/ttyACM0)")
    p.add_argument("--baud", type=int, default=None, help="override MAVLINK_BAUD")
    p.add_argument("--camera", default=None, help="override CAMERA_SOURCE (csi:0 | v4l2:/dev/videoN | file:<path>)")
    p.add_argument("--no-vision", action="store_true", help="disable camera/detector/matcher")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--log-level", default=None)
    p.add_argument("--check", action="store_true",
                   help="check the whole system (host, LLM, planner, cameras, YOLO/CLIP, Pixhawk, web, files) and exit; "
                        "uses the running main.py if there is one. Extra flags: --quick --json --only a,b")
    p.add_argument("--quick", action="store_true", help="with --check: skip camera/thermal/YOLO/CLIP")
    p.add_argument("--json", action="store_true", help="with --check: JSON output")
    p.add_argument("--standalone", action="store_true", help="with --check: ignore a running main.py and touch the devices")
    p.add_argument("--only", default="", help="with --check: comma list of groups (host,llm,planner,camera,thermal,yolo,clip,vehicle,web,files)")
    return p.parse_args(argv)


def build_settings(args: argparse.Namespace, base: Optional[Settings] = None) -> Settings:
    s = base or get_settings()
    upd: dict[str, Any] = {}
    if args.vehicle:
        upd["vehicle_mode"] = args.vehicle
    if args.bench_home:
        upd["bench_home"] = args.bench_home
    if args.demo:
        args.mock_vehicle = True
        upd.update(demo_mode=True, vehicle_mode="sim")
    if args.mock_vehicle:
        upd["mavlink_url"] = f"udpin:127.0.0.1:{MOCK_PORT}"
    if args.mavlink_url:
        if args.demo:
            raise SystemExit("--demo always uses the pretend vehicle on loopback; drop --mavlink-url")
        upd["mavlink_url"] = args.mavlink_url
    if args.baud:
        upd["mavlink_baud"] = args.baud
    if args.camera:
        upd["camera_source"] = args.camera
    if args.host:
        upd["web_host"] = args.host
    if args.port:
        upd["web_port"] = args.port
    if args.data_dir:
        upd["data_dir"] = Path(args.data_dir)
    if args.log_level:
        upd["log_level"] = args.log_level
    return s.model_copy(update=upd) if upd else s


def check_vehicle_mode(s: Settings) -> None:
    """Refuse dangerous combinations before anything is opened."""
    from mission import is_simulated_link
    try:
        bh = bench_home_for(s)
    except BenchHomeError as exc:
        raise SystemExit(f"refusing to start: {exc}")
    if bh is not None:
        log.warning("BENCH HOME %.7f, %.7f (%.0f m AMSL): upload test only - NOT a GPS home; a real Pixhawk home wins", *bh)
    if s.vehicle_mode == "sim" and not is_simulated_link(s.mavlink_url, s.dry_run):
        raise SystemExit(f"VEHICLE_MODE=sim but MAVLINK_URL={s.mavlink_url!r} is not a simulator link. "
                         f"Use --vehicle readonly to look at real hardware.")
    if s.vehicle_mode == "full" and not is_simulated_link(s.mavlink_url, s.dry_run):
        log.warning("VEHICLE_MODE=full on a NON-simulated link %s: arming %s", s.mavlink_url,
                    "PERMITTED (ALLOW_REAL_ARM=true, DRY_RUN=false)" if s.real_arm_permitted else
                    "still refused (ALLOW_REAL_ARM/DRY_RUN)")


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as so:
        so.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            so.bind((host if host not in ("", "0.0.0.0") else "0.0.0.0", port))
            return True
        except OSError:
            return False


def start_mock(speed: float, s: Optional[Settings] = None) -> Any:
    from scripts.mock_vehicle import MockVehicle
    demo = bool(s is not None and s.demo_mode)
    extra: dict[str, Any] = dict(hover=True, hover_max_s=s.demo_max_s, climb_ms=10.0, descend_ms=10.0) if demo else {}
    mock = MockVehicle(f"udpout:127.0.0.1:{MOCK_PORT}", home=FIELD_HOME, home_amsl_m=FIELD_AMSL_M,
                       time_scale=speed, **extra)
    mock.start()
    log.info("mock vehicle at %.7f, %.7f (x%.1f speed)", *FIELD_HOME, speed)
    return mock


def build_app(s: Settings, *, vision: bool = True, vehicle: Any = None, pipeline: Any = None, llm: Any = None,
              manage_connection: bool = True):
    """Wire controller + web app. Returns (app, controller, hub). Used by main() and the e2e tests."""
    import mission
    from controller_real import RealController
    from pipeline import NullPipeline, VisionPipeline
    from webapp.files import make_router
    from webapp.server import Hub, create_app

    vehicle = vehicle or mission.create_backend(s)
    holder: dict[str, Any] = {}
    if pipeline is None:
        if vision:
            pipeline = VisionPipeline(s, on_alert=lambda a: holder["ctrl"].push_alert(a),
                                      telemetry_fn=vehicle.telemetry)
        else:
            pipeline = NullPipeline()
    ctrl = RealController(s, vehicle, llm=llm, pipeline=pipeline, manage_connection=manage_connection)
    holder["ctrl"] = ctrl
    hub = Hub()
    app = create_app(ctrl, s, hub=hub)
    app.include_router(make_router(s, ctrl))
    import syscheck
    app.include_router(syscheck.make_router(s, ctrl))      # GET /api/system/check: the same report for this process
    from webapp import stream
    if s.demo_mode and vision:
        import demo                                       # ground demo: /api/search + best match boxed red on /live
        view = demo.DemoView(s, pipeline, ctrl)
        app.include_router(demo.make_router(view))
        app.include_router(stream.make_router(demo.make_source(pipeline, s, view), s))
        return app, ctrl, hub
    app.include_router(stream.make_router(stream.PipelineSource(pipeline, s), s))   # /live + /stream/*.mjpg (pipeline cameras)
    return app, ctrl, hub


async def run(args: argparse.Namespace) -> int:
    import uvicorn

    s = build_settings(args)
    logging.getLogger().setLevel(s.log_level)
    check_vehicle_mode(s)
    if not port_free(s.web_host, s.web_port):
        log.error("port %s is in use (another drone-brain or `python -m webapp.server --fake`?)", s.web_port)
        return 2
    mock = start_mock(args.mock_speed, s) if args.mock_vehicle else None
    app, ctrl, _hub = build_app(s, vision=not args.no_vision)

    class _Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):  # we own SIGINT/SIGTERM (graceful shutdown below)
            yield

    server = _Server(uvicorn.Config(app, host=s.web_host, port=s.web_port, log_level="info", lifespan="on",
                                    timeout_graceful_shutdown=3))
    servers = [server]
    tls = Path(s.data_dir) / "tls"
    if s.https and (tls / "server.crt").is_file() and (tls / "server.key").is_file() and port_free(s.web_host, s.https_port):
        # same app over HTTPS (lifespan only on the plain server, so the controller starts once)
        servers.append(_Server(uvicorn.Config(app, host=s.web_host, port=s.https_port, log_level="info", lifespan="off",
                                              timeout_graceful_shutdown=3, ssl_certfile=str(tls / "server.crt"),
                                              ssl_keyfile=str(tls / "server.key"))))
    elif s.https:
        log.warning("HTTPS=true but no certificate in %s (scripts/make_cert.sh) or port %s busy: HTTP only", tls, s.https_port)
    loop = asyncio.get_running_loop()

    def _stop(signame: str) -> None:
        log.warning("%s: shutting down", signame)
        for srv in servers:
            srv.should_exit = True
        # cameras are released right away and the process ends within ~3 s even if a stream/thread blocks
        import shutdown
        shutdown.arm_hard_exit(3.0)
        # armed in AUTO with our mission -> RTL first (bounded 2 s); mid-upload -> cancel; pilot/RTL/readonly -> nothing
        threading.Thread(target=ctrl.shutdown_rtl, args=(2.0,), daemon=True, name="shutdown-rtl").start()
        threading.Thread(target=shutdown.release_all, args=(2.5,), daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop, sig.name)
    log.info("drone-brain: vehicle=%s link=%s camera=%s vision=%s data=%s ui=http://%s:%s/",
             s.vehicle_mode, s.mavlink_url, s.camera_source, not args.no_vision, s.data_dir, s.web_host, s.web_port)
    t0 = time.time()
    try:
        # lifespan shutdown -> ctrl.detach(): vision, flight record, vehicle link
        await asyncio.gather(*(srv.serve() for srv in servers))
    finally:
        if mock is not None:
            mock.stop()
        log.info("drone-brain stopped after %.0f s", time.time() - t0)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.check:
        import shutdown
        import syscheck
        shutdown.install(exit_code=130)      # SIGINT/SIGTERM: release both cameras, exit within ~3 s
        sub = [a for a, on in (("--quick", args.quick), ("--json", args.json), ("--standalone", args.standalone)) if on]
        if args.only:
            sub += ["--only", args.only]
        if args.mavlink_url:
            sub += ["--mavlink-url", args.mavlink_url]
        if args.baud:
            sub += ["--baud", str(args.baud)]
        if args.port:
            return syscheck.cli(sub, settings=get_settings().model_copy(update={"web_port": args.port}))
        return syscheck.cli(sub)
    lvl = args.log_level or get_settings().log_level
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
