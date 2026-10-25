"""FastAPI app: one WebSocket (/ws), REST fallbacks, static phone page, /data images.

Wiring from main.py (inside the running asyncio loop):

    hub = Hub()
    app = create_app(controller, settings, hub=hub)       # controller uses hub.broadcast[_threadsafe]
    server = uvicorn.Server(uvicorn.Config(app, host=settings.web_host, port=settings.web_port))
    await asyncio.gather(server.serve(), other_tasks...)

Dev: `python -m webapp.server --fake [--port 8765]`.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import contextlib
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import models
from config import HARD_MAX_ALT_M, Settings
from webapp.controller import Controller
from webapp.coords import parse_corners
from webapp.maps import register_map_routes

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_TEXT_LEN = 2000
CHAT_SID_RE = re.compile(r"[A-Za-z0-9_-]{4,64}")   # phone chat id (app.js CHAT_SID)
SEND_TIMEOUT_S = 3.0
CHAT_TIMEOUT_S = 120.0      # planner/LLM can be slow
COMMAND_TIMEOUT_S = 30.0
MAX_WS_MESSAGE_BYTES = 1_000_000   # > this is rejected (the connection survives)
WS_RATE_PER_S = 20.0               # sustained messages per second per phone (abort is exempt)
WS_RATE_BURST = 40
WS_MAX_INFLIGHT = 6                # concurrent chat/plan/target/confirm tasks per phone (abort is exempt)


class Hub:
    """Tracks connected phones and fans messages out to them."""

    def __init__(self) -> None:
        self._clients: dict[WebSocket, asyncio.Lock] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        self._clients[ws] = asyncio.Lock()

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.pop(ws, None)

    async def send(self, ws: WebSocket, msg: models.WsMessage) -> bool:
        """Send to one client; drops the client on failure. Returns success."""
        lock = self._clients.get(ws)
        if lock is None:
            return False
        try:
            async with lock:
                await asyncio.wait_for(ws.send_text(msg.model_dump_json()), SEND_TIMEOUT_S)
            return True
        except Exception as e:  # closed, slow, etc.
            log.info("dropping ws client: %s", e.__class__.__name__)
            self.disconnect(ws)
            return False

    async def broadcast(self, msg: models.WsMessage) -> None:
        """Send to every connected client concurrently; a slow client cannot block others."""
        clients = list(self._clients)
        if clients:
            await asyncio.gather(*(self.send(ws, msg) for ws in clients))

    def broadcast_threadsafe(self, msg: models.WsMessage) -> Optional[concurrent.futures.Future]:
        """Callable from any non-async thread. Returns the Future (or None if the app loop
        is not running yet, in which case the message is dropped)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            log.debug("broadcast_threadsafe: no running loop, dropped %s", msg.type)
            return None
        return asyncio.run_coroutine_threadsafe(self.broadcast(msg), loop)


class TextBody(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LEN)
    session: Optional[str] = Field(None, max_length=64)   # REST chat: optional per-phone draft id


class CoordsBody(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ClientLogBody(BaseModel):
    """Browser-side error report (window.onerror, unhandled rejections, WebGL context loss, map failures)."""
    level: str = Field("error", max_length=10)
    message: str = Field(max_length=1000)
    source: str = Field("", max_length=300)       # file:line:col
    stack: str = Field("", max_length=3000)
    page: str = Field("", max_length=300)
    ua: str = Field("", max_length=300)


class RateLimit:
    """Sliding window: at most `n` events per `per_s` seconds (client_log must not become a log-flood vector)."""

    def __init__(self, n: int = 30, per_s: float = 60.0) -> None:
        self.n, self.per_s, self._t = n, per_s, []

    def allow(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        self._t = [t for t in self._t if now - t < self.per_s]
        if len(self._t) >= self.n:
            return False
        self._t.append(now)
        return True


class TokenBucket:
    """Per-connection message limiter: `rate` tokens/s, up to `burst`."""

    def __init__(self, rate: float = WS_RATE_PER_S, burst: int = WS_RATE_BURST) -> None:
        self.rate, self.burst, self.tokens, self.t = rate, float(burst), float(burst), time.monotonic()

    def take(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class ConfirmBody(BaseModel):
    plan_id: str = Field(min_length=1, max_length=100)


def _chat(role: str, text: str) -> models.WsMessage:
    return models.WsMessage(type="chat", data={"role": role, "text": text, "ts": time.time()})


def _error(text: str) -> models.WsMessage:
    return models.WsMessage(type="error", data={"message": text, "ts": time.time()})


class _RevalidatedStatic(StaticFiles):
    """Phones must not keep an old app.css/app.js next to a new index.html (an unstyled 512 px logo, stale UI):
    revalidate every load (cheap on the hotspot: unchanged files answer 304)."""

    async def get_response(self, path: str, scope) -> Any:
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


def create_app(controller: Controller, settings: Settings, hub: Optional[Hub] = None) -> FastAPI:
    hub = hub or Hub()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.bind_loop(asyncio.get_running_loop())
        attach = getattr(controller, "attach", None)   # optional hook (FakeController)
        if callable(attach):
            attach(hub)
        try:
            yield
        finally:
            detach = getattr(controller, "detach", None)
            if callable(detach):
                await detach()

    app = FastAPI(title="drone-brain", lifespan=lifespan)
    app.state.hub = hub

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default echoes the offending input, which breaks JSON encoding for NaN/Infinity (-> 500).
        detail = [{"loc": [str(x) for x in e.get("loc", ())], "msg": str(e.get("msg", "invalid")), "type": e.get("type", "")}
                  for e in exc.errors()]
        return JSONResponse({"detail": detail}, status_code=422)
    app.state.controller = controller

    if getattr(settings, "https", False):
        from fastapi.responses import RedirectResponse

        ca = Path(settings.data_dir) / "tls" / "kestrel-ca.crt"

        @app.get("/kestrel-ca.crt", include_in_schema=False)
        def ca_cert() -> FileResponse:
            # the phone installs this once; afterwards it trusts https://<jetson>:<https_port>/ without warnings
            if not ca.is_file():
                raise HTTPException(status_code=404, detail="no CA (scripts/make_cert.sh)")
            return FileResponse(ca, media_type="application/x-x509-ca-cert", filename="kestrel-ca.crt")

        @app.middleware("http")
        async def _to_https(request: Request, call_next):
            # plain HTTP only redirects to HTTPS (and serves the CA certificate); registered before the captive-portal
            # middleware, so a phone's connectivity probe goes captive -> http://<ip>/ -> https://<ip>:<port>/
            if request.url.scheme != "http" or request.url.path == "/kestrel-ca.crt":
                return await call_next(request)
            if request.method not in ("GET", "HEAD"):       # commands only over HTTPS
                return JSONResponse({"detail": f"use https on port {settings.https_port}"}, status_code=403)
            url = request.url.replace(scheme="https", hostname=request.url.hostname or "10.42.0.1", port=settings.https_port)
            return RedirectResponse(str(url), status_code=302)

    if getattr(settings, "captive_portal", False):
        import ipaddress
        from fastapi.responses import RedirectResponse

        @app.middleware("http")
        async def _captive_portal(request: Request, call_next):
            # On the hotspot every DNS name resolves to the Jetson (dnsmasq) and port 80 is redirected here, so a
            # phone's "is there internet?" probe (connectivitycheck.gstatic.com/generate_204, captive.apple.com/...)
            # arrives with a foreign Host: answer with the app, and the phone shows its "Sign in to network" page.
            host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
            try:
                ipaddress.ip_address(host)
                own = True
            except ValueError:
                own = host in ("", "localhost", "testserver") or host.endswith(".local")
            if own or request.method not in ("GET", "HEAD"):
                return await call_next(request)
            local = request.scope.get("server") or ("10.42.0.1", settings.web_port)
            # plain port 80 (redirected to the app on the hotspot): some phones only open a sign-in page for that form
            return RedirectResponse(f"http://{local[0]}/", status_code=302)

    boot_id = uuid.uuid4().hex[:12]     # new on every start: the phone drops its chat thread from an earlier run

    def status_msg() -> models.WsMessage:
        return models.WsMessage(type="status", data={**controller.snapshot(), "boot_id": boot_id})

    async def guarded(coro: Awaitable[Any], timeout: float) -> Any:
        return await asyncio.wait_for(coro, timeout)

    # ---- command implementations: broadcast results to all phones, return primary message ----
    async def do_chat(text: str, session: Optional[str] = None) -> models.WsMessage:
        await hub.broadcast(_chat("user", text))
        if session:       # each phone gets its own mission draft (set before the controller task is created)
            try:
                import assistant
                assistant.CHAT_SESSION.set(session)
            except Exception:  # noqa: BLE001
                pass
        try:
            res = await guarded(controller.handle_chat(text), CHAT_TIMEOUT_S)
        except Exception as e:
            log.exception("handle_chat failed")
            msg = _error(f"Chat failed: {e.__class__.__name__}: {e}")
            await hub.broadcast(msg)
            return msg
        if isinstance(res, models.PlanProposal):
            msg = models.WsMessage(type="plan", data=res.model_dump())
        else:
            msg = _chat("assistant", str(res))
        await hub.broadcast(msg)
        await hub.broadcast(status_msg())
        return msg

    async def do_plan_area(req: models.AreaPlanRequest) -> models.WsMessage:
        """Polygon survey from the map UI. Result (plan / rejection text) goes to all phones."""
        try:
            res = await guarded(controller.plan_area(req), CHAT_TIMEOUT_S)
        except Exception as e:
            log.exception("plan_area failed")
            msg = _error(f"Planning failed: {e.__class__.__name__}: {e}")
            await hub.broadcast(msg)
            return msg
        if isinstance(res, models.PlanProposal):
            msg = models.WsMessage(type="plan", data=res.model_dump())
        else:
            # rejection / explanation: sent as an error so the UI keeps the user on the Area step
            msg = _error(str(res))
        await hub.broadcast(msg)
        await hub.broadcast(status_msg())
        return msg

    async def do_confirm(plan_id: str) -> models.WsMessage:
        try:
            res = await guarded(controller.confirm(plan_id), COMMAND_TIMEOUT_S)
            msg = _chat("system", str(res))
        except Exception as e:
            log.exception("confirm failed")
            msg = _error(f"Confirm failed: {e.__class__.__name__}: {e}")
        await hub.broadcast(msg)
        await hub.broadcast(status_msg())
        return msg

    async def do_go(plan_id: str) -> models.WsMessage:
        """GO (notes/go-flow.md): arm + start the staged plan. Controllers without go() refuse."""
        fn = getattr(controller, "go", None)
        try:
            if not callable(fn):
                raise RuntimeError("this controller has no GO (upload/readonly build)")
            # deep readiness run + arm/start. Shielded: a slow reply must never cancel GO half way (state "arming")
            res = await guarded(asyncio.shield(fn(plan_id)), COMMAND_TIMEOUT_S + 30.0)
            msg = _chat("system", str(res))
        except Exception as e:
            log.exception("go failed")
            msg = _error(f"GO failed: {e.__class__.__name__}: {e}")
        await hub.broadcast(msg)
        await hub.broadcast(status_msg())
        return msg

    async def do_abort() -> models.WsMessage:
        try:
            res = await guarded(controller.abort(), COMMAND_TIMEOUT_S)
            msg = _chat("system", str(res))
        except Exception as e:
            log.exception("abort failed")
            msg = _error(f"ABORT FAILED: {e.__class__.__name__}: {e}. Use the RC transmitter!")
        await hub.broadcast(msg)
        await hub.broadcast(status_msg())
        return msg

    async def do_target(text: str) -> models.WsMessage:
        try:
            tgt = await guarded(controller.update_target(text), CHAT_TIMEOUT_S)
            msg = _chat("system", f"Target updated: {tgt.description}")
        except Exception as e:
            log.exception("update_target failed")
            msg = _error(f"Target update failed: {e.__class__.__name__}: {e}")
        await hub.broadcast(msg)
        await hub.broadcast(status_msg())
        return msg

    # ---- WebSocket ----
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        if getattr(settings, "https", False) and ws.url.scheme == "ws":
            await ws.close(code=1008)                       # HTTPS mode: commands only over wss
            return
        await hub.connect(ws)
        tasks: set[asyncio.Task] = set()
        bucket = TokenBucket()
        session_id = "ws-" + uuid.uuid4().hex[:12]
        last_limit_note = 0.0

        def spawn(coro: Awaitable[Any]) -> None:
            t = asyncio.ensure_future(coro)
            tasks.add(t)
            t.add_done_callback(tasks.discard)

        try:
            await hub.send(ws, status_msg())
            while True:
                frame = await ws.receive()
                if frame.get("type") == "websocket.disconnect":
                    break
                raw = frame.get("text")
                try:
                    if raw is None:
                        raise ValueError("binary frames are not supported; send JSON text")
                    if len(raw) > MAX_WS_MESSAGE_BYTES:
                        raise ValueError(f"message too large (limit {MAX_WS_MESSAGE_BYTES // 1000} kB)")
                    cmd = json.loads(raw)
                    if not isinstance(cmd, dict):
                        raise ValueError("expected JSON object")
                    kind = cmd.get("type")
                    if kind != "abort" and not bucket.take():   # Abort is never rate-limited
                        now = time.monotonic()
                        if now - last_limit_note > 1.0:
                            last_limit_note = now
                            raise ValueError("too many messages; slow down")
                        continue
                    if kind != "abort" and len(tasks) >= WS_MAX_INFLIGHT:
                        raise ValueError("too many requests in progress; wait for them to finish")
                    if kind in ("chat", "target"):
                        text = cmd.get("text")
                        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_LEN:
                            raise ValueError(f"'text' must be a non-empty string up to {MAX_TEXT_LEN} chars")
                        # the phone's own chat id keeps its draft across reconnects (screen lock, WiFi blip, reload);
                        # a new id per connection made the assistant "forget" the area/target (2026-10-21)
                        sid = cmd.get("session")
                        sid = sid if isinstance(sid, str) and CHAT_SID_RE.fullmatch(sid) else session_id
                        spawn(do_chat(text.strip(), sid) if kind == "chat" else do_target(text.strip()))
                    elif kind == "plan_area":
                        try:
                            req = models.AreaPlanRequest.model_validate(
                                {k: v for k, v in cmd.items() if k != "type"})
                        except Exception as e:  # pydantic ValidationError
                            raise ValueError(f"invalid plan_area request: {e}") from e
                        spawn(do_plan_area(req))
                    elif kind == "confirm":
                        pid = cmd.get("plan_id")
                        if not isinstance(pid, str) or not pid:
                            raise ValueError("'plan_id' required")
                        spawn(do_confirm(pid))
                    elif kind == "go":
                        pid = cmd.get("plan_id")
                        if not isinstance(pid, str) or not pid or len(pid) > 100:
                            raise ValueError("'plan_id' required")
                        spawn(do_go(pid))
                    elif kind == "abort":
                        spawn(do_abort())
                    elif kind == "ping":   # keepalive; reply is lightweight
                        await hub.send(ws, models.WsMessage(type="telemetry", data=controller.snapshot().get("telemetry", {})))
                    else:
                        raise ValueError(f"unknown message type: {kind!r}")
                except ValueError as e:   # includes JSONDecodeError
                    await hub.send(ws, _error(f"Bad message: {e}"))
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("ws error")
        finally:
            hub.disconnect(ws)
            # in-flight commands (esp. abort) must still complete; do not cancel them.

    # ---- REST fallbacks ----
    @app.post("/api/chat")
    async def api_chat(body: TextBody) -> dict:
        sid = body.session if body.session and CHAT_SID_RE.fullmatch(body.session) else None
        return (await do_chat(body.text.strip(), sid)).model_dump()

    @app.post("/api/confirm")
    async def api_confirm(body: ConfirmBody) -> dict:
        return (await do_confirm(body.plan_id)).model_dump()

    def _reply(msg: models.WsMessage) -> dict:
        d = msg.model_dump()
        d["message"] = msg.data.get("text") or msg.data.get("message") or ""
        return d

    @app.post("/api/go")
    async def api_go(body: ConfirmBody) -> dict:
        """{"plan_id"} -> the chat/error message (as /api/confirm) + "message": its text."""
        return _reply(await do_go(body.plan_id))

    @app.post("/api/readiness/refresh")
    async def api_readiness_refresh() -> dict:
        fn = getattr(controller, "refresh_readiness", None)
        rd = await guarded(fn(), COMMAND_TIMEOUT_S + 30.0) if callable(fn) else None
        await hub.broadcast(status_msg())
        return {"readiness": rd}

    @app.get("/api/flights/last/summary")
    async def api_last_summary() -> Any:
        """The newest flight summary ({text, flight_dir, source, ts, images}) or null. Persistent across restarts."""
        return controller.snapshot().get("last_flight_summary")

    @app.post("/api/plan_area")
    async def api_plan_area(body: models.AreaPlanRequest) -> dict:
        return (await do_plan_area(body)).model_dump()

    @app.post("/api/parse_coords")
    async def api_parse_coords(body: CoordsBody) -> dict:
        """Parse pasted 'lat, lon' lines into corners (plain Python parser)."""
        home = None
        try:
            h = controller.snapshot().get("home")
            if isinstance(h, dict) and h.get("lat") is not None and h.get("lon") is not None:
                home = (float(h["lat"]), float(h["lon"]))
        except Exception:  # noqa: BLE001  (a broken snapshot must not break coordinate parsing)
            home = None
        try:
            pts = parse_corners(body.text, home=home)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"corners": [p.model_dump() for p in pts]}

    @app.get("/api/limits")
    async def api_limits() -> dict:
        """Numbers the planning UI needs for live footprint / GSD / person-size feedback."""
        import safety  # Phase 7: effective ceiling = min(MAX_ALT_M, LAANC_CEILING_M, 120), floor >= OBSTACLE_CLEARANCE_M
        lim = safety.limits_info(settings)
        return {
            **lim,
            "min_alt_m": lim["effective_min_alt_m"], "max_alt_m": min(lim["effective_max_alt_m"], HARD_MAX_ALT_M),
            "default_alt_m": settings.default_alt_m, "default_speed_ms": settings.default_speed_ms,
            "max_speed_ms": settings.max_speed_ms, "overlap": settings.survey_overlap,
            "hfov_deg": settings.eff_hfov_deg, "vfov_deg": settings.eff_vfov_deg,   # of the corrected (rotated) image
            "sensor_width_mm": settings.eff_sensor_width_mm, "focal_length_mm": settings.focal_length_mm,
            "image_width_px": settings.eff_image_width_px, "person_height_m": settings.person_height_m,
            "min_person_px": settings.min_person_px, "geofence_radius_m": settings.geofence_radius_m,
            "max_area_m2": settings.max_area_m2, "min_area_m2": settings.min_area_m2,
            "max_path_m": settings.max_path_m, "frame_interval_s": settings.frame_interval_s,
        }

    @app.post("/api/abort")
    async def api_abort() -> dict:
        return (await do_abort()).model_dump()

    @app.post("/api/target")
    async def api_target(body: TextBody) -> dict:
        return (await do_target(body.text.strip())).model_dump()

    @app.get("/api/status")
    async def api_status() -> dict:
        return status_msg().model_dump()

    client_log_limit = RateLimit(30, 60.0)
    app.state.client_log_limit = client_log_limit

    @app.post("/api/client_log")
    async def api_client_log(body: ClientLogBody, request: Request) -> JSONResponse:
        """Browser errors end up in the server log (rate-limited: 30 per minute)."""
        if not client_log_limit.allow():
            return JSONResponse({"ok": False, "dropped": "rate limited"}, status_code=429)
        who = request.client.host if request.client else "?"
        lvl = {"info": logging.INFO, "warn": logging.WARNING}.get(body.level, logging.WARNING)
        log.log(lvl, "client[%s] %s: %s%s%s | page=%s ua=%s", who, body.level, body.message.replace("\n", " ")[:500],
                    f" @ {body.source}" if body.source else "", f" | {body.stack.replace(chr(10), ' / ')[:600]}" if body.stack else "",
                    body.page, body.ua[:80])
        return JSONResponse({"ok": True})

    # ---- static + data ----
    @app.get("/data/{path:path}")
    async def data_file(path: str) -> FileResponse:
        root = Path(settings.data_dir).resolve()
        try:
            target = (root / path).resolve()
            target.relative_to(root)
        except (ValueError, OSError):
            raise HTTPException(status_code=404, detail="not found")
        if not target.is_file() or target.suffix.lower() not in IMAGE_SUFFIXES \
                or any(part.startswith(".") for part in target.relative_to(root).parts):
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(target, headers={"Cache-Control": "public, max-age=3600"})

    register_map_routes(app, settings)
    app.mount("/static", _RevalidatedStatic(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def main(argv: Optional[list[str]] = None) -> None:
    import uvicorn
    from config import get_settings
    from webapp.controller import FakeController

    p = argparse.ArgumentParser(description="drone-brain web app (dev server)")
    p.add_argument("--fake", action="store_true", help="use FakeController (UI development)")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    args = p.parse_args(argv)
    if not args.fake:
        p.error("only --fake is supported standalone; the real controller is wired by main.py")

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(FakeController(settings.data_dir), settings)
    uvicorn.run(app, host=args.host or "127.0.0.1", port=args.port or settings.web_port, log_level="info")


if __name__ == "__main__":
    main()
