"""Controller protocol the orchestrator implements, plus a FakeController for UI dev/tests.

The web layer knows nothing about planner / MAVLink / vision. main.py provides an object
implementing `Controller`; the web layer calls it and pushes results to phones.

snapshot() contract (JSON-serialisable dict, sent as the `status` WS message):
    {
      "telemetry":     models.Telemetry.model_dump(),
      "mission_state": "idle" | "planned" | "uploading" | "flying" | "rtl" | ...,
      "target":        models.Target.model_dump() | None,
      "alerts":        [models.Alert.model_dump(), ...]   # recent, oldest first
      "plan":          models.PlanProposal.model_dump() | None   # pending plan awaiting Confirm
      "home":          {"lat": .., "lon": ..} | None   # (Phase 2b, optional) launch point for the map UI
      "route":         [[lat, lon], ...] | None         # (optional) flown/active route, for the map
      "readiness":     {ts, running, go_allowed, go_reason, items: [{id, label, status, detail, blocking}]} | None
      "staged":        {plan_id, uploaded_ts, expires_ts} | None      # notes/go-flow.md (only while "staged")
      "last_flight_summary": {text, flight_dir, source, ts, images: [{url, caption, conf, lat, lon, t}]} | None
    }
Missing keys are tolerated by the UI.

Optional controller methods (notes/go-flow.md; the server refuses politely when they are missing):
    async go(plan_id) -> str                  # GO for the staged plan (arm + start)
    async refresh_readiness() -> dict | None  # re-run the readiness checks now
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

import models

if TYPE_CHECKING:  # pragma: no cover
    from webapp.server import Hub

log = logging.getLogger(__name__)


@runtime_checkable
class Controller(Protocol):
    async def handle_chat(self, text: str) -> models.PlanProposal | str:
        """User message. Returns a PlanProposal, or a str (assistant reply / rejection)."""
        ...

    async def plan_area(self, req: models.AreaPlanRequest) -> models.PlanProposal | str:
        """Polygon survey from the map UI (Phase 2b). Build with survey.build_polygon_survey using the
        current vehicle position as home, run safety.validate, parse req.target_text into a Target
        (planner; empty -> "any person"). Return PlanProposal (even when safety is not ok, so the
        phone can show BLOCKED reasons) or a user-facing str if the polygon is unusable
        (survey.PolygonError text) or a mission is already running."""
        ...

    async def confirm(self, plan_id: str) -> str:
        """User confirmed plan `plan_id`. Returns a human-readable result string."""
        ...

    async def abort(self) -> str:
        """Abort mission and return to launch (RTL). Must be fast and never raise if avoidable."""
        ...

    async def update_target(self, text: str) -> models.Target:
        """Change the search target mid-mission (planner parses `text`)."""
        ...

    def snapshot(self) -> dict:
        """Current status (see module docstring). Must be cheap and non-blocking."""
        ...


# ---------------------------------------------------------------------------
# Fake controller
# ---------------------------------------------------------------------------

# KSU Marietta campus, "Nuesoft Technologies Field" (OSM way 180244662) soccer field centre
_HOME_LAT, _HOME_LON = 33.9371904, -84.5271293


def _offset_latlon(north_m: float, east_m: float) -> tuple[float, float]:
    lat = _HOME_LAT + north_m / 111_320.0
    lon = _HOME_LON + east_m / (111_320.0 * math.cos(math.radians(_HOME_LAT)))
    return lat, lon


def fake_plan(text: str = "", alt_m: float = 20.0) -> models.PlanProposal:
    """A small lawnmower plan: 60 m x 40 m rectangle 30 m north of home."""
    unsafe = "unsafe" in text.lower()
    if unsafe:
        alt_m = 200.0
    lanes = [-15.0, 0.0, 15.0]  # east offsets
    pts: list[tuple[float, float]] = []
    for i, e in enumerate(lanes):
        a, b = (30.0, 90.0) if i % 2 == 0 else (90.0, 30.0)
        pts += [(a, e), (b, e)]
    wps = []
    for i, (n, e) in enumerate(pts):
        lat, lon = _offset_latlon(n, e)
        wps.append(models.Waypoint(seq=i, north_m=n, east_m=e, lat=lat, lon=lon, alt_m=alt_m))
    path = 0.0
    prev = (0.0, 0.0)
    for n, e in pts + [(0.0, 0.0)]:
        path += math.hypot(n - prev[0], e - prev[1])
        prev = (n, e)
    mission = models.Mission(length_m=60, width_m=40, direction_deg=0, offset_m=30,
                             altitude_m=alt_m, speed_ms=5)
    plan = models.SurveyPlan(
        home=models.GeoPoint(lat=_HOME_LAT, lon=_HOME_LON), mission=mission, waypoints=wps,
        altitude_m=alt_m, speed_ms=5.0, v_max_ms=7.5, footprint_w_m=28.0, footprint_l_m=16.0,
        line_spacing_m=15.0, path_length_m=round(path, 1), gsd_m=0.0146, person_px=116.0,
        warnings=["Fake plan: for UI development only"],
    )
    if unsafe:
        safety = models.SafetyResult(ok=False, errors=[f"Altitude {alt_m:.0f} m exceeds max 60 m"],
                                     warnings=[])
    else:
        safety = models.SafetyResult(ok=True, warnings=["Battery margin 35% (fake)"])
    return models.PlanProposal(plan_id=uuid.uuid4().hex[:8], plan=plan,
                               target=models.Target(description="person in red jacket",
                                                    attributes=["red jacket"]),
                               safety=safety)


def _placeholder_jpeg(path: Path, label: str) -> None:
    from PIL import Image, ImageDraw  # pillow is installed on the Jetson

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (320, 240), (40, 60, 40))
    d = ImageDraw.Draw(img)
    d.rectangle([110, 50, 210, 210], outline=(255, 60, 60), width=4)
    d.ellipse([145, 70, 175, 100], fill=(230, 190, 150))
    d.rectangle([140, 100, 180, 170], fill=(200, 30, 30))
    d.text((8, 8), label, fill=(255, 255, 255))
    img.save(path, "JPEG", quality=80)


class FakeController:
    """In-memory controller for tests and `python -m webapp.server --fake`."""

    def __init__(self, data_dir: Path, alert_interval_s: float = 6.0, telemetry_interval_s: float = 1.0,
                 background: bool = True, settings: Optional[Any] = None) -> None:
        self.data_dir = Path(data_dir)
        self._settings = settings
        self._pos = (_HOME_LAT, _HOME_LON)
        self._heading = 0.0
        self._route: list[tuple[float, float]] = []
        self._route_dist = 0.0
        self._seq: Optional[int] = None
        self.alert_interval_s = alert_interval_s
        self.telemetry_interval_s = telemetry_interval_s
        self.background = background
        self.state = "idle"
        self.target: Optional[models.Target] = None
        self.pending: Optional[models.PlanProposal] = None
        self.alerts: list[dict[str, Any]] = []
        self.calls: list[tuple[str, Any]] = []   # for tests
        self.battery = 100.0
        self.alt = 0.0
        self._n_alerts = 0
        self._hub: Optional["Hub"] = None
        self._tasks: list[asyncio.Task] = []
        self.staged_plan_id: Optional[str] = None
        self.staged_ts: Optional[float] = None
        self.readiness: Optional[dict] = None
        self.last_flight_summary: Optional[dict] = None
        self._flight_alerts: list[dict] = []
        self._flight_start = 0.0
        self._aborted = False

    # --- Controller protocol ---
    async def handle_chat(self, text: str) -> models.PlanProposal | str:
        self.calls.append(("chat", text))
        await asyncio.sleep(0)
        low = text.lower()
        if self.state in ("flying", "uploading", "arming"):
            return "A mission is already in progress. Use Abort to return home first."
        if self.state == "staged":
            return "A mission is uploaded and waiting for GO. Press Abort to discard it first, then plan again."
        if "search" in low or "survey" in low or "unsafe" in low:
            self.pending = fake_plan(text)
            self.state = "planned"
            self.target = self.pending.target
            return self.pending
        return f"(fake) I can plan searches, e.g. 'search 60x40 m north of home at 20 m for a person in a red jacket'. You said: {text}"

    async def plan_area(self, req: models.AreaPlanRequest) -> models.PlanProposal | str:
        """Real survey + safety code with the fake home (the soccer field)."""
        import survey
        import safety
        from config import get_settings

        self.calls.append(("plan_area", req))
        await asyncio.sleep(0)
        if self.state in ("flying", "uploading", "arming", "rtl"):
            return "A mission is already in progress. Use Abort to return home first."
        if self.state == "staged":
            return "A mission is uploaded and waiting for GO. Press Abort to discard it first, then plan again."
        st = self._settings or get_settings()
        home = models.GeoPoint(lat=_HOME_LAT, lon=_HOME_LON)
        alt = req.altitude_m if req.altitude_m is not None else st.default_alt_m
        try:
            plan = survey.build_polygon_survey(req.corners, alt, home, st, speed_ms=req.speed_ms,
                                               overlap=req.overlap)
        except survey.PolygonError as e:
            return str(e)
        text = (req.target_text or "").strip()
        target = models.Target(description=text or "any person", attributes=[])
        res = safety.validate(plan, st)
        self.pending = models.PlanProposal(plan_id=uuid.uuid4().hex[:8], plan=plan, target=target, safety=res)
        self.state = "planned"
        self.target = target
        return self.pending

    async def confirm(self, plan_id: str) -> str:
        self.calls.append(("confirm", plan_id))
        if not self.pending or self.pending.plan_id != plan_id:
            return "Unknown or expired plan; ask again."
        if not self.pending.safety.ok:
            return "Plan failed safety checks; not flying."
        if self.state == "staged":
            return "This plan is already uploaded and waiting for GO. Press GO to start, or Abort to discard it."
        self.state = "staged"            # notes/go-flow.md: Confirm = upload + verify only
        self.staged_plan_id = plan_id
        self.staged_ts = time.time()
        self.readiness = self._fake_readiness()
        return ("Plan confirmed (fake): mission uploaded and verified, STAGED (not armed). Review it, then press GO "
                "to arm and start, or Abort to discard.")

    def _fake_readiness(self) -> dict:
        ids = [("link", "MAVLink link", True, "heartbeat 0.4 s ago"), ("disarmed", "Vehicle disarmed", True, "disarmed"),
               ("gps", "GPS", True, "3D fix, 14 satellites, HDOP 0.7"), ("ekf", "EKF", True, "EKF healthy"),
               ("prearm", "Pre-arm checks", True, "pre-arm checks pass"), ("rc", "RC receiver", True, "8 channels"),
               ("battery", "Battery", True, f"15.2 V, {self.battery:.0f}%"), ("fence", "Geofence", True, "fence enabled"),
               ("failsafe", "Failsafes", True, "RTL at 20 m, failsafes set"), ("home", "Home unchanged", True, "unchanged since upload"),
               ("mission", "Mission on vehicle = staged plan", True, "items match the staged plan"),
               ("geofence_alt", "Altitude vs ceiling", True, "plan 20 m <= ceiling 60 m"),
               ("arm_guard", "Real-flight lock", True, "simulator/mock link (fake)"),
               ("vision", "Vision pipeline", False, "vision running (fake)"), ("legal", "Legal checklist", True, "checklist acknowledged")]
        items = [{"id": i, "label": lab, "status": "ok", "detail": d, "blocking": b} for i, lab, b, d in ids]
        if self.battery < 50:
            items[6].update(status="warn", detail=f"battery {self.battery:.0f}% < 50%")
        return {"ts": time.time(), "running": False, "go_allowed": True, "go_reason": "", "items": items}

    async def refresh_readiness(self) -> Optional[dict]:
        self.calls.append(("refresh_readiness", None))
        if self.state != "staged":
            return None
        self.readiness = self._fake_readiness()
        return self.readiness

    async def go(self, plan_id: str) -> str:
        self.calls.append(("go", plan_id))
        if self.state != "staged" or not self.pending:
            return f"GO refused: no mission is staged (state {self.state}). Confirm a plan first."
        if plan_id != self.staged_plan_id or self.pending.plan_id != plan_id:
            return "GO refused: that is not the plan uploaded to the vehicle. Review the staged plan and press GO again."
        self.readiness = self._fake_readiness()
        if not self.readiness["go_allowed"]:
            return f"GO refused: {self.readiness['go_reason']}"
        self.state = "flying"
        self.staged_plan_id = self.staged_ts = None
        self.readiness = None
        self.alerts = []
        self._flight_alerts = []
        self._flight_start = time.time()
        self._aborted = False
        plan = self.pending.plan
        self._route = [(plan.home.lat, plan.home.lon)] + [(w.lat, w.lon) for w in plan.waypoints] \
            + [(plan.home.lat, plan.home.lon)]
        self._route_dist = 0.0
        self._speed = plan.speed_ms
        self._alt_target = plan.altitude_m
        self.pending = None
        return "GO (fake): armed, AUTO, mission started."

    async def abort(self) -> str:
        self.calls.append(("abort", None))
        if self.state == "staged":
            self.state = "planned"
            self.staged_plan_id = self.staged_ts = None
            self.readiness = None
            return "ABORT: the staged plan was discarded; nothing was sent to the drone (fake)."
        if self.state == "flying":
            self._aborted = True
        self.state = "rtl"
        self.pending = None
        return "ABORT: returning to launch (fake)."

    def make_summary(self) -> dict:
        """Fake post-flight summary (same shape as flight_summary.summarize)."""
        best: dict[int, dict] = {}
        for a in self._flight_alerts:
            if a["track_id"] not in best or (a.get("c_avg") or 0) > (best[a["track_id"]].get("c_avg") or 0):
                best[a["track_id"]] = a
        tracks = sorted(best.values(), key=lambda a: a.get("c_avg") or 0, reverse=True)
        imgs = []
        for a in tracks[:6]:
            t = round(a["ts"] - self._flight_start, 1) if self._flight_start else None
            cap = (f"{'Target match' if a['level'] == 'target_match' else 'Heat signature' if a['level'] == 'heat' else 'Person'} "
                   f"{(a.get('c_avg') or 0):.0%} at {a['lat']:.5f}, {a['lon']:.5f}"
                   + (f", T+{int(t) // 60}:{int(t) % 60:02d}" if t is not None else ""))
            imgs.append({"url": a["image_url"], "caption": cap, "conf": a.get("c_avg"), "lat": a["lat"], "lon": a["lon"], "t": t})
        dur = time.time() - self._flight_start if self._flight_start else 0.0
        text = (f"(fake) Search for {self.target.description if self.target else 'any person'}: flew "
                f"{int(dur) // 60}:{int(dur) % 60:02d} min. "
                + ("Aborted by the operator. " if self._aborted else "Mission completed, returned home. ")
                + (f"{len(self._flight_alerts)} alert(s) on {len(tracks)} track(s); best {imgs[0]['caption']}."
                   if imgs else "No detections."))
        return {"text": text, "flight_dir": "fake", "source": "template", "ts": time.time(), "images": imgs}

    async def update_target(self, text: str) -> models.Target:
        self.calls.append(("target", text))
        self.target = models.Target(description=text.strip() or "unknown", attributes=[])
        return self.target

    def snapshot(self) -> dict:
        tel = models.Telemetry(
            connected=True, mode={"flying": "AUTO", "rtl": "RTL"}.get(self.state, "STABILIZE"),
            armed=self.state in ("flying", "rtl"), lat=self._pos[0], lon=self._pos[1],
            alt_rel_m=round(self.alt, 1), heading_deg=round(self._heading, 1),
            groundspeed_ms=getattr(self, "_speed", 5.0) if self.state == "flying" else 0.0,
            battery_pct=round(self.battery, 1), voltage_v=15.2, gps_fix=3, satellites=14,
            mission_seq=self._seq if self.state == "flying" else None,
            mission_count=(len(self._route) - 1) if self.state == "flying" and self._route else None,
            last_heartbeat_ts=time.time(),
        )
        return {
            "telemetry": tel.model_dump(),
            "mission_state": self.state,
            "target": self.target.model_dump() if self.target else None,
            "alerts": list(self.alerts[-20:]),
            "plan": self.pending.model_dump() if self.pending else None,
            "home": {"lat": _HOME_LAT, "lon": _HOME_LON},
            "route": [list(p) for p in self._route] if self.state in ("flying", "rtl") else None,
            "readiness": self.readiness if self.state == "staged" else None,
            "staged": ({"plan_id": self.staged_plan_id, "uploaded_ts": self.staged_ts, "expires_ts": self.staged_ts + 900.0}
                       if self.state == "staged" and self.staged_ts else None),
            "last_flight_summary": self.last_flight_summary,
        }

    def _advance(self, dt: float) -> None:
        """Move the fake drone along the confirmed route (flat-earth interpolation)."""
        if len(self._route) < 2:
            return
        self._route_dist += getattr(self, "_speed", 5.0) * dt
        remaining = self._route_dist
        for i, (a, b) in enumerate(zip(self._route, self._route[1:])):
            dn = (b[0] - a[0]) * 111_320.0
            de = (b[1] - a[1]) * 111_320.0 * math.cos(math.radians(a[0]))
            seg = math.hypot(dn, de)
            if remaining <= seg or i == len(self._route) - 2:
                f = 1.0 if seg == 0 else min(1.0, remaining / seg)
                self._pos = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                if seg > 0.01:
                    self._heading = math.degrees(math.atan2(de, dn)) % 360.0
                self._seq = i + 1
                if f >= 1.0 and i == len(self._route) - 2:
                    self.state = "rtl"
                return
            remaining -= seg

    # --- fake background activity (UI dev) ---
    def attach(self, hub: "Hub") -> None:
        """Called by the app on startup: start fake telemetry/alert generators."""
        self._hub = hub
        if self.background:
            loop = asyncio.get_running_loop()
            self._tasks = [loop.create_task(self._telemetry_loop()), loop.create_task(self._alert_loop())]

    async def detach(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks = []

    def make_alert(self) -> models.Alert:
        self._n_alerts += 1
        n = self._n_alerts
        rel = f"fake/alert_{n % 5}.jpg"
        p = self.data_dir / rel
        if not p.exists():
            _placeholder_jpeg(p, f"fake alert {n % 5}")
        match = n % 3 == 0
        return models.Alert(
            level="target_match" if match else ("heat" if n % 4 == 0 else "person"), track_id=n, ts=time.time(), c_yolo=0.86,
            c_match=0.82 if match else None, c_final=0.84 if match else None, c_avg=0.85, n_frames=3,
            reason="Red jacket and backpack visible" if match else "Person standing in open field",
            image_url=f"/data/{rel}", lat=self._pos[0] + 0.00005 * (n % 3), lon=self._pos[1] + 0.00005 * (n % 4), alt_rel_m=20.0,
            target_description=self.target.description if self.target else None,
        )

    async def _telemetry_loop(self) -> None:
        while True:
            await asyncio.sleep(self.telemetry_interval_s)
            if self.state == "flying":
                self.alt = min(getattr(self, "_alt_target", 20.0), self.alt + 4.0)
                self.battery = max(0.0, self.battery - 0.2)
                self._advance(self.telemetry_interval_s)
            elif self.state == "rtl":
                self.alt = max(0.0, self.alt - 3.0)
                if self.alt == 0.0:
                    self.state = "idle"
                    self._pos = (_HOME_LAT, _HOME_LON)
                    self._route = []
                    if self._flight_start:
                        self.last_flight_summary = self.make_summary()
                        self._flight_start = 0.0
                        if self._hub:
                            await self._hub.broadcast(models.WsMessage(type="flight_summary", data=self.last_flight_summary))
            if self._hub:
                await self._hub.broadcast(models.WsMessage(type="telemetry", data=self.snapshot()["telemetry"]))

    async def _alert_loop(self) -> None:
        while True:
            await asyncio.sleep(self.alert_interval_s)
            if self.state != "flying":
                continue
            alert = self.make_alert().model_dump()
            self.alerts.append(alert)
            self._flight_alerts.append(alert)
            self.alerts = self.alerts[-50:]
            if self._hub:
                await self._hub.broadcast(models.WsMessage(type="alert", data=alert))
