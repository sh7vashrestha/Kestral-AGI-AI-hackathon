"""RealController: the orchestrator behind the phone UI (Phase 7). Implements webapp.controller.Controller.

    ctrl = RealController(settings, vehicle, llm=LLMClient(settings=settings), pipeline=VisionPipeline(...))
    app = create_app(ctrl, settings, hub)     # lifespan calls ctrl.attach(hub) / ctrl.detach()

Flow
----
chat / plan_area -> planner (LLM) -> survey with the Pixhawk HOME_POSITION as home -> safety.validate
(+ the uploaded item list) -> mission.preflight_check (read-only param reads) -> PlanProposal
(always returned when the geometry is buildable, BLOCKED reasons inside) -> files in data/plans/<plan_id>/.

confirm(plan_id): latest plan only, safety ok, legal checklist acknowledged, plan home == current Pixhawk
home (else the plan is recalculated), preflight re-run. Then per VEHICLE_MODE:
    sim / full  -> upload + read-back verify only -> state "staged" (notes/go-flow.md); a readiness checker runs
                   while staged; a separate go(plan_id) re-runs it (incl. mission re-download) and then
                   FlightStorage + vision recording -> vehicle.start_staged (arm guard still applies)
    upload      -> upload + read-back verify only (refused while armed); no mode change / arm / start
    readonly    -> refused: "read-only mode: plan saved, not uploaded"

Home tracking: HOME_POSITION is polled every telemetry tick (2 Hz). A change is pushed to the phones
(status), logged to telemetry.jsonl during a flight, alerted mid-flight, and a pending plan whose home moved
more than HOME_SHIFT_REPLAN_M (or HOME_ALT_SHIFT_M vertically) is rebuilt from the same request.

State machine: idle -> planned -> uploading -> staged -> (GO) arming -> flying -> rtl -> landed (-> planned on the
next plan). Abort while staged -> planned (nothing sent); while arming -> cancel (RTL if already armed).
Link loss while arming/flying -> RTL (sim/full) + alert; RTL is re-sent when the link returns.
A critical background task (telemetry loop, vision worker) failing while arming/flying -> RTL.
In flight (arming/flying/rtl) informational system chat lines are not broadcast (the phone is usually out of WiFi
range): they go to the flight record. After landing flight_summary writes summary.md and it is pushed once and kept
in the snapshot (last_flight_summary).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
import uuid
from collections import deque
from typing import Any, Awaitable, Callable, Optional

import flight_summary
import mission_diff
import models
import planner
import safety
import survey
from config import Settings, bench_home_for
from llm_client import LLMError
from mission import (ArmRefused, HomeFix, ReadOnlyViolation, VehicleBackend, VehicleError, arm_guard,
                     downgrade_preflight_for_bench, home_shift_m, preflight_check)
from models import AreaPlanRequest, PlanProposal, PreflightResult, SafetyResult, Target, WsMessage
from plan_files import to_geojson, to_wpl
from storage import FlightStorage, PlanStore

log = logging.getLogger("controller")

BUSY_STATES = ("uploading", "arming", "flying", "rtl", "manual")   # manual: the pilot changed the mode; not our mission
ACTIVE_STATES = ("arming", "flying")         # our flight is (about to be) in the air: link loss / task crash -> RTL
QUIET_STATES = ("arming", "flying", "rtl")   # informational system chat goes to the flight record, not the phones
RTL_MODES = ("RTL", "SMART_RTL", "AUTO_RTL", "LAND")
TELEMETRY_HZ = 2.0
ARMING_HOME_WINDOW_S = 20.0      # home change this soon after confirm = ArduPilot's reset at arming

BENCH_CHIP = "BENCH HOME \u2014 not from GPS (upload test only)"
BENCH_UPLOAD_NOTE = ("ArduPilot replaces mission item 0 with its real home when it gets a GPS fix / arms. The survey "
                     "waypoints are absolute lat/lon (they stay at the field) while their altitudes are relative to "
                     "wherever the vehicle arms.")
WP_HINT = "Mission Planner -> Plan -> Load WP File"

STAGED_TEXT = ("A mission is uploaded and waiting for GO. Press Abort to discard it first (nothing is sent to the "
               "drone), then plan again.")

# Removed at the operator's request (2026-10-21). Add items back here to bring the "Before the first flight"
# dialog and its Confirm gate back.
LEGAL_CHECKLIST: list[str] = []


def _chat(role: str, text: str) -> WsMessage:
    return WsMessage(type="chat", data={"role": role, "text": text, "ts": time.time()})


def _err(text: str) -> WsMessage:
    return WsMessage(type="error", data={"message": text, "ts": time.time()})


class RealController:
    def __init__(self, settings: Settings, vehicle: VehicleBackend, *, llm: Any = None, pipeline: Any = None,
                 manage_connection: bool = True, connect_timeout_s: float = 10.0,
                 param_timeout_s: float = 3.0) -> None:
        from pipeline import NullPipeline

        self.s = settings
        self.vehicle = vehicle
        self.mode = settings.vehicle_mode
        if llm is None:
            from llm_client import LLMClient
            llm = LLMClient(settings=settings)
        self.llm = llm
        self.pipeline = pipeline if pipeline is not None else NullPipeline()
        self.plan_store = PlanStore(settings)
        self.manage_connection = manage_connection
        self.connect_timeout_s = connect_timeout_s
        self.param_timeout_s = param_timeout_s

        self.state = "idle"
        self.pending: Optional[PlanProposal] = None
        self._pending_src: Optional[dict[str, Any]] = None
        self.target: Optional[Target] = None
        self.alerts: deque[dict] = deque(maxlen=50)
        self.legal_ack_ts: Optional[float] = None
        self.flight: Optional[FlightStorage] = None
        self.last_flight_dir: Optional[str] = None
        self._flight_meta: dict[str, Any] = {}
        self._route: Optional[list[list[float]]] = None
        self._home_seen: Optional[HomeFix] = None
        self._tel = models.Telemetry()
        self._hub: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._tasks: list[asyncio.Task] = []
        self._plan_lock = asyncio.Lock()
        self._abort_requested = False
        self._rtl_on_restore = False
        self._vehicle_connected = not manage_connection
        self._saw_armed = False
        self._saw_auto = False                 # the vehicle was seen flying our mission in AUTO
        self._confirming = False               # one Confirm at a time (double tap / two phones)
        self._manual_alerted = False
        self._gps_alerted = False
        self._last_mode: Optional[str] = None
        self._st_alert_ts = time.time()
        self._shutdown_rtl_done = False
        self._own_rtl_ts = 0.0
        self._tick = 0
        self._last_pos: Optional[tuple[float, float]] = None
        self.task_failures: dict[str, int] = {}
        self.events: deque[tuple[float, str, dict]] = deque(maxlen=200)   # for tests / debugging
        self.last_upload: Optional[dict] = None
        self.last_vehicle_check: Optional[dict] = None
        self._last_st_ts = 0.0
        self.vehicle_hint: Optional[str] = None
        # --bench-home (upload mode ONLY; raises BenchHomeError in any other mode). Never used by arming paths.
        self.bench = bench_home_for(settings)
        self._bench_ignored_noted = False
        self.readback: list[dict] = []                  # what the Pixhawk holds (last upload / last "Read mission")
        self.readback_info: Optional[dict] = None       # {plan_id, ts, source, diff, files}
        self._uploaded: Optional[tuple[str, list[models.MissionItem], Optional[float]]] = None
        # Upload -> review -> GO (notes/go-flow.md)
        self.staged: Optional[dict[str, Any]] = None    # {plan_id, items, uploaded_ts, home_version}
        self.readiness: Optional[dict] = None           # snapshot["readiness"] (None unless staged)
        self._readiness_lock = asyncio.Lock()
        self._readiness_running = False
        self._ready_pre: Optional[PreflightResult] = None          # param cross-check of the last deep run
        self._mission_check: Optional[tuple[str, str, float]] = None   # (status, detail, ts) of the last re-download
        self._last_deep_ts = 0.0
        self._going = False                    # one GO at a time
        self._detaching = False
        self._summary_tasks: set[asyncio.Task] = set()
        self.last_flight_summary: Optional[dict] = flight_summary.load_last(settings.data_dir)

        vehicle.on_link_lost = self._cb_link_lost
        vehicle.on_link_restored = self._cb_link_restored

    # ================================================================== lifecycle
    def attach(self, hub: Any) -> None:
        """Called by create_app's lifespan inside the running loop."""
        self._hub = hub
        self._loop = asyncio.get_running_loop()
        self._plan_lock = asyncio.Lock()
        self._readiness_lock = asyncio.Lock()
        self.pipeline.start()
        self._spawn("vehicle-link", self._vehicle_link_loop, critical=False)
        self._spawn("telemetry", self._telemetry_loop, critical=True)
        self._spawn("vision-watchdog", self._vision_watchdog, critical=False)
        self._spawn("readiness", self._readiness_loop, critical=False)

    async def detach(self) -> None:
        await asyncio.to_thread(self.shutdown_rtl)
        self._detaching = True
        for t in list(self._summary_tasks):
            t.cancel()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks = []
        await asyncio.to_thread(self.pipeline.stop)
        self._finish_flight("shutdown")
        if self.manage_connection:
            await asyncio.to_thread(self.vehicle.close)
        log.info("controller stopped")

    def _spawn(self, name: str, fn: Callable[[], Awaitable[None]], *, critical: bool) -> None:
        self._tasks.append(asyncio.get_running_loop().create_task(self._supervise(name, fn, critical), name=name))

    async def _supervise(self, name: str, fn: Callable[[], Awaitable[None]], critical: bool) -> None:
        """Watchdog: restart a failed task with exponential backoff; RTL if a critical one fails mid-flight."""
        backoff = 1.0
        while True:
            t0 = time.monotonic()
            try:
                await fn()
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.task_failures[name] = self.task_failures.get(name, 0) + 1
                log.exception("task %s failed (#%d)", name, self.task_failures[name])
                self._event("task_failed", task=name, error=repr(exc))
                if critical and self.state in ACTIVE_STATES:
                    await self._emergency_rtl(f"internal task '{name}' failed: {exc.__class__.__name__}")
                if time.monotonic() - t0 > 60:
                    backoff = 1.0
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    # ================================================================== helpers
    def _event(self, kind: str, **data: Any) -> None:
        self.events.append((time.time(), kind, data))

    async def _broadcast(self, msg: WsMessage) -> None:
        if self._hub is not None:
            try:
                await self._hub.broadcast(msg)
            except Exception:  # noqa: BLE001
                log.exception("broadcast failed")

    async def _broadcast_status(self) -> None:
        await self._broadcast(WsMessage(type="status", data=self.snapshot()))

    async def _say(self, text: str) -> None:
        """Informational system chat line. In flight (QUIET_STATES) it is not broadcast (the user: "no need to
        broadcast the text"; the phone is usually out of WiFi range): it goes to the flight record instead.
        Safety errors (_err), command replies and target alerts are NOT routed through here."""
        if self.state in QUIET_STATES:
            self._event("note", text=text)
            if self.flight is not None:
                self.flight.append("telemetry", {"event": "note", "text": text, "ts": time.time()})
            return
        await self._broadcast(_chat("system", text))

    def _set_state(self, st: str) -> None:
        if st != self.state:
            log.info("mission state %s -> %s", self.state, st)
            self._event("state", old=self.state, new=st)
            self.state = st
            if self.flight is not None:
                self.flight.append("telemetry", {"event": "state", "state": st, "ts": time.time()})

    async def _vthread(self, fn: Callable, *a: Any, timeout: Optional[float] = None) -> Any:
        coro = asyncio.to_thread(fn, *a)
        return await (asyncio.wait_for(coro, timeout) if timeout else coro)

    def _home_tol_exceeded(self, a_lat: float, a_lon: float, a_amsl: Optional[float], hf: HomeFix) -> tuple[bool, float, float]:
        shift = home_shift_m(models.GeoPoint(lat=a_lat, lon=a_lon), hf)
        dalt = 0.0 if a_amsl is None else hf.amsl_m - a_amsl
        return (shift > self.s.home_shift_replan_m or abs(dalt) > self.s.home_alt_shift_m), shift, dalt

    def _effective_home(self) -> tuple[Optional[HomeFix], str]:
        """(home, source). The Pixhawk's HOME_POSITION ALWAYS wins; the bench home is only a stand-in for an
        upload test (upload mode) while the Pixhawk has no home."""
        real = self.vehicle.home_position()
        if real is not None:
            if self.bench is not None and not self._bench_ignored_noted:
                self._bench_ignored_noted = True
                log.warning("BENCH_HOME ignored: the Pixhawk reports a real home (%.7f, %.7f)", real.lat, real.lon)
            return real, "pixhawk"
        if self.bench is not None and self.mode == "upload":
            lat, lon, amsl = self.bench
            return HomeFix(lat=lat, lon=lon, amsl_m=amsl, ts=time.time(), version=0), "bench"
        return None, "pixhawk"

    def _bench_snapshot(self) -> Optional[dict]:
        if self.bench is None:
            return None
        active = self.vehicle.home_position() is None
        return {"lat": self.bench[0], "lon": self.bench[1], "amsl_m": self.bench[2], "active": active, "text": BENCH_CHIP,
                "note": None if active else "Bench home ignored: the Pixhawk reports its real home (GPS)."}

    # ================================================================== Controller protocol
    async def handle_chat(self, text: str) -> PlanProposal | str:
        """Chat -> assistant.Assistant (Hardening A): per-session draft, one structured LLM action per turn, Python
        decides numbers/limits. Plans go through the normal _propose path; the summary is broadcast as a chat line."""
        import assistant as _assistant
        busy = self.state in BUSY_STATES
        asst = self._chat_assistant()
        rep = await asst.handle(text, session=_assistant.current_session(), busy=busy,
                                pending_plan=self.pending.plan if self.pending is not None else None,
                                map_req=self._chat_map_req())
        self._event("chat", action=rep.action, llm_action=rep.llm_action, llm_ok=rep.llm_ok)
        if rep.action == "update_target" and rep.target is not None:
            if not busy:
                return rep.text
            await self._chat_set_target(rep.target)
            return f"Target updated: {rep.target.description}"
        if rep.action != "plan" or rep.src is None:
            return rep.text
        if self.state in BUSY_STATES:
            return _assistant.BUSY_TEXT
        if self.state == "staged":
            return STAGED_TEXT
        res = await self._propose(rep.src)
        if not isinstance(res, PlanProposal):
            return f"{rep.text} But I can't build it yet: {res}"
        await self._broadcast(_chat("assistant", self._chat_summary(rep.text, res)))
        return res

    async def plan_area(self, req: AreaPlanRequest) -> PlanProposal | str:
        if self.state in BUSY_STATES:
            return "A mission is already in progress. Use Abort to return home first."
        if self.state == "staged":
            return STAGED_TEXT
        text = (req.target_text or "").strip()
        target = Target(description="any person")
        if text:
            target = await self._parse_target(text)
        return await self._propose({"kind": "area", "req": req, "target": target})

    async def confirm(self, plan_id: str) -> str:
        if self._confirming:
            return "Another Confirm is already being processed; wait for it to finish."
        self._confirming = True
        try:
            return await self._confirm_locked(plan_id)
        finally:
            self._confirming = False

    async def _confirm_locked(self, plan_id: str) -> str:
        p = self.pending
        if p is None or p.plan_id != plan_id:
            return "Unknown or expired plan (it may have been recalculated). Review the latest plan and confirm again."
        if not p.safety.ok:
            return "Plan failed safety checks; not flying."
        if self.state in BUSY_STATES:
            return "A mission is already in progress."
        if self.state == "staged":
            return "This plan is already uploaded and waiting for GO. Press GO to start, or Abort to discard it."
        if LEGAL_CHECKLIST and self.legal_ack_ts is None:
            return "Please acknowledge the legal checklist before the first flight of this session."
        hf, src = self._effective_home()
        if hf is None:
            return "No Pixhawk home position: cannot verify the plan's home. Wait for a 3D GPS fix."
        moved, shift, dalt = self._home_tol_exceeded(p.plan.home.lat, p.plan.home.lon, p.home_amsl_m, hf)
        if src != p.home_source:
            asyncio.get_running_loop().create_task(self._rebuild_pending(hf, shift, dalt, force=True))
            return (f"The home source changed ({p.home_source} -> {src}) since this plan was made: not uploading. "
                    f"The plan is being recalculated; review it and confirm again.")
        if moved:
            asyncio.get_running_loop().create_task(self._rebuild_pending(hf, shift, dalt))
            return (f"Home moved {shift:.0f} m (altitude {dalt:+.1f} m) since this plan was made: not uploading. "
                    f"The plan is being recalculated; review it and confirm again.")
        tel = self.vehicle.telemetry()
        if not tel.connected:
            return "No MAVLink link to the Pixhawk; not flying."
        if tel.armed:
            if self.mode == "upload":
                return "Upload refused: the vehicle is ARMED (a mission must not change under an armed vehicle)."
            return "The vehicle is already ARMED; land and disarm before starting a new mission."
        if src == "pixhawk" and tel.lat is not None and tel.lon is not None:
            d = home_shift_m(models.GeoPoint(lat=tel.lat, lon=tel.lon), hf)
            if d > self.s.home_shift_replan_m:
                return (f"The drone is {d:.0f} m from its Pixhawk home. ArduPilot resets home to the arming spot, "
                        f"so this plan (geofence, RTL point, relative altitudes) would be wrong. Carry the drone "
                        f"back to home, or re-plan once the Pixhawk home matches its position.")
        pre = await self._vthread(lambda: preflight_check(self.vehicle, self.s, p.plan, timeout=self.param_timeout_s))
        if src == "bench":
            pre = downgrade_preflight_for_bench(pre)      # upload test on the bench: shown as warnings, never blocks
        if not pre.ok:
            return "Pre-flight check failed: " + "; ".join(pre.errors)

        if self.mode == "readonly":
            self._event("confirm_refused_readonly", plan_id=plan_id)
            return (f"Read-only mode: plan saved, not uploaded (data/plans/{plan_id}/mission.waypoints). "
                    f"Restart with --vehicle upload to send it to the Pixhawk for review, or sim/full to fly it.")
        if self.mode == "upload":
            return await self._upload_for_review(p)

        # ---- stage (sim / full): upload + read-back verify only; GO arms and starts (notes/go-flow.md) ----
        return await self._stage(p)

    async def abort(self) -> str:
        if self.mode in ("readonly", "upload"):
            return (f"{'Read-only' if self.mode == 'readonly' else 'Upload-only'} mode: drone-brain cannot command "
                    f"the vehicle. Use the RC transmitter (RTL switch / mode) now!")
        tel = self.vehicle.telemetry()
        if self.state == "staged":                       # nothing is armed: discard the staged plan, send nothing
            self._unstage("operator abort")
            return ("ABORT: the staged plan was discarded; nothing was sent to the drone. The mission stays on the "
                    "Pixhawk until it is replaced or cleared. The plan is kept: Confirm to upload it again.")
        if self.state in ("uploading", "arming"):
            self._abort_requested = True
            self.vehicle.cancel_flight()                 # stop the upload / GO; never arm
        if not tel.connected:
            if self.state in ACTIVE_STATES:
                self._rtl_on_restore = True              # send RTL the moment the link returns
            return ("ABORT NOT SENT: no MAVLink link to the Pixhawk. Use the RC transmitter NOW (RTL switch). "
                    "RTL will be sent automatically if the link returns.")
        if self.state == "manual":
            return (f"The pilot has taken over (mode {tel.mode}): drone-brain does not fight the pilot and sent nothing. "
                    f"Use the RC transmitter.")
        if self.state == "uploading" and not tel.armed:
            return "ABORT: the upload was cancelled; the drone will not arm."
        if self.state == "arming" and not tel.armed:
            return "ABORT: GO was cancelled before arming; the drone will not arm."
        if not tel.armed and self.state not in ("uploading", "arming"):
            if self.state not in BUSY_STATES and self.pending is not None:
                self.pending = None
                self._pending_src = None
                self._set_state("idle")
            return "The vehicle is disarmed on the ground; nothing to abort (pending plan cleared)."
        if self.state == "rtl" and tel.armed and tel.mode in RTL_MODES:
            return f"Already returning ({tel.mode}); nothing more to send."
        if self.flight is not None:
            self._flight_meta["aborted"] = True
        ok, msg = await self._do_rtl("operator ABORT")
        return ("ABORT: return to launch commanded (RTL)." if ok
                else f"ABORT FAILED: {msg}. Use the RC transmitter!")

    async def update_target(self, text: str) -> Target:
        """Target tab / mid-flight: assistant.parse_target (typos, other languages, people only; LLM down -> the
        operator's own words). The pipeline then builds validated CLIP prompts (clip_prompts.LlmPromptBuilder)."""
        target = await self._chat_assistant().parse_target(text)
        await self._chat_set_target(target)
        return target

    # ---- chat helpers (Hardening A) ----
    def _chat_assistant(self) -> Any:
        asst = getattr(self, "_chat_asst", None)
        if asst is None:
            import assistant as _assistant
            asst = self._chat_asst = _assistant.Assistant(self.llm, self.s)
            asst.heading_fn = self._compass_heading
        return asst

    def _compass_heading(self) -> Optional[float]:
        """The drone's heading for 'in front of the drone' (None without a live link)."""
        tel = self.vehicle.telemetry()
        return tel.heading_deg if tel.connected and tel.heading_deg is not None else None

    def _chat_map_req(self) -> Optional[AreaPlanRequest]:
        src = self._pending_src
        if src and src.get("kind") == "area":
            return src.get("req")
        return None

    async def _chat_set_target(self, target: Target) -> None:
        self.target = target
        await asyncio.to_thread(self.pipeline.update_target, target)
        if self.flight is not None and getattr(self.pipeline, "matcher", None) is None:
            self.flight.record_target(target)
        self._event("target", description=target.description)

    @staticmethod
    def _chat_summary(text: str, p: PlanProposal) -> str:
        pl = p.plan
        bits = [text]
        if pl.estimated_flight_s:
            bits.append(f"{pl.estimated_flight_s / 60.0:.1f} min in the air, {pl.lane_count or 0} lanes.")
        if p.safety.ok:
            bits.append("Check the map, then press Confirm.")
        else:
            bits.append(f"BLOCKED: {p.safety.errors[0] if p.safety.errors else 'safety checks failed'}")
        return " ".join(bits)

    def snapshot(self) -> dict:
        tel = self._tel if self._loop is not None else self.vehicle.telemetry()
        hf, src = self._effective_home()
        home = None if hf is None else {"lat": hf.lat, "lon": hf.lon, "amsl_m": hf.amsl_m, "version": hf.version,
                                        "source": src}
        return {
            "telemetry": tel.model_dump(),
            "mission_state": self.state,
            "target": self.target.model_dump() if self.target else None,
            "alerts": list(self.alerts)[-20:],
            "plan": self.pending.model_dump() if self.pending else None,
            "home": home,
            "route": self._route if self.state in BUSY_STATES else None,
            "vehicle_mode": self.mode,
            "demo": bool(getattr(self.s, "demo_mode", False)),
            "legal_ack": self.legal_ack_ts is not None,
            "vision": self.pipeline.status(),
            "flight_dir": self.last_flight_dir,
            "limits": safety.limits_info(self.s),
            "statustext": self._statustexts()[-15:],
            "last_upload": ({k: v for k, v in self.last_upload.items() if k not in ("readback", "diff")}
                            if self.last_upload else None),
            "vehicle_hint": self.vehicle_hint,
            "home_source": src,
            "bench_home": self._bench_snapshot(),
            "readback": self.readback,
            "readback_info": self.readback_info,
            # notes/go-flow.md
            "readiness": ({**self.readiness, "running": self._readiness_running or self.readiness.get("running", False)}
                          if self.state == "staged" and self.readiness is not None else
                          ({"ts": None, "running": True, "go_allowed": False, "go_reason": "checks running",
                            "items": []} if self.state == "staged" else None)),
            "staged": (None if self.staged is None or self.state != "staged" else
                       {"plan_id": self.staged["plan_id"], "uploaded_ts": self.staged["uploaded_ts"],
                        "expires_ts": self.staged["uploaded_ts"] + self.s.staged_max_age_s}),
            "last_flight_summary": self.last_flight_summary,
        }

    def _statustexts(self) -> list[dict]:
        fn = getattr(self.vehicle, "statustexts", None)
        try:
            return fn() if fn else []
        except Exception:  # noqa: BLE001
            return []

    # ================================================================== upload-only mode / vehicle check
    async def _upload_core(self, p: PlanProposal, fence: Optional[list]) -> tuple[dict, dict, dict] | str:
        """upload_for_review + read-back bookkeeping (phone overlay, upload.json, readback.waypoints). Leaves the
        state at "uploading" on success; "planned" + error text on failure. Shared by upload mode and staging."""
        self._set_state("uploading")
        await self._broadcast_status()
        t0 = time.time()
        try:
            res = await self._vthread(self.vehicle.upload_for_review, p.plan, fence, timeout=120.0)
        except (VehicleError, ValueError, asyncio.TimeoutError) as exc:
            self._set_state("planned")
            self._event("upload_failed", plan_id=p.plan_id, error=str(exc))
            return f"Upload to the Pixhawk FAILED: {exc}"
        readback = res.pop("readback", [])
        try:
            planned: Optional[list[models.MissionItem]] = self.vehicle.build_mission_items(p.plan)
        except ValueError:
            planned = None
        info = self._set_readback(readback, planned, p.plan_id, "upload", p.home_amsl_m)
        rec = {"plan_id": p.plan_id, "ts": time.time(), "duration_s": round(time.time() - t0, 2),
               "url": self.s.mavlink_url, "home_source": p.home_source, **res,
               "readback": self.readback, "diff": info["diff"], "files": info["files"]}
        if p.home_source == "bench" and self.bench is not None:
            rec["bench_home"] = {"lat": self.bench[0], "lon": self.bench[1], "amsl_m": self.bench[2], "text": BENCH_CHIP}
        self.last_upload = rec
        if planned is not None:
            self._uploaded = (p.plan_id, planned, p.home_amsl_m)
        files: dict[str, Any] = {"upload.json": rec}
        if readback:
            files["readback.waypoints"] = to_wpl(readback, p.home_amsl_m)
        self.plan_store.save(p.plan_id, files)
        self._event("uploaded", **{k: v for k, v in rec.items() if k not in ("readback", "diff")})
        return res, info, rec

    async def _upload_for_review(self, p: PlanProposal) -> str:
        tel = self.vehicle.telemetry()
        if tel.armed:
            return "Upload refused: the vehicle is ARMED (a mission must not change under an armed vehicle)."
        fence = None
        if self.s.upload_fence_polygon:
            from plan_files import area_outline
            fence = area_outline(p.plan)
        out = await self._upload_core(p, fence)
        if isinstance(out, str):
            return out
        self._set_state("planned")
        res, info, _rec = out
        fence_txt = ""
        if res.get("fence"):
            f = res["fence"]
            fence_txt = f" Fence polygon: {f['vertices']} vertices, verify {'OK' if f['verified'] else 'FAILED'} (FENCE_* params untouched)."
        bench_txt = ""
        if p.home_source == "bench" and self.bench is not None:
            bench_txt = (f" {BENCH_CHIP}: home {self.bench[0]:.7f}, {self.bench[1]:.7f} ({self.bench[2]:.0f} m AMSL) was "
                         f"written to mission item 0. {BENCH_UPLOAD_NOTE}")
        files_txt = (f" Files: /api/plans/{p.plan_id}/mission.waypoints (planned) and /api/plans/{p.plan_id}/readback.waypoints "
                     f"(on the Pixhawk) - {WP_HINT}.")
        if not res["verified"]:
            return (f"Uploaded {res['items']} items but the read-back verify FAILED - do not use this mission; "
                    f"clear it and try again.{fence_txt} Planned vs uploaded: {info['diff']['summary']['text']}.{bench_txt}{files_txt}")
        return (f"Uploaded {res['items']} mission items to the Pixhawk, read-back verify OK.{fence_txt} "
                f"Planned vs uploaded: {info['diff']['summary']['text']}.{bench_txt} "
                f"Open Mission Planner/QGC -> Read mission. The drone will NOT take off from this app in upload mode. "
                f"Note: if the RC transmitter switches to AUTO and the vehicle is armed, the Pixhawk could fly this "
                f"mission - keep props off on the bench.{files_txt}")

    # ================================================================== staged mission / readiness / GO (go-flow.md)
    async def _stage(self, p: PlanProposal) -> str:
        """Confirm in sim/full: upload + read-back verify, then wait (state "staged") for a separate GO."""
        self._abort_requested = False
        reset = getattr(self.vehicle, "reset_cancel", None)
        if callable(reset):
            reset()                            # an earlier Abort must not cancel this upload; a new one will
        out = await self._upload_core(p, None)
        if isinstance(out, str):
            return out
        res, info, _rec = out
        if not res["verified"]:
            self._set_state("planned")
            return (f"Uploaded {res['items']} items but the read-back verify FAILED: not staged, do not fly this "
                    f"mission. Planned vs uploaded: {info['diff']['summary']['text']}. Confirm again to re-upload.")
        if self._abort_requested or self.pending is not p:
            self._set_state("planned")
            return "ABORT was pressed during the upload: the mission is on the Pixhawk but NOT staged; nothing will arm."
        try:
            items = self.vehicle.build_mission_items(p.plan)
        except ValueError as exc:            # cannot happen after a verified upload; never stage without items
            self._set_state("planned")
            return f"Mission items: {exc}"
        hf = self.vehicle.home_position()
        self.staged = {"plan_id": p.plan_id, "items": items, "uploaded_ts": time.time(),
                       "home_version": None if hf is None else hf.version}
        self._mission_check = ("ok", f"{len(items)} items verified at upload", time.time())
        self._ready_pre = p.preflight if isinstance(p.preflight, PreflightResult) else None
        self.readiness = None
        self._set_state("staged")
        self._event("staged", plan_id=p.plan_id, items=len(items))
        self._kick_readiness(deep=True)
        n = len(p.plan.waypoints)
        return (f"Mission uploaded and verified ({n} waypoints, {p.plan.altitude_m:.0f} m): STAGED, the drone is NOT "
                f"armed. Review it on the map or in Mission Planner (Read mission) while drone-brain runs the "
                f"pre-flight checks. Press GO to arm and start, or Abort to discard. Expires in "
                f"{self.s.staged_max_age_s / 60.0:.0f} min.")

    def _unstage(self, reason: str, state: str = "planned") -> None:
        if self.staged is not None:
            self._event("unstaged", plan_id=self.staged["plan_id"], reason=reason)
        self.staged = None
        self.readiness = None
        self._mission_check = None
        if self.state == "staged":
            self._set_state(state)

    def _staged_expired(self) -> bool:
        st = self.staged
        return st is not None and time.time() - st["uploaded_ts"] > self.s.staged_max_age_s

    def _kick_readiness(self, deep: bool) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        t = loop.create_task(self._readiness_and_push(deep))
        self._summary_tasks.add(t)            # keep a reference (cancelled on detach)
        t.add_done_callback(self._summary_tasks.discard)

    async def _readiness_and_push(self, deep: bool) -> None:
        try:
            await self._run_readiness(deep=deep)
        except Exception:  # noqa: BLE001 - a check must never take the controller down
            log.exception("readiness run failed")
        await self._broadcast_status()

    async def _readiness_loop(self) -> None:
        while True:
            await asyncio.sleep(self.s.readiness_interval_s)
            if self.state != "staged" or self._going:
                continue
            if self._staged_expired():
                self._unstage("expired")
                await self._broadcast(_err(f"The staged plan expired after {self.s.staged_max_age_s / 60.0:.0f} min "
                                           f"without GO: press Confirm to upload and check it again."))
                await self._broadcast_status()
                continue
            await self._readiness_and_push(deep=time.time() - self._last_deep_ts >= self.s.readiness_deep_every_s)

    async def refresh_readiness(self) -> Optional[dict]:
        """POST /api/readiness/refresh: a deep run now (incl. mission re-download). None unless staged."""
        if self.state != "staged":
            return None
        await self._run_readiness(deep=True)
        await self._broadcast_status()
        return self.readiness

    async def _run_readiness(self, deep: bool) -> Optional[dict]:
        """One readiness pass (reuses vehicle_check.run_vehicle_check). Light: telemetry + the SYS_STATUS pre-arm bit,
        param rows from the last deep run. Deep: + MAV_CMD_RUN_PREARM_CHECKS, param reads and a mission re-download
        compared with the staged items. Returns the readiness dict, or None if nothing is staged (any more)."""
        from vehicle_check import build_readiness, run_vehicle_check
        async with self._readiness_lock:
            st, p = self.staged, self.pending
            if st is None or p is None or self.state != "staged":
                return None
            self._readiness_running = True
            if self.readiness is not None:
                self.readiness = {**self.readiness, "running": True}
            try:
                pre = None if deep else self._ready_pre
                rep = await self._vthread(lambda: run_vehicle_check(
                    self.vehicle, self.s, prearm_wait_s=self.s.readiness_prearm_wait_s, param_timeout_s=self.param_timeout_s,
                    plan=p.plan, run_prearm=deep, read_frame=False, preflight=pre), timeout=90.0)
                mc = None
                if deep and rep["link"]["connected"] and self.vehicle.heartbeat_ok():
                    mc = await self._check_vehicle_mission(st, p)
            except (VehicleError, asyncio.TimeoutError) as exc:
                mc = None
                rep = {"link": {"connected": self.vehicle.telemetry().connected}, "findings": {
                    "link": {"fail": [f"vehicle check failed: {exc}"], "warn": []}}}
            finally:
                self._readiness_running = False
            if self.staged is not st:             # aborted / GO / expired / re-staged meanwhile: the result is stale
                return None
            if deep and rep.get("preflight"):
                self._ready_pre = PreflightResult.model_validate(rep["preflight"])
            if mc is not None:
                self._mission_check = mc
                self._last_deep_ts = time.time()
            res = {"ts": time.time(), "running": False, **build_readiness(rep, self._readiness_extra(st, p))}
            self.readiness = res
            self._event("readiness", deep=deep, go_allowed=res["go_allowed"], reason=res["go_reason"])
            return res

    async def _check_vehicle_mission(self, st: dict, p: PlanProposal) -> tuple[str, str, float]:
        got: list[models.MissionItem] = []
        try:
            same = await self._vthread(lambda: self.vehicle.verify_mission(st["items"], readback=got), timeout=60.0)
        except (VehicleError, asyncio.TimeoutError) as exc:
            return "fail", f"could not read the mission back from the Pixhawk: {exc}", time.time()
        self._set_readback(got, st["items"], st["plan_id"], "read", p.home_amsl_m)    # the phone sees what is there
        if not same:
            return ("fail", f"the mission on the Pixhawk changed since upload ({len(got)} items; Mission Planner / QGC "
                            f"edit?). Abort, then Confirm again to re-upload.", time.time())
        return "ok", f"{len(got)} items match the staged plan", time.time()

    def _readiness_extra(self, st: dict, p: PlanProposal) -> dict[str, tuple[str, str]]:
        """Readiness items run_vehicle_check cannot know (notes/go-flow.md item ids)."""
        tel = self.vehicle.telemetry()
        ex: dict[str, tuple[str, str]] = {}
        ex["disarmed"] = ("fail", "vehicle is ARMED") if tel.armed else ("ok", "disarmed")
        hf = self.vehicle.home_position()
        if hf is None:
            ex["home"] = ("fail", "the Pixhawk reports no home position")
        else:
            moved, shift, dalt = self._home_tol_exceeded(p.plan.home.lat, p.plan.home.lon, p.home_amsl_m, hf)
            far = (home_shift_m(models.GeoPoint(lat=tel.lat, lon=tel.lon), hf)
                   if tel.lat is not None and tel.lon is not None else 0.0)
            if moved:
                ex["home"] = ("fail", f"home moved {shift:.0f} m (altitude {dalt:+.1f} m) since upload: Abort and "
                                      f"Confirm again (the plan is recalculated)")
            elif far > self.s.home_shift_replan_m:
                ex["home"] = ("fail", f"the drone is {far:.0f} m from its home; ArduPilot resets home at arming. "
                                      f"Carry it back to home")
            elif hf.version != st["home_version"]:
                ex["home"] = ("ok", f"refined by {shift:.1f} m since upload (within {self.s.home_shift_replan_m:.0f} m)")
            else:
                ex["home"] = ("ok", "unchanged since upload")
        mc = self._mission_check
        if mc is None:
            ex["mission"] = ("pending", "not checked yet")
        else:
            ex["mission"] = (mc[0], f"{mc[1]} ({time.time() - mc[2]:.0f} s ago)")
        ceil, src = safety.ceiling_info(self.s)
        alt = p.plan.altitude_m
        ex["geofence_alt"] = (("ok", f"plan {alt:.0f} m <= ceiling {ceil:.0f} m ({src})") if alt <= ceil + 1e-6
                              else ("fail", f"plan altitude {alt:.0f} m is above the {src} ceiling {ceil:.0f} m"))
        ok, why = arm_guard(self.s)
        ex["arm_guard"] = (("ok", why) if ok else
                           ("fail", "LOCKED (safety, not an error): GO cannot arm the real drone. To fly, a human must "
                                    "set ALLOW_REAL_ARM=true in .env (and DRY_RUN=false) - in the drone-brain/.env file "
                                    "on the Jetson, not a Pixhawk parameter"))
        vs = self.pipeline.status() if self.pipeline is not None else {}
        alive = self.pipeline.alive()
        vstate = vs.get("state") if isinstance(vs, dict) else None
        ex["vision"] = (("ok", f"vision {vstate or 'running'}") if alive and vstate not in ("disabled", "failed", "error")
                        else ("warn", f"vision {vstate or 'down'}: {vs.get('reason', '') if isinstance(vs, dict) else ''}".strip()))
        ex["legal"] = (("ok", "checklist acknowledged" if LEGAL_CHECKLIST else "no checklist")
                       if self.legal_ack_ts is not None or not LEGAL_CHECKLIST else ("fail", "acknowledge the legal checklist"))
        return ex

    async def go(self, plan_id: str) -> str:
        """GO (notes/go-flow.md): only for the staged plan, after a fresh deep readiness run."""
        if self.mode not in ("sim", "full"):
            return (f"{self.mode} mode: GO is not available - drone-brain does not arm or start missions in this mode. "
                    f"Restart with --vehicle sim/full to fly from the app.")
        if self._going:
            return "GO is already being processed; wait for it to finish."
        self._going = True
        try:
            return await self._go_locked(plan_id)
        finally:
            self._going = False

    async def _go_locked(self, plan_id: str) -> str:
        st, p = self.staged, self.pending
        if self.state != "staged" or st is None or p is None:
            return (f"GO refused: no mission is staged (state {self.state}). Confirm a plan first: GO only starts a "
                    f"mission that was uploaded and verified.")
        if st["plan_id"] != plan_id or p.plan_id != plan_id:
            return ("GO refused: that is not the plan uploaded to the vehicle (it may have been replaced). Review the "
                    "staged plan and press GO again.")
        if self._staged_expired():
            self._unstage("expired")
            return (f"GO refused: the staged plan expired after {self.s.staged_max_age_s / 60.0:.0f} min. Press Confirm "
                    f"to upload and check it again.")
        rd = await self._run_readiness(deep=True)
        await self._broadcast_status()
        if rd is None or self.staged is not st or self.state != "staged":
            return "GO refused: the staged plan changed while checking (Abort?). Nothing was armed."
        if not rd["go_allowed"]:
            return f"GO refused: {rd['go_reason']}"
        # ---- arm + start (arm guard still applies inside vehicle.arm) ----
        self._abort_requested = False
        reset = getattr(self.vehicle, "reset_cancel", None)
        if callable(reset):
            reset()                                   # from here on an Abort cancels start_staged
        self._start_flight(p)
        self._saw_auto = self._manual_alerted = self._gps_alerted = False
        self._last_mode = None
        self._set_state("arming")
        await self._broadcast_status()
        try:
            await self._vthread(self.vehicle.start_staged, st["items"])
        except ArmRefused as exc:
            self._finish_flight("arm refused")
            self.staged, self.readiness = None, None
            self._set_state("planned")
            return f"Not flying: {exc}"
        except (VehicleError, ValueError, NotImplementedError) as exc:
            log.error("start_staged failed: %s", exc)
            if self.vehicle.telemetry().armed or getattr(self.vehicle, "go_armed", False):
                self.staged, self.readiness = None, None
                await self._emergency_rtl(f"mission start failed after arming: {exc}")
                return f"Mission start failed after arming ({exc}); RTL commanded."
            self._finish_flight("start failed")
            if self._abort_requested or "no longer matches" in str(exc):
                self.staged, self.readiness = None, None
                self._set_state("planned")
                return f"Not flying (nothing armed): {exc}. Confirm again to re-upload."
            self._set_state("staged")                 # still uploaded + verified: fix the cause and press GO again
            self._kick_readiness(deep=True)
            return f"GO failed, the drone is NOT armed: {exc}. The plan is still staged; press GO again or Abort."
        self.staged, self.readiness = None, None
        self.pending = None
        self._pending_src = None
        self._set_state("flying")
        self._saw_armed = True
        self._saw_auto = True          # start_staged verified AUTO + armed: a later mode change is the pilot / a failsafe
        self._flight_meta["armed"] = True
        if self._abort_requested:
            await self._do_rtl("abort requested during arming")
            return "Mission started, but ABORT was requested meanwhile: RTL commanded."
        n = len(p.plan.waypoints)
        if getattr(self.s, "demo_mode", False):
            return (f"DEMO search started for \"{p.target.description}\": nothing flies. Open Live to watch the camera; "
                    f"the best match is boxed red. Abort ends the search and writes the summary.")
        return (f"GO: armed, AUTO, mission started ({n} waypoints, {p.plan.altitude_m:.0f} m). Recording to "
                f"{self.last_flight_dir}. Out of WiFi range the phone cannot reach the drone: the RC transmitter "
                f"(RTL switch) is the in-flight abort.")

    def _set_readback(self, got: list[models.MissionItem], planned: Optional[list[models.MissionItem]],
                      plan_id: Optional[str], source: str, home_amsl: Optional[float]) -> dict:
        """Remember what the Pixhawk holds (+ the diff against `planned`) for the phones."""
        diff = mission_diff.compare_missions(planned, got, plan_id)
        files = {}
        if plan_id:
            files = {"mission.waypoints": f"/api/plans/{plan_id}/mission.waypoints"}
            if got:
                files["readback.waypoints"] = f"/api/plans/{plan_id}/readback.waypoints"
        self.readback = mission_diff.readback_view(got)
        self.readback_info = {"plan_id": plan_id, "ts": time.time(), "source": source, "diff": diff, "files": files,
                              "hint": WP_HINT, "home_amsl_m": home_amsl}
        return self.readback_info

    async def read_vehicle_mission(self) -> str:
        """Download the mission from the Pixhawk (a read: allowed in readonly and upload mode) and refresh the phone's
        overlay + Planned-vs-uploaded table. Compared with the pending plan, else the last uploaded one."""
        if self.state in BUSY_STATES:
            return "A mission is in progress; not reading now."
        if not self.vehicle.telemetry().connected:
            return "No MAVLink link to the Pixhawk."
        try:
            got = await self._vthread(self.vehicle.download_mission, timeout=60.0)
        except (VehicleError, asyncio.TimeoutError) as exc:
            return f"Reading the mission from the Pixhawk FAILED: {exc}"
        plan_id, planned, amsl = None, None, None
        if self.pending is not None:
            try:
                plan_id, planned, amsl = (self.pending.plan_id, self.vehicle.build_mission_items(self.pending.plan),
                                          self.pending.home_amsl_m)
            except ValueError:
                plan_id = None
        elif self._uploaded is not None:
            plan_id, planned, amsl = self._uploaded
        info = self._set_readback(got, planned, plan_id, "read", amsl)
        if plan_id and got:
            self.plan_store.save(plan_id, {"readback.waypoints": to_wpl(got, amsl)})
        self._event("mission_read", items=len(got), plan_id=plan_id)
        what = f"vs plan {plan_id}" if plan_id else "(no plan to compare with)"
        return f"Read {len(got)} mission items from the Pixhawk {what}: {info['diff']['summary']['text']}."

    async def clear_vehicle_mission(self) -> str:
        if self.mode == "readonly":
            return "Read-only mode: cannot clear the mission on the Pixhawk."
        if self.state in BUSY_STATES:
            return "A mission is in progress; not clearing."
        if self.state == "staged":
            return "A mission is staged for GO; press Abort first, then clear it."
        if self.vehicle.telemetry().armed:
            return "Clear refused: the vehicle is ARMED."
        try:
            await self._vthread(self.vehicle.clear_mission, timeout=30.0)
        except (VehicleError, asyncio.TimeoutError) as exc:
            return f"Clearing the mission FAILED: {exc}"
        self._event("mission_cleared")
        self.last_upload = None
        return "Mission cleared on the Pixhawk (MISSION_CLEAR_ALL accepted)."

    async def vehicle_check(self) -> dict:
        from vehicle_check import run_vehicle_check
        import json
        rep = await self._vthread(lambda: run_vehicle_check(self.vehicle, self.s, param_timeout_s=self.param_timeout_s),
                                  timeout=90.0)
        self.last_vehicle_check = rep
        try:
            path = self.s.data_dir / "vehicle_check.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rep, indent=1, default=str))
        except OSError as exc:
            log.error("saving vehicle_check.json failed: %s", exc)
        return rep

    # ================================================================== alerts (vision threads)
    def push_alert(self, alert: models.Alert) -> None:
        """Pipeline/matcher callback; safe from any thread. (Storage logging is done by the matcher/pipeline.)"""
        d = alert.model_dump()
        loop = self._loop
        if loop is None or loop.is_closed():
            self.alerts.append(d)
            return
        loop.call_soon_threadsafe(self._alert_on_loop, d)

    def _alert_on_loop(self, d: dict) -> None:
        self.alerts.append(d)
        self._flight_meta["alerts"] = self._flight_meta.get("alerts", 0) + 1
        self._event("alert", level=d.get("level"), track_id=d.get("track_id"))
        asyncio.get_running_loop().create_task(self._broadcast(WsMessage(type="alert", data=d)))

    # ================================================================== legal checklist
    def legal_status(self) -> dict:
        return {"items": LEGAL_CHECKLIST, "acked": self.legal_ack_ts is not None, "ack_ts": self.legal_ack_ts,
                "laanc_ceiling_m": self.s.laanc_ceiling_m}

    def acknowledge_legal(self, accepted: Optional[list[bool]] = None) -> dict:
        if LEGAL_CHECKLIST and accepted is not None and (len(accepted) != len(LEGAL_CHECKLIST) or not all(accepted)):
            raise ValueError("every checklist item must be ticked")
        self.legal_ack_ts = time.time()
        log.warning("legal checklist acknowledged by the operator")
        self._event("legal_ack")
        return self.legal_status()

    # ================================================================== planning
    async def _parse_target(self, text: str, strict: bool = False) -> Target:
        try:
            return await planner.aupdate_target_from_message(text, self.llm, self.s)
        except planner.PlannerError as exc:
            if isinstance(exc.__cause__, LLMError) or not strict:
                log.warning("target parse fell back to raw text (%s)", exc)
                return Target(description=text.strip()[:500], attributes=[])
            raise
        except Exception as exc:  # noqa: BLE001 - LLM down etc.: never block on the description
            log.warning("target parse failed (%s); using raw text", exc)
            return Target(description=text.strip()[:500], attributes=[])

    def _build_plan(self, src: dict[str, Any], home: models.GeoPoint) -> models.SurveyPlan:
        s = self.s
        if src["kind"] == "area":
            req: AreaPlanRequest = src["req"]
            alt = req.altitude_m if req.altitude_m is not None else s.default_alt_m
            return survey.build_polygon_survey(req.corners, alt, home, s, speed_ms=req.speed_ms, overlap=req.overlap)
        m: models.Mission = src["mission"]
        if m.shape == "polygon" and m.corners:
            return survey.build_polygon_survey(m.corners, m.altitude_m, home, s, speed_ms=m.speed_ms)
        return survey.build_survey(m, home, s)

    async def _propose(self, src: dict[str, Any]) -> PlanProposal | str:
        async with self._plan_lock:
            if self.state == "staged":         # the staged plan is only replaced after an Abort
                return STAGED_TEXT
            res = await self._make_proposal(src)
            if isinstance(res, PlanProposal):
                self.pending = res
                self._pending_src = src
                self.target = res.target
                await asyncio.to_thread(self.pipeline.update_target, res.target)
                if self.state not in BUSY_STATES:
                    self._set_state("planned")
            return res

    async def _make_proposal(self, src: dict[str, Any]) -> PlanProposal | str:
        tel = self.vehicle.telemetry()
        if not tel.connected:
            return "No MAVLink link to the Pixhawk: cannot plan without its home position."
        hf, hsrc = self._effective_home()
        if hsrc == "pixhawk" and (tel.gps_fix or 0) < 3:
            return (f"Waiting for Pixhawk home position - needs 3D GPS fix "
                    f"(GPS fix type {tel.gps_fix if tel.gps_fix is not None else '-'}, {tel.satellites or 0} satellites).")
        if hf is None:
            return "Waiting for Pixhawk home position - needs 3D GPS fix (the Pixhawk has not reported HOME_POSITION yet)."
        home = hf.geopoint()
        try:
            plan = self._build_plan(src, home)
        except survey.PolygonError as exc:
            return str(exc)
        except ValueError as exc:
            return f"Cannot build a survey: {exc}"
        errors: list[str] = []
        items: list[models.MissionItem] = []
        try:
            items = self.vehicle.build_mission_items(plan)
        except ValueError as exc:
            errors.append(f"Mission items: {exc}")
        res = safety.validate(plan, self.s, items=items or None)
        pre = await self._vthread(lambda: preflight_check(self.vehicle, self.s, plan, timeout=self.param_timeout_s))
        if hsrc == "bench":
            pre = downgrade_preflight_for_bench(pre)      # expected on the bench; safety errors (res) still block
        errors = res.errors + errors + pre.errors
        warnings = list(res.warnings) + [w for w in pre.warnings if w not in res.warnings]
        if hsrc == "bench":
            warnings.insert(0, BENCH_CHIP)
        if self.mode == "readonly":
            warnings.append("Read-only mode: this plan is saved for download but will NOT be uploaded.")
        elif self.mode == "upload":
            warnings.append("Upload mode: Confirm uploads this mission to the Pixhawk for review (Mission Planner/QGC "
                            "-> Read). This app will NOT arm or take off.")
        plan_id = uuid.uuid4().hex[:8]
        files = {n: f"/api/plans/{plan_id}/{n}" for n in PlanStore.FILES}
        prop = PlanProposal(plan_id=plan_id, plan=plan, target=src["target"],
                            safety=SafetyResult(ok=not errors, errors=list(dict.fromkeys(errors)),
                                                warnings=list(dict.fromkeys(warnings))),
                            preflight=pre, home_amsl_m=hf.amsl_m, limits=safety.limits_info(self.s), files=files,
                            home_source=hsrc, bench_notice=(f"{BENCH_CHIP}. {BENCH_UPLOAD_NOTE}" if hsrc == "bench" else None))
        self._save_plan_files(prop, items)
        self._event("plan", plan_id=plan_id, ok=prop.safety.ok, home_version=hf.version, home_source=hsrc)
        return prop

    def _save_plan_files(self, p: PlanProposal, items: list[models.MissionItem]) -> None:
        files: dict[str, Any] = {"plan.json": p, "mission.geojson": to_geojson(p.plan, plan_id=p.plan_id,
                                                                               home_amsl_m=p.home_amsl_m, home_source=p.home_source),
                                 "preflight.json": p.preflight or {}}
        if items:
            files["mission.waypoints"] = to_wpl(items, p.home_amsl_m)
        self.plan_store.save(p.plan_id, files)

    async def _rebuild_pending(self, hf: HomeFix, shift: float, dalt: float, force: bool = False) -> None:
        async with self._plan_lock:
            src, old = self._pending_src, self.pending
            if src is None or old is None or self.state in BUSY_STATES or self.state == "staged":
                return                             # staged: readiness "home" fails instead (Abort, then re-plan)
            moved, shift, dalt = self._home_tol_exceeded(old.plan.home.lat, old.plan.home.lon, old.home_amsl_m, hf)
            if not moved and not force:
                return
            res = await self._make_proposal(src)
            if isinstance(res, PlanProposal):
                self.pending = res
                what = (f"Home moved {shift:.0f} m (altitude {dalt:+.1f} m)" if moved else
                        f"The Pixhawk now reports its real home (the bench home is ignored)")
                text = (f"{what} - plan recalculated "
                        f"({old.plan_id} -> {res.plan_id}). Review it and confirm again.")
                self._event("replan", old=old.plan_id, new=res.plan_id, shift=shift, dalt=dalt)
                await self._broadcast(WsMessage(type="plan", data=res.model_dump()))
                await self._say(text)
            else:
                self.pending, self._pending_src = None, None
                self._set_state("idle")
                await self._broadcast(_err(f"Home moved {shift:.0f} m and the plan could not be rebuilt: {res}"))
            await self._broadcast_status()

    # ================================================================== flight records
    def _start_flight(self, p: PlanProposal) -> None:
        self._finish_flight("superseded")
        self.alerts.clear()                           # the map shows this flight's alerts only (status resends them)
        fs = FlightStorage(self.s)
        fs.set_plan(p.plan)
        hf = self.vehicle.home_position()
        self._flight_meta = {"plan_id": p.plan_id, "vehicle_mode": self.mode, "start_ts": time.time(),
                             "target": p.target.description, "home": None if hf is None else
                             {"lat": hf.lat, "lon": hf.lon, "amsl_m": hf.amsl_m}, "home_changes": 0,
                             "distance_m": 0.0, "max_alt_m": 0.0, "alerts": 0, "link_losses": 0}
        if getattr(self.s, "demo_mode", False):
            self._flight_meta["demo"] = True            # ground demo: nothing flew (flight_summary says so)
        fs.update_summary(**self._flight_meta, state="arming")
        fs.record_target(p.target)
        try:
            items = self.vehicle.build_mission_items(p.plan)
            fs.write_text("mission.waypoints", to_wpl(items, p.home_amsl_m))
        except ValueError:
            pass
        import json
        fs.write_text("mission.geojson", json.dumps(to_geojson(p.plan, plan_id=p.plan_id, home_amsl_m=p.home_amsl_m)))
        self.flight = fs
        self.last_flight_dir = fs.rel(fs.root)
        self._route = [[p.plan.home.lat, p.plan.home.lon]] + [[w.lat, w.lon] for w in p.plan.waypoints] + \
            [[p.plan.home.lat, p.plan.home.lon]]
        self._last_pos = None
        self.pipeline.begin_flight(fs)
        if self.target is not None:
            self.pipeline.update_target(self.target)
        log.info("flight record: %s", fs.root)

    def _finish_flight(self, reason: str) -> None:
        fs = self.flight
        if fs is None:
            return
        self.flight = None
        try:
            self.pipeline.end_flight()
        except Exception:  # noqa: BLE001
            log.exception("pipeline.end_flight failed")
        meta = dict(self._flight_meta)
        meta.update(end_reason=reason, end_state=self.state, end_ts=time.time(),
                    duration_s=round(time.time() - meta.get("start_ts", time.time()), 1),
                    distance_m=round(meta.get("distance_m", 0.0), 1), vision=self.pipeline.status())
        fs.update_summary(**meta)
        fs.close()
        log.info("flight record closed (%s): %s", reason, fs.root)
        if meta.get("armed"):                  # it flew (or at least armed): summary for the operator after landing
            self._schedule_summary(fs.root)

    def _schedule_summary(self, root: Any) -> None:
        """flight_summary in the background: never blocks or delays RTL / safety paths. Shutdown -> template only."""
        loop = self._loop
        if self._detaching or loop is None or loop.is_closed() or not loop.is_running():
            try:
                self.last_flight_summary = flight_summary.summarize_template(root, self.s.data_dir)
            except Exception:  # noqa: BLE001
                log.exception("flight summary (template) failed")
            return
        t = loop.create_task(self._summarize_flight(root), name="flight-summary")
        self._summary_tasks.add(t)
        t.add_done_callback(self._summary_tasks.discard)

    async def _summarize_flight(self, root: Any) -> None:
        try:
            res = await flight_summary.summarize(root, self.llm, data_dir=self.s.data_dir,
                                                 timeout_s=self.s.flight_summary_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("flight summary failed")
            return
        self.last_flight_summary = res
        self._event("flight_summary", source=res["source"], flight_dir=res["flight_dir"], images=len(res["images"]))
        await self._broadcast(WsMessage(type="flight_summary", data=res))
        await self._broadcast_status()

    # ================================================================== RTL
    async def _do_rtl(self, reason: str) -> tuple[bool, str]:
        if self.mode in ("readonly", "upload"):
            return False, f"{self.mode} mode (no vehicle control)"
        log.warning("RTL: %s", reason)
        self._own_rtl_ts = time.time()
        self._event("rtl", reason=reason)
        if self.flight is not None:
            self.flight.append("telemetry", {"event": "rtl", "reason": reason, "ts": time.time()})
            self._flight_meta.setdefault("rtl_reasons", []).append(reason)
        try:
            await self._vthread(self.vehicle.rtl, timeout=25.0)
        except (VehicleError, asyncio.TimeoutError) as exc:
            log.error("RTL failed: %s", exc)
            return False, f"{exc.__class__.__name__}: {exc}"
        if self.state in ACTIVE_STATES:
            self._set_state("rtl")
        return True, "ok"

    async def _emergency_rtl(self, reason: str) -> None:
        await self._broadcast(_err(f"SAFETY: {reason}. Commanding RTL."))
        ok, msg = await self._do_rtl(reason)
        if not ok:
            await self._broadcast(_err(f"RTL FAILED ({msg}). Use the RC transmitter!"))
        await self._broadcast_status()

    def _cb_link_lost(self) -> None:          # vehicle callback thread
        if self._loop is not None and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._on_link_lost(), self._loop)

    def _cb_link_restored(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._on_link_restored(), self._loop)

    async def _on_link_lost(self) -> None:
        self._event("link_lost", state=self.state)
        if self.flight is not None:
            self._flight_meta["link_losses"] = self._flight_meta.get("link_losses", 0) + 1
            self.flight.append("telemetry", {"event": "link_lost", "ts": time.time()})
        if self.state not in ACTIVE_STATES:      # uploading / staged never arm: GO re-checks the link
            await self._broadcast(_err("MAVLink link to the Pixhawk lost."
                                       + (" The staged plan cannot be started until it returns (GO is blocked)."
                                          if self.state == "staged" else "")))
            return
        if self.mode in ("readonly", "upload"):
            await self._broadcast(_err(f"MAVLink link lost ({self.mode} mode): use the RC transmitter."))
            return
        self._rtl_on_restore = True
        if self.state == "arming":
            self._abort_requested = True
            self.vehicle.cancel_flight()          # never arm after a link loss during GO
        await self._emergency_rtl("MAVLink link to the Pixhawk lost while flying")

    async def _on_link_restored(self) -> None:
        self._event("link_restored", state=self.state)
        await self._say("MAVLink link restored.")
        if self._rtl_on_restore and self.state in ("arming", "flying", "rtl"):
            self._rtl_on_restore = False
            tel = self.vehicle.telemetry()
            if tel.armed and tel.mode not in RTL_MODES:
                await self._emergency_rtl("link restored after a loss during the flight")

    # ================================================================== background loops
    def _no_vehicle_hint(self) -> str:
        url = self.s.mavlink_url
        if url.startswith("serial:"):
            return (f"No MAVLink heartbeat on {url} @ {self.s.mavlink_baud} - check the USB cable / baud rate, and that "
                    f"nothing else (Mission Planner, QGC, another main.py) has the port open")
        return (f"No vehicle on {url} - start SITL (scripts/sitl_run.sh), or use --mock-vehicle, or "
                f"--vehicle readonly --mavlink-url serial:/dev/ttyACM0 --baud 115200")

    async def _vehicle_link_loop(self) -> None:
        backoff = 2.0
        t_fail: Optional[float] = None
        last_summary = 0.0
        while not self._vehicle_connected:
            try:
                await self._vthread(self.vehicle.connect, self.connect_timeout_s)
                self._vehicle_connected = True
                self.vehicle_hint = None
                log.info("vehicle connected (%s mode)", self.mode)
                await self._broadcast_status()
            except Exception as exc:  # noqa: BLE001 - VehicleError, serial/OS errors (port busy, no device)
                now = time.monotonic()
                if t_fail is None:      # say it once, loudly, with the fix; then retry quietly with backoff
                    t_fail = last_summary = now
                    self.vehicle_hint = self._no_vehicle_hint()
                    log.warning("%s (%s: %s); retrying quietly", self.vehicle_hint, type(exc).__name__, exc)
                    await self._broadcast_status()
                elif now - last_summary >= 60.0:
                    last_summary = now
                    log.warning("still no vehicle after %.0f s: %s", now - t_fail, self.vehicle_hint)
                else:
                    log.debug("vehicle connect failed: %s (retry in %.0fs)", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _vision_watchdog(self) -> None:
        backoff = 2.0
        last_sig: Any = None
        last_log = 0.0
        while True:
            await asyncio.sleep(1.0)
            if self.pipeline.alive():
                backoff = max(2.0, backoff / 2)
                continue
            st = self.pipeline.status()
            sig = (st.get("state"), st.get("reason"))
            if sig != last_sig or time.monotonic() - last_log > 60.0:
                log.error("vision worker died/stalled: %s", st)      # once per distinct state, then once a minute
                last_sig, last_log = sig, time.monotonic()
            self._event("vision_failed", status=self.pipeline.status())
            if getattr(self.pipeline, "critical", False) and self.state in ACTIVE_STATES:
                await self._emergency_rtl("vision pipeline crashed during the search")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            await asyncio.to_thread(self.pipeline.restart)

    async def _telemetry_loop(self) -> None:
        period = 1.0 / TELEMETRY_HZ
        while True:
            t0 = time.monotonic()
            tel = self.vehicle.telemetry()
            self._tel = tel
            self._tick += 1
            await self._broadcast(WsMessage(type="telemetry", data=tel.model_dump()))
            if self.flight is not None and self._tick % int(TELEMETRY_HZ) == 0:
                self.flight.append("telemetry", {**tel.model_dump(), "state": self.state})
                self._flight_stats(tel)
            changed = self._update_state(tel)
            await self._watch_flight(tel)
            st = self._statustexts()
            if st and st[-1]["ts"] > self._last_st_ts:
                self._last_st_ts = st[-1]["ts"]
                changed = True
            await self._check_home()
            if changed:
                await self._broadcast_status()
            await asyncio.sleep(max(0.05, period - (time.monotonic() - t0)))

    async def _watch_flight(self, tel: models.Telemetry) -> None:
        """Tell the operator what the vehicle itself decided (mode changes, failsafes, GPS loss, warnings)."""
        if self.state not in BUSY_STATES:
            self._st_alert_ts = time.time()
            self._last_mode = None
            return
        if not tel.connected:          # a short heartbeat gap must not make us forget the last known mode
            return
        if self.state == "manual" and not self._manual_alerted:
            self._manual_alerted = True
            await self._broadcast(_err(f"PILOT CONTROL: the vehicle left AUTO (now {tel.mode}). drone-brain no longer treats "
                                       f"this as its mission and sends no commands. Fly/land with the RC transmitter."))
        elif self._last_mode is not None and tel.mode != self._last_mode and self.state not in ("manual", "arming") \
                and time.time() - self._own_rtl_ts > 15.0:          # arming: start_staged sets AUTO itself
            await self._broadcast(_err(f"Vehicle mode changed {self._last_mode} -> {tel.mode} (failsafe or pilot)."
                                       + (" The vehicle is returning on its own." if tel.mode in RTL_MODES else "")))
        self._last_mode = tel.mode
        fix = tel.gps_fix
        if fix is not None and fix < 3 and not self._gps_alerted:
            self._gps_alerted = True
            await self._broadcast(_err(f"GPS fix LOST in flight (fix type {fix}). The Pixhawk failsafe decides what happens "
                                       f"(see vehicle messages); be ready on the RC transmitter."))
        elif fix is not None and fix >= 3:
            self._gps_alerted = False
        for d in self._statustexts():
            if d["ts"] > self._st_alert_ts and d["severity"] <= 4:
                self._st_alert_ts = max(self._st_alert_ts, d["ts"])
                await self._broadcast(_err(f"Vehicle: {d['text']}"))

    def shutdown_rtl(self, timeout_s: float = 2.0) -> Optional[str]:
        """SIGTERM policy (sim/full): never leave an armed vehicle flying our mission without a plan. Mid-upload ->
        cancel (never arm); armed in AUTO -> command RTL before exiting. Pilot control / already RTL / readonly+upload
        -> nothing. Safe from any thread; bounded by timeout_s; returns what it did."""
        if self._shutdown_rtl_done or self.mode not in ("sim", "full"):
            return None
        self._shutdown_rtl_done = True
        try:
            tel = self.vehicle.telemetry()
            if self.state in ("uploading", "arming"):
                self.vehicle.cancel_flight()
            if self.state not in ACTIVE_STATES or not tel.connected or not tel.armed or tel.mode in RTL_MODES:
                return "cancelled upload" if self.state in ("uploading", "arming") else None
        except Exception:  # noqa: BLE001
            log.exception("shutdown_rtl: telemetry")
            return None
        import threading
        res: list[str] = []

        def run() -> None:
            try:
                self.vehicle.rtl()
                res.append("RTL commanded")
            except Exception as exc:  # noqa: BLE001
                res.append(f"RTL FAILED: {exc}")
        th = threading.Thread(target=run, daemon=True, name="shutdown-rtl")
        th.start()
        th.join(timeout_s)
        out = res[0] if res else "RTL pending"
        log.warning("shutdown while flying: %s", out)
        return out

    def _flight_stats(self, tel: models.Telemetry) -> None:
        if tel.lat is None or tel.lon is None:
            return
        if self._last_pos is not None:
            self._flight_meta["distance_m"] = self._flight_meta.get("distance_m", 0.0) + home_shift_m(
                models.GeoPoint(lat=self._last_pos[0], lon=self._last_pos[1]), models.GeoPoint(lat=tel.lat, lon=tel.lon))
        self._last_pos = (tel.lat, tel.lon)
        self._flight_meta["max_alt_m"] = max(self._flight_meta.get("max_alt_m", 0.0), tel.alt_rel_m or 0.0)

    def _update_state(self, tel: models.Telemetry) -> bool:
        before = self.state
        if not tel.connected:
            return False
        if tel.armed:
            self._saw_armed = True
            if self.flight is not None:
                self._flight_meta["armed"] = True
            if tel.mode == "AUTO" and self.state in ACTIVE_STATES:
                self._saw_auto = True
        if self.state == "staged" and tel.armed:
            # armed by the RC / a ground station while waiting for GO: not our flight; never fight it
            self._unstage("armed externally", state="manual")
            self._manual_alerted = True
            loop = self._loop
            if loop is not None and not loop.is_closed():
                loop.create_task(self._broadcast(_err(
                    f"The vehicle was ARMED outside drone-brain (mode {tel.mode}) while the plan was staged: GO is "
                    f"cancelled and drone-brain sends no commands. Fly/land with the RC transmitter.")))
        if self.state == "flying":
            if tel.mode in RTL_MODES:
                self._set_state("rtl")
            elif not tel.armed and self._saw_armed:
                self._set_state("landed")
            elif tel.armed and self._saw_auto and tel.mode != "AUTO":
                self._set_state("manual")      # pilot / failsafe left AUTO: no longer our mission, never fight it
        elif self.state in ("rtl", "manual"):
            if not tel.armed:
                self._set_state("landed")
        if self.state == "landed" and before != "landed":
            self._saw_armed = False
            self._finish_flight("landed")
            self._route = None
        return self.state != before

    async def _check_home(self) -> None:
        hf = self.vehicle.home_position()
        if hf is None or (self._home_seen is not None and hf.version == self._home_seen.version):
            return
        old, self._home_seen = self._home_seen, hf
        shift = home_shift_m(old, hf) if old else 0.0
        dalt = (hf.amsl_m - old.amsl_m) if old else 0.0
        rec = {"event": "home_changed", "ts": time.time(), "lat": hf.lat, "lon": hf.lon, "amsl_m": hf.amsl_m,
               "version": hf.version, "shift_m": round(shift, 2), "dalt_m": round(dalt, 2), "state": self.state}
        self._event("home", **rec)
        if self.flight is not None:
            self.flight.append("telemetry", rec)
            if old is not None:
                self._flight_meta["home_changes"] = self._flight_meta.get("home_changes", 0) + 1
        await self._broadcast_status()
        if self.state in BUSY_STATES and old is not None:
            since = time.time() - self._flight_meta.get("start_ts", 0.0)
            if since < ARMING_HOME_WINDOW_S and self.state in ACTIVE_STATES:
                await self._say(f"Home reset at arming ({shift:.1f} m, alt {dalt:+.1f} m).")
            else:
                await self._broadcast(_err(f"WARNING: Pixhawk home changed mid-flight by {shift:.0f} m "
                                           f"(alt {dalt:+.1f} m). RTL will return to the NEW home."))
            return
        if self.bench is not None and old is None:
            await self._say("The Pixhawk now reports its real home (GPS): BENCH_HOME is ignored from now on.")
        if self.state == "staged":
            if old is not None:
                self._kick_readiness(deep=False)      # readiness "home" decides (fails if it moved too far)
            return
        p = self.pending
        if p is not None:
            moved, shift_p, dalt_p = self._home_tol_exceeded(p.plan.home.lat, p.plan.home.lon, p.home_amsl_m, hf)
            if moved or p.home_source == "bench":
                await self._rebuild_pending(hf, shift_p, dalt_p, force=p.home_source == "bench")
