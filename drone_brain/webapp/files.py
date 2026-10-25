"""Plan / flight file downloads, legal checklist and pre-flight endpoints (Phase 7).

Mounted by main.py:  app.include_router(make_router(settings, controller))

    GET  /api/plans                              recent plans: [{plan_id, mtime, files}]
    GET  /api/plans/{plan_id}                    plan.json
    GET  /api/plans/{plan_id}/{name}             plan.json | mission.waypoints | mission.geojson | preflight.json
    GET  /api/flights                            recent flight records
    GET  /api/flights/{flight_id}/{name}         flight.json | *.jsonl | mission.waypoints | mission.geojson
    GET  /api/legal                              checklist items + whether this session acknowledged them
    POST /api/legal/ack   {"accepted": [true, ...]}
    GET  /api/preflight                          run mission.preflight_check() now (read-only param reads)
    GET  /api/vehicle/check[?cached=1]            full read-only vehicle check (also saved to data/vehicle_check.json)
    POST /api/vehicle/clear_mission              MISSION_CLEAR_ALL (upload/sim/full modes, refused while armed)
    POST /api/vehicle/read_mission               download the mission from the Pixhawk (a read: every mode) and refresh the
                                                 phone's read-back overlay + Planned-vs-uploaded table (status.readback*)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

import models
from config import Settings
from storage import LOG_KINDS, PlanStore

MEDIA = {".waypoints": "text/plain; charset=utf-8", ".geojson": "application/geo+json", ".json": "application/json",
         ".jsonl": "application/x-ndjson"}
PLAN_EXTRA_FILES = {"upload.json", "readback.waypoints"}
FLIGHT_FILES = {"flight.json", "mission.waypoints", "mission.geojson"} | {f"{k}.jsonl" for k in LOG_KINDS}


class LegalAck(BaseModel):
    accepted: Optional[list[bool]] = None


def _safe_id(v: str) -> str:
    if not v or len(v) > 64 or not all(c.isalnum() or c in "-_" for c in v):
        raise HTTPException(status_code=404, detail="not found")
    return v


def _file(path: Path, download_name: Optional[str] = None, root: Optional[Path] = None) -> FileResponse:
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    if root is not None:        # a symlink inside the data dir must not lead out of it
        try:
            path.resolve().relative_to(root.resolve())
        except (ValueError, OSError):
            raise HTTPException(status_code=404, detail="not found")
    headers = {"Cache-Control": "no-cache"}
    if download_name:
        headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
    return FileResponse(path, media_type=MEDIA.get(path.suffix, "application/octet-stream"), headers=headers)


def make_router(settings: Settings, controller: Any) -> APIRouter:
    r = APIRouter()
    store = PlanStore(settings)
    flights_root = Path(settings.data_dir) / "flights"

    @r.get("/api/plans")
    async def list_plans(limit: int = 50) -> dict:
        return {"plans": store.list(limit=max(1, min(limit, 500)))}

    @r.get("/api/plans/{plan_id}")
    async def get_plan(plan_id: str) -> dict:
        p = store.root / _safe_id(plan_id) / "plan.json"
        if not p.is_file():
            raise HTTPException(status_code=404, detail="not found")
        try:
            if not p.resolve().is_relative_to(store.root.resolve()):
                raise HTTPException(status_code=404, detail="not found")
            return json.loads(p.read_text())
        except (OSError, ValueError):
            raise HTTPException(status_code=500, detail="plan file unreadable")

    @r.get("/api/plans/{plan_id}/{name}")
    async def get_plan_file(plan_id: str, name: str) -> FileResponse:
        if name not in PlanStore.FILES and name not in PLAN_EXTRA_FILES:
            raise HTTPException(status_code=404, detail="not found")
        pid = _safe_id(plan_id)
        return _file(store.root / pid / name, f"{pid}_{name}" if name.endswith(".waypoints") or name.startswith("mission") else None,
                     root=store.root)

    @r.get("/api/flights")
    async def list_flights(limit: int = 50) -> dict:
        out = []
        if flights_root.is_dir():
            dirs = sorted((d for d in flights_root.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
            for d in dirs[:max(1, min(limit, 500))]:
                out.append({"flight_id": d.name, "files": sorted(f.name for f in d.iterdir() if f.name in FLIGHT_FILES)})
        return {"flights": out}

    @r.get("/api/flights/{flight_id}/{name}")
    async def get_flight_file(flight_id: str, name: str) -> FileResponse:
        if name not in FLIGHT_FILES:
            raise HTTPException(status_code=404, detail="not found")
        fid = _safe_id(flight_id)
        return _file(flights_root / fid / name, f"{fid}_{name}" if name.startswith("mission") else None, root=flights_root)

    # ---- legal checklist (session state lives in the controller) ----
    @r.get("/api/legal")
    async def legal() -> dict:
        fn = getattr(controller, "legal_status", None)
        if fn is None:
            return {"items": [], "acked": True}
        return fn()

    @r.post("/api/legal/ack")
    async def legal_ack(body: LegalAck, request: Request) -> dict:
        fn = getattr(controller, "acknowledge_legal", None)
        if fn is None:
            raise HTTPException(status_code=404, detail="not supported")
        try:
            res = fn(body.accepted)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        hub = getattr(request.app.state, "hub", None)
        if hub is not None:
            await hub.broadcast(models.WsMessage(type="status", data=controller.snapshot()))
        return res

    @r.get("/api/preflight")
    async def preflight() -> dict:
        vehicle = getattr(controller, "vehicle", None)
        if vehicle is None:
            raise HTTPException(status_code=404, detail="no vehicle")
        from mission import preflight_check
        try:
            res = await asyncio.wait_for(asyncio.to_thread(preflight_check, vehicle, settings, None, 3.0), 60)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"pre-flight check failed: {exc}")
        return res.model_dump()

    @r.get("/api/vehicle/check")
    async def vehicle_check(cached: bool = False) -> dict:
        if cached and getattr(controller, "last_vehicle_check", None):
            return controller.last_vehicle_check
        fn = getattr(controller, "vehicle_check", None)
        if fn is None:
            raise HTTPException(status_code=404, detail="not supported")
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"vehicle check failed: {exc}")

    @r.post("/api/vehicle/clear_mission")
    async def clear_mission(request: Request) -> dict:
        fn = getattr(controller, "clear_vehicle_mission", None)
        if fn is None:
            raise HTTPException(status_code=404, detail="not supported")
        text = await fn()
        hub = getattr(request.app.state, "hub", None)
        msg = models.WsMessage(type="chat", data={"role": "system", "text": text, "ts": __import__("time").time()})
        if hub is not None:
            await hub.broadcast(msg)
            await hub.broadcast(models.WsMessage(type="status", data=controller.snapshot()))
        return msg.model_dump()

    @r.post("/api/vehicle/read_mission")
    async def read_mission(request: Request) -> dict:
        fn = getattr(controller, "read_vehicle_mission", None)
        if fn is None:
            raise HTTPException(status_code=404, detail="not supported")
        try:
            text = await asyncio.wait_for(fn(), 90)
        except Exception as exc:  # noqa: BLE001
            text = f"Reading the mission from the Pixhawk FAILED: {exc.__class__.__name__}: {exc}"
        hub = getattr(request.app.state, "hub", None)
        msg = models.WsMessage(type="chat", data={"role": "system", "text": text, "ts": __import__("time").time()})
        if hub is not None:
            await hub.broadcast(msg)
            await hub.broadcast(models.WsMessage(type="status", data=controller.snapshot()))
        return msg.model_dump()

    return r
