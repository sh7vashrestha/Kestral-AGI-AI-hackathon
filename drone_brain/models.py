"""Pydantic models for every piece of data that crosses a module boundary.

Shared contract: modules import from here instead of defining their own copies.
Coordinate convention: local frame is metres North/East of HOME; bearings are
degrees clockwise from true north.
"""
from __future__ import annotations

import time
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------- Planner (LLM intent) ----------------------------------------

class LatLon(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Mission(BaseModel):
    """Search area intent.

    shape="rectangle": rectangle whose near edge is centred `offset_m` from home along
    `direction_deg`; it extends `length_m` along that bearing and `width_m` across it
    (length_m / width_m are required for rectangles).
    shape="polygon": area given by `corners` (3-8 lat/lon points in perimeter order, the
    phone UI uses 4); length/width/direction/offset are unused (Phase 2b)."""
    shape: Literal["rectangle", "polygon"] = "rectangle"
    length_m: Optional[float] = Field(None, gt=0, description="extent along direction_deg (rectangle)")
    width_m: Optional[float] = Field(None, gt=0, description="extent across direction_deg (rectangle)")
    direction_deg: float = Field(0.0, ge=0, lt=360, description="bearing from home, 0=N, 90=E")
    offset_m: float = Field(0.0, ge=0, description="distance from home to near edge")
    centered: bool = Field(False, description="rectangle centred on home (offset_m ignored); direction_deg = long axis")
    altitude_m: float = Field(gt=0, description="relative to home")
    speed_ms: Optional[float] = Field(None, gt=0)
    corners: Optional[list[LatLon]] = Field(None, min_length=3, max_length=8)

    @model_validator(mode="after")
    def _check_shape(self) -> "Mission":
        if self.shape == "rectangle":
            if self.length_m is None or self.width_m is None:
                raise ValueError("rectangle mission requires length_m and width_m")
        elif not self.corners:
            raise ValueError("polygon mission requires corners")
        return self


class AreaPlanRequest(BaseModel):
    """Phone UI -> controller: plan a survey of a tapped polygon (Phase 2b)."""
    corners: list[LatLon] = Field(min_length=3, max_length=8)
    altitude_m: Optional[float] = Field(None, gt=0)
    overlap: Optional[float] = Field(None, ge=0, lt=0.9)
    speed_ms: Optional[float] = Field(None, gt=0)
    target_text: Optional[str] = Field(None, max_length=2000)


class Target(BaseModel):
    description: str = Field(min_length=1)
    attributes: list[str] = Field(default_factory=list)


class Intent(BaseModel):
    """What the planner returns. Either mission+target, or an error message for the user."""
    mission: Optional[Mission] = None
    target: Optional[Target] = None
    error: Optional[str] = None


# ---------- Survey / safety ---------------------------------------------

class GeoPoint(BaseModel):
    lat: float
    lon: float
    alt_m: float = 0.0  # relative to home


class Waypoint(BaseModel):
    seq: int
    north_m: float
    east_m: float
    lat: float
    lon: float
    alt_m: float


class SurveyPlan(BaseModel):
    home: GeoPoint
    mission: Mission
    waypoints: list[Waypoint]           # survey legs only; TAKEOFF/RTL added by mission.py
    altitude_m: float
    speed_ms: float                     # already clamped to v_max
    v_max_ms: float
    footprint_w_m: float
    footprint_l_m: float
    line_spacing_m: float
    path_length_m: float                # home -> waypoints -> home
    gsd_m: float
    person_px: float
    warnings: list[str] = Field(default_factory=list)
    # Phase 2b additions (all optional so older callers/tests are unaffected)
    area_m2: Optional[float] = None
    lane_count: Optional[int] = None
    estimated_flight_s: Optional[float] = None
    battery_note: Optional[str] = None


class SafetyResult(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PreflightCheck(BaseModel):
    """One row of the Pixhawk parameter cross-check (Phase 7)."""
    param: str
    value: Optional[float] = None       # as read from the vehicle (None = not read / missing)
    requirement: str
    ok: bool
    severity: Literal["error", "warning"] = "error"
    message: str = ""


class PreflightResult(BaseModel):
    """mission.preflight_check(): read-only PARAM_REQUEST_READ comparison with .env (Phase 7)."""
    ok: bool
    checks: list[PreflightCheck] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    effective_max_alt_m: Optional[float] = None
    ceiling_source: Optional[str] = None        # "LAANC" | "MAX_ALT_M" | "120 m hard limit"
    rtl_alt_m: Optional[float] = None
    fence: Optional[str] = None                 # human-readable fence status
    home_source: str = "pixhawk"                # "bench" = failures downgraded to warnings (upload test, no GPS home)
    ts: float = Field(default_factory=time.time)


class PlanProposal(BaseModel):
    """Shown on the phone for the Confirm tap."""
    plan_id: str
    plan: SurveyPlan
    target: Target
    safety: SafetyResult
    created_ts: float = Field(default_factory=time.time)
    # Phase 7 additions (optional)
    preflight: Optional[PreflightResult] = None
    home_amsl_m: Optional[float] = None         # Pixhawk home altitude the plan was computed against
    limits: Optional[dict] = None               # effective ceiling / floor and their sources
    files: Optional[dict] = None                # download URLs (mission.waypoints, mission.geojson, ...)
    home_source: str = "pixhawk"                # "pixhawk" (HOME_POSITION) | "bench" (BENCH_HOME, upload test only)
    bench_notice: Optional[str] = None          # set when home_source == "bench": the unmistakable warning text


# ---------- Vehicle -----------------------------------------------------

class Telemetry(BaseModel):
    connected: bool = False
    mode: str = "UNKNOWN"
    armed: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_rel_m: Optional[float] = None
    heading_deg: Optional[float] = None
    groundspeed_ms: Optional[float] = None
    battery_pct: Optional[float] = None
    voltage_v: Optional[float] = None
    gps_fix: Optional[int] = None
    satellites: Optional[int] = None
    mission_seq: Optional[int] = None
    mission_count: Optional[int] = None
    last_heartbeat_ts: Optional[float] = None
    # Phase 7 live flight state (optional)
    landed_state: Optional[str] = None      # EXTENDED_SYS_STATE: ON_GROUND | TAKEOFF | IN_AIR | LANDING | UNDEFINED
    system_status: Optional[str] = None     # HEARTBEAT system_status: STANDBY | ACTIVE | CRITICAL | ...
    hdop: Optional[float] = None
    heartbeat_age_s: Optional[float] = None
    vehicle_type: Optional[str] = None      # QUADROTOR, HEXAROTOR, ...
    ts: float = Field(default_factory=time.time)


# ---------- Vision ------------------------------------------------------

class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    frame_id: int
    ts: float
    bbox: BBox
    c_yolo: float
    frame_path: Optional[str] = None
    telemetry: Optional[Telemetry] = None   # latest snapshot at capture time
    track_id: Optional[int] = None
    c_heat: Optional[float] = None          # Phase 5: thermal evidence 0..1 (None = no thermal data)
    c_person: Optional[float] = None        # Phase 5: fused person confidence (preferred over c_yolo)


class HotSpot(BaseModel):
    """Warm blob in a thermal frame (Phase 5). bbox in thermal pixels."""
    bbox: BBox
    peak_c: float
    mean_c: float
    delta_c: float
    area_px: int
    score: float = Field(ge=0, le=1)


class MatchResult(BaseModel):
    """JSON schema the VLM must return (also used for response_format)."""
    is_human: bool
    matches: bool
    match_confidence: float = Field(ge=0, le=1)
    reason: str


class ClipPrompts(BaseModel):
    """Per-target text prompts for the CLIP backend (added in Phase 6)."""
    target_description: str
    positive: list[str] = Field(min_length=1)
    negative: list[str] = Field(default_factory=list)


class Alert(BaseModel):
    level: Literal["person", "target_match", "heat"]
    track_id: int
    ts: float
    c_yolo: float
    c_match: Optional[float] = None
    c_final: Optional[float] = None
    c_avg: Optional[float] = None
    n_frames: int = 1
    reason: str = ""
    image_url: Optional[str] = None          # served by webapp from data_dir
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_rel_m: Optional[float] = None
    target_description: Optional[str] = None


# ---------- WebSocket envelope -----------------------------------------

class WsMessage(BaseModel):
    """Every server→phone push: {"type": ..., "data": {...}}.
    types: chat | plan | status | telemetry | alert | error | flight_summary (notes/go-flow.md)"""
    type: Literal["chat", "plan", "status", "telemetry", "alert", "error", "flight_summary"]
    data: dict


# ---------- Vehicle mission items (added in Phase 4) -------------------

class MissionItem(BaseModel):
    """One MAVLink mission item (MISSION_ITEM_INT semantics, but lat/lon in degrees).

    mission.py converts lat/lon to int32 x1e7 on the wire. `frame` uses MAVLink
    MAV_FRAME values (0 GLOBAL, 2 MISSION, 6 GLOBAL_RELATIVE_ALT_INT).
    """
    seq: int
    command: int                 # MAV_CMD_* id (16 WAYPOINT, 22 TAKEOFF, 20 RTL, 178 CHANGE_SPEED)
    frame: int = 6
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    lat: float = 0.0             # degrees
    lon: float = 0.0             # degrees
    alt: float = 0.0             # metres (relative to home for frame 6)
    autocontinue: int = 1
