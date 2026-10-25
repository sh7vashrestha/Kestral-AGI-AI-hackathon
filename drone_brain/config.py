"""Typed settings loaded from drone-brain/.env (see .env.example).

This is the only global object in the project. Import with:
    from config import get_settings
    s = get_settings()
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]      # repo root: .env, data/, models/

# Absolute legal/physical ceiling; MAX_ALT_M can never be configured above this.
HARD_MAX_ALT_M = 120.0

# KSU Marietta soccer field centre (mock / SITL start; `--bench-home field`)
FIELD_HOME = (33.9371904, -84.5271293)
FIELD_AMSL_M = 300.0


class BenchHomeError(ValueError):
    """BENCH_HOME / --bench-home is invalid or used in a vehicle mode that does not allow it."""


def parse_bench_home(text: str) -> Optional[tuple[float, float, float]]:
    """'LAT,LON[,AMSL]' or 'field' -> (lat, lon, amsl_m); '' -> None. Raises BenchHomeError."""
    t = (text or "").strip()
    if not t:
        return None
    if t.lower() == "field":
        return (FIELD_HOME[0], FIELD_HOME[1], FIELD_AMSL_M)
    parts = [p.strip() for p in t.replace(";", ",").split(",") if p.strip()]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise BenchHomeError(f"BENCH_HOME {text!r}: expected LAT,LON[,AMSL] or 'field'") from None
    if len(vals) not in (2, 3):
        raise BenchHomeError(f"BENCH_HOME {text!r}: expected LAT,LON[,AMSL] or 'field'")
    lat, lon = vals[0], vals[1]
    amsl = vals[2] if len(vals) == 3 else FIELD_AMSL_M
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or not -500 <= amsl <= 9000:
        raise BenchHomeError(f"BENCH_HOME {text!r}: coordinates out of range")
    return (lat, lon, amsl)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- Safety switches -------------------------------------------------
    dry_run: bool = True                 # True: SITL / mock only
    allow_real_arm: bool = False         # Only a human sets this in .env. Never set by code.
    # Phase 7: how main.py may use the vehicle link.
    #   sim      - simulator/mock only (loopback or wildcard UDP with DRY_RUN); full control
    #   readonly - any link (e.g. the real Pixhawk): receive telemetry, heartbeats, stream/param READS only;
    #              mission upload, mode change, arm, RTL, param set are refused before any I/O
    #   full     - any link, full control; arming still needs real_arm_permitted on real hardware
    #   upload   - readonly + mission upload/verify/clear (review in Mission Planner/QGC); never mode/arm/RTL/params
    vehicle_mode: Literal["sim", "readonly", "upload", "full"] = "sim"
    # Ground demo (main.py --demo / ./demo.sh): a pretend drone on loopback hovers at home, nothing can fly; GO starts
    # the live camera search (eye-level CLIP prompts, best match boxed red on /live). demo_max_s: search ends by itself.
    demo_mode: bool = False
    demo_max_s: float = Field(600.0, ge=0)
    # Indoor upload test only: used as the planning home while the Pixhawk has no HOME_POSITION. Accepted ONLY with
    # VEHICLE_MODE=upload (anything else refuses to start); a real Pixhawk home always wins. "" = off.
    bench_home: str = ""                 # "LAT,LON[,AMSL]" or "field"
    upload_fence_polygon: bool = False   # upload mode: also upload the search area as an inclusion fence polygon
                                         # (replaces the Pixhawk's fence polygon; FENCE_* params are never touched)

    # --- LLM (single shared llama-server) --------------------------------
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_timeout_s: float = 60.0
    llm_retries: int = 2
    llm_max_tokens: int = 512
    llm_temperature: float = 0.1
    assistant_timeout_s: float = Field(45.0, gt=0)   # Hardening A: one chat turn (LLM) before the "use the Survey tab" fallback

    # --- MAVLink ---------------------------------------------------------
    # SITL default; real hardware: serial:/dev/ttyTHS1 with MAVLINK_BAUD=921600
    mavlink_url: str = "udpin:0.0.0.0:14550"
    mavlink_baud: int = 921600
    mavlink_timeout_s: float = 5.0
    heartbeat_loss_s: float = 3.0
    autopilot: str = "ardupilot"         # only 'ardupilot' implemented

    # --- Camera ----------------------------------------------------------
    # csi:<sensor-id> | v4l2:/dev/videoN | file:<video file or image folder>
    camera_source: str = "csi:0"
    camera_width: int = 1920
    camera_height: int = 1080
    capture_fps: float = Field(1.0, gt=0, le=60)             # Δt = 1 / capture_fps
    # Optics used for footprint / GSD math. Defaults: IMX477 (active 5.952 x 3.348 mm in the 16:9 modes)
    # with an ASSUMED 6 mm CS lens (HFOV 52.8, VFOV 31.2). The user MUST set the real lens in .env.
    camera_hfov_deg: float = 52.8
    camera_vfov_deg: float = 31.2
    sensor_width_mm: float = 5.952
    focal_length_mm: float = 6.0
    image_width_px: int = 1920
    sensor_height_mm: float = 3.348      # IMX477 active height in the 16:9 modes (needed when rotated 90/270)
    # Orientation applied AT CAPTURE (every consumer sees the corrected image). Rotation is clockwise, applied first,
    # then the flip (mirror of the rotated image). 90/270 turn 1920x1080 into 1080x1920 and swap the FOVs.
    camera_rotate: int = 180             # 0 | 90 | 180 | 270
    camera_flip: str = "none"            # none | horizontal | vertical

    # --- Thermal (FLIR Lepton 3.x on PureThermal) --------------------------
    # v4l2:/dev/video0 | file:<folder of .npy (degC) / 16-bit .png (centikelvin)> | none
    thermal_source: str = "none"
    thermal_hfov_deg: float = 57.0       # Lepton 3.5; Lepton 3.0 = 56 deg
    thermal_vfov_deg: float = 44.0
    thermal_width: int = 160
    thermal_height: int = 120
    thermal_rotate: int = 0              # 0 | 90 | 180 | 270 clockwise, applied to the 160x120 degC array at capture
    thermal_flip: str = "none"           # none | horizontal | vertical (after the rotation)
    thermal_delta_c: float = 4.0         # blob must be this much warmer than local background
    thermal_min_c: float = 24.0
    thermal_max_c: float = 42.0
    thermal_max_age_s: float = 0.5
    thermal_weight: float = 0.5          # w in C_person = 1-(1-C_yolo)(1-w*C_heat)
    thermal_recheck_conf: float = 0.35
    thermal_stuck_pixels: str = "67,64"  # static bad pixels "x,y;x,y" (160x120 coords)
    thermal_alert_interval_s: float = 10.0
    thermal_min_person_m: float = 0.3
    thermal_max_person_m: float = 1.8
    rgb_dark_threshold: float = 25.0     # mean gray below this -> skip YOLO/CLIP, heat alerts only

    # --- Detection / matching -------------------------------------------
    yolo_model: str = "yolo26n.pt"       # .engine preferred once exported
    yolo_conf_threshold: float = 0.8
    yolo_imgsz: int = 640
    match_backend: str = "clip"         # clip | vlm : who scores crops against the target
    vlm_confirm_top: bool = False        # clip backend: send a track's best crop once to the VLM for a reason
    match_threshold: float = 0.7
    clip_model: str = "ViT-B-32"           # OpenCLIP model name (drone-brain-clip experiments)
    clip_pretrained: str = "laion2b_s34b_b79k"   # ... and its pretrained weights tag
    clip_use_trt: bool = False           # CLIP image tower via TensorRT: off (the FP16 engine gave wrong rankings on this Jetson)
    clip_match_cutoff: float = Field(0.6, ge=0, le=1)  # CLIP: positive-prompt probability needed for matches=True (tuned in Hardening D, notes/hardening-D.md)
    clip_prompts_llm: bool = True        # build per-target CLIP prompts with the planner LLM (fallback: default prompts)
    min_frames: int = 3                  # VLM/CLIP results needed before a target_match alert (Hardening D: 3; see match_strong_threshold)
    match_strong_threshold: float = Field(0.93, ge=0, le=1)  # a track with >= 2 results and C_avg >= this alerts without waiting for min_frames
    match_min_px: int = 20               # boxes with both sides below this are tracked and get a person alert but are not scored by CLIP
    crop_padding: float = 0.2
    crop_min_px: int = 224
    vlm_queue_size: int = 32             # pending crops (one per track); CLIP drains ~60/s, so 32 covers a 30-person frame
    alert_min_interval_s: float = 5.0
    tracker_iou: float = 0.3
    tracker_gps_radius_m: float = 8.0

    # --- Survey / mission limits ----------------------------------------
    survey_overlap: float = Field(0.25, ge=0.0, lt=0.9)
    default_alt_m: float = 20.0
    default_speed_ms: float = 3.0
    max_alt_m: float = 60.0
    min_alt_m: float = 5.0
    geofence_radius_m: float = 300.0
    max_path_m: float = 3000.0
    max_speed_ms: float = 10.0
    min_person_px: int = 80
    # Hardening A: minimum person height (px) per target detail level; the chat assistant picks the altitude from these
    detail_px_person: int = Field(80, gt=0)        # person only
    detail_px_clothing: int = Field(120, gt=0)     # clothing colour
    detail_px_fine: int = Field(160, gt=0)         # fine detail (logo, glasses, pattern)
    max_area_m2: float = 200000.0        # polygon survey upper bound (Phase 2b)
    min_area_m2: float = 100.0           # polygon survey lower bound
    est_endurance_min: float = 15.0      # assumed usable flight time, for the battery note only
    person_height_m: float = 1.7
    # Phase 7 airspace: effective ceiling = min(MAX_ALT_M, LAANC_CEILING_M, 120); floor = max(MIN_ALT_M, OBSTACLE_CLEARANCE_M)
    laanc_ceiling_m: Optional[float] = Field(None, gt=0)   # altitude approved in the LAANC authorization (per flight)
    obstacle_clearance_m: float = Field(15.0, ge=0)        # min survey + RTL altitude above home (poles, goals, trees)
    home_shift_replan_m: float = Field(5.0, gt=0)          # pending plan is rebuilt if the Pixhawk home moves more
    home_alt_shift_m: float = Field(2.0, gt=0)             # ... or its altitude changes more than this
    # Upload -> review -> GO (notes/go-flow.md): Confirm stages the mission, a separate GO arms and starts it
    readiness_interval_s: float = Field(5.0, gt=0)         # light readiness re-check while staged (telemetry only)
    readiness_deep_every_s: float = Field(30.0, gt=0)      # + param reads, pre-arm command, mission re-download
    readiness_prearm_wait_s: float = Field(1.5, ge=0)      # how long to collect PreArm STATUSTEXT after the 401 command
    staged_max_age_s: float = Field(900.0, gt=0)           # a staged plan older than this must be confirmed again
    batt_warn_pct: float = Field(50.0, ge=0, le=100)       # readiness: battery below this is a warning (< 30 % blocks)
    flight_summary_timeout_s: float = Field(60.0, gt=0)    # Gemma flight summary; on timeout the template text is used

    # --- Web / storage ---------------------------------------------------
    web_host: str = "0.0.0.0"
    web_port: int = 8000
    captive_portal: bool = False         # hotspot: any other host name (phone connectivity checks) -> redirect to the app
    https: bool = False                  # also serve the app over HTTPS (scripts/make_cert.sh); plain HTTP then only redirects
    https_port: int = 8443
    data_dir: Path = ROOT / "data"
    log_level: str = "INFO"

    # --- Offline map (Phase 2b) -----------------------------------------
    # Relative paths resolve inside <data_dir>/maps. Empty = no satellite layer.
    map_satellite_mbtiles: str = ""

    @field_validator("laanc_ceiling_m", mode="before")
    @classmethod
    def _empty_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("camera_rotate", "thermal_rotate", mode="before")
    @classmethod
    def _rotate_ok(cls, v):
        r = int(str(v).strip()) % 360
        if r not in (0, 90, 180, 270):
            raise ValueError("rotation must be 0, 90, 180 or 270")
        return r

    @field_validator("camera_flip", "thermal_flip", mode="before")
    @classmethod
    def _flip_ok(cls, v):
        f = str(v).strip().lower() or "none"
        if f not in ("none", "horizontal", "vertical"):
            raise ValueError("flip must be none, horizontal or vertical")
        return f

    @field_validator("max_alt_m")
    @classmethod
    def _cap_alt(cls, v: float) -> float:
        if v > HARD_MAX_ALT_M:
            raise ValueError(f"MAX_ALT_M={v} exceeds hard limit {HARD_MAX_ALT_M}")
        return v

    # ---- effective optics: what the corrected (rotated) image actually sees --------------------------------
    @property
    def cam_swapped(self) -> bool:
        return self.camera_rotate % 180 == 90

    @property
    def eff_hfov_deg(self) -> float:
        """Horizontal FOV of the image after CAMERA_ROTATE (= the sensor's vertical FOV at 90/270)."""
        return self.camera_vfov_deg if self.cam_swapped else self.camera_hfov_deg

    @property
    def eff_vfov_deg(self) -> float:
        return self.camera_hfov_deg if self.cam_swapped else self.camera_vfov_deg

    @property
    def eff_camera_width(self) -> int:
        return self.camera_height if self.cam_swapped else self.camera_width

    @property
    def eff_camera_height(self) -> int:
        return self.camera_width if self.cam_swapped else self.camera_height

    @property
    def eff_image_width_px(self) -> int:
        """Pixels along the image's horizontal axis (image_width_px scaled to the rotated frame)."""
        if not self.cam_swapped:
            return self.image_width_px
        return int(round(self.image_width_px * self.camera_height / self.camera_width))

    @property
    def eff_sensor_width_mm(self) -> float:
        """Size (mm) of the sensor axis that ends up horizontal in the corrected image."""
        return self.sensor_height_mm if self.cam_swapped else self.sensor_width_mm

    @property
    def thermal_swapped(self) -> bool:
        return self.thermal_rotate % 180 == 90

    @property
    def eff_thermal_hfov_deg(self) -> float:
        return self.thermal_vfov_deg if self.thermal_swapped else self.thermal_hfov_deg

    @property
    def eff_thermal_vfov_deg(self) -> float:
        return self.thermal_hfov_deg if self.thermal_swapped else self.thermal_vfov_deg

    @property
    def eff_thermal_size(self) -> tuple[int, int]:
        """(width, height) of the corrected thermal array."""
        return (self.thermal_height, self.thermal_width) if self.thermal_swapped else (self.thermal_width, self.thermal_height)

    @property
    def map_dir(self) -> Path:
        return Path(self.data_dir) / "maps"

    @property
    def frame_interval_s(self) -> float:
        return 1.0 / self.capture_fps

    @property
    def real_arm_permitted(self) -> bool:
        """Arming non-simulated hardware needs both switches set by a human."""
        return self.allow_real_arm and not self.dry_run


def bench_home_for(settings: Settings) -> Optional[tuple[float, float, float]]:
    """The validated bench home, or None when unset. Refuses (BenchHomeError) in every vehicle mode but `upload`:
    a made-up home must never reach a path that can arm or fly."""
    bh = parse_bench_home(getattr(settings, "bench_home", ""))
    if bh is not None and settings.vehicle_mode != "upload":
        raise BenchHomeError(f"--bench-home / BENCH_HOME is only allowed with --vehicle upload "
                             f"(VEHICLE_MODE={settings.vehicle_mode!r}): a bench home is not a GPS home and must never "
                             f"be used where the vehicle can arm or fly.")
    return bh


@lru_cache
def get_settings() -> Settings:
    return Settings()
