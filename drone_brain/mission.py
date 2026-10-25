"""Vehicle link + mission handling (Phase 4).

Design
------
* ``VehicleBackend`` is the interface the rest of drone-brain uses.
  ``ArduPilotBackend`` implements it with pymavlink; a ``PX4Backend`` can be added
  later and selected through ``settings.autopilot`` (see ``create_backend``).
* One background **reader thread** owns every ``recv_match`` call. It keeps a
  lock-protected ``models.Telemetry`` snapshot up to date, sends our own 1 Hz
  heartbeat, watches link health and routes protocol replies (mission handshake,
  COMMAND_ACK, PARAM_VALUE) to whichever caller is waiting for them. Callers never
  read the socket themselves, so telemetry never stalls and callers cannot steal
  each other's messages.
* Every wait has a timeout and raises a descriptive ``VehicleError``.

SAFETY
------
``arm()`` refuses unless the link is a simulator/mock (loopback UDP/TCP, or a
wildcard listen address while ``DRY_RUN`` is true) or ``settings.real_arm_permitted``
(``ALLOW_REAL_ARM=true`` and ``DRY_RUN=false``, both set by a human). Nothing in
this module ever changes those settings.

ACCESS LEVELS (Phase 7, ``VEHICLE_MODE``)
---------------------------------------
* ``readonly``: for looking at the real Pixhawk. The backend may receive messages, send our heartbeat,
  request message intervals / data streams (incl. HOME_POSITION), read parameters (PARAM_REQUEST_READ /
  PARAM_REQUEST_LIST), download the mission (MISSION_REQUEST_LIST / _INT + the closing MISSION_ACK: a read)
  and run MAV_CMD_RUN_PREARM_CHECKS (does not arm). Everything else raises
  ``ReadOnlyViolation`` before any I/O.
* ``upload``: readonly + mission upload / verify / MISSION_CLEAR_ALL (for reviewing a plan in
  Mission Planner / QGC). set_mode, arm/disarm, mission start, param set, RTL and fly_plan are still refused.
  Uploads are refused while the vehicle is armed.
* ``sim`` / ``full``: everything (arming still behind ``arm_guard``).
A second allow-list in ``_send`` rejects any outgoing message the access level does not permit.

HOME (Phase 7)
--------------
HOME_POSITION is tracked continuously (requested at connect, on arming, while unknown, and
streamed at 0.2 Hz). ``home_position()`` returns the latest ``HomeFix`` (lat, lon, AMSL,
version); ``on_home_changed(old, new)`` fires (off the reader thread) on every change.
"""
from __future__ import annotations

import abc
import collections
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Optional

# Use MAVLink 2 from the start so pymavlink never swaps dialect modules mid-run.
os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil  # noqa: E402

from config import HARD_MAX_ALT_M, Settings, get_settings  # noqa: E402
from models import GeoPoint, MissionItem, PreflightCheck, PreflightResult, SurveyPlan, Telemetry  # noqa: E402

log = logging.getLogger("mission")

# ---------------------------------------------------------------------------
# MAVLink constants (numeric, so they do not depend on the loaded dialect)
# ---------------------------------------------------------------------------
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_DO_CHANGE_SPEED = 178
MAV_CMD_MISSION_START = 300
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_SET_MESSAGE_INTERVAL = 511
MAV_CMD_REQUEST_MESSAGE = 512

MAV_FRAME_GLOBAL = 0
MAV_FRAME_MISSION = 2
MAV_FRAME_GLOBAL_RELATIVE_ALT_INT = 6

MAV_RESULT_ACCEPTED = 0
MAV_PARAM_TYPE_REAL32 = 9
MAV_RESULT_IN_PROGRESS = 5
MAV_MISSION_ACCEPTED = 0

MAV_TYPE_GCS = 6
MAV_TYPE_ONBOARD_CONTROLLER = 18
MAV_AUTOPILOT_INVALID = 8
MAV_STATE_ACTIVE = 4
MAV_COMP_ID_ONBOARD_COMPUTER = 191
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_MODE_FLAG_SAFETY_ARMED = 128

# Non-vehicle heartbeat senders we must not mistake for the autopilot.
_NON_VEHICLE_TYPES = {MAV_TYPE_GCS, MAV_TYPE_ONBOARD_CONTROLLER, 5, 27, 30, 31, 32, 34}
_COPTER_TYPES = {2, 3, 4, 13, 14, 15}

# Message ids we request at a fixed rate.
_MSG_SYS_STATUS = 1
_MSG_GPS_RAW_INT = 24
_MSG_GLOBAL_POSITION_INT = 33
_MSG_MISSION_CURRENT = 42
_MSG_VFR_HUD = 74
_MSG_HOME_POSITION = 242

ARDUCOPTER_MODES: dict[str, int] = {
    "STABILIZE": 0, "ACRO": 1, "ALT_HOLD": 2, "AUTO": 3, "GUIDED": 4, "LOITER": 5,
    "RTL": 6, "CIRCLE": 7, "LAND": 9, "DRIFT": 11, "SPORT": 13, "FLIP": 14,
    "AUTOTUNE": 15, "POSHOLD": 16, "BRAKE": 17, "THROW": 18, "AVOID_ADSB": 19,
    "GUIDED_NOGPS": 20, "SMART_RTL": 21, "FLOWHOLD": 22, "FOLLOW": 23, "ZIGZAG": 24,
    "SYSTEMID": 25, "AUTOROTATE": 26, "AUTO_RTL": 27,
}
ARDUCOPTER_MODE_NAMES: dict[int, str] = {v: k for k, v in ARDUCOPTER_MODES.items()}

# Routed protocol replies (everything else only feeds telemetry).
_PROTO_TYPES = (
    "MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK", "MISSION_COUNT",
    "MISSION_ITEM_INT", "COMMAND_ACK", "PARAM_VALUE",
)

_RESULT_NAMES = {
    0: "ACCEPTED", 1: "TEMPORARILY_REJECTED", 2: "DENIED", 3: "UNSUPPORTED",
    4: "FAILED", 5: "IN_PROGRESS", 6: "CANCELLED",
}
_MISSION_ACK_NAMES = {
    0: "ACCEPTED", 1: "ERROR", 2: "UNSUPPORTED_FRAME", 3: "UNSUPPORTED", 4: "NO_SPACE",
    5: "INVALID", 6: "INVALID_PARAM1", 7: "INVALID_PARAM2", 8: "INVALID_PARAM3",
    9: "INVALID_PARAM4", 10: "INVALID_PARAM5_X", 11: "INVALID_PARAM6_Y",
    12: "INVALID_PARAM7", 13: "INVALID_SEQUENCE", 14: "DENIED", 15: "OPERATION_CANCELLED",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class VehicleError(RuntimeError):
    """Base class for every error raised by this module."""


class LinkTimeout(VehicleError):
    """No (valid) reply from the vehicle within the allowed time."""


class CommandRejected(VehicleError):
    """The vehicle answered COMMAND_ACK / MISSION_ACK with a non-accepted result."""

    def __init__(self, message: str, result: Optional[int] = None) -> None:
        super().__init__(message)
        self.result = result


class ArmRefused(VehicleError):
    """arm() was blocked by the software safety guard (not by the vehicle)."""


class ReadOnlyViolation(VehicleError):
    """A vehicle-changing operation was attempted on a read-only backend (nothing was sent)."""


MAV_CMD_RUN_PREARM_CHECKS = 401
MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION = 5001
MAV_MISSION_TYPE_MISSION = 0
MAV_MISSION_TYPE_FENCE = 1

# Outgoing messages a read-only backend may send (second line of defence behind the method guards).
_RO_SEND_METHODS = {"heartbeat_send", "request_data_stream_send", "param_request_read_send", "param_request_list_send",
                    # downloading the mission is a read (the ack only closes the download conversation)
                    "mission_request_list_send", "mission_request_int_send", "mission_ack_send"}
_RO_COMMANDS = {MAV_CMD_SET_MESSAGE_INTERVAL, MAV_CMD_REQUEST_MESSAGE, MAV_CMD_RUN_PREARM_CHECKS}
# ... plus, in upload mode, the mission protocol
_UPLOAD_SEND_METHODS = {"mission_count_send", "mission_item_int_send", "mission_clear_all_send"}
# verify_mission tolerances (also used by mission_diff for the phone's planned-vs-uploaded table)
POS_TOL_DEG = 1e-6          # ~11 cm
ALT_TOL_M = 0.05
_ACCESS_BY_MODE = {"readonly": "readonly", "upload": "upload", "sim": "control", "full": "control"}
_ACCESS_RANK = {"readonly": 0, "upload": 1, "control": 2}
# Latest copy of these messages is kept (vehicle check / live state).
_RAW_KEEP = {"EKF_STATUS_REPORT", "SYS_STATUS", "GPS_RAW_INT", "RC_CHANNELS", "AUTOPILOT_VERSION",
             "EXTENDED_SYS_STATE", "MISSION_CURRENT", "BATTERY_STATUS", "HOME_POSITION", "GLOBAL_POSITION_INT"}
_MSG_RC_CHANNELS = 65
_MSG_AUTOPILOT_VERSION = 148
_MSG_EKF_STATUS_REPORT = 193
_MSG_EXTENDED_SYS_STATE = 245
LANDED_STATES = {0: "UNDEFINED", 1: "ON_GROUND", 2: "IN_AIR", 3: "TAKEOFF", 4: "LANDING"}
SYSTEM_STATUS = {0: "UNINIT", 1: "BOOT", 2: "CALIBRATING", 3: "STANDBY", 4: "ACTIVE", 5: "CRITICAL",
                 6: "EMERGENCY", 7: "POWEROFF", 8: "FLIGHT_TERMINATION"}
MAV_TYPES = {1: "FIXED_WING", 2: "QUADROTOR", 3: "COAXIAL", 4: "HELICOPTER", 10: "GROUND_ROVER", 11: "SURFACE_BOAT",
             12: "SUBMARINE", 13: "HEXAROTOR", 14: "OCTOROTOR", 15: "TRICOPTER", 19: "VTOL_TAILSITTER_DUOROTOR",
             20: "VTOL_TAILSITTER_QUADROTOR", 21: "VTOL_TILTROTOR", 29: "DODECAROTOR"}


@dataclass(frozen=True)
class HomeFix:
    """Vehicle home as last reported by HOME_POSITION."""
    lat: float
    lon: float
    amsl_m: float
    ts: float
    version: int

    def geopoint(self) -> GeoPoint:
        return GeoPoint(lat=self.lat, lon=self.lon, alt_m=0.0)


def home_shift_m(a: "HomeFix | GeoPoint", b: "HomeFix | GeoPoint") -> float:
    """Horizontal distance between two home positions (flat-earth, metres)."""
    import math
    dn = (b.lat - a.lat) * 111_320.0
    de = (b.lon - a.lon) * 111_320.0 * math.cos(math.radians((a.lat + b.lat) / 2))
    return math.hypot(dn, de)


# ---------------------------------------------------------------------------
# Arming guard
# ---------------------------------------------------------------------------
_LOOPBACK_PREFIXES = ("127.",)
_LOOPBACK_NAMES = {"localhost", "::1"}
_WILDCARD_HOSTS = {"0.0.0.0", ""}
_NET_SCHEMES = {"udp", "udpin", "udpout", "tcp", "tcpin"}


def _split_url(url: str) -> tuple[str, str]:
    """('udpin', '0.0.0.0:14550') ; plain '/dev/xyz' paths report scheme 'serial'."""
    if url.startswith("/dev/"):
        return "serial", url
    scheme, _, rest = url.partition(":")
    return scheme.lower(), rest


def _url_host(rest: str) -> str:
    host = rest.rsplit(":", 1)[0] if ":" in rest else rest
    return host.strip("[]").lower()


def is_simulated_link(url: str, dry_run: bool) -> bool:
    """True if `url` can only be a simulator/mock.

    * loopback UDP/TCP (127.x, localhost, ::1) is always a simulator;
    * a wildcard listen address (udpin:0.0.0.0:14550) counts only when DRY_RUN is true
      (the SITL may run on a laptop sending to the Jetson);
    * serial devices, broadcast and any other host never count.
    """
    scheme, rest = _split_url(url)
    if scheme not in _NET_SCHEMES:
        return False
    host = _url_host(rest)
    if host in _LOOPBACK_NAMES or host.startswith(_LOOPBACK_PREFIXES):
        return True
    return host in _WILDCARD_HOSTS and dry_run


def arm_guard(settings: Settings) -> tuple[bool, str]:
    """(allowed, reason). Pure function of the settings; used by arm() and tests."""
    if is_simulated_link(settings.mavlink_url, settings.dry_run):
        return True, "simulator/mock link"
    if settings.real_arm_permitted:
        return True, "real arming permitted (ALLOW_REAL_ARM=true and DRY_RUN=false)"
    scheme, _ = _split_url(settings.mavlink_url)
    return False, (
        f"arming refused: link {settings.mavlink_url!r} ({scheme}) is not a simulator and "
        f"real arming is not permitted (DRY_RUN={settings.dry_run}, "
        f"ALLOW_REAL_ARM={settings.allow_real_arm}). A human must set ALLOW_REAL_ARM=true "
        f"and DRY_RUN=false in .env (drone-brain/.env on the Jetson, not a Pixhawk parameter)."
    )


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class StatusTextLog:
    """De-duplicated STATUSTEXT history. ArduPilot repeats e.g. PreArm messages every ~30 s while disarmed:
    each distinct text is kept once with a repeat counter, first/last-seen times, and moves to the end when it
    repeats. add() returns the line to log (first sight, or a summary every `summary_s`) or None."""

    def __init__(self, maxlen: int = 100, summary_s: float = 600.0) -> None:
        self.maxlen, self.summary_s = maxlen, summary_s
        self._d: "collections.OrderedDict[str, dict]" = collections.OrderedDict()

    def add(self, text: str, severity: int, now: float) -> Optional[str]:
        e = self._d.get(text)
        if e is None:
            self._d[text] = {"ts": now, "first_ts": now, "severity": severity, "text": text, "count": 1,
                             "_logged_ts": now, "_logged_count": 1}
            while len(self._d) > self.maxlen:
                self._d.popitem(last=False)
            return text
        e["count"] += 1
        e["ts"], e["severity"] = now, severity
        self._d.move_to_end(text)
        if now - e["_logged_ts"] >= self.summary_s:
            n = e["count"] - e["_logged_count"]
            e["_logged_ts"], e["_logged_count"] = now, e["count"]
            return f"{text} (repeated {n}x in the last {self.summary_s / 60:.0f} min, {e['count']} total)"
        return None

    def entries(self, since: float = 0.0) -> list[dict]:
        return [{k: v for k, v in e.items() if not k.startswith("_")} for e in self._d.values() if e["ts"] >= since]


class VehicleBackend(abc.ABC):
    """What main.py / webapp / safety use. Implement per autopilot firmware."""

    on_link_lost: Optional[Callable[[], None]] = None
    on_link_restored: Optional[Callable[[], None]] = None
    on_home_changed: Optional[Callable[[Optional["HomeFix"], "HomeFix"], None]] = None
    readonly: bool = False
    access: str = "control"

    @abc.abstractmethod
    def connect(self, timeout: float = 10.0) -> None: ...

    def cancel_flight(self) -> None:
        """Ask a running fly_plan() to stop before it arms (abort during upload). Default: nothing to cancel."""

    def reset_cancel(self) -> None:
        """Forget an earlier cancel_flight() before start_staged() (the caller clears it when GO is accepted, so an
        Abort that arrives between GO and the start of start_staged is not lost). Default: nothing to reset."""

    def start_staged(self, items: list[MissionItem]) -> None:
        """GO for a mission already uploaded + verified (upload_for_review): re-verify -> AUTO -> arm -> start."""
        raise NotImplementedError

    def home_position(self) -> Optional[HomeFix]:
        """Latest HOME_POSITION (non-blocking). Default: derived from get_home()."""
        h = self.get_home(timeout=0.0)
        return None if h is None else HomeFix(h.lat, h.lon, 0.0, time.time(), 0)

    def get_param(self, name: str, timeout: Optional[float] = None) -> Optional[float]:  # pragma: no cover
        raise NotImplementedError

    def preflight_check(self, plan: Optional[SurveyPlan] = None) -> PreflightResult:
        return preflight_check(self, self.settings, plan)  # type: ignore[attr-defined]

    @abc.abstractmethod
    def close(self) -> None: ...

    @abc.abstractmethod
    def telemetry(self) -> Telemetry: ...

    @abc.abstractmethod
    def heartbeat_ok(self) -> bool: ...

    @abc.abstractmethod
    def get_home(self, timeout: float = 5.0) -> Optional[GeoPoint]: ...

    @abc.abstractmethod
    def build_mission_items(self, plan: SurveyPlan) -> list[MissionItem]: ...

    @abc.abstractmethod
    def upload_mission(self, items: list[MissionItem]) -> None: ...

    @abc.abstractmethod
    def download_mission(self) -> list[MissionItem]: ...

    @abc.abstractmethod
    def verify_mission(self, items: list[MissionItem]) -> bool: ...

    @abc.abstractmethod
    def set_param(self, name: str, value: float) -> None: ...

    @abc.abstractmethod
    def set_mode(self, name: str) -> None: ...

    @abc.abstractmethod
    def arm(self) -> None: ...

    @abc.abstractmethod
    def disarm(self, force: bool = False) -> None: ...

    @abc.abstractmethod
    def start_mission(self) -> None: ...

    @abc.abstractmethod
    def rtl(self) -> None: ...

    @abc.abstractmethod
    def fly_plan(self, plan: SurveyPlan) -> None: ...


def create_backend(settings: Optional[Settings] = None, **kwargs) -> VehicleBackend:
    """Factory keyed on settings.autopilot."""
    s = settings or get_settings()
    kind = s.autopilot.strip().lower()
    if kind == "ardupilot":
        return ArduPilotBackend(s, **kwargs)
    raise ValueError(f"unsupported AUTOPILOT={s.autopilot!r} (implemented: 'ardupilot')")


# ---------------------------------------------------------------------------
# ArduPilot implementation
# ---------------------------------------------------------------------------
class ArduPilotBackend(VehicleBackend):
    """ArduCopter over pymavlink. Thread-safe; one instance per vehicle link."""

    HEARTBEAT_PERIOD_S = 1.0
    RECV_POLL_S = 0.2
    ITEM_RETRIES = 4
    COMMAND_RETRIES = 3
    RECONNECT_AFTER_S = 10.0

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        source_system: int = 245,
        on_link_lost: Optional[Callable[[], None]] = None,
        on_link_restored: Optional[Callable[[], None]] = None,
        on_home_changed: Optional[Callable[[Optional[HomeFix], HomeFix], None]] = None,
        readonly: Optional[bool] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.source_system = source_system
        self.on_link_lost = on_link_lost
        self.on_link_restored = on_link_restored
        self.on_home_changed = on_home_changed
        mode = getattr(self.settings, "vehicle_mode", "sim")
        if readonly is not None:
            self.access = "readonly" if readonly else "control"
        else:
            self.access = _ACCESS_BY_MODE.get(mode, "readonly")
        self.readonly = self.access == "readonly"
        self.timeout_s = max(0.2, self.settings.mavlink_timeout_s)

        self._conn = None  # mavutil connection; only created by connect()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cancel_fly = threading.Event()   # abort during upload: fly_plan()/upload_mission() stop before arming
        self.go_armed = False                  # set once start_staged/fly_plan armed the vehicle

        # One condition/lock guards telemetry, home and the protocol queues.
        self._cv = threading.Condition()
        self._send_lock = threading.Lock()
        self._op_lock = threading.RLock()   # one request/response conversation at a time

        self._tel = Telemetry()
        self._home: Optional[GeoPoint] = None
        self.home_amsl_m: Optional[float] = None
        self._home_fix: Optional[HomeFix] = None
        self.home_version = 0
        self._last_home_req = 0.0
        self._home_req_due: list[float] = []   # extra HOME_POSITION requests (e.g. just after arming)
        self._last_fix: Optional[int] = None
        self._raw: dict[str, tuple[float, dict]] = {}
        self._status_log = StatusTextLog()
        self._sysid: Optional[int] = None
        self._compid: int = 1
        self._proto: dict[str, Deque] = {t: deque(maxlen=256) for t in _PROTO_TYPES}
        self._statustext: Deque[str] = deque(maxlen=10)

        self._link_up = False
        self._ever_up = False
        self._last_hb_sent = 0.0
        self._last_stream_req = 0.0
        self._last_gpi = 0.0
        self._last_reconnect = 0.0
        self.reconnect_count = 0

    # ------------------------------------------------------------------ connection
    def _open_connection(self):
        url = self.settings.mavlink_url
        scheme, rest = _split_url(url)
        if scheme == "serial":
            device = rest
        elif scheme in _NET_SCHEMES:
            device = url
        else:
            raise VehicleError(
                f"unsupported MAVLINK_URL {url!r}: use udpin:/udpout:/tcp:/tcpin: host:port "
                f"or serial:/dev/ttyXXX"
            )
        log.info("opening MAVLink link %s", url)
        return mavutil.mavlink_connection(
            device,
            baud=self.settings.mavlink_baud,
            source_system=self.source_system,
            source_component=MAV_COMP_ID_ONBOARD_COMPUTER,
        )

    def connect(self, timeout: float = 10.0) -> None:
        """Open the link, start the reader thread, wait for the first autopilot heartbeat."""
        if self._thread is not None and self._thread.is_alive():
            raise VehicleError("already connected; call close() first")
        self._stop.clear()
        self._conn = self._open_connection()
        self._thread = threading.Thread(target=self._reader_loop, name="mavlink-reader", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._sysid is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cv.wait(min(remaining, 0.2))
            ok = self._sysid is not None
        if not ok:
            self.close()
            raise LinkTimeout(
                f"no autopilot HEARTBEAT on {self.settings.mavlink_url} within {timeout:.1f}s"
            )
        log.info("connected to system %s component %s", self._sysid, self._compid)

    def close(self) -> None:
        self._stop.set()
        th = self._thread
        if th is not None and th is not threading.current_thread():
            th.join(timeout=2.0)
        self._thread = None
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover - best effort
                log.debug("error closing link", exc_info=True)
        with self._cv:
            self._link_up = False
            self._sysid = None
            self._tel = Telemetry()
            self._cv.notify_all()

    # ------------------------------------------------------------------ telemetry
    def telemetry(self) -> Telemetry:
        with self._cv:
            tel = self._tel.model_copy()
        tel.connected = self.heartbeat_ok()
        tel.ts = time.time()
        if tel.last_heartbeat_ts is not None:
            tel.heartbeat_age_s = round(tel.ts - tel.last_heartbeat_ts, 2)
        return tel

    def raw(self, mtype: str) -> Optional[tuple[float, dict]]:
        """(receive time, msg.to_dict()) of the latest message of that type (see _RAW_KEEP)."""
        with self._cv:
            return self._raw.get(mtype)

    def statustexts(self, since: float = 0.0) -> list[dict]:
        """STATUSTEXT log: [{"ts", "severity", "text"}], oldest first."""
        with self._cv:
            return self._status_log.entries(since)

    def heartbeat_ok(self) -> bool:
        with self._cv:
            ts = self._tel.last_heartbeat_ts
        return ts is not None and (time.time() - ts) < self.settings.heartbeat_loss_s

    def home_position(self) -> Optional[HomeFix]:
        with self._cv:
            return self._home_fix

    def get_home(self, timeout: float = 5.0) -> Optional[GeoPoint]:
        with self._cv:
            if self._home is not None:
                return self._home.model_copy()
        if self._conn is None or timeout <= 0:
            return None
        self._send_command_nowait(MAV_CMD_REQUEST_MESSAGE, _MSG_HOME_POSITION)
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._home is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(min(remaining, 0.2))
            return self._home.model_copy()

    # ------------------------------------------------------------------ reader thread
    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            conn = self._conn
            if conn is None:
                break
            try:
                msg = conn.recv_match(blocking=True, timeout=self.RECV_POLL_S)
            except Exception as exc:  # serial unplugged, socket error, ...
                log.warning("MAVLink read error: %s", exc)
                self._stop.wait(1.0)
                self._reconnect()
                continue
            now = time.time()
            if msg is not None:
                try:
                    self._dispatch(msg, now)
                except Exception:  # never let a malformed message kill the reader
                    log.exception("error handling %s", msg.get_type())
            try:
                self._housekeeping(now)
            except Exception:
                log.exception("housekeeping error")

    def _dispatch(self, msg, now: float) -> None:
        mtype = msg.get_type()
        if mtype == "BAD_DATA":
            return
        src_sys = msg.get_srcSystem()
        if src_sys == self.source_system:
            return  # our own packets echoed back
        if mtype == "HEARTBEAT":
            self._on_heartbeat(msg, now)
            return
        if self._sysid is None or src_sys != self._sysid:
            return

        with self._cv:
            tel = self._tel
            if mtype in _RAW_KEEP:
                try:
                    self._raw[mtype] = (now, msg.to_dict())
                except Exception:  # noqa: BLE001
                    pass
            if mtype == "GLOBAL_POSITION_INT":
                tel.lat = msg.lat / 1e7
                tel.lon = msg.lon / 1e7
                tel.alt_rel_m = msg.relative_alt / 1000.0
                if msg.hdg != 65535:
                    tel.heading_deg = msg.hdg / 100.0
                self._last_gpi = now
            elif mtype == "VFR_HUD":
                tel.groundspeed_ms = float(msg.groundspeed)
            elif mtype == "SYS_STATUS":
                tel.battery_pct = None if msg.battery_remaining < 0 else float(msg.battery_remaining)
                tel.voltage_v = None if msg.voltage_battery in (65535, 0xFFFF) else msg.voltage_battery / 1000.0
            elif mtype == "GPS_RAW_INT":
                tel.gps_fix = int(msg.fix_type)
                tel.satellites = None if msg.satellites_visible == 255 else int(msg.satellites_visible)
                tel.hdop = None if msg.eph in (65535, 0xFFFF) else round(msg.eph / 100.0, 2)
                if tel.gps_fix >= 3 and (self._last_fix is None or self._last_fix < 3):
                    self._home_req_due += [now + 1.0, now + 5.0]   # home is set shortly after the first 3D fix
                self._last_fix = tel.gps_fix
            elif mtype == "MISSION_CURRENT":
                tel.mission_seq = int(msg.seq)
                total = getattr(msg, "total", 0)
                if total not in (0, 65535):
                    tel.mission_count = int(total)
            elif mtype == "HOME_POSITION":
                self._on_home(msg, now)
            elif mtype == "EXTENDED_SYS_STATE":
                tel.landed_state = LANDED_STATES.get(int(msg.landed_state), str(msg.landed_state))
            elif mtype == "STATUSTEXT":
                text = msg.text if isinstance(msg.text, str) else bytes(msg.text).decode("utf-8", "replace")
                text = text.rstrip("\x00")
                self._statustext.append(text)
                line = self._status_log.add(text, int(msg.severity), now)
                if line is not None:
                    log.info("vehicle: %s", line)
            elif mtype in self._proto:
                self._proto[mtype].append(msg)
            tel.ts = now
            self._cv.notify_all()

    def _on_home(self, msg, now: float) -> None:
        """Called with self._cv held."""
        lat, lon, amsl = msg.latitude / 1e7, msg.longitude / 1e7, msg.altitude / 1000.0
        if lat == 0.0 and lon == 0.0:
            return  # not set yet
        old = self._home_fix
        changed = old is None or abs(old.lat - lat) > 1e-7 or abs(old.lon - lon) > 1e-7 or abs(old.amsl_m - amsl) > 0.05
        if not changed:
            self._home_fix = HomeFix(old.lat, old.lon, old.amsl_m, now, old.version)
            return
        self.home_version += 1
        new = HomeFix(lat, lon, amsl, now, self.home_version)
        self._home_fix = new
        self._home = new.geopoint()
        self.home_amsl_m = amsl
        if old is None:
            log.info("home position: %.7f, %.7f, %.1f m AMSL", lat, lon, amsl)
        else:
            log.warning("home position changed by %.1f m (alt %+.1f m): %.7f, %.7f, %.1f m AMSL",
                        home_shift_m(old, new), amsl - old.amsl_m, lat, lon, amsl)
        cb = self.on_home_changed
        if cb is not None:
            self._fire(lambda: cb(old, new))

    def _on_heartbeat(self, msg, now: float) -> None:
        if msg.type in _NON_VEHICLE_TYPES or msg.autopilot == MAV_AUTOPILOT_INVALID:
            return
        src_sys, src_comp = msg.get_srcSystem(), msg.get_srcComponent()
        restored = False
        first = False
        with self._cv:
            if self._sysid is None:
                self._sysid, self._compid = src_sys, src_comp
                first = True
            elif src_sys != self._sysid or src_comp != self._compid:
                return
            tel = self._tel
            tel.last_heartbeat_ts = now
            was_armed = tel.armed
            tel.armed = bool(msg.base_mode & MAV_MODE_FLAG_SAFETY_ARMED)
            tel.system_status = SYSTEM_STATUS.get(int(msg.system_status), str(msg.system_status))
            tel.vehicle_type = MAV_TYPES.get(int(msg.type), f"TYPE_{int(msg.type)}")
            try:
                self._raw["HEARTBEAT"] = (now, msg.to_dict())
            except Exception:  # noqa: BLE001
                pass
            if tel.armed and not was_armed:
                # ArduPilot resets home to the current position when arming
                self._home_req_due += [now + 0.3, now + 2.0]
            if msg.type in _COPTER_TYPES:
                tel.mode = ARDUCOPTER_MODE_NAMES.get(int(msg.custom_mode), f"MODE_{int(msg.custom_mode)}")
            else:
                tel.mode = "UNKNOWN"
            tel.connected = True
            if not self._link_up:
                self._link_up = True
                restored = self._ever_up
                self._ever_up = True
            self._cv.notify_all()
        if first or restored:
            self._request_streams()
        if restored:
            log.warning("MAVLink link restored")
            self._fire(self.on_link_restored)

    def _housekeeping(self, now: float) -> None:
        # Our own heartbeat, 1 Hz.
        if now - self._last_hb_sent >= self.HEARTBEAT_PERIOD_S:
            self._last_hb_sent = now
            self._send(
                "heartbeat_send", MAV_TYPE_ONBOARD_CONTROLLER, MAV_AUTOPILOT_INVALID, 0, 0, MAV_STATE_ACTIVE
            )
        # Link health.
        with self._cv:
            hb = self._tel.last_heartbeat_ts
            up = self._link_up
        if up and hb is not None and now - hb >= self.settings.heartbeat_loss_s:
            with self._cv:
                self._link_up = False
                self._tel.connected = False
                self._cv.notify_all()
            log.error("MAVLink heartbeat lost (%.1fs)", now - hb)
            self._fire(self.on_link_lost)
        # HOME_POSITION: scheduled requests (arming / first fix), and every 5 s while unknown.
        if self._link_up and self._sysid is not None:
            due = [t for t in self._home_req_due if t <= now]
            if due or (self._home_fix is None and now - self._last_home_req > 5.0):
                self._home_req_due = [t for t in self._home_req_due if t > now]
                self._last_home_req = now
                self._send_command_nowait(MAV_CMD_REQUEST_MESSAGE, _MSG_HOME_POSITION)
        # Re-ask for streams if position data dried up while the link is alive.
        if self._link_up and now - self._last_gpi > 3.0 and now - self._last_stream_req > 5.0:
            self._request_streams()
        # Prolonged loss on a stream-oriented link: reopen the socket/port.
        if not self._link_up and self._ever_up and hb is not None:
            if now - hb > self.RECONNECT_AFTER_S and now - self._last_reconnect > self.RECONNECT_AFTER_S:
                scheme, _ = _split_url(self.settings.mavlink_url)
                if scheme != "udpin":  # a listening UDP socket needs no reconnect
                    self._reconnect()

    def _reconnect(self) -> None:
        self._last_reconnect = time.time()
        if self._stop.is_set():
            return
        self.reconnect_count += 1
        log.warning("reconnecting MAVLink (attempt %d)", self.reconnect_count)
        try:
            new = self._open_connection()
        except Exception as exc:
            log.error("reconnect failed: %s", exc)
            return
        with self._send_lock:
            old, self._conn = self._conn, new
        if old is not None:
            try:
                old.close()
            except Exception:
                pass

    def _fire(self, cb: Optional[Callable[[], None]]) -> None:
        """Run a user callback off the reader thread so it may call rtl() etc. safely."""
        if cb is None:
            return

        def _run() -> None:
            try:
                cb()
            except Exception:
                log.exception("link callback raised")

        threading.Thread(target=_run, name="mavlink-callback", daemon=True).start()

    def _request_streams(self) -> None:
        self._last_stream_req = time.time()
        if self._sysid is None:
            return
        # Legacy request (works on every ArduPilot) + per-message intervals.
        self._send("request_data_stream_send", self._sysid, self._compid, 0, 4, 1)
        for msg_id, hz in (
            (_MSG_GLOBAL_POSITION_INT, 5), (_MSG_VFR_HUD, 5), (_MSG_SYS_STATUS, 2),
            (_MSG_GPS_RAW_INT, 2), (_MSG_MISSION_CURRENT, 2), (_MSG_HOME_POSITION, 0.2),
            (_MSG_EKF_STATUS_REPORT, 1), (_MSG_EXTENDED_SYS_STATE, 1), (_MSG_RC_CHANNELS, 1),
        ):
            self._send_command_nowait(MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)
        if "AUTOPILOT_VERSION" not in self._raw:
            self._send_command_nowait(MAV_CMD_REQUEST_MESSAGE, _MSG_AUTOPILOT_VERSION)
        self._last_home_req = time.time()
        self._send_command_nowait(MAV_CMD_REQUEST_MESSAGE, _MSG_HOME_POSITION)

    # ------------------------------------------------------------------ send helpers
    def _send(self, method: str, *args) -> None:
        if self.access != "control":
            allowed = method in _RO_SEND_METHODS or (
                method == "command_long_send" and len(args) > 2 and int(args[2]) in _RO_COMMANDS) or (
                self.access == "upload" and method in _UPLOAD_SEND_METHODS)
            if not allowed:
                raise ReadOnlyViolation(f"{self.access} vehicle link: refusing to send {method} {args[2:3]}")
        conn = self._conn
        if conn is None:
            raise VehicleError("not connected")
        with self._send_lock:
            try:
                getattr(conn.mav, method)(*args)
            except OSError as exc:
                raise VehicleError(f"send failed ({method}): {exc}") from exc

    def _send_command_nowait(self, command: int, *params: float, confirmation: int = 0) -> None:
        p = list(params) + [0.0] * (7 - len(params))
        self._send(
            "command_long_send", self._sysid or 0, self._compid, command, confirmation, *p[:7]
        )

    # ------------------------------------------------------------------ waiting helpers
    def _clear(self, *types: str) -> None:
        with self._cv:
            for t in types:
                self._proto[t].clear()

    def _wait_any(
        self,
        types: tuple[str, ...],
        timeout: float,
        pred: Optional[Callable[[object], bool]] = None,
    ):
        """Pop the next queued protocol message of `types` accepted by `pred`, or None."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                for t in types:
                    q = self._proto[t]
                    while q:
                        m = q.popleft()
                        if pred is None or pred(m):
                            return m
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)

    def _wait_state(self, pred: Callable[[Telemetry], bool], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            while not pred(self._tel):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(remaining)
            return True

    def _refuse_readonly(self, action: str, need: str = "control") -> None:
        """Raise before any I/O unless the access level permits `need` ("upload" or "control")."""
        if _ACCESS_RANK[self.access] < _ACCESS_RANK[need]:
            mode = {"readonly": "read-only", "upload": "upload-only"}[self.access]
            raise ReadOnlyViolation(
                f"{mode} mode: {action} refused (VEHICLE_MODE={self.access}). Nothing was sent to the vehicle; "
                f"use the RC transmitter / ground station for control.")

    def _refuse_if_armed(self, action: str) -> None:
        if self.telemetry().armed:
            raise VehicleError(f"{action} refused: the vehicle is ARMED (a mission must not change under an armed vehicle)")

    def _require_connected(self) -> None:
        if self._conn is None or self._sysid is None:
            raise VehicleError("not connected to a vehicle (call connect() first)")

    def _recent_status(self) -> str:
        with self._cv:
            return " | ".join(list(self._statustext)[-3:])

    # ------------------------------------------------------------------ commands
    def _command(self, command: int, *params: float, name: str = "") -> None:
        """COMMAND_LONG with ACK wait, resend (confirmation++) on timeout."""
        if command not in _RO_COMMANDS:
            self._refuse_readonly(name or f"command {command}")
        self._require_connected()
        label = name or f"command {command}"
        with self._op_lock:
            for attempt in range(self.COMMAND_RETRIES):
                self._clear("COMMAND_ACK")
                self._send_command_nowait(command, *params, confirmation=attempt)
                ack = self._wait_any(
                    ("COMMAND_ACK",), self.timeout_s, lambda m: m.command == command
                )
                if ack is None:
                    log.warning("%s: no COMMAND_ACK (attempt %d/%d)", label, attempt + 1, self.COMMAND_RETRIES)
                    continue
                if ack.result == MAV_RESULT_IN_PROGRESS:
                    ack = self._wait_any(
                        ("COMMAND_ACK",), 10.0,
                        lambda m: m.command == command and m.result != MAV_RESULT_IN_PROGRESS,
                    )
                    if ack is None:
                        raise LinkTimeout(f"{label}: stuck IN_PROGRESS for 10s")
                if ack.result == MAV_RESULT_ACCEPTED:
                    return
                status = self._recent_status()
                raise CommandRejected(
                    f"{label} rejected: {_RESULT_NAMES.get(ack.result, ack.result)}"
                    + (f" [{status}]" if status else ""),
                    result=int(ack.result),
                )
            raise LinkTimeout(f"{label}: no COMMAND_ACK after {self.COMMAND_RETRIES} attempts")

    def get_param(self, name: str, timeout: Optional[float] = None) -> Optional[float]:
        """PARAM_REQUEST_READ (read-only; allowed in readonly mode). None if the vehicle does not
        answer (ArduPilot ignores unknown parameter names)."""
        self._require_connected()
        pid = name.encode()
        if len(pid) > 16:
            raise ValueError(f"parameter name too long: {name!r}")
        t = self.timeout_s if timeout is None else timeout

        def match(m, _n=name) -> bool:
            pid_ = m.param_id if isinstance(m.param_id, str) else bytes(m.param_id).decode()
            return pid_.rstrip("\x00") == _n

        with self._op_lock:
            for attempt in range(self.COMMAND_RETRIES):
                self._clear("PARAM_VALUE")
                self._send("param_request_read_send", self._sysid, self._compid, pid, -1)
                reply = self._wait_any(("PARAM_VALUE",), t / self.COMMAND_RETRIES if attempt < 2 else t, match)
                if reply is not None:
                    return float(reply.param_value)
        log.warning("get_param %s: no PARAM_VALUE reply", name)
        return None

    def set_param(self, name: str, value: float) -> None:
        self._refuse_readonly(f"set_param {name}")
        self._require_connected()
        pid = name.encode()
        if len(pid) > 16:
            raise ValueError(f"parameter name too long: {name!r}")
        with self._op_lock:
            for attempt in range(self.COMMAND_RETRIES):
                self._clear("PARAM_VALUE")
                self._send(
                    "param_set_send", self._sysid, self._compid, pid, float(value),
                    MAV_PARAM_TYPE_REAL32,
                )

                def match(m, _n=name) -> bool:
                    pid_ = m.param_id if isinstance(m.param_id, str) else bytes(m.param_id).decode()
                    return pid_.rstrip("\x00") == _n

                reply = self._wait_any(("PARAM_VALUE",), self.timeout_s, match)
                if reply is None:
                    log.warning("set_param %s: no PARAM_VALUE (attempt %d)", name, attempt + 1)
                    continue
                if abs(reply.param_value - value) > 1e-4 * max(1.0, abs(value)):
                    raise CommandRejected(
                        f"set_param {name}={value} not applied (vehicle reports {reply.param_value})"
                    )
                log.info("param %s = %s", name, reply.param_value)
                return
            raise LinkTimeout(f"set_param {name}: no PARAM_VALUE reply")

    def set_mode(self, name: str) -> None:
        mode = name.strip().upper()
        self._refuse_readonly(f"set_mode {mode}")
        if mode not in ARDUCOPTER_MODES:
            raise ValueError(f"unknown ArduCopter mode {name!r}; known: {sorted(ARDUCOPTER_MODES)}")
        with self._op_lock:
            self._command(
                MAV_CMD_DO_SET_MODE, MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, ARDUCOPTER_MODES[mode],
                name=f"set_mode {mode}",
            )
            if not self._wait_state(lambda t: t.mode == mode, self.timeout_s):
                raise LinkTimeout(
                    f"set_mode {mode}: HEARTBEAT still reports {self.telemetry().mode} "
                    f"[{self._recent_status()}]"
                )
        log.info("mode -> %s", mode)

    def arm(self) -> None:
        self._refuse_readonly("arm")
        allowed, reason = arm_guard(self.settings)
        if not allowed:
            log.error(reason)
            raise ArmRefused(reason)
        log.warning("ARMING (%s)", reason)
        with self._op_lock:
            self._command(MAV_CMD_COMPONENT_ARM_DISARM, 1, 0, name="arm")
            if not self._wait_state(lambda t: t.armed, self.timeout_s):
                raise LinkTimeout(f"arm accepted but HEARTBEAT not armed [{self._recent_status()}]")

    def disarm(self, force: bool = False) -> None:
        self._refuse_readonly("disarm")
        with self._op_lock:
            self._command(
                MAV_CMD_COMPONENT_ARM_DISARM, 0, 21196 if force else 0, name="disarm"
            )
            if not self._wait_state(lambda t: not t.armed, self.timeout_s):
                raise LinkTimeout("disarm accepted but HEARTBEAT still armed")

    def start_mission(self) -> None:
        self._refuse_readonly("mission start")
        self._command(MAV_CMD_MISSION_START, 0, 0, name="mission start")

    def rtl(self) -> None:
        """Return to launch. Prefers the RTL flight mode, falls back to NAV_RETURN_TO_LAUNCH."""
        self._refuse_readonly("RTL")
        try:
            self.set_mode("RTL")
            return
        except VehicleError as exc:
            log.error("set_mode RTL failed (%s); trying MAV_CMD_NAV_RETURN_TO_LAUNCH", exc)
        self._command(MAV_CMD_NAV_RETURN_TO_LAUNCH, name="rtl command")

    # ------------------------------------------------------------------ mission building
    def build_mission_items(self, plan: SurveyPlan) -> list[MissionItem]:
        """home placeholder, TAKEOFF, CHANGE_SPEED, survey waypoints, RTL."""
        if plan.altitude_m > HARD_MAX_ALT_M:
            raise ValueError(f"plan altitude {plan.altitude_m} m exceeds hard limit {HARD_MAX_ALT_M} m")
        if plan.altitude_m <= 0:
            raise ValueError("plan altitude must be positive")
        if not plan.waypoints:
            raise ValueError("plan has no waypoints")
        rel = MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
        items = [
            # seq 0: ArduPilot's home slot (ignored on upload, replaced by the real home).
            MissionItem(seq=0, command=MAV_CMD_NAV_WAYPOINT, frame=MAV_FRAME_GLOBAL,
                        lat=plan.home.lat, lon=plan.home.lon, alt=0.0),
            MissionItem(seq=1, command=MAV_CMD_NAV_TAKEOFF, frame=rel, alt=plan.altitude_m),
            MissionItem(seq=2, command=MAV_CMD_DO_CHANGE_SPEED, frame=MAV_FRAME_MISSION,
                        param1=1.0, param2=plan.speed_ms, param3=-1.0, param4=0.0),
        ]
        for wp in plan.waypoints:
            items.append(
                MissionItem(seq=len(items), command=MAV_CMD_NAV_WAYPOINT, frame=rel,
                            lat=wp.lat, lon=wp.lon, alt=wp.alt_m)
            )
        items.append(MissionItem(seq=len(items), command=MAV_CMD_NAV_RETURN_TO_LAUNCH, frame=MAV_FRAME_MISSION))
        return items

    # ------------------------------------------------------------------ mission protocol
    def _send_item(self, it: MissionItem, mission_type: int = MAV_MISSION_TYPE_MISSION) -> None:
        self._send(
            "mission_item_int_send", self._sysid, self._compid, it.seq, it.frame, it.command,
            0, it.autocontinue,  # `current` flag is unused on upload
            it.param1, it.param2, it.param3, it.param4,
            int(round(it.lat * 1e7)), int(round(it.lon * 1e7)), float(it.alt), mission_type,
        )

    def upload_mission(self, items: list[MissionItem], mission_type: int = MAV_MISSION_TYPE_MISSION) -> None:
        self._refuse_readonly("mission upload", need="upload")
        self._require_connected()
        self._refuse_if_armed("mission upload")      # every mode: a mission must never change under an armed vehicle
        mt = mission_type
        if not items:
            raise ValueError("cannot upload an empty mission")
        items = sorted(items, key=lambda i: i.seq)
        if [i.seq for i in items] != list(range(len(items))):
            raise ValueError("mission item seq values must be 0..N-1 without gaps")
        n = len(items)
        with self._op_lock:
            self._clear("MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK")
            last_sent: Optional[int] = None  # seq of the item we answered last
            idle_retries = 0
            self._send("mission_count_send", self._sysid, self._compid, n, mt)
            while True:
                if self._cancel_fly.is_set():
                    raise VehicleError("mission upload cancelled by the operator")
                m = self._wait_any(("MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"), self.timeout_s,
                                   lambda x: getattr(x, "mission_type", 0) == mt)
                if m is None:
                    idle_retries += 1
                    if idle_retries > self.ITEM_RETRIES:
                        raise LinkTimeout(
                            f"mission upload stalled after item {last_sent} of {n} "
                            f"(no request/ack for {self.timeout_s:.1f}s x{self.ITEM_RETRIES})"
                        )
                    log.warning("upload: timeout, retry %d (last item %s)", idle_retries, last_sent)
                    if last_sent is None:
                        self._send("mission_count_send", self._sysid, self._compid, n, mt)
                    else:
                        self._send_item(items[last_sent], mt)
                    continue
                idle_retries = 0
                if m.get_type() == "MISSION_ACK":
                    if m.type == MAV_MISSION_ACCEPTED:
                        if last_sent != n - 1:
                            log.warning("MISSION_ACK before last item was sent; ignoring")
                            continue
                        break
                    raise CommandRejected(
                        f"mission upload rejected: {_MISSION_ACK_NAMES.get(m.type, m.type)}", result=int(m.type)
                    )
                seq = int(m.seq)
                if not 0 <= seq < n:
                    raise VehicleError(f"vehicle requested mission item {seq}, outside 0..{n - 1}")
                self._send_item(items[seq], mt)
                last_sent = seq
        if mt == MAV_MISSION_TYPE_MISSION:
            with self._cv:
                self._tel.mission_count = n
        log.info("uploaded %d %s items", n, "mission" if mt == 0 else f"type-{mt}")

    def clear_mission(self, mission_type: int = MAV_MISSION_TYPE_MISSION) -> None:
        """MISSION_CLEAR_ALL (upload access). Refused while armed in upload mode."""
        self._refuse_readonly("mission clear", need="upload")
        self._require_connected()
        self._refuse_if_armed("mission clear")
        with self._op_lock:
            for attempt in range(self.COMMAND_RETRIES):
                self._clear("MISSION_ACK")
                self._send("mission_clear_all_send", self._sysid, self._compid, mission_type)
                m = self._wait_any(("MISSION_ACK",), self.timeout_s, lambda x: getattr(x, "mission_type", 0) == mission_type)
                if m is None:
                    continue
                if m.type != MAV_MISSION_ACCEPTED:
                    raise CommandRejected(f"mission clear rejected: {_MISSION_ACK_NAMES.get(m.type, m.type)}", int(m.type))
                if mission_type == MAV_MISSION_TYPE_MISSION:
                    with self._cv:
                        self._tel.mission_count = 0
                log.info("cleared mission (type %d)", mission_type)
                return
        raise LinkTimeout("mission clear: no MISSION_ACK")

    def download_mission(self, mission_type: int = MAV_MISSION_TYPE_MISSION) -> list[MissionItem]:
        self._refuse_readonly("mission download", need="readonly")     # a read: allowed in every mode
        self._require_connected()
        mt = mission_type
        with self._op_lock:
            n: Optional[int] = None
            for attempt in range(self.ITEM_RETRIES):
                self._clear("MISSION_COUNT")
                self._send("mission_request_list_send", self._sysid, self._compid, mt)
                m = self._wait_any(("MISSION_COUNT",), self.timeout_s, lambda x: getattr(x, "mission_type", 0) == mt)
                if m is not None:
                    n = int(m.count)
                    break
                log.warning("download: no MISSION_COUNT (attempt %d)", attempt + 1)
            if n is None:
                raise LinkTimeout("download_mission: no MISSION_COUNT reply")
            items: list[MissionItem] = []
            for seq in range(n):
                got: Optional[MissionItem] = None
                for attempt in range(self.ITEM_RETRIES):
                    self._clear("MISSION_ITEM_INT")
                    self._send("mission_request_int_send", self._sysid, self._compid, seq, mt)
                    m = self._wait_any(("MISSION_ITEM_INT",), self.timeout_s,
                                       lambda x, s=seq: x.seq == s and getattr(x, "mission_type", 0) == mt)
                    if m is not None:
                        got = MissionItem(
                            seq=int(m.seq), command=int(m.command), frame=int(m.frame),
                            param1=float(m.param1), param2=float(m.param2),
                            param3=float(m.param3), param4=float(m.param4),
                            lat=m.x / 1e7, lon=m.y / 1e7, alt=float(m.z),
                            autocontinue=int(m.autocontinue),
                        )
                        break
                    log.warning("download: no item %d (attempt %d)", seq, attempt + 1)
                if got is None:
                    raise LinkTimeout(f"download_mission: item {seq}/{n} never arrived")
                items.append(got)
            self._send("mission_ack_send", self._sysid, self._compid, MAV_MISSION_ACCEPTED, mt)
        return items

    def verify_mission(self, items: list[MissionItem], mission_type: int = MAV_MISSION_TYPE_MISSION,
                       readback: Optional[list] = None) -> bool:
        """Read the mission back and compare command/lat/lon/alt. Item 0 (home slot) is
        compared by command only: ArduPilot substitutes the real home position.
        `readback` (a list) receives the downloaded items."""
        got = self.download_mission(mission_type)
        if readback is not None:
            readback.extend(got)
        want = sorted(items, key=lambda i: i.seq)
        if len(got) != len(want):
            log.error("verify: item count %d != %d", len(got), len(want))
            return False
        for a, b in zip(want, got):
            if a.command != b.command:
                log.error("verify: seq %d command %d != %d", a.seq, a.command, b.command)
                return False
            if a.seq == 0 and mission_type == MAV_MISSION_TYPE_MISSION:
                continue
            if (
                abs(a.lat - b.lat) > POS_TOL_DEG or abs(a.lon - b.lon) > POS_TOL_DEG or abs(a.alt - b.alt) > ALT_TOL_M
            ):
                log.error("verify: seq %d position mismatch %s vs %s", a.seq, (a.lat, a.lon, a.alt), (b.lat, b.lon, b.alt))
                return False
        return True

    # ------------------------------------------------------------------ checks / review upload
    def run_prearm_checks(self, wait_s: float = 3.0) -> dict:
        """MAV_CMD_RUN_PREARM_CHECKS (401, does NOT arm; allowed read-only) and collect the PreArm/Arm
        STATUSTEXT lines for `wait_s`."""
        self._require_connected()
        t0 = time.time()
        ack: Optional[str] = None
        try:
            self._command(MAV_CMD_RUN_PREARM_CHECKS, name="run prearm checks")
            ack = "ACCEPTED"
        except CommandRejected as exc:
            ack = _RESULT_NAMES.get(exc.result, str(exc.result))   # ArduPilot: DENIED/FAILED when checks fail
        except LinkTimeout:
            ack = "NO_ACK"
        time.sleep(max(0.0, wait_s - (time.time() - t0)))
        lines = [d for d in self.statustexts(since=t0 - 0.5)
                 if d["text"].startswith(("PreArm", "Arm", "PREARM", "prearm"))]
        return {"ack": ack, "messages": lines}

    @staticmethod
    def build_fence_items(corners: list[tuple[float, float]]) -> list[MissionItem]:
        """Inclusion polygon (MAV_MISSION_TYPE_FENCE) from (lat, lon) corners. Does not touch FENCE_* params."""
        n = len(corners)
        if n < 3:
            raise ValueError("fence polygon needs >= 3 vertices")
        return [MissionItem(seq=i, command=MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION, frame=MAV_FRAME_GLOBAL,
                            param1=float(n), lat=lat, lon=lon, alt=0.0) for i, (lat, lon) in enumerate(corners)]

    def upload_for_review(self, plan: SurveyPlan, fence_corners: Optional[list[tuple[float, float]]] = None) -> dict:
        """Upload + read-back verify the mission (and optionally an inclusion fence polygon) WITHOUT changing
        mode, arming or starting anything. Upload access; refused while armed."""
        self._refuse_readonly("mission upload", need="upload")
        self._require_connected()
        self._refuse_if_armed("mission upload")
        items = self.build_mission_items(plan)
        self.upload_mission(items)
        readback: list[MissionItem] = []
        ok = self.verify_mission(items, readback=readback)
        out = {"items": len(items), "verified": ok, "readback": readback, "fence": None}
        if fence_corners:
            fitems = self.build_fence_items(fence_corners)
            self.upload_mission(fitems, mission_type=MAV_MISSION_TYPE_FENCE)
            fback: list[MissionItem] = []
            fok = self.verify_mission(fitems, mission_type=MAV_MISSION_TYPE_FENCE, readback=fback)
            out["fence"] = {"vertices": len(fitems), "verified": fok}
        return out

    # ------------------------------------------------------------------ high level
    def cancel_flight(self) -> None:
        self._cancel_fly.set()

    def _check_cancel(self, step: str) -> None:
        if self._cancel_fly.is_set():
            raise VehicleError(f"cancelled by the operator before {step}; not arming")

    def reset_cancel(self) -> None:
        self._cancel_fly.clear()

    def _flight_precheck(self, what: str) -> None:
        self._refuse_readonly(what)
        self._require_connected()
        if not self.heartbeat_ok():
            raise VehicleError("refusing to fly: no recent HEARTBEAT")
        fix = self.telemetry().gps_fix
        if fix is not None and fix < 3:
            raise VehicleError(f"refusing to fly: GPS fix type {fix} (<3)")

    def fly_plan(self, plan: SurveyPlan) -> None:
        """upload -> verify -> AUTO_OPTIONS=3 -> mode AUTO -> arm -> MISSION_START.

        Raises (and does not arm) on any failure before the arm step. Arming is still
        subject to the arm guard.
        """
        self._refuse_readonly("fly_plan")
        self._cancel_fly.clear()
        self._flight_precheck("fly_plan")
        items = self.build_mission_items(plan)
        self.upload_mission(items)
        self._check_cancel("verify")
        if not self.verify_mission(items):
            raise VehicleError("mission read-back verification failed; not arming")
        self._arm_and_start(items)

    def start_staged(self, items: list[MissionItem]) -> None:
        """GO (notes/go-flow.md): the mission was uploaded + verified earlier by upload_for_review and the operator
        reviewed it. Read it back once more (a Mission Planner edit since then = refuse) -> AUTO_OPTIONS=3 -> AUTO ->
        arm -> MISSION_START. Nothing is uploaded. Does NOT clear a pending cancel (see reset_cancel). Raises (and
        does not arm) on any failure before the arm step; arming is still subject to the arm guard."""
        self._flight_precheck("start_staged")
        self.go_armed = False
        if self.telemetry().armed:           # armed by someone else (RC / ground station): not our flight to start
            raise VehicleError("GO refused: the vehicle is already ARMED (by the RC transmitter or a ground station)")
        self._check_cancel("verify")
        if not self.verify_mission(items):
            raise VehicleError("the mission on the vehicle no longer matches the staged plan; not arming")
        self._arm_and_start(items)

    def _arm_and_start(self, items: list[MissionItem]) -> None:
        # AUTO_OPTIONS bit0: allow arming in AUTO; bit1: take off without throttle raise.
        self._check_cancel("setting AUTO")
        if not self.heartbeat_ok():
            raise VehicleError("link lost during upload; not arming")
        self.set_param("AUTO_OPTIONS", 3)
        self._check_cancel("setting AUTO")
        self.set_mode("AUTO")
        self._check_cancel("arming")
        self.arm()
        self.go_armed = True                         # remembered: telemetry may already show disarmed (RTL + land)
        self._check_cancel("starting the mission")   # armed already: the controller sees the error and commands RTL
        self.start_mission()
        log.info("mission started: %d items", len(items))


# ---------------------------------------------------------------------------
# Pre-flight parameter cross-check (Phase 7) - PARAM_REQUEST_READ only
# ---------------------------------------------------------------------------
PREFLIGHT_PARAMS = ("RTL_ALT", "FENCE_ENABLE", "FENCE_TYPE", "FENCE_ALT_MAX", "FENCE_RADIUS", "FENCE_ACTION",
                    "BATT_FS_LOW_ACT", "FS_BATT_ENABLE", "FS_THR_ENABLE", "WPNAV_SPEED")
# ArduCopter 4.8 renamed these to SI units; read the new name when the old one does not answer.
PARAM_ALIASES = {"RTL_ALT": ("RTL_ALT_M", 100.0), "WPNAV_SPEED": ("WP_SPD", 100.0)}
_BATT_ACTIONS = {0: "None", 1: "Land", 2: "RTL", 3: "SmartRTL or RTL", 4: "SmartRTL or Land", 5: "Terminate",
                 6: "Auto DO_LAND_START or RTL", 7: "Brake or Land"}
_BATT_OK = {1, 2, 3, 4, 6, 7}
_FENCE_ACTIONS = {0: "Report only", 1: "RTL or Land", 2: "Always Land", 3: "SmartRTL or RTL or Land",
                  4: "Brake or Land", 5: "SmartRTL or Land"}
_THR_ACTIONS = {0: "Disabled", 1: "Always RTL", 2: "Continue mission in AUTO", 3: "Always Land",
                4: "SmartRTL or RTL", 5: "SmartRTL or Land", 6: "Auto DO_LAND_START or RTL", 7: "Brake or Land"}


def _plan_max_dist(plan: SurveyPlan) -> float:
    import math
    from survey import geo_to_local
    pts = [(w.lat, w.lon) for w in plan.waypoints]
    if plan.mission.corners:
        pts += [(c.lat, c.lon) for c in plan.mission.corners]
    return max((math.hypot(*geo_to_local(a, b, plan.home)) for a, b in pts), default=0.0)


def evaluate_preflight(params: dict[str, Optional[float]], settings: Settings,
                       plan: Optional[SurveyPlan] = None) -> PreflightResult:
    """Pure comparison of read parameter values with .env limits (table in notes/phase7-requirements.md)."""
    from safety import ceiling_info, effective_min_alt
    params = dict(params)
    names = {"RTL_ALT": "RTL_ALT", "WPNAV_SPEED": "WPNAV_SPEED"}
    for old, (new, factor) in PARAM_ALIASES.items():
        if params.get(old) is None and params.get(new) is not None:
            params[old] = params[new] * factor      # internal: legacy units (cm, cm/s)
            names[old] = new
    ceil, src = ceiling_info(settings)
    clearance = effective_min_alt(settings)
    rows: list[PreflightCheck] = []

    def row(param: str, req: str, ok: bool, msg: str, sev: str = "error") -> None:
        name = names.get(param, param)
        rows.append(PreflightCheck(param=name, value=params.get(name), requirement=req, ok=ok,
                                   severity=sev, message=msg))

    def missing(param: str, req: str) -> bool:
        if params.get(param) is None:
            alias = PARAM_ALIASES.get(param)
            row(param, req, False, f"{param}{'/' + alias[0] if alias else ''} could not be read from the Pixhawk")
            return True
        return False

    req = f">= {clearance:.0f} m and <= {ceil:.0f} m ({src})"
    rtl_m: Optional[float] = None
    if not missing("RTL_ALT", req):
        rtl_m = params["RTL_ALT"] / 100.0
        if params["RTL_ALT"] <= 0:
            row("RTL_ALT", req, False, "RTL_ALT=0 returns at the current altitude; set it explicitly")
        elif rtl_m < clearance:
            row("RTL_ALT", req, False, f"RTL altitude {rtl_m:.0f} m is below the obstacle clearance {clearance:.0f} m")
        elif rtl_m > ceil:
            row("RTL_ALT", req, False, f"RTL altitude {rtl_m:.0f} m is above the {src} ceiling {ceil:.0f} m")
        else:
            row("RTL_ALT", req, True, f"RTL at {rtl_m:.0f} m")

    if not missing("FENCE_ENABLE", "1"):
        on = int(params["FENCE_ENABLE"]) == 1
        row("FENCE_ENABLE", "1", on, "fence enabled" if on else "geofence is DISABLED on the Pixhawk")

    req = "altitude (bit0) + circle (bit1)"
    if not missing("FENCE_TYPE", req):
        ft = int(params["FENCE_TYPE"])
        ok = (ft & 3) == 3
        row("FENCE_TYPE", req, ok, f"type bits {ft:#x}" + ("" if ok else ": altitude and circle fences must both be on"))

    req = f"<= {ceil:.0f} m"
    if not missing("FENCE_ALT_MAX", req):
        fam = params["FENCE_ALT_MAX"]
        if fam > ceil + 1e-6:
            row("FENCE_ALT_MAX", req, False, f"fence ceiling {fam:.0f} m is above the {src} ceiling {ceil:.0f} m")
        elif plan is not None and plan.altitude_m >= fam:
            row("FENCE_ALT_MAX", req, False, f"plan altitude {plan.altitude_m:.0f} m reaches the fence ceiling {fam:.0f} m")
        else:
            row("FENCE_ALT_MAX", req, True, f"fence ceiling {fam:.0f} m")

    req = f"<= {settings.geofence_radius_m:.0f} m"
    if not missing("FENCE_RADIUS", req):
        fr = params["FENCE_RADIUS"]
        far = _plan_max_dist(plan) if plan is not None else 0.0
        if fr > settings.geofence_radius_m + 1e-6:
            row("FENCE_RADIUS", req, False, f"fence radius {fr:.0f} m is larger than GEOFENCE_RADIUS_M {settings.geofence_radius_m:.0f} m")
        elif plan is not None and far >= fr:
            row("FENCE_RADIUS", req, False, f"plan reaches {far:.0f} m from home, outside the Pixhawk fence radius {fr:.0f} m")
        else:
            row("FENCE_RADIUS", req, True, f"fence radius {fr:.0f} m")

    req = "RTL or Land (not 0)"
    if not missing("FENCE_ACTION", req):
        fa = int(params["FENCE_ACTION"])
        row("FENCE_ACTION", req, fa != 0, _FENCE_ACTIONS.get(fa, f"action {fa}"))

    req = "RTL or Land"
    if params.get("BATT_FS_LOW_ACT") is not None:
        a = int(params["BATT_FS_LOW_ACT"])
        row("BATT_FS_LOW_ACT", req, a in _BATT_OK, "low-battery failsafe: " + _BATT_ACTIONS.get(a, str(a)))
    elif params.get("FS_BATT_ENABLE") is not None:
        a = int(params["FS_BATT_ENABLE"])
        row("FS_BATT_ENABLE", req, a in (1, 2), "battery failsafe: " + {0: "Disabled", 1: "Land", 2: "RTL"}.get(a, str(a)))
    else:
        row("BATT_FS_LOW_ACT", req, False, "battery failsafe parameter (BATT_FS_LOW_ACT / FS_BATT_ENABLE) could not be read")

    req = "enabled (RC loss)"
    if not missing("FS_THR_ENABLE", req):
        a = int(params["FS_THR_ENABLE"])
        if a == 0:
            row("FS_THR_ENABLE", req, False, "RC-loss (throttle) failsafe is DISABLED")
        elif a == 2:
            row("FS_THR_ENABLE", req, False, "RC loss: continue mission in AUTO (no RTL on RC loss during the search)", "warning")
        else:
            row("FS_THR_ENABLE", req, True, "RC loss: " + _THR_ACTIONS.get(a, str(a)))

    want = plan.speed_ms if plan is not None else settings.default_speed_ms
    req = f">= plan speed {want:.1f} m/s"
    if params.get("WPNAV_SPEED") is None:
        row("WPNAV_SPEED", req, False, "WPNAV_SPEED could not be read", "warning")
    else:
        v = params["WPNAV_SPEED"] / 100.0
        row("WPNAV_SPEED", req, v >= want - 1e-6,
            f"{v:.1f} m/s" + ("" if v >= want - 1e-6 else " (DO_CHANGE_SPEED overrides it in the mission, but RTL/resume use it)"),
            "warning")

    errors = [f"Pixhawk {r.param}: {r.message}" for r in rows if not r.ok and r.severity == "error"]
    warnings = [f"Pixhawk {r.param}: {r.message}" for r in rows if not r.ok and r.severity == "warning"]
    fence = None
    if params.get("FENCE_ENABLE") is not None:
        if int(params["FENCE_ENABLE"]) == 1:
            fence = (f"enabled, type {int(params.get('FENCE_TYPE') or 0):#x}, radius {params.get('FENCE_RADIUS') or 0:.0f} m, "
                     f"alt max {params.get('FENCE_ALT_MAX') or 0:.0f} m, action "
                     f"{_FENCE_ACTIONS.get(int(params.get('FENCE_ACTION') or 0), '?')}")
        else:
            fence = "DISABLED"
    return PreflightResult(ok=not errors, checks=rows, errors=errors, warnings=warnings,
                           effective_max_alt_m=ceil, ceiling_source=src, rtl_alt_m=rtl_m, fence=fence)


def read_preflight_params(backend: VehicleBackend, names: tuple[str, ...] = PREFLIGHT_PARAMS,
                          timeout: Optional[float] = None) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    hb = getattr(backend, "heartbeat_ok", None)

    def read(n: str) -> Optional[float]:
        if callable(hb) and not hb():          # link lost: do not wait a timeout per parameter (= "could not be read")
            return None
        try:
            return backend.get_param(n, timeout=timeout)
        except VehicleError as exc:
            log.warning("preflight: reading %s failed: %s", n, exc)
            return None

    renamed: set = getattr(backend, "_renamed_params", set())     # old names this firmware did not answer
    for n in names:
        if n == "FS_BATT_ENABLE" and out.get("BATT_FS_LOW_ACT") is not None:
            continue  # legacy name only needed on old firmware
        out[n] = None if n in renamed else read(n)
        if out[n] is None and n in PARAM_ALIASES:
            out[PARAM_ALIASES[n][0]] = read(PARAM_ALIASES[n][0])
            if out[PARAM_ALIASES[n][0]] is not None:
                renamed.add(n)                # next time ask for the new name directly (no timeout per check)
    try:
        backend._renamed_params = renamed
    except AttributeError:
        pass
    return out


def preflight_check(backend: VehicleBackend, settings: Optional[Settings] = None,
                    plan: Optional[SurveyPlan] = None, timeout: Optional[float] = None) -> PreflightResult:
    """Read the failsafe/fence parameters (PARAM_REQUEST_READ only - never PARAM_SET) and compare with .env."""
    s = settings or get_settings()
    try:
        params = read_preflight_params(backend, timeout=timeout)
    except VehicleError as exc:
        return PreflightResult(ok=False, errors=[f"Pre-flight check failed: {exc}"])
    return evaluate_preflight(params, s, plan)


def downgrade_preflight_for_bench(pre: PreflightResult) -> PreflightResult:
    """Upload test on the bench (BENCH_HOME, no GPS): the Pixhawk-state failures that are expected indoors (fence /
    battery / RC-loss parameters, plan vs fence ...) must not block an UPLOAD. Every failed error row becomes a warning
    row; the table is unchanged otherwise. Safety errors (safety.validate) are NOT touched by this."""
    rows = [r.model_copy(update={"severity": "warning"}) if (not r.ok and r.severity == "error") else r for r in pre.checks]
    tag = " (bench: would block a flight, does not block this upload)"
    warns = list(pre.warnings) + [e + tag for e in pre.errors]
    return pre.model_copy(update={"ok": True, "errors": [], "warnings": list(dict.fromkeys(warns)), "checks": rows,
                                  "home_source": "bench"})


__all__ = [
    "ARDUCOPTER_MODES", "ArduPilotBackend", "ArmRefused", "CommandRejected", "HomeFix", "LinkTimeout",
    "ReadOnlyViolation", "VehicleBackend", "VehicleError", "arm_guard", "create_backend", "evaluate_preflight",
    "downgrade_preflight_for_bench", "home_shift_m", "is_simulated_link", "preflight_check",
]
