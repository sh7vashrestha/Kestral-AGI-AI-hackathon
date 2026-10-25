#!/usr/bin/env python3
"""Pure-pymavlink fake ArduCopter for tests and bench work. No hardware, no serial.

It talks to *localhost UDP only* (refuses anything else) and speaks enough MAVLink for
drone-brain's mission.py: heartbeat, telemetry streams, mission upload/download, params,
arm/disarm, mode changes, MISSION_START, RTL, and a small kinematic sim that flies the
uploaded mission in AUTO, then RTL -> land -> auto-disarm.

Connection styles (pick the one matching the side you run mission.py on):

* default ``udpout:127.0.0.1:14550``: mock *sends* to a GCS that listens on 14550, i.e.
  mission.py's default ``MAVLINK_URL=udpin:0.0.0.0:14550`` works unchanged.
* ``udpin:127.0.0.1:14551``: mock listens, GCS uses ``udpout:127.0.0.1:14551``.

CLI::

    python scripts/mock_vehicle.py --url udpout:127.0.0.1:14550 --time-scale 5

As a module (tests)::

    mock = MockVehicle("udpout:127.0.0.1:%d" % port, time_scale=30)
    mock.start(); ...; mock.stop()
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil  # noqa: E402

log = logging.getLogger("mock_vehicle")

M_PER_DEG_LAT = 111_319.5

# custom_mode numbers (ArduCopter)
MODE_STABILIZE, MODE_ALT_HOLD, MODE_AUTO, MODE_GUIDED, MODE_LOITER, MODE_RTL, MODE_LAND = 0, 2, 3, 4, 5, 6, 9
_KNOWN_MODES = {0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}

MAV_RESULT_ACCEPTED, MAV_RESULT_DENIED, MAV_RESULT_UNSUPPORTED, MAV_RESULT_FAILED = 0, 2, 3, 4
MAV_MISSION_ACCEPTED, MAV_MISSION_INVALID_SEQUENCE, MAV_MISSION_NO_SPACE = 0, 13, 4


@dataclass
class _Item:
    seq: int
    frame: int
    command: int
    p1: float
    p2: float
    p3: float
    p4: float
    lat: float   # degrees
    lon: float
    alt: float


def _is_loopback_url(url: str) -> bool:
    scheme, _, rest = url.partition(":")
    host = rest.rsplit(":", 1)[0]
    return scheme in ("udpin", "udpout", "udp") and (host.startswith("127.") or host == "localhost")


class MockVehicle:
    """Fake ArduCopter. Thread-safe state reads via simple attributes."""

    def __init__(
        self,
        url: str = "udpout:127.0.0.1:14550",
        *,
        home: tuple[float, float] = (-35.363261, 149.165230),
        home_amsl_m: float = 584.0,
        time_scale: float = 1.0,
        sysid: int = 1,
        compid: int = 1,
        hb_rate_hz: float = 1.0,
        telemetry_hz: float = 10.0,
        climb_ms: float = 2.5,
        descend_ms: float = 1.5,
        rtl_speed_ms: float = 5.0,
        gps_fix: int = 3,
        hover: bool = False,
        hover_max_s: float = 0.0,
    ) -> None:
        if not _is_loopback_url(url):
            raise ValueError(f"mock vehicle only talks to loopback UDP, got {url!r}")
        self.url = url
        # Fixed local origin for the kinematics (north/east are metres from it). Home starts here but can move
        # (set_home(), or ArduPilot-style reset to the current position when arming).
        self.origin_lat, self.origin_lon = home
        self.home_lat, self.home_lon = home
        self.home_amsl_m = home_amsl_m
        self.reset_home_on_arm = True
        self.send_home = True          # False: never report HOME_POSITION (like ArduPilot before its first fix)
        self.time_scale = time_scale
        self.sysid, self.compid = sysid, compid
        self.hb_period = 1.0 / hb_rate_hz
        self.tel_period = 1.0 / telemetry_hz
        self.climb_ms, self.descend_ms, self.rtl_speed_ms = climb_ms, descend_ms, rtl_speed_ms
        self.gps_fix = gps_fix
        # hover (demo): AUTO climbs to the TAKEOFF altitude above home and stays there (no waypoints flown);
        # after hover_max_s (>0) it returns by itself (RTL), like a finished mission
        self.hover, self.hover_max_s = hover, hover_max_s

        # --- observable vehicle state
        self.armed = False
        self.mode = MODE_STABILIZE
        self.north = 0.0
        self.east = 0.0
        self.alt = 0.0
        self.heading = 0.0
        self.groundspeed = 0.0
        self.cmd_speed = 5.0
        self.flight_time_s = 0.0
        self.mission: list[_Item] = []
        self.fence: list[_Item] = []           # MAV_MISSION_TYPE_FENCE items
        self.prearm_failures: list[str] = []   # extra "PreArm: ..." reasons for RUN_PREARM_CHECKS
        self.seq = 0
        self.mission_running = False
        self.mission_done = False
        # Defaults satisfy drone-brain's pre-flight check (notes/phase7-requirements.md) with stock .env limits.
        self.params: dict[str, float] = {
            "AUTO_OPTIONS": 0.0, "WPNAV_SPEED": 500.0, "RTL_ALT": 2000.0,
            "FS_GCS_ENABLE": 0.0, "SYSID_MYGCS": 255.0,
            "FENCE_ENABLE": 1.0, "FENCE_TYPE": 3.0, "FENCE_ALT_MAX": 60.0, "FENCE_RADIUS": 300.0,
            "FENCE_ACTION": 1.0, "BATT_FS_LOW_ACT": 2.0, "FS_THR_ENABLE": 1.0,
        }
        self.statustexts: list[str] = []
        self.received_types: dict[str, int] = {}
        self.commands_received: list[int] = []

        # --- fault injection / link control
        self.mute = False                      # True: send nothing (simulates link loss)
        self.drop_inbound: dict[str, int] = {}  # message type -> number of next msgs to ignore
        self.mission_capacity = 700            # MISSION_COUNT above this -> MISSION_ACK NO_SPACE
        self.upload_ack_error: Optional[int] = None   # answer the next MISSION_COUNT with this MISSION_ACK code
        self.fail_upload_at: Optional[int] = None     # after item N is received, send ACK error (once) and stop
        self.upload_fail_code = MAV_MISSION_INVALID_SEQUENCE
        self.swap_requests = False             # request items 1,0,2,3.. (out of order) instead of 0,1,2..
        self.duplicate_requests = False        # send every MISSION_REQUEST_INT twice
        self.corrupt_readback = False          # download returns altitudes shifted by 5 m (verify must fail)
        self.battery_pct_override: Optional[int] = None
        self.garbage_every_s: Optional[float] = None   # inject random bytes into the stream

        # --- protocol state
        self._up_count = 0
        self._up_items: list[_Item] = []
        self._up_active = False
        self._up_type = 0
        self._conn = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._t_hb = self._t_tel = self._t_home = 0.0

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "MockVehicle":
        if self._thread is not None:
            raise RuntimeError("already started")
        self._conn = mavutil.mavlink_connection(
            self.url, source_system=self.sysid, source_component=self.compid
        )
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mock-vehicle", daemon=True)
        self._thread.start()
        log.info("mock vehicle up on %s (time_scale=%s)", self.url, self.time_scale)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> "MockVehicle":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def wait_until(self, pred: Callable[["MockVehicle"], bool], timeout: float, poll: float = 0.05) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if pred(self):
                return True
            time.sleep(poll)
        return pred(self)

    @property
    def lat(self) -> float:
        return self.origin_lat + self.north / M_PER_DEG_LAT

    @property
    def lon(self) -> float:
        return self.origin_lon + self.east / (M_PER_DEG_LAT * math.cos(math.radians(self.origin_lat)))

    def _to_local(self, lat: float, lon: float) -> tuple[float, float]:
        return ((lat - self.origin_lat) * M_PER_DEG_LAT,
                (lon - self.origin_lon) * M_PER_DEG_LAT * math.cos(math.radians(self.origin_lat)))

    def set_home(self, lat: float, lon: float, amsl_m: Optional[float] = None, *, announce: bool = True) -> None:
        """Move the home position (like MAV_CMD_DO_SET_HOME / re-arming elsewhere). Thread-safe."""
        with self._lock:
            self.home_lat, self.home_lon = lat, lon
            if amsl_m is not None:
                self.home_amsl_m = amsl_m
            if announce:
                self._home_msg()

    # --- fault injection helpers used by the edge-case tests
    def pilot_set_mode(self, mode: int) -> None:
        """The pilot flips the RC mode switch."""
        with self._lock:
            self._set_mode(mode)

    def trigger_gps_loss(self) -> None:
        """GPS fix lost + EKF failsafe (ArduCopter default FS_EKF_ACTION=1: LAND)."""
        with self._lock:
            self.gps_fix = 0
            self._statustext("EKF failsafe", 2)
            self._statustext("GPS Glitch", 4)
            if self.armed:
                self._set_mode(MODE_LAND)

    def trigger_battery_failsafe(self) -> None:
        with self._lock:
            self.battery_pct_override = 8
            self._statustext("Battery 1 low 14.1V used 900 mAh", 3)
            self._statustext("Failsafe: Battery low", 1)
            if self.armed:
                self._set_mode(MODE_RTL)

    def trigger_fence_breach(self) -> None:
        with self._lock:
            self._statustext("Fence breach: altitude", 1)
            if self.armed:
                self._set_mode(MODE_RTL)

    def move_home(self, north_m: float, east_m: float, d_alt_m: float = 0.0) -> None:
        """Shift home by metres (test helper)."""
        lat = self.home_lat + north_m / M_PER_DEG_LAT
        lon = self.home_lon + east_m / (M_PER_DEG_LAT * math.cos(math.radians(self.home_lat)))
        self.set_home(lat, lon, self.home_amsl_m + d_alt_m)

    # ------------------------------------------------------------------ main loop
    def _run(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            try:
                msg = self._conn.recv_match(blocking=True, timeout=0.02)
            except Exception:
                log.exception("mock recv error")
                time.sleep(0.05)
                msg = None
            if msg is not None and msg.get_type() != "BAD_DATA":
                try:
                    with self._lock:
                        self._handle(msg)
                except Exception:
                    log.exception("mock failed handling %s", msg.get_type())
            now = time.monotonic()
            dt = now - last
            if dt >= 0.02:
                last = now
                with self._lock:
                    self._step(dt * self.time_scale)
                    self._emit(now)

    # ------------------------------------------------------------------ sending
    def _tx(self, name: str, *args) -> None:
        if self.mute or self._conn is None:
            return
        try:
            getattr(self._conn.mav, name)(*args)
        except Exception:
            log.debug("mock send failed", exc_info=True)

    def _statustext(self, text: str, severity: int = 6) -> None:
        self.statustexts.append(text)
        self._tx("statustext_send", severity, text.encode()[:50])

    def _heartbeat(self) -> None:
        base = 0x01 | (0x80 if self.armed else 0)  # custom-mode enabled (+armed)
        self._tx("heartbeat_send", 2, 3, base, self.mode, 4 if self.armed else 3)

    def _home_msg(self) -> None:
        if not self.send_home:
            return
        self._tx(
            "home_position_send", int(self.home_lat * 1e7), int(self.home_lon * 1e7),
            int(self.home_amsl_m * 1000), 0, 0, 0, [1, 0, 0, 0], 0, 0, 0,
        )

    def _emit(self, now: float) -> None:
        if self.garbage_every_s and now - getattr(self, "_t_garbage", 0.0) >= self.garbage_every_s:
            self._t_garbage = now
            try:
                self._conn.write(os.urandom(48))
            except Exception:
                pass
        if now - self._t_hb >= self.hb_period:
            self._t_hb = now
            self._heartbeat()
        if now - self._t_home >= 1.0:
            self._t_home = now
            self._home_msg()
        if now - self._t_tel >= self.tel_period:
            self._t_tel = now
            ms = int(now * 1000) & 0xFFFFFFFF
            self._tx(
                "global_position_int_send", ms, int(self.lat * 1e7), int(self.lon * 1e7),
                int((self.home_amsl_m + self.alt) * 1000), int(self.alt * 1000), 0, 0, 0,
                int((self.heading % 360) * 100),
            )
            self._tx("vfr_hud_send", self.groundspeed, self.groundspeed, int(self.heading) % 360, 50,
                     self.home_amsl_m + self.alt, 0.0)
            pct = max(0, int(100 - self.flight_time_s * 0.02))
            if self.battery_pct_override is not None:
                pct = self.battery_pct_override
            present = 0x1 | 0x2 | 0x4 | 0x8 | 0x20 | 0x10000 | 0x200000 | 0x2000000 | 0x10000000
            health = present if not self._prearm_reasons() else present & ~0x10000000
            if self.gps_fix < 3:
                health &= ~0x20
            self._tx("sys_status_send", present, present, health, 100, int(15800 - 3 * (100 - pct) * 10), 0, pct,
                     0, 0, 0, 0, 0, 0)
            self._tx("extended_sys_state_send", 0, 2 if (self.armed and self.alt > 0.3) else 1)
            ekf_flags = 0x1 | 0x2 | 0x4 | 0x8 | 0x20 | (0x10 if self.gps_fix >= 3 else 0)
            self._tx("ekf_status_report_send", ekf_flags, 0.1, 0.1, 0.1, 0.05, 0.0)
            self._tx("rc_channels_send", ms, 8, *([1500] * 18), 200)
            self._tx("gps_raw_int_send", ms * 1000, self.gps_fix, int(self.lat * 1e7), int(self.lon * 1e7),
                     int((self.home_amsl_m + self.alt) * 1000), 100, 100, int(self.groundspeed * 100), 0, 12)
            self._tx("mission_current_send", self.seq)

    def _prearm_reasons(self) -> list[str]:
        r = list(self.prearm_failures)
        if self.gps_fix < 3:
            r.append("PreArm: GPS 1: Bad fix")
        return r

    # ------------------------------------------------------------------ inbound
    def _for_me(self, msg) -> bool:
        ts = getattr(msg, "target_system", None)
        return ts in (None, 0, self.sysid)

    def _handle(self, msg) -> None:
        t = msg.get_type()
        self.received_types[t] = self.received_types.get(t, 0) + 1
        if self.drop_inbound.get(t, 0) > 0:
            self.drop_inbound[t] -= 1
            log.info("mock dropping inbound %s (fault injection)", t)
            return
        if t == "HEARTBEAT" or not self._for_me(msg):
            return
        gcs = (msg.get_srcSystem(), msg.get_srcComponent())
        if t == "COMMAND_LONG":
            self._on_command(msg)
        elif t == "SET_MODE":
            if msg.custom_mode in _KNOWN_MODES:
                self._set_mode(int(msg.custom_mode))
        elif t == "PARAM_SET":
            self._on_param_set(msg)
        elif t == "PARAM_REQUEST_READ":
            name = self._pname(msg.param_id)
            if name in self.params:
                self._param_value(name)
        elif t == "MISSION_COUNT":
            self._on_mission_count(msg)
        elif t == "MISSION_ITEM_INT":
            self._on_mission_item(msg)
        elif t == "MISSION_REQUEST_LIST":
            mt = getattr(msg, "mission_type", 0)
            self._tx("mission_count_send", gcs[0], gcs[1], len(self.fence if mt == 1 else self.mission), mt)
        elif t in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
            self._on_mission_request(msg)
        elif t == "MISSION_CLEAR_ALL":
            mt = getattr(msg, "mission_type", 0)
            if mt == 1:
                self.fence = []
            else:
                self.mission, self.seq, self.mission_running = [], 0, False
            self._tx("mission_ack_send", gcs[0], gcs[1], MAV_MISSION_ACCEPTED, mt)
        # REQUEST_DATA_STREAM, MISSION_ACK, ... are accepted silently.

    @staticmethod
    def _pname(raw) -> str:
        return (raw if isinstance(raw, str) else bytes(raw).decode()).rstrip("\x00")

    def _ack(self, msg, result: int) -> None:
        self._tx("command_ack_send", msg.command, result, 0, 0,
                 msg.get_srcSystem(), msg.get_srcComponent())

    def _on_command(self, msg) -> None:
        c = msg.command
        self.commands_received.append(int(c))
        if c == 400:  # ARM_DISARM
            if msg.param1 >= 0.5:
                if self.armed:
                    return self._ack(msg, MAV_RESULT_ACCEPTED)
                if self.gps_fix < 3:
                    self._statustext("PreArm: Need 3D Fix", 4)
                    return self._ack(msg, MAV_RESULT_FAILED)
                if self.prearm_failures:
                    self._statustext(self.prearm_failures[0], 4)
                    return self._ack(msg, MAV_RESULT_FAILED)
                if self.mode == MODE_AUTO and not int(self.params["AUTO_OPTIONS"]) & 1:
                    self._statustext("Arm: Mode not armable", 4)
                    return self._ack(msg, MAV_RESULT_FAILED)
                self.armed = True
                self.flight_time_s = 0.0
                self.mission_done = False
                if self.reset_home_on_arm:  # ArduPilot: home := current position at arming
                    self.home_lat, self.home_lon = self.lat, self.lon
                    self._home_msg()
                self._statustext("Arming motors")
                self._ack(msg, MAV_RESULT_ACCEPTED)
                self._heartbeat()
            else:
                if self.armed and self.alt > 0.3 and int(msg.param2) != 21196:
                    self._statustext("Disarm: not landed", 4)
                    return self._ack(msg, MAV_RESULT_FAILED)
                self.armed = False
                self.mission_running = False
                self._ack(msg, MAV_RESULT_ACCEPTED)
                self._heartbeat()
        elif c == 300:  # MISSION_START
            if not self.armed or self.mode != MODE_AUTO or len(self.mission) < 2:
                self._statustext("Mission start denied", 4)
                return self._ack(msg, MAV_RESULT_FAILED)
            if self.seq < 1 or self.mission_done:
                self.seq, self.mission_done = 1, False
            self.mission_running = True
            self._ack(msg, MAV_RESULT_ACCEPTED)
        elif c == 20:  # NAV_RETURN_TO_LAUNCH
            self._set_mode(MODE_RTL)
            self._ack(msg, MAV_RESULT_ACCEPTED)
        elif c == 176:  # DO_SET_MODE
            mode = int(msg.param2)
            if mode not in _KNOWN_MODES:
                return self._ack(msg, MAV_RESULT_FAILED)
            self._set_mode(mode)
            self._ack(msg, MAV_RESULT_ACCEPTED)
        elif c in (511, 512):  # SET_MESSAGE_INTERVAL, REQUEST_MESSAGE
            if c == 512 and int(msg.param1) == 242:
                self._home_msg()
            if c == 512 and int(msg.param1) == 148:
                self._tx("autopilot_version_send", 0xFFFF, (4 << 24) | (6 << 16) | (3 << 8) | 255, 0, 0, 0,
                         b"mock0000", b"mock0000", b"mock0000", 0x1209, 0x5740, 0, b"\x00" * 18)
            self._ack(msg, MAV_RESULT_ACCEPTED)
        elif c == 401:  # RUN_PREARM_CHECKS
            reasons = self._prearm_reasons()
            for r in reasons:
                self._statustext(r, 2)
            self._ack(msg, MAV_RESULT_ACCEPTED if not reasons else MAV_RESULT_FAILED)
        else:
            self._ack(msg, MAV_RESULT_UNSUPPORTED)

    def _set_mode(self, mode: int) -> None:
        if mode != self.mode:
            log.info("mock mode %s -> %s", self.mode, mode)
            self.mode = mode
            if mode != MODE_AUTO:
                self.mission_running = False
            self._heartbeat()

    def _param_value(self, name: str) -> None:
        idx = list(self.params).index(name)
        self._tx("param_value_send", name.encode(), self.params[name], 9, len(self.params), idx)

    def _on_param_set(self, msg) -> None:
        name = self._pname(msg.param_id)
        if name not in self.params:
            return  # ArduPilot silently ignores unknown parameters
        self.params[name] = float(msg.param_value)
        self._param_value(name)

    # --- mission protocol
    def _on_mission_count(self, msg) -> None:
        gcs = (msg.get_srcSystem(), msg.get_srcComponent())
        mt = getattr(msg, "mission_type", 0)
        if mt == 0:
            self.mission_running = False
        if msg.count == 0:
            if mt == 1:
                self.fence = []
            else:
                self.mission, self.seq = [], 0
            return self._tx("mission_ack_send", gcs[0], gcs[1], MAV_MISSION_ACCEPTED, mt)
        if self.upload_ack_error is not None:
            code, self.upload_ack_error = self.upload_ack_error, None
            return self._tx("mission_ack_send", gcs[0], gcs[1], code, mt)
        if msg.count > self.mission_capacity:
            return self._tx("mission_ack_send", gcs[0], gcs[1], MAV_MISSION_NO_SPACE, mt)
        self._up_count, self._up_items, self._up_active, self._up_type = int(msg.count), [], True, mt
        self._request_item(gcs, 0, mt)

    def _request_item(self, gcs, seq: int, mt: int) -> None:
        """MISSION_REQUEST_INT with optional fault injection (duplicated / swapped requests)."""
        if self.swap_requests and seq in (0, 1) and self._up_count > 2:
            seq = 1 - seq
        self._tx("mission_request_int_send", gcs[0], gcs[1], seq, mt)
        if self.duplicate_requests:
            self._tx("mission_request_int_send", gcs[0], gcs[1], seq, mt)

    def _on_mission_item(self, msg) -> None:
        gcs = (msg.get_srcSystem(), msg.get_srcComponent())
        if not self._up_active:
            return
        n = len(self._up_items)
        if self.fail_upload_at is not None and msg.seq >= self.fail_upload_at:
            self.fail_upload_at, self._up_active = None, False
            return self._tx("mission_ack_send", gcs[0], gcs[1], self.upload_fail_code, self._up_type)
        if self.swap_requests and msg.seq == 1 and n == 0:     # we asked for 1 first: accept it out of order
            self._up_items.append(_Item(1, int(msg.frame), int(msg.command), msg.param1, msg.param2, msg.param3,
                                        msg.param4, msg.x / 1e7, msg.y / 1e7, float(msg.z)))
            return self._tx("mission_request_int_send", gcs[0], gcs[1], 0, self._up_type)
        if self.swap_requests and msg.seq == 0 and len(self._up_items) == 1 and self._up_items[0].seq == 1:
            self._up_items.insert(0, _Item(0, int(msg.frame), int(msg.command), msg.param1, msg.param2, msg.param3,
                                           msg.param4, msg.x / 1e7, msg.y / 1e7, float(msg.z)))
            n = 2
            if n < self._up_count:
                return self._tx("mission_request_int_send", gcs[0], gcs[1], 2, self._up_type)
        elif msg.seq == n:
            self._up_items.append(
                _Item(int(msg.seq), int(msg.frame), int(msg.command), msg.param1, msg.param2, msg.param3,
                      msg.param4, msg.x / 1e7, msg.y / 1e7, float(msg.z))
            )
            n += 1
        elif msg.seq != n - 1:  # not the next one and not a duplicate retry
            self._up_active = False
            return self._tx("mission_ack_send", gcs[0], gcs[1], MAV_MISSION_INVALID_SEQUENCE, self._up_type)
        if n >= self._up_count:
            self._up_active = False
            if self._up_type == 1:
                self.fence = self._up_items
            else:
                self.mission, self.seq, self.mission_done = self._up_items, 0, False
            self._up_items = []
            self._tx("mission_ack_send", gcs[0], gcs[1], MAV_MISSION_ACCEPTED, self._up_type)
            log.info("mock accepted %s with %d items", "fence" if self._up_type == 1 else "mission", n)
        else:
            self._request_item(gcs, n, self._up_type)

    def _on_mission_request(self, msg) -> None:
        gcs = (msg.get_srcSystem(), msg.get_srcComponent())
        mt = getattr(msg, "mission_type", 0)
        lst = self.fence if mt == 1 else self.mission
        if not 0 <= msg.seq < len(lst):
            return self._tx("mission_ack_send", gcs[0], gcs[1], MAV_MISSION_INVALID_SEQUENCE, mt)
        it = lst[msg.seq]
        lat, lon, alt, frame = it.lat, it.lon, it.alt, it.frame
        if self.corrupt_readback and it.command == 16 and it.seq > 0:
            alt += 5.0
        if it.seq == 0 and mt == 0:  # like ArduPilot: slot 0 always reports the real home
            lat, lon, alt, frame = self.home_lat, self.home_lon, self.home_amsl_m, 0
        self._tx("mission_item_int_send", gcs[0], gcs[1], it.seq, frame, it.command,
                 1 if it.seq == self.seq else 0, 1, it.p1, it.p2, it.p3, it.p4,
                 int(round(lat * 1e7)), int(round(lon * 1e7)), alt, mt)

    # ------------------------------------------------------------------ kinematics
    def _step(self, dt_total: float) -> None:
        if not self.armed:
            self.groundspeed = 0.0
            return
        # Sub-step so large time-scale ticks cannot overshoot waypoints.
        n = max(1, int(math.ceil(dt_total / 0.1)))
        dt = dt_total / n
        for _ in range(n):
            self.flight_time_s += dt
            if self.mode == MODE_AUTO:
                self._step_auto(dt)
            elif self.mode == MODE_RTL:
                self._step_rtl(dt)
            elif self.mode == MODE_LAND:
                self._step_land(dt)
            else:
                self.groundspeed = 0.0

    def _climb_to(self, target: float, dt: float) -> bool:
        d = target - self.alt
        step = (self.climb_ms if d > 0 else self.descend_ms) * dt
        if abs(d) <= step:
            self.alt = target
            return True
        self.alt += math.copysign(step, d)
        return False

    def _move_to(self, n: float, e: float, speed: float, dt: float) -> float:
        """Move toward (n,e) at `speed`; returns remaining distance."""
        dn, de = n - self.north, e - self.east
        dist = math.hypot(dn, de)
        if dist > 0.01:
            self.heading = math.degrees(math.atan2(de, dn)) % 360
        step = speed * dt
        if dist <= step:
            self.north, self.east, self.groundspeed = n, e, 0.0
            return 0.0
        self.north += dn / dist * step
        self.east += de / dist * step
        self.groundspeed = speed
        return dist - step

    def _step_auto(self, dt: float) -> None:
        self.groundspeed = 0.0
        if not self.mission_running or self.seq >= len(self.mission):
            return
        it = self.mission[self.seq]
        if self.hover:
            alt = next((m.alt for m in self.mission if m.command == 22), 10.0)
            self._climb_to(alt, dt)
            if self.hover_max_s > 0 and self.flight_time_s >= self.hover_max_s:
                self.mission_running, self.mission_done = False, True
                self._statustext("Demo search time is up: returning")
                self._set_mode(MODE_RTL)
            return
        if it.command == 22:  # TAKEOFF
            if self.alt < 0.1 and not int(self.params["AUTO_OPTIONS"]) & 2:
                return  # real ArduPilot would wait for the pilot to raise throttle
            if self._climb_to(it.alt, dt):
                self.seq += 1
        elif it.command == 178:  # DO_CHANGE_SPEED
            if it.p2 > 0:
                self.cmd_speed = it.p2
            self.seq += 1
        elif it.command == 16:  # WAYPOINT
            if it.seq == 0:
                self.seq += 1
                return
            n, e = self._to_local(it.lat, it.lon)
            self._climb_to(it.alt, dt)
            if self._move_to(n, e, self.cmd_speed, dt) <= 0.5:
                self.seq += 1
        elif it.command == 20:  # RTL
            self.seq += 1
            self.mission_running, self.mission_done = False, True
            self._set_mode(MODE_RTL)
        else:
            self.seq += 1
        if self.seq >= len(self.mission) and self.mission_running:
            self.mission_running, self.mission_done = False, True

    def _step_rtl(self, dt: float) -> None:
        hn, he = self._to_local(self.home_lat, self.home_lon)
        if self.alt < self.params.get("RTL_ALT", 0.0) / 100.0 and math.hypot(hn - self.north, he - self.east) > 0.5:
            self._climb_to(self.params["RTL_ALT"] / 100.0, dt)   # climb to RTL_ALT first
            return
        if self._move_to(hn, he, self.rtl_speed_ms, dt) <= 0.5:
            self._step_land(dt)

    def _step_land(self, dt: float) -> None:
        self.groundspeed = 0.0
        if self._climb_to(0.0, dt):
            self.armed = False
            self.mission_running = False
            self._statustext("Land complete, disarmed")
            self._heartbeat()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fake ArduCopter for drone-brain (localhost UDP only)")
    ap.add_argument("--url", default="udpout:127.0.0.1:14550",
                    help="udpout:127.0.0.1:PORT (send to a listening GCS, default) or udpin:127.0.0.1:PORT")
    ap.add_argument("--time-scale", type=float, default=1.0, help="simulation speed-up factor")
    ap.add_argument("--home", default="-35.363261,149.165230", help="lat,lon")
    ap.add_argument("--hb-rate", type=float, default=1.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    lat, lon = (float(x) for x in args.home.split(","))
    mock = MockVehicle(args.url, home=(lat, lon), time_scale=args.time_scale, hb_rate_hz=args.hb_rate)
    mock.start()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.wait(1.0):
            log.info("mode=%d armed=%s seq=%d alt=%.1f pos=(%.0f,%.0f)", mock.mode, mock.armed,
                     mock.seq, mock.alt, mock.north, mock.east)
    finally:
        mock.stop()


if __name__ == "__main__":
    main()
