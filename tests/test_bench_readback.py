"""--bench-home (upload mode only) and the read-back view: gating, home precedence, preflight downgrade, diff, e2e on the mock."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import main  # noqa: E402
import mission_diff as md  # noqa: E402
import test_e2e as e2e  # noqa: E402
from config import FIELD_AMSL_M, FIELD_HOME, BenchHomeError, bench_home_for, parse_bench_home  # noqa: E402
from conftest import make_settings  # noqa: E402
from mission import ALT_TOL_M, POS_TOL_DEG, ArduPilotBackend, ReadOnlyViolation  # noqa: E402
from mission_diff import compare_missions, readback_view  # noqa: E402
from models import MissionItem  # noqa: E402

BENCH_TXT = "BENCH HOME — not from GPS (upload test only)"


# ------------------------------------------------------------------ parsing / gating
def test_parse_bench_home():
    assert parse_bench_home("") is None and parse_bench_home("  ") is None
    assert parse_bench_home("field") == (FIELD_HOME[0], FIELD_HOME[1], FIELD_AMSL_M)
    assert parse_bench_home("33.9,-84.5") == (33.9, -84.5, FIELD_AMSL_M)
    assert parse_bench_home("33.9, -84.5, 250") == (33.9, -84.5, 250.0)
    for bad in ("abc", "1", "1,2,3,4", "95,0", "0,200", "33.9,-84.5,99999"):
        with pytest.raises(BenchHomeError):
            parse_bench_home(bad)
    assert main.FIELD_HOME == FIELD_HOME and main.FIELD_AMSL_M == FIELD_AMSL_M


@pytest.mark.parametrize("mode", ["sim", "readonly", "full"])
def test_bench_home_refused_outside_upload(mode):
    s = make_settings(vehicle_mode=mode, bench_home="field", mavlink_url="udpin:127.0.0.1:14999")
    with pytest.raises(BenchHomeError, match="only allowed with --vehicle upload"):
        bench_home_for(s)
    with pytest.raises(SystemExit, match="refusing to start"):
        main.check_vehicle_mode(s)
    from controller_real import RealController
    with pytest.raises(BenchHomeError):
        RealController(s, object(), llm=object(), manage_connection=False)


def test_bench_home_allowed_in_upload_and_cli():
    s = make_settings(vehicle_mode="upload", bench_home="field", mavlink_url="serial:/dev/ttyACM0")
    assert bench_home_for(s) == (FIELD_HOME[0], FIELD_HOME[1], FIELD_AMSL_M)
    main.check_vehicle_mode(s)                      # no SystemExit
    args = main.parse_args(["--vehicle", "upload", "--bench-home", "field", "--mavlink-url", "serial:/dev/ttyACM0"])
    assert main.build_settings(args, make_settings()).bench_home == "field"
    # CLI flag without --vehicle upload: the default mode is sim, so it must refuse
    args = main.parse_args(["--bench-home", "field"])
    with pytest.raises(SystemExit):
        main.check_vehicle_mode(main.build_settings(args, make_settings()))
    assert bench_home_for(make_settings(vehicle_mode="sim")) is None            # off by default, any mode


# ------------------------------------------------------------------ readonly may download (a read)
def test_readonly_download_allowed_and_sends_only_reads():
    port = e2e.free_udp_port()
    s = make_settings(vehicle_mode="readonly", mavlink_url=f"udpin:127.0.0.1:{port}")
    mock = e2e.MockVehicle(f"udpout:127.0.0.1:{port}", home=FIELD_HOME, time_scale=1).start()
    from mock_vehicle import _Item
    mock.mission = [_Item(0, 0, 16, 0, 0, 0, 0, FIELD_HOME[0], FIELD_HOME[1], 300.0),
                    _Item(1, 6, 22, 0, 0, 0, 0, 0.0, 0.0, 20.0)]
    be = ArduPilotBackend(s)
    try:
        be.connect(timeout=10)
        got = be.download_mission()
        assert [g.command for g in got] == [16, 22] and got[1].alt == 20.0
        assert {"MISSION_COUNT", "MISSION_ITEM_INT", "MISSION_CLEAR_ALL", "PARAM_SET", "SET_MODE"}.isdisjoint(mock.received_types)
        with pytest.raises(ReadOnlyViolation):
            be.upload_mission(got)
        for m in ("mission_count_send", "mission_item_int_send", "mission_clear_all_send"):
            with pytest.raises(ReadOnlyViolation):
                be._send(m, 1, 1, 0)
    finally:
        be.close()
        mock.stop()


# ------------------------------------------------------------------ diff
def _items():
    return [MissionItem(seq=0, command=16, frame=0, lat=33.9, lon=-84.5),
            MissionItem(seq=1, command=22, frame=6, alt=20.0),
            MissionItem(seq=2, command=178, frame=2, param1=1.0, param2=5.0, param3=-1.0),
            MissionItem(seq=3, command=16, frame=6, lat=33.93745, lon=-84.5274, alt=20.0),
            MissionItem(seq=4, command=16, frame=6, lat=33.93790, lon=-84.5274, alt=20.0),
            MissionItem(seq=5, command=20, frame=2)]


def _copy(items, chg):
    return [i.model_copy(update=chg.get(i.seq, {})) for i in items]


def test_diff_all_match_and_tolerances_are_verifys():
    assert POS_TOL_DEG == md.POS_TOL_DEG and ALT_TOL_M == md.ALT_TOL_M
    w = _items()
    d = compare_missions(w, _copy(w, {3: {"lat": w[3].lat + 5e-8}, 0: {"lat": 0.0, "lon": 0.0}, 4: {"frame": 3}}), "p1")
    assert d["summary"]["all_ok"] and d["summary"]["matched"] == d["summary"]["total"] == 6
    assert d["summary"]["text"].startswith("6/6 items match, max error 0.")
    assert [r["ok"] for r in d["rows"]] == [True] * 6
    assert d["rows"][0]["note"].startswith("home slot")                # item 0: command only
    # exactly as strict as verify_mission
    ok = compare_missions(w, _copy(w, {3: {"lat": w[3].lat + 0.9 * POS_TOL_DEG, "alt": 20.0 + 0.9 * ALT_TOL_M}}))
    assert ok["summary"]["all_ok"]
    bad = compare_missions(w, _copy(w, {3: {"lat": w[3].lat + 2 * POS_TOL_DEG}, 4: {"alt": 20.0 + 2 * ALT_TOL_M}}))
    assert [r["ok"] for r in bad["rows"]] == [True, True, True, False, False, True]
    assert bad["rows"][3]["d_h_cm"] == pytest.approx(22.3, abs=0.3) and "position" in bad["rows"][3]["note"]
    assert bad["rows"][4]["d_alt_m"] == pytest.approx(0.1) and "altitude" in bad["rows"][4]["note"]
    assert bad["summary"]["matched"] == 4 and "2 DIFFER" in bad["summary"]["text"] and not bad["summary"]["all_ok"]


def test_diff_non_positional_items_and_structure():
    w = _items()
    d = compare_missions(w, _copy(w, {2: {"param2": 7.0}, 1: {"alt": 25.0}, 5: {"command": 21}}))
    rows = {r["seq"]: r for r in d["rows"]}
    assert rows[2]["ok"] is False and "p2" in rows[2]["note"] and rows[2]["positional"] is False and rows[2]["d_h_cm"] is None
    assert rows[1]["ok"] is False and rows[1]["d_alt_m"] == 5.0            # TAKEOFF height
    assert rows[5]["ok"] is False and "LAND" in rows[5]["note"]
    # missing / extra items, frame class mismatch
    d = compare_missions(w, _copy(w, {3: {"frame": 0}})[:5] + [MissionItem(seq=6, command=16, lat=1.0, lon=1.0, alt=1.0)])
    rows = {r["seq"]: r for r in d["rows"]}
    assert rows[5]["readback"] is None and "missing" in rows[5]["note"]
    assert rows[6]["planned"] is None and "extra" in rows[6]["note"]
    assert "frame" in rows[3]["note"] and not rows[3]["ok"]
    assert compare_missions(None, w)["compared"] is False
    v = readback_view(w)
    assert [x["name"] for x in v] == ["WAYPOINT", "TAKEOFF", "DO_CHANGE_SPEED", "WAYPOINT", "WAYPOINT", "RTL"]
    assert v[3]["positional"] and not v[1]["positional"] and v[3]["frame_name"] == "REL_ALT"


# ------------------------------------------------------------------ end-to-end on the mock (upload mode)
class NoHomeMock(e2e.MockVehicle):
    """Indoors: no GPS fix, never reports HOME_POSITION."""

    def __init__(self, *a, **kw):
        kw["gps_fix"] = 0
        super().__init__(*a, **kw)
        self.send_home = False


@pytest.fixture
def no_home_mock(monkeypatch):
    monkeypatch.setattr(e2e, "MockVehicle", NoHomeMock)


FAR = [{"lat": 33.9371904 + d, "lon": -84.5271293 + e} for d, e in ((0.004, 0.0), (0.004, 0.001), (0.0045, 0.001), (0.0045, 0.0))]


def test_no_bench_home_means_no_planning_without_gps(tmp_path, no_home_mock):
    with e2e.Rig(tmp_path, vehicle_mode="upload") as rig:
        m = rig.plan()
        assert m["type"] == "error" and "3D GPS fix" in m["data"]["message"]


def test_bench_home_end_to_end(tmp_path, no_home_mock):
    with e2e.Rig(tmp_path, vehicle_mode="upload", bench_home="field") as rig:
        rig.mock.params["FENCE_ENABLE"] = 0.0            # expected on the bench; must not block the upload
        rig.mock.params["BATT_FS_LOW_ACT"] = 0.0
        snap = rig.client.get("/api/status").json()["data"]
        assert snap["home_source"] == "bench" and snap["bench_home"]["active"] and snap["home"]["source"] == "bench"
        assert snap["home"]["lat"] == pytest.approx(FIELD_HOME[0]) and snap["readback"] == []
        m = rig.plan()
        assert m["type"] == "plan", m
        p = m["data"]
        assert p["home_source"] == "bench" and p["safety"]["ok"], p["safety"]
        assert p["plan"]["home"]["lat"] == pytest.approx(FIELD_HOME[0]) and p["home_amsl_m"] == FIELD_AMSL_M
        assert BENCH_TXT in p["safety"]["warnings"] and BENCH_TXT in p["bench_notice"]
        assert p["preflight"]["ok"] and p["preflight"]["home_source"] == "bench" and not p["preflight"]["errors"]
        assert any("FENCE_ENABLE" in w and "bench" in w for w in p["preflight"]["warnings"])
        assert any(c["param"] == "FENCE_ENABLE" and not c["ok"] and c["severity"] == "warning" for c in p["preflight"]["checks"])
        pdir = tmp_path / "plans" / p["plan_id"]
        assert json.loads((pdir / "plan.json").read_text())["home_source"] == "bench"
        assert json.loads((pdir / "preflight.json").read_text())["home_source"] == "bench"
        gj = json.loads((pdir / "mission.geojson").read_text())
        assert gj["features"][2]["properties"]["home_source"] == "bench"
        rig.ack()
        txt = rig.post("/api/confirm", {"plan_id": p["plan_id"]})["data"]["text"]
        n = len(p["plan"]["waypoints"]) + 4
        assert txt.startswith(f"Uploaded {n} mission items to the Pixhawk, read-back verify OK")
        assert BENCH_TXT in txt and "replaces mission item 0" in txt and "absolute lat/lon" in txt
        assert f"/api/plans/{p['plan_id']}/readback.waypoints" in txt and "Load WP File" in txt
        assert f"{n}/{n} items match" in txt
        assert rig.mock.mission[0].lat == pytest.approx(FIELD_HOME[0], abs=1e-6)       # bench home in item 0
        assert not rig.mock.armed and not {176, 300, 400, 20} & set(rig.mock.commands_received)
        up = json.loads((pdir / "upload.json").read_text())
        assert up["home_source"] == "bench" and up["verified"] and up["diff"]["summary"]["all_ok"]
        # snapshot carries the read-back for the phone
        snap = rig.client.get("/api/status").json()["data"]
        assert len(snap["readback"]) == n and snap["readback"][3]["name"] == "WAYPOINT"
        assert {"seq", "name", "command", "lat", "lon", "alt", "frame"} <= set(snap["readback"][3])
        info = snap["readback_info"]
        assert info["source"] == "upload" and info["diff"]["summary"]["matched"] == n and info["diff"]["summary"]["all_ok"]
        assert info["files"]["readback.waypoints"].endswith("/readback.waypoints")
        assert rig.client.get(info["files"]["mission.waypoints"]).status_code == 200
        assert rig.client.get(info["files"]["readback.waypoints"]).status_code == 200
        assert "readback" not in snap["last_upload"]
        # tamper with what the Pixhawk holds (as if edited in Mission Planner), then read it again
        rig.mock.mission[3].lat += 3e-5            # ~3.3 m
        rig.mock.mission[4].alt += 2.0
        msg = rig.post("/api/vehicle/read_mission")["data"]["text"]
        assert f"Read {n} mission items" in msg and "DIFFER" in msg
        snap = rig.client.get("/api/status").json()["data"]
        d = snap["readback_info"]["diff"]
        assert snap["readback_info"]["source"] == "read" and not d["summary"]["all_ok"] and d["summary"]["matched"] == n - 2
        assert [r["seq"] for r in d["rows"] if not r["ok"]] == [3, 4]
        assert d["rows"][3]["d_h_cm"] == pytest.approx(333.6, abs=2)
        assert not rig.mock.armed and not {176, 300, 400, 20} & set(rig.mock.commands_received)
        assert "PARAM_SET" not in rig.mock.received_types


def test_bench_home_safety_still_blocks(tmp_path, no_home_mock):
    with e2e.Rig(tmp_path, vehicle_mode="upload", bench_home="field") as rig:
        m = rig.post("/api/plan_area", {"corners": FAR, "altitude_m": 20})
        assert m["type"] == "plan"
        p = m["data"]
        assert p["home_source"] == "bench" and not p["safety"]["ok"]
        assert any("geofence" in e for e in p["safety"]["errors"])
        rig.ack()
        assert "failed safety" in rig.post("/api/confirm", {"plan_id": p["plan_id"]})["data"]["text"]
        assert rig.mock.mission == []


def test_real_home_overrides_bench_home(tmp_path):
    # the mock reports a real home at the field; the bench home elsewhere must be ignored
    with e2e.Rig(tmp_path, vehicle_mode="upload", bench_home="34.5,-85.5,100") as rig:
        snap = rig.client.get("/api/status").json()["data"]
        assert snap["home_source"] == "pixhawk" and snap["home"]["lat"] == pytest.approx(FIELD_HOME[0])
        assert snap["bench_home"]["active"] is False and "ignored" in snap["bench_home"]["note"]
        p = rig.plan()["data"]
        assert p["home_source"] == "pixhawk" and p["bench_notice"] is None
        assert p["plan"]["home"]["lat"] == pytest.approx(FIELD_HOME[0]) and p["home_amsl_m"] == 300.0
        assert BENCH_TXT not in p["safety"]["warnings"] and p["preflight"]["home_source"] == "pixhawk"
        assert json.loads((tmp_path / "plans" / p["plan_id"] / "plan.json").read_text())["home_source"] == "pixhawk"


def test_preflight_downgrade_only_in_upload_with_bench(tmp_path):
    # upload mode, real home, FENCE_ENABLE=0: the preflight error still blocks (no bench downgrade)
    with e2e.Rig(tmp_path, vehicle_mode="upload", bench_home="field") as rig:
        rig.mock.params["FENCE_ENABLE"] = 0.0
        p = rig.plan()["data"]
        assert p["home_source"] == "pixhawk" and not p["preflight"]["ok"] and not p["safety"]["ok"]
    with e2e.Rig(tmp_path, vehicle_mode="upload") as rig:
        rig.mock.params["FENCE_ENABLE"] = 0.0
        p = rig.plan()["data"]
        assert not p["safety"]["ok"] and any("FENCE_ENABLE" in e for e in p["safety"]["errors"])


def test_readonly_controller_reads_mission(tmp_path):
    with e2e.Rig(tmp_path, vehicle_mode="readonly") as rig:
        from mock_vehicle import _Item
        rig.mock.mission = [_Item(0, 0, 16, 0, 0, 0, 0, FIELD_HOME[0], FIELD_HOME[1], 300.0), _Item(1, 2, 20, 0, 0, 0, 0, 0, 0, 0)]
        txt = rig.post("/api/vehicle/read_mission")["data"]["text"]
        assert "Read 2 mission items" in txt and "no plan to compare" in txt
        snap = rig.client.get("/api/status").json()["data"]
        assert [i["name"] for i in snap["readback"]] == ["WAYPOINT", "RTL"] and snap["readback_info"]["diff"]["compared"] is False
        assert {"MISSION_COUNT", "MISSION_ITEM_INT", "MISSION_CLEAR_ALL"}.isdisjoint(rig.mock.received_types)
