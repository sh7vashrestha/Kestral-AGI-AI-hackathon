"""Phase 2b web tests: plan_area (WS + REST), map info, PMTiles Range, MBTiles, traversal."""
from __future__ import annotations

import sqlite3
import struct

import pytest
from fastapi.testclient import TestClient

import models
from config import Settings
from survey import local_to_geo
from webapp.controller import FakeController, _HOME_LAT, _HOME_LON
from webapp.server import Hub, create_app

JPEG = b"\xff\xd8\xff\xe0 fake jpeg tile"


def make_pmtiles(path, size=4096):
    h = bytearray(127)
    h[:7] = b"PMTiles"
    h[7] = 3
    h[99] = 1           # MVT
    h[100], h[101] = 0, 17
    struct.pack_into("<4i", h, 102, int(-84.54e7), int(33.92e7), int(-84.51e7), int(33.95e7))
    h[118] = 15
    struct.pack_into("<2i", h, 119, int(-84.525e7), int(33.935e7))
    path.write_bytes(bytes(h) + bytes(i % 251 for i in range(size - 127)))


def make_mbtiles(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE metadata (name text, value text)")
    con.execute("CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob)")
    con.executemany("INSERT INTO metadata VALUES (?,?)", [("name", "sat"), ("format", "jpg"),
                                                          ("bounds", "-84.54,33.92,-84.51,33.95"),
                                                          ("minzoom", "0"), ("maxzoom", "18")])
    # XYZ (x=1, y=0) at z=1 is TMS row 1
    con.execute("INSERT INTO tiles VALUES (1, 1, 1, ?)", (JPEG,))
    con.commit()
    con.close()


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "data"
    maps = data / "maps"
    maps.mkdir(parents=True)
    make_pmtiles(maps / "area.pmtiles")
    (maps / "bad.pmtiles").write_bytes(b"not pmtiles")
    make_mbtiles(maps / "sat.mbtiles")
    (data / "secret.pmtiles").write_bytes(b"outside the maps dir")
    settings = Settings(_env_file=None, data_dir=data, map_satellite_mbtiles="sat.mbtiles",
                        camera_hfov_deg=90.0, camera_vfov_deg=60.0, sensor_width_mm=4.8,
                        focal_length_mm=3.67, image_width_px=1920)
    ctrl = FakeController(data, background=False, settings=settings)
    app = create_app(ctrl, settings, hub=Hub())
    with TestClient(app) as client:
        yield client, ctrl, settings


def square(n0=40.0, size=80.0):
    from models import GeoPoint
    home = GeoPoint(lat=_HOME_LAT, lon=_HOME_LON)
    pts = [(n0, -size / 2), (n0, size / 2), (n0 + size, size / 2), (n0 + size, -size / 2)]
    return [{"lat": a, "lon": b} for a, b in (local_to_geo(n, e, home) for n, e in pts)]


def recv_until(ws, typ, limit=10):
    for _ in range(limit):
        m = ws.receive_json()
        if m["type"] == typ:
            return m
    raise AssertionError(f"no {typ}")


# ---- plan_area ----

def test_plan_area_ws(env):
    client, ctrl, _ = env
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        ws.send_json({"type": "plan_area", "corners": square(), "altitude_m": 20, "target_text": "man in red"})
        m = recv_until(ws, "plan")
        p = models.PlanProposal.model_validate(m["data"])
        assert p.safety.ok and p.plan.mission.shape == "polygon" and p.target.description == "man in red"
        assert p.plan.estimated_flight_s and len(p.plan.waypoints) >= 2
        ws.send_json({"type": "confirm", "plan_id": p.plan_id})
        assert "confirmed" in recv_until(ws, "chat")["data"]["text"].lower()
        assert ctrl.state == "staged"
        ws.send_json({"type": "go", "plan_id": p.plan_id})
        assert "GO" in recv_until(ws, "chat")["data"]["text"]
    assert ctrl.state == "flying"


def test_plan_area_rest_and_blocked(env):
    client, ctrl, _ = env
    r = client.post("/api/plan_area", json={"corners": square(), "altitude_m": 20}).json()
    assert r["type"] == "plan" and r["data"]["safety"]["ok"]
    ctrl.state = "idle"
    far = square(n0=400)
    r = client.post("/api/plan_area", json={"corners": far, "altitude_m": 20}).json()
    assert r["type"] == "plan" and not r["data"]["safety"]["ok"]
    assert any("geofence" in e for e in r["data"]["safety"]["errors"])


def test_plan_area_bowtie_rejected(env):
    client, ctrl, _ = env
    c = square()
    bow = [c[0], c[2], c[1], c[3]]
    r = client.post("/api/plan_area", json={"corners": bow, "altitude_m": 20}).json()
    assert r["type"] == "error" and "crosses itself" in r["data"]["message"]


def test_plan_area_bad_requests(env):
    client, _, _ = env
    assert client.post("/api/plan_area", json={"corners": square()[:2]}).status_code == 422
    with client.websocket_connect("/ws") as ws:
        recv_until(ws, "status")
        ws.send_json({"type": "plan_area", "corners": [1, 2, 3]})
        assert "Bad message" in recv_until(ws, "error")["data"]["message"]


def test_limits(env):
    client, _, s = env
    d = client.get("/api/limits").json()
    assert d["max_alt_m"] == s.max_alt_m and d["hfov_deg"] == 90.0 and d["max_area_m2"] == s.max_area_m2


def test_snapshot_has_home(env):
    client, _, _ = env
    d = client.get("/api/status").json()["data"]
    assert d["home"]["lat"] == pytest.approx(_HOME_LAT)


# ---- map ----

def test_map_info(env):
    client, _, _ = env
    d = client.get("/api/map/info").json()
    ids = [l["id"] for l in d["layers"]]
    assert ids == ["area", "satellite"]          # bad.pmtiles ignored
    assert d["has_map"] and d["bounds"][0] == pytest.approx(-84.54)
    assert d["center"] == pytest.approx([33.9371904, -84.5271293])   # KSU field is inside the map: opens there
    assert d["layers"][0]["url"] == "/maps/area.pmtiles"
    assert d["layers"][1]["tiles"] == ["/tiles/sat/{z}/{x}/{y}.jpg"]
    assert d["notice"] == ""          # airspace banner removed (operator request)


def test_map_info_empty(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    with TestClient(create_app(FakeController(tmp_path, background=False), settings)) as c:
        d = c.get("/api/map/info").json()
    assert d["has_map"] is False and d["layers"] == [] and d["bounds"] is None


def test_pmtiles_range(env):
    client, _, _ = env
    full = client.get("/maps/area.pmtiles")
    assert full.status_code == 200 and len(full.content) == 4096 and full.headers["accept-ranges"] == "bytes"
    r = client.get("/maps/area.pmtiles", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == full.content[10:20]
    assert r.headers["content-range"] == "bytes 10-19/4096"
    r = client.get("/maps/area.pmtiles", headers={"Range": "bytes=4000-"})
    assert r.status_code == 206 and r.content == full.content[4000:]
    r = client.get("/maps/area.pmtiles", headers={"Range": "bytes=-100"})
    assert r.content == full.content[-100:]
    r = client.get("/maps/area.pmtiles", headers={"Range": "bytes=0-99999"})
    assert r.status_code == 206 and len(r.content) == 4096
    assert client.get("/maps/area.pmtiles", headers={"Range": "bytes=5000-"}).status_code == 416
    assert client.get("/maps/area.pmtiles", headers={"Range": "garbage"}).status_code == 416
    assert client.head("/maps/area.pmtiles").headers["content-length"] == "4096"


def test_pmtiles_traversal_and_names(env):
    client, _, _ = env
    for url in ("/maps/../secret.pmtiles", "/maps/%2e%2e/secret.pmtiles", "/maps/..%2fsecret.pmtiles",
                "/maps/sat.mbtiles", "/maps/nope.pmtiles", "/maps/.hidden.pmtiles"):
        assert client.get(url).status_code == 404, url


def test_sat_tiles(env):
    client, _, _ = env
    r = client.get("/tiles/sat/1/1/0.jpg")      # XYZ y=0 -> TMS row 1
    assert r.status_code == 200 and r.content == JPEG and r.headers["content-type"] == "image/jpeg"
    assert client.get("/tiles/sat/1/0/0.jpg").status_code == 404
    assert client.get("/tiles/sat/1/5/0.jpg").status_code == 404
    assert client.get("/tiles/sat/1/1/0.gif").status_code == 404


def test_sat_absent(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path, map_satellite_mbtiles="missing.mbtiles")
    with TestClient(create_app(FakeController(tmp_path, background=False), settings)) as c:
        assert c.get("/tiles/sat/1/1/0.jpg").status_code == 404
        assert c.get("/api/map/info").json()["layers"] == []


# ---- coordinate paste ----

def test_parse_coords_rest(env):
    client, _, _ = env
    txt = "33.9372, -84.5271\n33.9372 -84.5260\n33.9380,-84.5260\n33°56'20\" N, 84°31'37\" W"
    r = client.post("/api/parse_coords", json={"text": txt})
    assert r.status_code == 200
    c = r.json()["corners"]
    assert len(c) == 4 and c[1] == {"lat": 33.9372, "lon": -84.526}
    assert c[3]["lat"] == pytest.approx(33.93889, abs=1e-4) and c[3]["lon"] == pytest.approx(-84.52694, abs=1e-4)


def test_parse_coords_errors(env):
    client, _, _ = env
    assert client.post("/api/parse_coords", json={"text": "1,2\n3,4"}).status_code == 422
    assert client.post("/api/parse_coords", json={"text": "a, b\n1,2\n3,4"}).status_code == 422
    assert client.post("/api/parse_coords", json={"text": "95, 10\n1,2\n3,4"}).status_code == 422


def test_parse_coords_hemisphere_order():
    from webapp.coords import parse_corners
    c = parse_corners("84.5W 33.9N\n84.4W 33.9N\n84.4W 34.0N")
    assert c[0].lat == 33.9 and c[0].lon == -84.5


# ---- static UI ----

def test_ui_is_offline_and_vendored(env):
    client, _, _ = env
    html = client.get("/").text
    assert "http://" not in html and "https://" not in html
    for path in ("/static/vendor/maplibre-gl.js", "/static/vendor/maplibre-gl.css", "/static/vendor/pmtiles.js",
                 "/static/app.js", "/static/js/mapview.js", "/static/js/geo.js", "/static/js/style.js"):
        r = client.get(path)
        assert r.status_code == 200, path
        if path.startswith("/static/js") or path.endswith("app.js"):
            assert "cdn." not in r.text and "unpkg" not in r.text and "https://" not in r.text, path
    assert "abort-btn" in html


def test_status_carries_boot_id(env):
    """The phone clears its chat thread when the boot id changes (app restarted, drafts gone)."""
    client, _, _ = env
    a = client.get("/api/status").json()["data"]["boot_id"]
    assert a and a == client.get("/api/status").json()["data"]["boot_id"]
