"""Offline map serving (Phase 2b): PMTiles with HTTP Range, MBTiles raster tiles, map info.

Everything is read from <data_dir>/maps (settings.map_dir). Nothing here touches the network.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import struct
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from config import FIELD_HOME, Settings

log = logging.getLogger(__name__)

AIRSPACE_NOTICE = ""   # removed at the operator's request (2026-10-21): the app does not check airspace
PMTILES_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*\.pmtiles$")
MAX_RANGE_BYTES = 16 * 1024 * 1024
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


def read_pmtiles_header(path: Path) -> Optional[dict]:
    """Parse the fixed 127-byte PMTiles v3 header. None if the file is not PMTiles v3."""
    try:
        with open(path, "rb") as f:
            h = f.read(127)
    except OSError:
        return None
    if len(h) < 127 or h[:7] != b"PMTiles" or h[7] != 3:
        return None
    min_zoom, max_zoom = h[100], h[101]
    min_lon, min_lat, max_lon, max_lat = (v / 1e7 for v in struct.unpack_from("<4i", h, 102))
    center_zoom = h[118]
    clon, clat = (v / 1e7 for v in struct.unpack_from("<2i", h, 119))
    return {"minzoom": min_zoom, "maxzoom": max_zoom, "bounds": [min_lon, min_lat, max_lon, max_lat],
            "center": [clat, clon], "center_zoom": center_zoom, "tile_type": h[99]}


def _sat_path(settings: Settings) -> Optional[Path]:
    name = (settings.map_satellite_mbtiles or "").strip()
    if not name:
        return None
    p = Path(name)
    if not p.is_absolute():
        p = Path(settings.map_dir) / p
    return p if p.is_file() else None


def _open_mbtiles(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0, check_same_thread=False)
    return con


def mbtiles_metadata(path: Path) -> dict:
    try:
        con = _open_mbtiles(path)
        try:
            return {k: v for k, v in con.execute("SELECT name, value FROM metadata")}
        finally:
            con.close()
    except sqlite3.Error as e:
        log.warning("cannot read mbtiles metadata %s: %s", path, e)
        return {}


def sniff_image(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _parse_range(header: str, size: int) -> Optional[tuple[int, int]]:
    """Single 'bytes=a-b' range -> inclusive (start, end); None = unsatisfiable/invalid."""
    m = _RANGE.match(header.strip())
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return None
    if m.group(1) == "":                      # suffix: last N bytes
        n = int(m.group(2))
        if n == 0:
            return None
        return max(0, size - n), size - 1
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else size - 1
    end = min(end, size - 1)
    if start >= size or start > end:
        return None
    return start, end


def register_map_routes(app: FastAPI, settings: Settings) -> None:
    def maps_root() -> Path:
        return Path(settings.map_dir).resolve()

    def list_pmtiles() -> list[Path]:
        root = maps_root()
        if not root.is_dir():
            return []
        return sorted(p for p in root.iterdir() if p.is_file() and PMTILES_NAME.match(p.name))

    @app.get("/api/map/info")
    def map_info() -> dict:
        layers: list[dict] = []
        bounds: Optional[list[float]] = None
        center: Optional[list[float]] = None
        for p in list_pmtiles():
            hd = read_pmtiles_header(p)
            if hd is None:
                log.warning("ignoring invalid pmtiles file %s", p.name)
                continue
            layers.append({"id": p.stem, "kind": "vector" if hd["tile_type"] == 1 else "raster",
                           "type": "pmtiles", "url": f"/maps/{p.name}", "minzoom": hd["minzoom"],
                           "maxzoom": hd["maxzoom"], "bounds": hd["bounds"], "size": p.stat().st_size})
            if bounds is None:
                bounds, center = hd["bounds"], hd["center"]
        sat = _sat_path(settings)
        if sat is not None:
            md = mbtiles_metadata(sat)
            fmt = (md.get("format") or "jpg").lower()
            ext = "png" if fmt == "png" else ("webp" if fmt == "webp" else "jpg")
            sb = None
            try:
                sb = [float(x) for x in md["bounds"].split(",")]
            except (KeyError, ValueError, AttributeError):
                pass
            layers.append({"id": "satellite", "kind": "raster", "type": "mbtiles",
                           "tiles": [f"/tiles/sat/{{z}}/{{x}}/{{y}}.{ext}"],
                           "minzoom": int(md.get("minzoom", 0) or 0), "maxzoom": int(md.get("maxzoom", 19) or 19),
                           "bounds": sb, "name": md.get("name", "satellite")})
            if bounds is None and sb and len(sb) == 4:
                bounds, center = sb, [(sb[1] + sb[3]) / 2, (sb[0] + sb[2]) / 2]
        if bounds and bounds[0] <= FIELD_HOME[1] <= bounds[2] and bounds[1] <= FIELD_HOME[0] <= bounds[3]:
            center = list(FIELD_HOME)  # fallback view (KSU field) for big maps (USA); the page moves to the drone home once known
        return {"has_map": bool(layers), "layers": layers, "bounds": bounds, "center": center,
                "attribution": "(c) OpenStreetMap contributors, Protomaps",
                "notice": AIRSPACE_NOTICE}

    @app.api_route("/maps/{name}", methods=["GET", "HEAD"])
    def pmtiles_file(name: str, request: Request) -> Response:
        if not PMTILES_NAME.match(name) or ".." in name:
            raise HTTPException(status_code=404, detail="not found")
        root = maps_root()
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise HTTPException(status_code=404, detail="not found")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        size = path.stat().st_size
        base = {"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"}
        rng = request.headers.get("range")
        if rng is None:
            if request.method == "HEAD":
                return Response(headers={**base, "Content-Length": str(size)},
                                media_type="application/octet-stream")
            return FileResponse(path, media_type="application/octet-stream", headers=base)
        parsed = _parse_range(rng, size)
        if parsed is None:
            return Response(status_code=416, headers={**base, "Content-Range": f"bytes */{size}"})
        start, end = parsed
        end = min(end, start + MAX_RANGE_BYTES - 1)
        headers = {**base, "Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1)}
        if request.method == "HEAD":
            return Response(status_code=206, headers=headers, media_type="application/octet-stream")
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(end - start + 1)
        return Response(content=data, status_code=206, headers=headers, media_type="application/octet-stream")

    @app.get("/tiles/sat/{z}/{x}/{y}.{ext}")
    def sat_tile(z: int, x: int, y: int, ext: str) -> Response:
        path = _sat_path(settings)
        if path is None or ext not in ("jpg", "jpeg", "png", "webp"):
            raise HTTPException(status_code=404, detail="no satellite layer")
        if not (0 <= z <= 24) or x < 0 or y < 0 or x >= (1 << z) or y >= (1 << z):
            raise HTTPException(status_code=404, detail="bad tile")
        tms_y = (1 << z) - 1 - y
        try:
            con = _open_mbtiles(path)
            try:
                row = con.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                                  (z, x, tms_y)).fetchone()
            finally:
                con.close()
        except sqlite3.Error as e:
            log.warning("mbtiles read failed: %s", e)
            raise HTTPException(status_code=503, detail="map database unavailable")
        if row is None:
            raise HTTPException(status_code=404, detail="no such tile")
        data = bytes(row[0])
        return Response(content=data, media_type=sniff_image(data),
                        headers={"Cache-Control": "public, max-age=86400"})
