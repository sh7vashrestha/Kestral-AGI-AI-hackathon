#!/usr/bin/env python3
"""One-time ONLINE step: cut an offline vector basemap for the flying area.

Extracts a bounding box from the public Protomaps daily planet build (OpenStreetMap data,
ODbL; (c) OpenStreetMap contributors) into data/maps/area.pmtiles using the official
`pmtiles extract` tool (HTTP range requests: only the needed tiles are downloaded).

    python scripts/fetch_map.py --center 33.9371904,-84.5271293 --radius-km 1.5
    python scripts/fetch_map.py --bbox minlon,minlat,maxlon,maxlat --maxzoom 17

If `pmtiles` is not on PATH the pinned go-pmtiles release is downloaded once into
data/maps/bin/. Satellite imagery is NOT downloaded here: drop your own raster .mbtiles into
data/maps/ and set MAP_SATELLITE_MBTILES=<file> in .env.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "maps"
PMTILES_VERSION = "1.31.2"
BUILD_URL = "https://build.protomaps.com/{date}.pmtiles"
log = logging.getLogger("fetch_map")


def bbox_from_center(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def find_tool(out_dir: Path) -> str:
    found = shutil.which("pmtiles")
    if found:
        return found
    local = out_dir / "bin" / "pmtiles"
    if local.is_file():
        return str(local)
    arch = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "x86_64", "amd64": "x86_64"}.get(platform.machine())
    if sys.platform != "linux" or arch is None:
        raise SystemExit("pmtiles CLI not found. Install it from https://github.com/protomaps/go-pmtiles/releases")
    url = (f"https://github.com/protomaps/go-pmtiles/releases/download/v{PMTILES_VERSION}/"
           f"go-pmtiles_{PMTILES_VERSION}_Linux_{arch}.tar.gz")
    log.info("downloading %s", url)
    (out_dir / "bin").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tgz = Path(td) / "pm.tgz"
        urllib.request.urlretrieve(url, tgz)  # noqa: S310 (pinned https URL)
        with tarfile.open(tgz) as t:
            member = next(m for m in t.getmembers() if m.name == "pmtiles")
            with t.extractfile(member) as src, open(local, "wb") as dst:  # type: ignore[arg-type]
                shutil.copyfileobj(src, dst)
    local.chmod(0o755)
    return str(local)


def latest_build(max_back: int = 14) -> str:
    today = dt.datetime.now(dt.timezone.utc).date()
    for i in range(1, max_back + 1):
        d = (today - dt.timedelta(days=i)).strftime("%Y%m%d")
        req = urllib.request.Request(BUILD_URL.format(date=d), method="HEAD", headers={"User-Agent": "drone-brain/0.1 (fetch_map)"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:  # noqa: S310
                if r.status == 200:
                    return d
        except Exception as e:  # noqa: BLE001
            log.info("build %s not available (%s)", d, e)
    raise SystemExit("no Protomaps daily build found; pass --build YYYYMMDD or --source URL")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bbox", help="minlon,minlat,maxlon,maxlat")
    g.add_argument("--center", help="lat,lon (use with --radius-km)")
    ap.add_argument("--radius-km", type=float, default=1.5)
    ap.add_argument("--maxzoom", type=int, default=18)
    ap.add_argument("--build", help="Protomaps build date YYYYMMDD (default: newest available)")
    ap.add_argument("--source", help="explicit source .pmtiles URL or path (overrides --build)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "area.pmtiles")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if a.bbox:
        bbox = tuple(float(x) for x in a.bbox.split(","))
        if len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            ap.error("--bbox must be minlon,minlat,maxlon,maxlat")
        centre = ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
    else:
        lat, lon = (float(x) for x in a.center.split(","))
        bbox = bbox_from_center(lat, lon, a.radius_km)
        centre = (lat, lon)
    if (bbox[2] - bbox[0]) * 111 > 60 or (bbox[3] - bbox[1]) * 111 > 60:
        ap.error("area larger than ~60 km; choose a smaller bbox")

    out: Path = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tool = find_tool(out.parent)
    build = a.build or latest_build()
    source = a.source or BUILD_URL.format(date=build)
    tmp = out.with_suffix(".part")
    cmd = [tool, "extract", source, str(tmp), f"--bbox={','.join(f'{v:.6f}' for v in bbox)}",
           f"--maxzoom={a.maxzoom}"]
    log.info("running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"extract failed: {e}")
    os.replace(tmp, out)
    meta = {"center": list(centre), "bbox": list(bbox), "maxzoom": a.maxzoom, "source": source,
            "build": build if not a.source else None, "size_bytes": out.stat().st_size,
            "attribution": "(c) OpenStreetMap contributors (ODbL), Protomaps"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    log.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
