"""Parse pasted 'lat, lon' lines into corners (done in Python, never by the LLM).

Accepted per point (one point per line, or points separated by ';'):
    33.9371, -84.5271          decimal, comma / tab / space separated
    33,9371; -84,5271          comma-decimal locale (';' between lat and lon)
    33.9371N 84.5271W          hemisphere letter before or after the number
    N33.9371 W84.5271
    33°56'13.9"N 84°31'37.7"W  degrees minutes seconds   (also 33°56.231'N = degrees + decimal minutes)
    (33.9371, -84.5271)        Google Maps copy formats, incl. maps URLs (@lat,lon / !3d..!4d.. / ?q=lat,lon)
Longitude-first input is detected from hemisphere letters only; a first value outside +-90 is rejected with a swap hint
(never silently swapped; pass home= to parse_corners to also catch swapped values that are individually valid).
A closing point (first corner repeated at the end, as in KML/GeoJSON) is dropped.
Everything else that is ambiguous is rejected with a message that says what to fix.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from models import LatLon

MAX_TEXT_CHARS = 20000
_N = r"\d+(?:[.,]\d+)?"
_HEMI_CH = "NSEWnsew"
_DEG_START = re.compile(rf"[-+]?\s*{_N}\s*°")
_DECIMAL_COMMA = re.compile(rf"^[-+NSEWnsew\s]*\d+,\d{{3,}}[NSEWnsew\s]*$")
_PLAIN = re.compile(rf"([{_HEMI_CH}])\s*([-+]?{_N})|([-+]?{_N})\s*([{_HEMI_CH}])?(?![\d.])")
_DMS = re.compile(
    rf"^\s*(?:([{_HEMI_CH}])\s*)?([-+])?\s*({_N})\s*°\s*(?:({_N})\s*'\s*)?(?:({_N})\s*\"\s*)?([{_HEMI_CH}])?\s*$")
_URL_PATTERNS = [
    re.compile(r"!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+)?)"),
    re.compile(r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)"),
    re.compile(r"[?&](?:q|ll|query|center)=(-?\d+(?:\.\d+)?)(?:,|%2C)\s*(-?\d+(?:\.\d+)?)", re.I),
]


def _normalise(text: str) -> str:
    tr = {"−": "-", "–": "-", "—": "-", "′": "'", "’": "'", "‘": "'",
          "″": '"', "”": '"', "“": '"', "º": "°", "˚": "°", " ": " ",
          " ": " ", "﻿": ""}
    for a, b in tr.items():
        text = text.replace(a, b)
    text = text.replace("''", '"')
    return text.replace("\r", "\n")


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _one(tok: str) -> tuple[float, str]:
    """Return (signed value, hemisphere letter or '')."""
    t = tok.strip()
    m = _DMS.match(t)
    if m:
        pre, sign, d, mi, se, suf = m.groups()
        val = _num(d) + (_num(mi) / 60.0 if mi else 0.0) + (_num(se) / 3600.0 if se else 0.0)
        if mi and _num(mi) >= 60 or se and _num(se) >= 60:
            raise ValueError(f"minutes/seconds in '{t}' must be below 60")
        neg = sign == "-"
        h = (pre or suf or "").upper()
        if pre and suf and pre.upper() != suf.upper():
            raise ValueError(f"'{t}' has two different hemisphere letters")
    else:
        m = _PLAIN.fullmatch(t)
        if not m:
            raise ValueError(f"cannot read '{t}' as a coordinate")
        if m.group(1):
            h, num = m.group(1).upper(), m.group(2)
        else:
            h, num = (m.group(4) or "").upper(), m.group(3)
        neg = num.startswith("-")
        val = _num(num.lstrip("+-"))
    if not math.isfinite(val):
        raise ValueError(f"'{t}' is not a finite number")
    if h in ("S", "W"):
        if neg:
            raise ValueError(f"'{t}' has both a minus sign and the letter {h}; use only one")
        neg = True
    elif h in ("N", "E") and neg:
        raise ValueError(f"'{t}' has a minus sign with the letter {h}; use only one")
    return (-val if neg else val), h


def _split_point(text: str) -> list[str]:
    """Split one point's text into its two coordinate tokens."""
    t = text.strip().strip("()[]{}").strip()
    if "°" in t:
        starts = [m.start() for m in _DEG_START.finditer(t)]
        if len(starts) == 2:
            first, second = t[:starts[1]].strip(), t[starts[1]:].strip()
            if first[:1] in _HEMI_CH and first[-1:] in _HEMI_CH and first[:1].isalpha():
                second, first = first[-1] + second, first[:-1].strip()   # prefix style 'N33° W84°'
            return [first.strip(",; "), second.strip(",; ")]
        return [p for p in re.split(r"[;,]", t) if p.strip()]
    if ";" in t:
        return [p for p in t.split(";") if p.strip()]
    if "," in t:
        decimal_commas = len(re.findall(r"(?<=\d),(?=\d)", t))
        if "." not in t and decimal_commas >= 2 and re.search(r"\s", t):
            return t.split()          # '33,9371 -84,5271'
        return [p for p in t.split(",") if p.strip()]
    return [m.group(0) for m in _PLAIN.finditer(t)]


def _from_url(line: str) -> Optional[tuple[float, float]]:
    if "http" not in line.lower() and "google" not in line.lower():
        return None
    for pat in _URL_PATTERNS:
        m = pat.search(line)
        if m:
            return float(m.group(1)), float(m.group(2))
    raise ValueError("could not find coordinates in that link; paste 'lat, lon' instead")


def _km_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    dn = math.radians(b[0] - a[0]) * 6371.0
    de = math.radians(((b[1] - a[1] + 180) % 360) - 180) * 6371.0 * math.cos(math.radians((a[0] + b[0]) / 2))
    return math.hypot(dn, de)


def parse_point(text: str) -> tuple[float, float]:
    """One point -> (lat, lon). Raises ValueError with a user-facing message."""
    u = _from_url(text)
    if u:
        a, b = u
        ha = hb = ""
    else:
        parts = _split_point(text)
        if len(parts) != 2:
            raise ValueError(f"expected 'lat, lon' (two numbers), got '{text.strip()}'")
        (a, ha), (b, hb) = _one(parts[0]), _one(parts[1])
    if ha in ("N", "S") and hb in ("N", "S") or ha in ("E", "W") and hb in ("E", "W"):
        raise ValueError(f"'{text.strip()}' gives two {'latitudes' if ha in ('N', 'S') else 'longitudes'}; "
                         f"need one latitude (N/S) and one longitude (E/W)")
    if ha in ("E", "W") or hb in ("N", "S"):     # written lon, lat
        a, b = b, a
    if not -90 <= a <= 90:
        hint = " Are latitude and longitude swapped? Write 'latitude, longitude'." if abs(b) <= 90 else ""
        raise ValueError(f"latitude {a:g} must be within +-90.{hint}")
    if not -180 <= b <= 180:
        raise ValueError(f"longitude {b:g} must be within +-180")
    return a, b


def parse_corners(text: str, max_points: int = 8, home: Optional[tuple[float, float]] = None) -> list[LatLon]:
    """Parse 3..max_points corners; raises ValueError (user-facing message).

    If ``home`` (lat, lon) is given and the corners are far from it while the lat/lon-swapped
    reading would be close, the input is rejected as probably swapped (never silently swapped)."""
    if not isinstance(text, str):
        raise ValueError("Paste the corners as text, one 'lat, lon' per line.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("That is far too much text for a list of corners.")
    text = _normalise(text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    pts: list[tuple[float, float]] = []
    for i, ln in enumerate(lines, start=1):
        # '33.9,-84.5;34.0,-84.6' = several points on one line; '33,9;-84,5' = one point
        if ";" in ln and "http" not in ln.lower():
            segs = [x for x in ln.split(";") if x.strip()]
            try:
                if len(segs) == 2 and all(_DECIMAL_COMMA.match(x) for x in segs):
                    pass                      # '33,9371; -84,5271': one point, comma decimals
                elif len(segs) >= 2 and all(len(_split_point(x)) == 2 for x in segs):
                    pts.extend(parse_point(x) for x in segs)
                    continue
            except ValueError as exc:
                raise ValueError(f"line {i}: {exc}") from None
        try:
            pts.append(parse_point(ln))
        except ValueError as exc:
            raise ValueError(f"line {i}: {exc}") from None
        if len(pts) > max_points + 1:
            break
    if len(pts) >= 4 and pts[0] == pts[-1]:
        pts.pop()                                # closing point of a KML/GeoJSON ring
    if len(pts) < 3:
        raise ValueError("Give at least 3 corners, one 'lat, lon' per line.")
    if len(pts) > max_points:
        raise ValueError(f"At most {max_points} corners are supported (got {len(pts)}).")
    seen: dict[tuple[float, float], int] = {}
    for i, p in enumerate(pts, start=1):
        key = (round(p[0], 7), round(p[1], 7))
        if key in seen:
            raise ValueError(f"Corner {i} is the same point as corner {seen[key]}. Remove the duplicate.")
        seen[key] = i
    if home is not None:
        c_lat = sum(p[0] for p in pts) / len(pts)
        c_lon = sum(p[1] for p in pts) / len(pts)
        d = _km_between(home, (c_lat, c_lon))
        if d > 5.0 and abs(c_lon) <= 90 and _km_between(home, (c_lon, c_lat)) < 5.0:
            raise ValueError(f"These corners are {d:.0f} km from home but would be next to home if latitude and "
                             f"longitude were swapped. Write each point as 'latitude, longitude'.")
    return [LatLon(lat=a, lon=b) for a, b in pts]
