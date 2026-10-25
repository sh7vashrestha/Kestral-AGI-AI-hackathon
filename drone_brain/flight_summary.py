"""Post-flight summary (notes/go-flow.md): a short plain-text report by Gemma (TEXT ONLY) + the best alert images.

    res = await summarize(flight_dir, llm, data_dir=settings.data_dir)    # never raises for LLM problems
    res == {"text", "flight_dir", "source": "gemma" | "template", "ts", "images": [{url, caption, conf, lat, lon, t}]}

The phone is usually out of WiFi range during the flight, so the result is persistent: <flight_dir>/summary.md (the
text) + summary.json (the dict above). `load_last(data_dir)` returns the newest one (after a restart too).

* Gemma gets a compact digest of flight.json / alerts.jsonl / telemetry.jsonl as text and never any pixels.
* Image captions are built here from the alert metadata (confidence, position, time), not by the LLM.
* LLM unavailable / timeout / empty answer -> `template_text()` (deterministic), source "template".
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

MAX_IMAGES = 6
MAX_ALERT_LINES = 10
MAX_WORDS = 220            # Gemma is asked for <= 200 words; a longer answer is cut here
LLM_MAX_TOKENS = 400
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
LEVEL_LABEL = {"target_match": "Target match", "person": "Person", "heat": "Heat signature"}

SYSTEM = ("You write the post-flight report of a search drone for its operator, who reads it on a phone after "
          "landing. Plain text only (no markdown, no lists with symbols), at most 200 words. Use ONLY the facts "
          "given; never invent detections, positions or numbers. Say: what was searched and for whom, how long and "
          "how far it flew, what was found (best confidence first, with position and time), anything abnormal "
          "(abort, RTL reason, mode changes, link losses, GPS or home warnings), and what the operator should check "
          "next (e.g. go to the best position, review the images). Heat signatures without a visual person are low "
          "priority (often sun-warmed ground or cars): mention them last and briefly, never as the best find.")


# ---------------------------------------------------------------- reading the flight record
def _read_jsonl(path: Path, limit: int = 20000) -> list[dict]:
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict):
                    out.append(d)
    except OSError:
        pass
    return out


def _conf(a: dict) -> float:
    for k in ("c_avg", "c_final", "c_yolo"):
        v = a.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return 0.0


def _mmss(s: Optional[float]) -> str:
    if s is None or s < 0:
        return "?"
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def best_per_track(alerts: list[dict]) -> list[dict]:
    """One alert per track (the highest confidence; a target match beats a plain person alert), best first.
    Heat-only alerts rank last: in daylight they are mostly sun-warmed ground and cars (field flight 2026-10-22)."""
    rank = {"target_match": 2, "person": 1, "heat": 0}
    best: dict[Any, dict] = {}
    for a in alerts:
        tid = a.get("track_id")
        cur = best.get(tid)
        key = (rank.get(a.get("level"), 0), _conf(a))
        if cur is None or key > (rank.get(cur.get("level"), 0), _conf(cur)):
            best[tid] = a
    return sorted(best.values(), key=lambda a: (a.get("level") != "heat", _conf(a)), reverse=True)


def digest(flight_dir: Path) -> dict:
    """Everything the summary needs, from the files the controller / pipeline already wrote."""
    flight_dir = Path(flight_dir)
    try:
        meta = json.loads((flight_dir / "flight.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    summ = meta.get("summary") or {}
    plan = meta.get("plan") or {}
    start = summ.get("start_ts") or meta.get("start_ts") or 0.0
    alerts = _read_jsonl(flight_dir / "alerts.jsonl")
    modes: list[str] = []
    events: list[dict] = []
    for row in _read_jsonl(flight_dir / "telemetry.jsonl"):
        if "event" in row:
            events.append(row)
            continue
        m = row.get("mode")
        if m and (not modes or modes[-1] != m):
            modes.append(m)
    targets = meta.get("targets") or []
    return {
        "flight_dir": flight_dir, "start_ts": start, "summary": summ, "plan": plan, "alerts": alerts,
        "tracks": best_per_track(alerts), "modes": modes, "events": events,
        "target": summ.get("target") or (targets[-1].get("description") if targets else None) or "any person",
    }


def _alert_line(a: dict, start: float) -> str:
    pos = (f"at {a['lat']:.6f}, {a['lon']:.6f}" if isinstance(a.get("lat"), (int, float)) and isinstance(a.get("lon"), (int, float))
           else "position unknown")
    t = _mmss((a.get("ts") or start) - start) if start else "?"
    lab = LEVEL_LABEL.get(a.get("level"), str(a.get("level")))
    txt = f"{lab} (track {a.get('track_id')}): confidence {_conf(a):.0%}, {pos}, T+{t}"
    if a.get("reason"):
        txt += f", {str(a['reason'])[:120]}"
    return txt


def digest_text(d: dict) -> str:
    """The text Gemma sees (no pixels, no file paths)."""
    s, p = d["summary"], d["plan"]
    lines = [f"Search target: {d['target']}."]
    if d["summary"].get("demo"):
        lines.append("GROUND DEMO: the drone did not fly. A camera on the ground searched the people in a room live; "
                     "ignore distance, altitude, positions and vehicle modes and report who was found and how confidently.")
    if p:
        lines.append(f"Plan: {len(p.get('waypoints') or [])} waypoints at {p.get('altitude_m', '?')} m, "
                     f"{p.get('speed_ms', '?')} m/s, planned path {p.get('path_length_m', '?')} m.")
    lines.append(f"Flight: duration {_mmss(s.get('duration_s'))} (min:s), distance flown {s.get('distance_m', 0):.0f} m, "
                 f"max altitude {s.get('max_alt_m', 0):.0f} m.")
    end = s.get("end_reason") or "unknown"
    lines.append(f"End: {end}" + (" (the operator pressed ABORT)" if s.get("aborted") else "") + ".")
    if s.get("rtl_reasons"):
        lines.append("RTL commanded because: " + "; ".join(str(r) for r in s["rtl_reasons"][:5]) + ".")
    if d["modes"]:
        lines.append("Vehicle modes in order: " + " -> ".join(d["modes"][:12]) + ".")
    lines.append(f"MAVLink link losses: {s.get('link_losses', 0)}. Home changes in flight: {s.get('home_changes', 0)}.")
    n = len(d["alerts"])
    if not n:
        lines.append("Alerts: none (nothing was detected).")
    else:
        lines.append(f"Alerts: {n} on {len(d['tracks'])} track(s). Best per track:")
        lines += ["- " + _alert_line(a, d["start_ts"]) for a in d["tracks"][:MAX_ALERT_LINES]]
    vis = s.get("vision") or {}
    if isinstance(vis, dict) and vis.get("state") not in (None, "running", "ok"):
        lines.append(f"Vision pipeline: {vis.get('state')} {vis.get('reason') or ''}".strip() + ".")
    return "\n".join(lines)


def template_text(d: dict) -> str:
    """Deterministic fallback when the LLM is unavailable."""
    s, p = d["summary"], d["plan"]
    if s.get("demo"):
        parts = [f"Ground demo search for {d['target']} (nothing flew), {_mmss(s.get('duration_s'))} min."]
        parts.append(f"{len(d['alerts'])} alert(s) on {len(d['tracks'])} person track(s)." if d["tracks"] else "No one was detected.")
        if d["tracks"]:
            a = d["tracks"][0]
            parts.append(f"Best: {LEVEL_LABEL.get(a.get('level'), a.get('level'))}, confidence {_conf(a):.0%}.")
        return " ".join(parts)
    parts = [f"Search for {d['target']}"
             + (f" over {len(p.get('waypoints') or [])} waypoints at {p.get('altitude_m', '?')} m" if p else "") + "."]
    parts.append(f"Flew {_mmss(s.get('duration_s'))} min, {s.get('distance_m', 0):.0f} m, max altitude {s.get('max_alt_m', 0):.0f} m.")
    end = s.get("end_reason") or "unknown"
    parts.append(f"Ended: {end}" + (", aborted by the operator" if s.get("aborted") else "") + ".")
    if s.get("rtl_reasons"):
        parts.append("RTL reason: " + str(s["rtl_reasons"][0]) + ".")
    if d["tracks"]:
        a = d["tracks"][0]
        parts.append(f"{len(d['alerts'])} alert(s) on {len(d['tracks'])} track(s). Best: " + _alert_line(a, d["start_ts"]) + ".")
    else:
        parts.append("No detections.")
    if len(d["modes"]) > 1:
        parts.append("Modes: " + " -> ".join(d["modes"][:12]) + ".")
    if s.get("link_losses"):
        parts.append(f"MAVLink link lost {s['link_losses']} time(s).")
    if s.get("home_changes"):
        parts.append(f"Home changed {s['home_changes']} time(s) in flight.")
    if d["tracks"]:
        parts.append("Review the images and check the best position.")
    return " ".join(parts)


def pick_images(d: dict, data_dir: Path, limit: int = MAX_IMAGES) -> list[dict]:
    """Up to `limit` saved alert crops (one per track, best confidence first) that exist under data_dir."""
    out: list[dict] = []
    root = Path(data_dir)
    for a in d["tracks"]:
        url = a.get("image_url")
        if not isinstance(url, str) or not url.startswith("/data/"):
            continue
        try:
            if not (root / url[len("/data/"):]).is_file():
                continue
        except (OSError, ValueError):
            continue
        t = round((a.get("ts") or 0.0) - d["start_ts"], 1) if d["start_ts"] and a.get("ts") else None
        lat, lon = a.get("lat"), a.get("lon")
        cap = f"{LEVEL_LABEL.get(a.get('level'), str(a.get('level')))} {_conf(a):.0%}"
        if a.get("level") == "target_match" and a.get("target_description"):
            cap += f" ({a['target_description']})"
        cap += f" at {lat:.5f}, {lon:.5f}" if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) else ", position unknown"
        if t is not None:
            cap += f", T+{_mmss(t)}"
        out.append({"url": url, "caption": cap, "conf": round(_conf(a), 3), "lat": lat, "lon": lon, "t": t})
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- LLM
def _clip_words(text: str, n: int = MAX_WORDS) -> str:
    words = text.split()
    return text.strip() if len(words) <= n else " ".join(words[:n]) + " ..."


async def ask_llm(llm: Any, text: str, timeout_s: float) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    fn = getattr(llm, "achat", None)
    if fn is not None:
        coro = fn(messages, max_tokens=LLM_MAX_TOKENS, extra=NO_THINK)
    else:
        coro = asyncio.to_thread(llm.chat, messages, max_tokens=LLM_MAX_TOKENS, extra=NO_THINK)
    out = await asyncio.wait_for(coro, timeout_s)
    if not isinstance(out, str) or not out.strip():
        raise ValueError("empty LLM answer")
    return _clip_words(out.replace("**", "").replace("#", ""))


async def summarize(flight_dir: Path, llm: Any = None, *, data_dir: Optional[Path] = None,
                    timeout_s: float = 60.0) -> dict:
    """Build, save (summary.md + summary.json) and return the summary. Never raises for LLM problems."""
    flight_dir = Path(flight_dir)
    data_dir = Path(data_dir) if data_dir is not None else flight_dir.parent.parent
    d = await asyncio.to_thread(digest, flight_dir)
    source, text = "template", ""
    if llm is not None:
        try:
            text = await ask_llm(llm, digest_text(d), timeout_s)
            source = "gemma"
        except Exception as exc:  # noqa: BLE001 - LLM down / timeout / bad answer: the template still informs
            log.warning("flight summary: LLM failed (%s: %s); using the template", type(exc).__name__, exc)
    if source != "gemma":
        text = template_text(d)
    try:
        rel = flight_dir.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        rel = flight_dir.name
    res = {"text": text, "flight_dir": rel, "source": source, "ts": time.time(),
           "images": await asyncio.to_thread(pick_images, d, data_dir)}
    await asyncio.to_thread(save, flight_dir, res)
    return res


def summarize_template(flight_dir: Path, data_dir: Optional[Path] = None) -> dict:
    """Synchronous template-only summary (shutdown path: no event loop work, no LLM)."""
    flight_dir = Path(flight_dir)
    data_dir = Path(data_dir) if data_dir is not None else flight_dir.parent.parent
    d = digest(flight_dir)
    try:
        rel = flight_dir.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        rel = flight_dir.name
    res = {"text": template_text(d), "flight_dir": rel, "source": "template", "ts": time.time(),
           "images": pick_images(d, data_dir)}
    save(flight_dir, res)
    return res


def save(flight_dir: Path, res: dict) -> None:
    try:
        (flight_dir / "summary.md").write_text(res["text"].strip() + "\n", encoding="utf-8")
        tmp = flight_dir / "summary.json.tmp"
        tmp.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
        tmp.replace(flight_dir / "summary.json")
    except OSError as exc:
        log.error("saving the flight summary in %s failed: %s", flight_dir, exc)


def load_last(data_dir: Path) -> Optional[dict]:
    """The newest flight's saved summary (summary.json, else summary.md as text only), or None."""
    root = Path(data_dir) / "flights"
    try:
        dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    except OSError:
        return None
    for d in dirs[:50]:
        js, md = d / "summary.json", d / "summary.md"
        try:
            if js.is_file():
                res = json.loads(js.read_text(encoding="utf-8"))
                if isinstance(res, dict) and isinstance(res.get("text"), str):
                    res.setdefault("images", [])
                    return res
            if md.is_file():
                return {"text": md.read_text(encoding="utf-8").strip(), "flight_dir": f"flights/{d.name}",
                        "source": "template", "ts": md.stat().st_mtime, "images": []}
        except (OSError, ValueError):
            continue
    return None
