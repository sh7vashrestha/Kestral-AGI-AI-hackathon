"""One command that checks the whole drone-brain system (host, LLM, planner, cameras, YOLO/CLIP, Pixhawk, web, files).

    python main.py --check [--quick] [--json] [--only vehicle,camera]
    .venv/bin/python scripts/system_check.py [same flags]

Read-only by construction: the vehicle is opened with VEHICLE_MODE=readonly (every command is refused before any
I/O); nothing is uploaded, armed, or set. If main.py is already running the devices (camera, thermal, serial
port, GPU) are NOT fought over: the report is fetched from the running app (/api/system/check), which answers from
its own live objects ("via running app"). Report is saved to data/system_check.json.

Library use:  report = run_all(settings, quick=False, only=None, app=None)  ->  dict with "checks", "summary", "ready".
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from config import Settings

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
TEST_HOME = (33.9371904, -84.5271293)          # fixed home for the planner check (KSU field)
GROUPS = ["host", "llm", "planner", "camera", "thermal", "yolo", "clip", "vehicle", "web", "files"]
HEAVY_MIN_FREE_GB = 1.2                         # loading YOLO + CLIP needs this much MemAvailable
DEADLINE_S = 75.0
ICON = {"ok": "✅", "warn": "⚠️ ", "fail": "❌", "skip": "⏭️ "}
COLOUR = {"ok": "\033[32m", "warn": "\033[33m", "fail": "\033[31m", "skip": "\033[90m"}
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
ARGUS_FIX = "sudo systemctl restart nvargus-daemon"


# ======================================================================================== plumbing
class Ctx:
    def __init__(self, s: Settings, quick: bool = False, app: Optional[dict] = None) -> None:
        self.s, self.quick, self.app = s, quick, app
        self.via = "running app" if app else None
        self.out: list[dict] = []
        self._lock = threading.Lock()

    def add(self, group: str, name: str, status: str, detail: str = "", fix: str = "", ms: Optional[float] = None,
            via: Optional[str] = None) -> dict:
        c = {"group": group, "name": name, "status": status, "detail": detail, "fix": fix,
             "ms": None if ms is None else round(ms), "via": via}
        with self._lock:
            self.out.append(c)
        return c


def meminfo() -> dict[str, float]:
    d: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            d[k] = float(v.split()[0]) / 1024 / 1024       # GB
    except OSError:
        pass
    return d


def mem_available_gb() -> float:
    return meminfo().get("MemAvailable", 0.0)


def busy_pids(path: str) -> list[int]:
    """PIDs (other than ours) that have `path` open. Scans /proc/*/fd (no fuser/lsof needed; own processes only)."""
    try:
        target = os.path.realpath(path)
    except OSError:
        return []
    me, pids = os.getpid(), []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            for fd in (d / "fd").iterdir():
                try:
                    if os.path.realpath(os.readlink(fd)) == target:
                        pids.append(int(d.name))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return pids


def pid_cmd(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()[:90]
    except OSError:
        return "?"


def other_main_py() -> list[int]:
    """PIDs of other running `main.py` orchestrators (they own the camera / serial port)."""
    out, me = [], os.getpid()
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) != me:
            cmd = pid_cmd(int(d.name))
            if re.search(r"python\S*\s+(\S*/)?main\.py(\s|$)", cmd) and "--check" not in cmd:
                out.append(int(d.name))
    return out


def run_cmd(args: list[str], timeout: float = 5.0) -> tuple[int, str]:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


# ======================================================================================== host
def check_host(c: Ctx) -> None:
    g = "host"
    # JetPack / L4T
    try:
        first = Path("/etc/nv_tegra_release").read_text().splitlines()[0]
        m = re.search(r"R(\d+).*REVISION:\s*([\d.]+)", first)
        l4t = f"L4T R{m.group(1)}.{m.group(2)}" if m else first[:50]
        rc, o = run_cmd(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"], 3)
        jp = f"JetPack {o.split('-')[0]} " if rc == 0 and o else ""
        c.add(g, "JetPack / L4T", "ok", f"{jp}({l4t})")
    except (OSError, IndexError):
        c.add(g, "JetPack / L4T", "warn", "/etc/nv_tegra_release not found: not a Jetson?")
    mem = meminfo()
    avail = mem.get("MemAvailable", 0.0)
    # headless?
    rc, o = run_cmd(["systemctl", "get-default"], 3)
    target = o.strip() if rc == 0 else "unknown"
    desktop = "graphical" in target
    if desktop and avail < 3.0:
        c.add(g, "Display", "warn", f"desktop session ({target}); costs 1-2 GB of the 8 GB shared RAM",
              "sudo systemctl set-default multi-user.target && sudo reboot (headless)")
    else:
        c.add(g, "Display", "ok", "desktop (graphical.target)" if desktop else f"headless ({target})")
    # RAM
    total = mem.get("MemTotal", 0.0)
    c.add(g, "RAM available", "warn" if avail < 2.0 else "ok",
          f"{avail:.1f} GB free of {total:.1f} GB" + (" (low: close the desktop/VS Code or stop unused services)" if avail < 2.0 else ""),
          "free memory: stop VS Code/Firefox, or `sudo systemctl set-default multi-user.target`" if avail < 2.0 else "")
    # swap
    try:
        rows = [ln.split() for ln in Path("/proc/swaps").read_text().splitlines()[1:] if ln.strip()]
    except OSError:
        rows = []
    zram = [r for r in rows if "zram" in r[0]]
    disk_swap = [r for r in rows if "zram" not in r[0]]
    gb = sum(int(r[2]) for r in disk_swap) / 1024 / 1024
    if zram:
        c.add(g, "Swap", "warn", f"zram swap is ON ({len(zram)} devices, lives in RAM)" + (f"; disk swap {gb:.0f} GB too" if disk_swap else ""),
              "sudo swapoff /dev/zram* ; sudo systemctl disable nvzramconfig")
    elif disk_swap:
        c.add(g, "Swap", "ok", f"zram off, disk swap {gb:.0f} GB ({Path(disk_swap[0][0]).name})")
    else:
        c.add(g, "Swap", "warn", "no swap at all (model loading can OOM)", "sudo fallocate -l 32G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile")
    # disk
    try:
        Path(c.s.data_dir).mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(c.s.data_dir).free / 1e9
        c.add(g, "Disk free", "fail" if free < 2 else ("warn" if free < 20 else "ok"), f"{free:.0f} GB free at {Path(c.s.data_dir).resolve()}",
              "free disk space (old flights in data/flights?)" if free < 20 else "")
    except OSError as exc:
        c.add(g, "Disk free", "fail", str(exc))
    # GPU
    t0 = time.perf_counter()
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            free_b, tot_b = torch.cuda.mem_get_info()
            c.add(g, "GPU (torch)", "ok", f"{name}, CUDA {torch.version.cuda}, torch {torch.__version__}, "
                  f"{free_b / 1e9:.1f}/{tot_b / 1e9:.1f} GB GPU-visible free", ms=(time.perf_counter() - t0) * 1000)
        else:
            c.add(g, "GPU (torch)", "fail", f"torch {torch.__version__} cannot see CUDA", "check the Jetson torch wheel / `tegrastats`; reboot")
    except Exception as exc:  # noqa: BLE001
        c.add(g, "GPU (torch)", "fail", f"{type(exc).__name__}: {exc}"[:120], "check the venv torch install")


# ======================================================================================== LLM + planner
def _llm_root(s: Settings) -> str:
    b = s.llm_base_url.rstrip("/")
    return b[:-3] if b.endswith("/v1") else b


def check_llm(c: Ctx) -> bool:
    import httpx
    g = "llm"
    root = _llm_root(c.s)
    t0 = time.perf_counter()
    try:
        r = httpx.get(root + "/health", timeout=3.0)
        ok = r.status_code == 200
        c.add(g, "llama-server /health", "ok" if ok else "fail", f"HTTP {r.status_code} {r.text.strip()[:60]}",
              "" if ok else "scripts/start_llama.sh (model still loading? wait 30 s)", ms=(time.perf_counter() - t0) * 1000)
        if not ok:
            return False
    except httpx.HTTPError as exc:
        c.add(g, "llama-server /health", "fail", f"not reachable at {root} ({type(exc).__name__})", "start it: scripts/start_llama.sh")
        for n in ("model / text-only", "tiny planner call", "JSON schema"):
            c.add(g, n, "skip", "llama-server is down")
        return False
    # model + vision
    try:
        props = httpx.get(root + "/props", timeout=3.0).json()
    except Exception:  # noqa: BLE001
        props = {}
    vision = bool((props.get("modalities") or {}).get("vision"))
    mm_cmd = _llama_has_mmproj()
    try:
        mid = httpx.get(root + "/v1/models", timeout=3.0).json()["data"][0]["id"]
    except Exception:  # noqa: BLE001
        mid = props.get("model_alias") or Path(str(props.get("model_path", "?"))).name
    if vision or mm_cmd:
        c.add(g, "model / text-only", "warn", f"{mid}: mmproj (vision) is loaded, wasting ~1 GB; images must never go to the LLM",
              "restart llama-server without --mmproj (scripts/start_llama.sh)")
    else:
        c.add(g, "model / text-only", "ok", f"{str(mid)[:60]} loaded, text-only (no mmproj)")
    # tiny call, thinking disabled
    from llm_client import LLMClient, LLMError
    from planner import NO_THINK
    from pydantic import BaseModel

    class _Ping(BaseModel):
        answer: str

    cl = LLMClient(settings=c.s, retries=0, timeout_s=40.0)
    t0 = time.perf_counter()
    try:
        txt = cl.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=12, extra=NO_THINK)
        ms = (time.perf_counter() - t0) * 1000
        c.add(g, "tiny planner call", "warn" if ms > 5000 else "ok", f"{ms / 1000:.2f} s, thinking off, reply {txt.strip()[:30]!r}",
              "slow: GPU busy or memory pressure (check tegrastats)" if ms > 5000 else "", ms=ms)
    except LLMError as exc:
        c.add(g, "tiny planner call", "fail", f"{type(exc).__name__}: {exc}"[:110], "check data/llama-server.log")
        c.add(g, "JSON schema", "skip", "chat failed")
        return False
    t0 = time.perf_counter()
    try:
        r = cl.chat_json([{"role": "user", "content": 'Return JSON {"answer": "ok"}'}], _Ping, max_tokens=40, extra=NO_THINK)
        c.add(g, "JSON schema", "ok", f"constrained JSON parsed ({r.answer!r})", ms=(time.perf_counter() - t0) * 1000)
    except LLMError as exc:
        c.add(g, "JSON schema", "fail", f"{type(exc).__name__}: {exc}"[:110], "the planner needs response_format json_schema support")
    return True


def _llama_has_mmproj() -> bool:
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            cmd = pid_cmd(int(d.name))
            if "llama-server" in cmd:
                try:
                    return "--mmproj" in Path(f"/proc/{d.name}/cmdline").read_bytes().decode(errors="replace")
                except OSError:
                    return False
    return False


def check_planner(c: Ctx, llm_ok: bool) -> None:
    g = "planner"
    if not llm_ok:
        c.add(g, "message to plan", "skip", "LLM unavailable")
        return
    import planner
    import safety
    import survey
    from llm_client import LLMClient
    from models import GeoPoint
    s = c.s
    alt = int(min(max(safety.effective_min_alt(s) + 10, 25), safety.effective_max_alt(s)))
    msg = f"search a 60 by 40 m area 30 m north of home at {alt} m altitude for a person in a red jacket"
    t0 = time.perf_counter()
    try:
        intent = planner.plan_from_message(msg, LLMClient(settings=s, retries=1, timeout_s=60.0), s)
        if intent.error or intent.mission is None:
            c.add(g, "message -> plan", "fail", f"planner: {intent.error or 'no mission'}"[:120], "see data/llama-server.log")
            return
        home = GeoPoint(lat=TEST_HOME[0], lon=TEST_HOME[1])
        plan = survey.build_survey(intent.mission, home, s)
        res = safety.validate(plan, s)
        ms = (time.perf_counter() - t0) * 1000
        tgt = intent.target.description if intent.target else "-"
        detail = (f"{len(plan.waypoints)} waypoints, {plan.path_length_m:.0f} m path at {plan.altitude_m:.0f} m; target {tgt!r}; "
                  f"safety {'OK' if res.ok else 'REJECTED: ' + '; '.join(res.errors)[:80]}")
        c.add(g, "message -> plan -> safety", "fail" if not res.ok else ("warn" if res.warnings else "ok"),
              detail + (f" (warn: {res.warnings[0][:50]})" if res.warnings and res.ok else ""),
              "check MAX_ALT_M / OBSTACLE_CLEARANCE_M / LAANC_CEILING_M in .env" if not res.ok else "", ms=ms)
    except Exception as exc:  # noqa: BLE001
        c.add(g, "message -> plan -> safety", "fail", f"{type(exc).__name__}: {exc}"[:120])


# ======================================================================================== RGB camera
def argus_journal(minutes: int = 2) -> Optional[str]:
    """The stuck-argus signature in the last `minutes` of `journalctl -u nvargus-daemon`, or None if clean/unreadable."""
    rc, o = run_cmd(["journalctl", "-u", "nvargus-daemon", "--since", f"{minutes} min ago", "--no-pager", "-q"], 6)
    if rc != 0 or not o.strip():
        return None
    for ln in o.splitlines()[::-1]:
        if re.search(r"No cameras available|SCF|Error|error|fail", ln):
            return ln.strip()[-110:]
    return None


def check_camera(c: Ctx) -> None:
    g, s = "camera", c.s
    name = "RGB (IMX477)"
    if c.quick:
        return c.add(g, name, "skip", "--quick")
    if c.app is not None:
        return _camera_via_app(c)
    busy = other_main_py()
    if busy and s.camera_source.startswith("csi"):
        return c.add(g, name, "skip", f"camera held by running main.py pid {busy[0]} (web API not answering); stop it first")
    from camera import Camera
    t0 = time.perf_counter()
    cam = Camera(s)
    cam.start()
    try:
        if not cam.wait_first(20.0):
            err = cam.last_error or "no frame"
            jr = argus_journal() if s.camera_source.startswith("csi") else None
            fix = ARGUS_FIX if s.camera_source.startswith("csi") else "check CAMERA_SOURCE and the cable"
            c.add(g, name, "fail", f"no frames from {s.camera_source}: {err}" + (f" | journal: {jr}" if jr else ""), fix)
            return
        first_ms = (time.perf_counter() - t0) * 1000
        seen: list[tuple[int, float]] = []
        end = time.monotonic() + 6.0
        last = -1
        while time.monotonic() < end and len(seen) < 10:
            f, ts, fid = cam.latest()
            if fid != last and f is not None:
                last = fid
                seen.append((fid, time.monotonic()))
            time.sleep(0.004)
        f, _, _ = cam.latest()
        h, w = f.shape[:2]
        fps = (len(seen) - 1) / (seen[-1][1] - seen[0][1]) if len(seen) > 1 and seen[-1][1] > seen[0][1] else 0.0
        gray = float(f[::8, ::8].mean())
        std = float(f[::8, ::8].std())
        try:
            import cv2
            d = Path(s.data_dir) / "system_check"
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / "rgb.jpg"), f)
        except Exception:  # noqa: BLE001
            pass
        probs = []
        fix = ""
        if len(seen) < 10:
            probs.append(f"only {len(seen)} frames in 6 s")
        if fps < 5:
            probs.append(f"{fps:.1f} fps is low")
        if (w, h) != (s.eff_camera_width, s.eff_camera_height):
            probs.append(f"{w}x{h}, expected {s.eff_camera_width}x{s.eff_camera_height} (CAMERA_ROTATE={s.camera_rotate})")
        if gray < s.rgb_dark_threshold or std < 3.0:
            probs.append(f"image is DARK/flat (mean {gray:.0f}, sd {std:.0f}): lens cap on?")
            fix = "remove the lens cap / check exposure; YOLO is skipped below RGB_DARK_THRESHOLD"
        if gray > 240:
            probs.append("image is blown out")
        c.add(g, name, "warn" if probs else "ok", f"{w}x{h}, {len(seen)} frames, {fps:.0f} fps, brightness {gray:.0f}/255"
              + (" | " + "; ".join(probs) if probs else "") + f" (first frame {first_ms / 1000:.1f} s)", fix, ms=first_ms)
    finally:
        cam.stop()


def _camera_via_app(c: Ctx) -> None:
    g, name = "camera", "RGB (IMX477)"
    v = c.app["snapshot"].get("vision") or {}
    cs = v.get("camera_status") or {}
    if v.get("state") == "disabled" and not cs:
        return c.add(g, name, "skip", f"vision disabled in the running app ({v.get('reason', '')[:60]})", via=c.via)
    st = cs.get("state")
    info = c.app.get("camera_info") or {}
    if st == "unavailable":
        return c.add(g, name, "fail", f"camera unavailable: {cs.get('error')}", ARGUS_FIX if c.s.camera_source.startswith("csi") else "check the camera cable / CAMERA_SOURCE",
                     via=c.via)
    if st != "ok":
        return c.add(g, name, "warn", f"camera state {st!r} (vision {v.get('state')})", "wait a few seconds and re-run", via=c.via)
    probs = []
    gray = info.get("mean_gray")
    if gray is not None and gray < c.s.rgb_dark_threshold:
        probs.append(f"image is DARK (mean {gray}): lens cap on?")
    if info.get("fps") is not None and info["fps"] < 5:
        probs.append(f"{info['fps']} fps is low")
    if info.get("age_s") is not None and info["age_s"] > 3:
        probs.append(f"last frame {info['age_s']} s old")
    c.add(g, name, "warn" if probs else "ok", f"{info.get('w', '?')}x{info.get('h', '?')}, {info.get('fps', '?')} fps, "
          f"brightness {gray if gray is not None else '?'}/255, {v.get('frames', 0)} frames since start"
          + (" | " + "; ".join(probs) if probs else ""), "remove the lens cap" if probs else "", via=c.via)


# ======================================================================================== thermal
def check_thermal(c: Ctx) -> None:
    g, s, name = "thermal", c.s, "Thermal (Lepton)"
    if c.quick:
        return c.add(g, name, "skip", "--quick")
    if s.thermal_source in ("", "none"):
        return c.add(g, name, "skip", "THERMAL_SOURCE=none")
    if c.app is not None:
        ti = c.app.get("thermal_info") or {}
        if not ti:
            return c.add(g, name, "warn", "the running app has no thermal stream (failed to open?)",
                         "check the USB cable; ls /dev/v4l/by-id; restart main.py", via=c.via)
        return _thermal_line(c, ti, via=c.via)
    path = s.thermal_source.partition(":")[2]
    if s.thermal_source.startswith("v4l2:"):
        if not Path(path).exists():
            return c.add(g, name, "fail", f"{path} not found (PureThermal unplugged?)", "replug USB; ls /dev/v4l/by-id")
        pids = busy_pids(path)
        if pids:
            return c.add(g, name, "skip", f"device busy: pid {pids[0]} ({pid_cmd(pids[0])[:50]})")
    from thermal import ThermalCamera
    cam = ThermalCamera(s).start()
    try:
        t0 = time.monotonic()
        while cam.frames_total < 20 and time.monotonic() - t0 < 15:
            time.sleep(0.05)
        if cam.frames_total < 3:
            return c.add(g, name, "fail", f"no frames from {path}: {cam.last_error}", "replug the PureThermal USB; check dmesg")
        dt = time.monotonic() - t0
        temp, _ = cam.latest()
        import numpy as np
        info = {"fps": round(cam.frames_total / dt, 1), "frames": cam.frames_total, "min_c": float(temp.min()),
                "median_c": float(np.median(temp)), "max_c": float(temp.max()), "stuck_px": int(cam.mask.sum()),
                "stuck_dyn": int(cam.stuck.mask.sum()), "error": cam.last_error}
        _thermal_line(c, info)
    finally:
        cam.stop()


def _thermal_line(c: Ctx, i: dict, via: Optional[str] = None) -> None:
    probs = []
    if i.get("fps", 0) < 4:
        probs.append(f"{i.get('fps')} fps is low (Lepton ~8.6)")
    if "min_c" in i:
        if i["max_c"] - i["min_c"] < 0.5:
            probs.append("flat image (shutter/FFC or covered?)")
        if not (-30 < i["median_c"] < 70):
            probs.append(f"median {i['median_c']:.0f} C is implausible")
        temps = f", {i['min_c']:.1f}/{i['median_c']:.1f}/{i['max_c']:.1f} C (min/med/max)"
    else:
        temps = ", no temperature frame yet"
    c.add("thermal", "Thermal (Lepton)", "warn" if probs else "ok",
          f"{i.get('fps')} fps{temps}, stuck pixels masked: {i.get('stuck_px', '?')}" + (" | " + "; ".join(probs) if probs else ""),
          "replug the PureThermal; run scripts/camera_test.py" if probs else "", via=via)


# ======================================================================================== YOLO + CLIP
def _free_models() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _heavy_ok(c: Ctx, group: str, name: str, need: float = HEAVY_MIN_FREE_GB) -> bool:
    avail = mem_available_gb()
    if avail < need:
        c.add(group, name, "skip", f"only {avail:.1f} GB RAM free (< {need:g}); not loading a second copy of the model. "
              "Run this while main.py is running (it reports its own loaded models) or stop llama-server first")
        return False
    return True


def _asset(rel: str):
    import cv2
    return cv2.imread(str(ROOT / "tests" / "assets" / rel))


def check_yolo(c: Ctx) -> None:
    g, name = "yolo", "YOLO person detector"
    if c.quick:
        return c.add(g, name, "skip", "--quick")
    if c.app is not None:
        mi = c.app.get("models_info") or {}
        v = c.app["snapshot"].get("vision") or {}
        if mi.get("detector"):
            w = mi.get("weights") or "?"
            return c.add(g, name, "warn" if not str(w).endswith(".engine") else "ok",
                         f"{mi['detector']} loaded ({w}); last loop {v.get('last_loop_ms', 0)} ms",
                         "export the TensorRT engine: scripts/export_models.py yolo" if not str(w).endswith(".engine") else "", via=c.via)
        return c.add(g, name, "warn" if v.get("state") != "disabled" else "fail",
                     f"detector not loaded (vision {v.get('state')}: {str(v.get('reason'))[:70]})", "see main.py log", via=c.via)
    from detector import Detector, resolve_weights
    try:
        w = resolve_weights(c.s.yolo_model)
    except FileNotFoundError as exc:
        return c.add(g, name, "fail", str(exc)[:110], "scripts/export_models.py yolo")
    if not _heavy_ok(c, g, name):
        return
    t0 = time.perf_counter()
    try:
        det = Detector(c.s)
        load_ms = (time.perf_counter() - t0) * 1000
        img = _asset("persons/p01_red_jacket_ski.jpg")
        times, n = [], 0
        for _ in range(5):
            t1 = time.perf_counter()
            res = det.detect(img, ts=time.time(), frame_id=0, telemetry=None)
            times.append((time.perf_counter() - t1) * 1000)
            n = len(res)
        med = statistics.median(times)
        probs = []
        if w.suffix != ".engine":
            probs.append("no TensorRT engine (slow)")
        if n == 0:
            probs.append("found no person in the test image")
        c.add(g, name, "warn" if probs else "ok", f"{w.name} loaded in {load_ms / 1000:.1f} s, inference {med:.0f} ms, {n} person(s) in test image"
              + (" | " + "; ".join(probs) if probs else ""), "scripts/export_models.py yolo" if probs else "", ms=med)
        del det
        _free_models()
    except Exception as exc:  # noqa: BLE001
        c.add(g, name, "fail", f"{type(exc).__name__}: {exc}"[:120], "scripts/export_models.py yolo; free GPU memory")


def check_clip(c: Ctx) -> None:
    g, name = "clip", "CLIP matcher"
    if c.quick:
        return c.add(g, name, "skip", "--quick")
    if c.app is not None:
        mi = c.app.get("models_info") or {}
        if mi.get("clip"):
            want_trt = getattr(c.s, "clip_use_trt", False)
            return c.add(g, name, "ok" if mi["clip"] == "tensorrt" or not want_trt else "warn",
                         f"loaded in the running app ({mi['clip']} image tower, backend {mi.get('backend')})",
                         "" if mi["clip"] == "tensorrt" or not want_trt else "export the engine: scripts/export_models.py clip", via=c.via)
        return c.add(g, name, "warn", f"CLIP not loaded in the running app (backend {mi.get('backend')}): person/heat alerts only",
                     "see main.py log (CUDA OOM?)", via=c.via)
    if not _heavy_ok(c, g, name, need=1.0):
        return
    t0 = time.perf_counter()
    try:
        from clip_encoder import ClipEncoder
        import numpy as np
        enc = ClipEncoder(c.s)
        load_s = time.perf_counter() - t0
        imgs = [_asset("persons/p01_red_jacket_ski.jpg"), _asset("persons/p02_white_cap_runner.jpg"), _asset("nonperson/n01_bus_body.jpg")]
        texts = ["a photo of a person in a red jacket", "a photo of a person in a white cap", "a photo of the side of a bus"]
        times = []
        for _ in range(3):
            t1 = time.perf_counter()
            fi = enc.encode_image(imgs)
            times.append((time.perf_counter() - t1) * 1000 / len(imgs))
        ft = enc.encode_text(texts)
        sim = fi @ ft.T
        right = int(sum(int(np.argmax(sim[i])) == i for i in range(3)))
        trt = enc.trt is not None
        probs = []
        if not trt and getattr(c.s, "clip_use_trt", False):
            from clip_encoder import ENGINE_PATH
            probs.append("torch fallback: " + ("TensorRT engine present but failed to load (GPU memory?)" if ENGINE_PATH.exists()
                                               else "no TensorRT engine"))
        if right < 3:
            probs.append(f"ranking sanity {right}/3")
        c.add(g, name, "fail" if right < 2 else ("warn" if probs else "ok"),
              f"{'TensorRT FP16' if trt else 'torch ' + str(enc.dtype).split('.')[-1]} image tower, loaded {load_s:.1f} s, "
              f"{statistics.median(times):.0f} ms/image, text-image ranking {right}/3 correct" + (" | " + "; ".join(probs) if probs else ""),
              "scripts/export_models.py clip" if probs else "", ms=statistics.median(times))
        del enc
    except Exception as exc:  # noqa: BLE001
        c.add(g, name, "fail", f"{type(exc).__name__}: {exc}"[:120], "free GPU memory; scripts/export_models.py clip")


# ======================================================================================== vehicle
def vehicle_hint(s: Settings) -> str:
    url = s.mavlink_url
    if url.startswith("serial:"):
        return f"check the USB cable/baud ({s.mavlink_baud}) and that nothing else has the port open"
    extra = " (a Pixhawk IS at /dev/ttyACM0: run with --mavlink-url serial:/dev/ttyACM0 --baud 115200)" if Path("/dev/ttyACM0").exists() else ""
    return (f"start SITL (scripts/sitl_run.sh), or --mock-vehicle, or --vehicle readonly --mavlink-url serial:/dev/ttyACM0 "
            f"--baud 115200{extra}")


def check_vehicle(c: Ctx) -> None:
    g, s = "vehicle", c.s
    if c.app is not None:
        rep = c.app.get("vehicle")
        if rep is None:
            return c.add(g, "Pixhawk", "fail", f"vehicle check failed in the running app: {c.app.get('vehicle_error', '?')}")
        return vehicle_lines(c, rep, via=c.via)
    url = s.mavlink_url
    tried: list[str] = []
    for u, baud, note in _vehicle_candidates(s):
        res = _probe_vehicle(c, s.model_copy(update={"mavlink_url": u, "mavlink_baud": baud}))
        if isinstance(res, dict):
            if note:
                c.add(g, "Pixhawk link", "warn", f"found the Pixhawk on {u} @ {baud}, but MAVLINK_URL in .env is {url}", note)
            return vehicle_lines(c, res)
        tried.append(res)
        if res.startswith("BUSY"):
            return c.add(g, "Pixhawk", "skip", res[5:])
    c.add(g, "Pixhawk heartbeat", "fail", f"no vehicle on {url}: " + "; ".join(tried)[:130], vehicle_hint(s))


def _vehicle_candidates(s: Settings) -> list[tuple[str, int, str]]:
    """Configured link first; if that is the SITL/UDP default and a Pixhawk is on USB, also try /dev/ttyACM0 (read-only)."""
    out = [(s.mavlink_url, s.mavlink_baud, "")]
    if not s.mavlink_url.startswith("serial:") and Path("/dev/ttyACM0").exists():
        out.append(("serial:/dev/ttyACM0", 115200, "run main.py with --vehicle readonly --mavlink-url serial:/dev/ttyACM0 --baud 115200 (or set MAVLINK_URL/BAUD in .env)"))
    return out


def _probe_vehicle(c: Ctx, s: Settings):
    """Read-only look at one link. Returns the vehicle_check report dict, or an error string ('BUSY ...' if the port is taken)."""
    url = s.mavlink_url
    if url.startswith("serial:"):
        dev = url.split(":", 1)[1]
        if not Path(dev).exists():
            return f"{dev} not found"
        pids = busy_pids(dev)
        if pids:
            return f"BUSY {dev} is busy: pid {pids[0]} ({pid_cmd(pids[0])[:50]}); not fighting for it"
    from mission import ArduPilotBackend
    from vehicle_check import run_vehicle_check
    ro = s.model_copy(update={"vehicle_mode": "readonly"})
    be = ArduPilotBackend(ro)
    assert be.readonly, "system_check must never use a writable vehicle backend"
    try:
        try:
            be.connect(timeout=8.0)
        except Exception as exc:  # noqa: BLE001
            return f"{url}: {type(exc).__name__}: {str(exc)[:70]}"
        time.sleep(2.0)      # let SYS_STATUS / EKF / GPS / HOME arrive
        return run_vehicle_check(be, ro, prearm_wait_s=3.0, param_timeout_s=3.0)
    except Exception as exc:  # noqa: BLE001
        return f"{url}: {type(exc).__name__}: {str(exc)[:80]}"
    finally:
        be.close()


def vehicle_lines(c: Ctx, r: dict, via: Optional[str] = None) -> None:
    """Turn a run_vehicle_check() report into one check line per aspect (no logic duplicated: the report decides)."""
    g = "vehicle"

    def add(name: str, st: str, detail: str, fix: str = "") -> None:
        c.add(g, name, st, detail, fix, via=via)

    link = r.get("link") or {}
    if "gps" not in r:
        reasons = "; ".join(r.get("reasons") or ["no heartbeat"])
        return add("Pixhawk heartbeat", "fail", f"{reasons} ({link.get('url')})", vehicle_hint(c.s))
    v, fw, fr = r.get("vehicle") or {}, r.get("firmware") or {}, r.get("frame") or {}
    add("Pixhawk heartbeat", "ok", f"link up, hb {link.get('heartbeat_age_s')} s, {v.get('type')} frame {fr.get('class')} {fr.get('type')}")
    fwv = fw.get("version")
    add("Firmware", "ok" if fwv else "warn", f"ArduCopter {fwv} ({fw.get('type')})" if fwv else "AUTOPILOT_VERSION not received",
        "" if fwv else "wait and re-run")
    armed = v.get("armed")
    add("Mode / armed", "warn" if armed else "ok", f"{v.get('mode')}, {'ARMED' if armed else 'disarmed'}, {v.get('landed_state') or v.get('system_status') or ''}",
        "disarm before planning" if armed else "")
    gp = r.get("gps") or {}
    sats = gp.get("satellites") or 0
    fix_ok = (gp.get("fix_type") or 0) >= 3
    hd = gp.get("hdop")
    if not fix_ok:
        add("GPS", "fail", f"{gp.get('fix')}, {sats} satellites", "take the drone outside with a clear sky; wait for 3D fix (>= 8 sats)")
    elif sats < 8 or (hd is not None and hd > 2.0):
        add("GPS", "warn" if hd is None or hd <= 2.0 else "fail", f"{gp.get('fix')}, {sats} sats, HDOP {hd}", "wait for more satellites / lower HDOP")
    else:
        add("GPS", "ok", f"{gp.get('fix')}, {sats} sats, HDOP {hd}")
    home = gp.get("home")
    add("Home position", "ok" if home else "fail", f"{home['lat']:.6f}, {home['lon']:.6f} ({home['amsl_m']:.0f} m)" if home else "not set (needs a 3D fix)",
        "" if home else "wait for the 3D fix; ArduPilot sets home at first fix")
    e = r.get("ekf") or {}
    lvl = e.get("level", "UNKNOWN")
    add("EKF", {"OK": "ok", "WARN": "warn", "BAD": "fail"}.get(lvl, "warn"),
        f"{lvl}" + (": " + ", ".join(f"{k} {x}" for k, x in (e.get("variances") or {}).items() if k != "terrain_alt") if e.get("available") else " (no EKF_STATUS_REPORT)"),
        "" if lvl == "OK" else "keep the drone still; check compass/GPS; wait 30-60 s after power-up")
    sn = r.get("sensors") or {}
    unh = sn.get("unhealthy") or []
    add("Sensors", "fail" if unh else ("ok" if sn.get("available") else "warn"),
        ("unhealthy: " + ", ".join(unh)) if unh else ("all enabled sensors healthy" if sn.get("available") else "no SYS_STATUS"),
        "calibrate / reconnect the unhealthy sensor in Mission Planner" if unh else "")
    pa = r.get("prearm") or {}
    msgs = [m.get("text", "") for m in pa.get("messages") or []]
    if msgs:
        add("Pre-arm checks", "fail", f"{len(msgs)} failure(s): " + " | ".join(dict.fromkeys(msgs))[:140], "fix the listed PreArm items (Mission Planner messages)")
    else:
        add("Pre-arm checks", "ok" if pa.get("prearm_bit_ok") is not False else "fail",
            f"no PreArm failures (cmd {pa.get('ack')})", "")
    bat = r.get("battery") or {}
    bp = bat.get("pct")
    add("Battery", "fail" if bp is not None and bp < 30 else ("ok" if bp is not None else "warn"),
        f"{bat.get('voltage_v')} V, {bp if bp is None else round(bp)}%", "charge the battery" if bp is not None and bp < 30 else "")
    rc = r.get("rc") or {}
    add("RC link", "ok" if rc.get("received") and rc.get("chancount") else "fail",
        f"{rc.get('chancount')} channels, rssi {rc.get('rssi')}" if rc.get("received") else "no RC channels (transmitter off / not bound?)",
        "" if rc.get("received") else "turn the transmitter on; bind the receiver (needed as the safety override)")
    for ck in (r.get("preflight") or {}).get("checks", []):
        st = "ok" if ck.get("ok") else ("warn" if ck.get("severity") == "warning" else "fail")
        add(f"param {ck['param']}", st, f"{ck.get('value')}: {ck.get('message') or ck.get('requirement')}"[:130],
            "" if st == "ok" else f"set {ck['param']} in Mission Planner (this app never writes params)")
    add("Vehicle verdict", "ok" if r.get("verdict") == "READY" else "fail",
        r.get("verdict", "?") + ("" if r.get("verdict") == "READY" else ": " + "; ".join(r.get("reasons") or [])[:140]))


# ======================================================================================== web / map / files
def check_web(c: Ctx) -> None:
    g, s = "web", c.s
    static = ROOT / "drone_brain" / "webapp" / "static"
    vend = [static / "vendor" / n for n in ("maplibre-gl.js", "maplibre-gl.css", "pmtiles.js")]
    miss = [p.name for p in vend if not p.exists()]
    c.add(g, "vendor JS (offline)", "fail" if miss else "ok", ("missing: " + ", ".join(miss)) if miss else
          "maplibre-gl, pmtiles bundled locally (no CDN)", "restore webapp/static/vendor from git" if miss else "")
    # map file
    mp = Path(s.map_dir)
    pm = sorted(mp.glob("*.pmtiles")) if mp.exists() else []
    if pm:
        from webapp.maps import read_pmtiles_header
        hd = read_pmtiles_header(pm[0])
        c.add(g, "offline map", "ok" if hd else "fail", f"{pm[0].name}, {pm[0].stat().st_size / 1e6:.0f} MB, zoom {hd['minzoom']}-{hd['maxzoom']}" if hd
              else f"{pm[0].name} is not a valid PMTiles v3 file", "" if hd else "scripts/fetch_map.py")
    else:
        c.add(g, "offline map", "warn", f"no .pmtiles in {mp}: the UI shows a plain grid", "scripts/fetch_map.py (needs internet once)")
    # static files serve
    t0 = time.perf_counter()
    try:
        from fastapi.testclient import TestClient
        from webapp.controller import FakeController
        from webapp.server import create_app
        with tempfile.TemporaryDirectory() as td:
            fake = FakeController(Path(td), background=False, settings=s)
            app = create_app(fake, s)
            bad = []
            with TestClient(app) as cl:
                for url in ("/", "/static/app.js", "/static/app.css", "/static/js/mapview.js", "/static/vendor/maplibre-gl.js",
                            "/api/limits", "/api/map/info", "/api/status"):
                    r = cl.get(url)
                    if r.status_code != 200:
                        bad.append(f"{url}={r.status_code}")
        c.add(g, "web app serves", "fail" if bad else "ok", ("bad: " + ", ".join(bad)) if bad else "index, JS, CSS, vendor, /api/limits, /api/map/info all HTTP 200",
              "check webapp/static", ms=(time.perf_counter() - t0) * 1000)
    except Exception as exc:  # noqa: BLE001
        c.add(g, "web app serves", "fail", f"{type(exc).__name__}: {exc}"[:120])


def check_files(c: Ctx) -> None:
    g, s = "files", c.s
    try:
        d = Path(s.data_dir)
        d.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=d, prefix=".syscheck", delete=True):
            pass
        c.add(g, "data dir writable", "ok", str(d.resolve()))
    except OSError as exc:
        c.add(g, "data dir writable", "fail", f"{s.data_dir}: {exc}", "fix permissions / DATA_DIR")
    from fusion import calibration_path
    cp = calibration_path(s)
    if cp.exists():
        try:
            j = json.loads(cp.read_text())
            c.add(g, "thermal calibration", "ok", f"{cp.name}, rms {j.get('rms_px', '?')} px, {j.get('n_points', '?')} points")
        except Exception as exc:  # noqa: BLE001
            c.add(g, "thermal calibration", "warn", f"{cp.name} unreadable ({exc}); default homography used", "scripts/calibrate_thermal.py")
    else:
        c.add(g, "thermal calibration", "warn", "no calibration file: DEFAULT homography (FOV ratio) is used; heat boxes will be off",
              "scripts/calibrate_thermal.py (hot object in view of both cameras)")
    # .env sanity
    env = ROOT / ".env"
    c.add(g, ".env present", "ok" if env.exists() else "warn", str(env) if env.exists() else "no .env: built-in defaults", "" if env.exists() else "cp .env.example .env")
    if s.allow_real_arm:
        c.add(g, "ALLOW_REAL_ARM", "warn", "TRUE: real arming is permitted when DRY_RUN=false. A human must have set this on purpose", "set ALLOW_REAL_ARM=false unless flying")
    else:
        c.add(g, "ALLOW_REAL_ARM", "ok", "false")
    c.add(g, "DRY_RUN", "ok" if s.dry_run else "warn", "true" if s.dry_run else "FALSE: non-simulated links are allowed to be commanded (with ALLOW_REAL_ARM)",
          "" if s.dry_run else "set DRY_RUN=true while testing")
    if abs(s.focal_length_mm - 6.0) < 1e-6 and abs(s.camera_hfov_deg - 52.8) < 1e-6:
        c.add(g, "lens values", "warn", "still the ASSUMED 6 mm lens (HFOV 52.8): footprint/GSD/line spacing are guesses",
              "measure the real lens; set FOCAL_LENGTH_MM, CAMERA_HFOV_DEG, CAMERA_VFOV_DEG in .env")
    else:
        c.add(g, "lens values", "ok", f"{s.focal_length_mm:g} mm, HFOV {s.camera_hfov_deg:g} / VFOV {s.camera_vfov_deg:g}")
    if s.laanc_ceiling_m is None:
        c.add(g, "LAANC ceiling", "warn", "LAANC_CEILING_M unset: only MAX_ALT_M/120 m limits apply",
              "set LAANC_CEILING_M to the altitude in your authorization")
    else:
        c.add(g, "LAANC ceiling", "ok", f"{s.laanc_ceiling_m:g} m")


# ======================================================================================== orchestration
def _worker(name: str, fn: Callable[[], None], c: Ctx, group: str) -> Callable[[], None]:
    def run() -> None:
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            log.exception("system check %s crashed", name)
            c.add(group, f"{name} (check crashed)", "fail", f"{type(exc).__name__}: {exc}"[:120])
        log.debug("system check task %s took %.1fs", name, time.perf_counter() - t0)
    return run


def run_all(s: Settings, *, quick: bool = False, only: Optional[list[str]] = None, app: Optional[dict] = None,
            deadline_s: float = DEADLINE_S) -> dict:
    """Run every check (groups in parallel threads) and return the report dict. `app` = data from the running
    process: {"snapshot", "vehicle", "camera_info", "thermal_info", "models_info"} -> device checks answer "via running app"."""
    t0 = time.time()
    c = Ctx(s, quick=quick, app=app)
    want = set(only) if only else set(GROUPS)
    unknown = want - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown group(s) {sorted(unknown)}; choose from {GROUPS}")
    llm_state = {"ok": False}

    def llm_and_planner() -> None:
        if "llm" in want:
            llm_state["ok"] = check_llm(c)
        else:
            llm_state["ok"] = True
        if "planner" in want:
            check_planner(c, llm_state["ok"])

    def ml() -> None:
        if "yolo" in want:
            check_yolo(c)
        if "clip" in want:
            check_clip(c)

    def fast() -> None:
        if "web" in want:
            check_web(c)
        if "files" in want:
            check_files(c)

    tasks: list[tuple[str, Callable[[], None], str]] = []
    if "host" in want:
        tasks.append(("host", lambda: check_host(c), "host"))
    if want & {"llm", "planner"}:
        tasks.append(("llm", llm_and_planner, "llm"))
    if "camera" in want:
        tasks.append(("camera", lambda: check_camera(c), "camera"))
    if "thermal" in want:
        tasks.append(("thermal", lambda: check_thermal(c), "thermal"))
    if "vehicle" in want:
        tasks.append(("vehicle", lambda: check_vehicle(c), "vehicle"))
    if want & {"yolo", "clip"}:
        tasks.append(("ml", ml, "yolo"))
    if want & {"web", "files"}:
        tasks.append(("fast", fast, "web"))
    threads = []
    for name, fn, grp in tasks:
        t = threading.Thread(target=_worker(name, fn, c, grp), name=f"syscheck-{name}", daemon=True)
        t.start()
        threads.append((name, grp, t))
    end = time.monotonic() + deadline_s
    for name, grp, t in threads:
        t.join(max(0.0, end - time.monotonic()))
        if t.is_alive():
            c.add(grp, f"{name} (timed out)", "fail", f"did not finish within {deadline_s:.0f} s")
    return build_report(c.out, s, quick=quick, only=only, via=c.via, t0=t0)


def build_report(checks: list[dict], s: Settings, *, quick: bool, only: Optional[list[str]], via: Optional[str], t0: float) -> dict:
    order = {g: i for i, g in enumerate(GROUPS)}
    seq = {id(x): i for i, x in enumerate(checks)}
    checks = sorted(checks, key=lambda x: (order.get(x["group"], 99), seq[id(x)]))
    summ = {k: sum(1 for x in checks if x["status"] == k) for k in ("ok", "warn", "fail", "skip")}
    return {"ts": time.time(), "duration_s": round(time.time() - t0, 1), "mode": "via running app" if via else "standalone",
            "quick": quick, "only": only, "summary": summ, "ready": summ["fail"] == 0, "checks": checks,
            "vehicle_mode": s.vehicle_mode, "link": s.mavlink_url}


def save_report(rep: dict, s: Settings) -> Optional[Path]:
    try:
        p = Path(s.data_dir) / "system_check.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=1, default=str))
        return p
    except OSError as exc:
        log.error("saving system_check.json failed: %s", exc)
        return None


# ======================================================================================== running app
def find_running_app(s: Settings, timeout: float = 1.5) -> Optional[dict]:
    """The /api/status of a drone-brain already serving on this machine's web port, or None."""
    import httpx
    port = s.web_port
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/status", timeout=timeout)
        j = r.json()
        if r.status_code == 200 and j.get("type") == "status":
            return j.get("data") or {}
    except Exception:  # noqa: BLE001
        pass
    return None


def fetch_from_app(s: Settings, quick: bool, only: Optional[list[str]]) -> dict:
    """GET /api/system/check from the running app; if it predates that endpoint (404), assemble the same context from
    its /api/status + /api/vehicle/check and run the non-device checks locally."""
    import httpx
    params: dict[str, Any] = {"quick": "true" if quick else "false"}
    if only:
        params["only"] = ",".join(only)
    base = f"http://127.0.0.1:{s.web_port}"
    r = httpx.get(base + "/api/system/check", params=params, timeout=120.0)
    if r.status_code != 404:
        r.raise_for_status()
        return r.json()
    snap = httpx.get(base + "/api/status", timeout=5.0).json().get("data") or {}
    rep, err = None, ""
    if not only or "vehicle" in only:
        try:
            vr = httpx.get(base + "/api/vehicle/check", timeout=60.0)
            vr.raise_for_status()
            rep = vr.json()
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
    return run_all(s, quick=quick, only=only, app={"snapshot": snap, "vehicle": rep, "vehicle_error": err})


def app_context(controller: Any, vehicle_report: Optional[dict], vehicle_error: str = "") -> dict:
    """Data the running process hands to run_all() (call from the app loop, then run run_all in a thread)."""
    p = getattr(controller, "pipeline", None)
    g = lambda n: (getattr(p, n)() if p is not None and hasattr(p, n) else {})   # noqa: E731
    return {"snapshot": controller.snapshot(), "vehicle": vehicle_report, "vehicle_error": vehicle_error,
            "camera_info": g("camera_info"), "thermal_info": g("thermal_info"), "models_info": g("models_info")}


def make_router(s: Settings, controller: Any):
    import asyncio

    from fastapi import APIRouter

    r = APIRouter()

    @r.get("/api/system/check")
    async def system_check(quick: bool = False, only: str = "") -> dict:
        groups = [x.strip() for x in only.split(",") if x.strip()] or None
        if groups and set(groups) - set(GROUPS):
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail=f"unknown group(s); choose from {GROUPS}")
        rep, err = None, ""
        if not groups or "vehicle" in groups:
            try:
                rep = await controller.vehicle_check()
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
        ctx = app_context(controller, rep, err)
        out = await asyncio.to_thread(run_all, s, quick=quick, only=groups, app=ctx)
        await asyncio.to_thread(save_report, out, s)
        return out

    return r


# ======================================================================================== rendering / CLI
def render(rep: dict, colour: Optional[bool] = None) -> str:
    if colour is None:
        colour = False
    w = shutil.get_terminal_size((120, 20)).columns
    paint = (lambda st, t: f"{COLOUR[st]}{t}{RESET}") if colour else (lambda st, t: t)
    lines = [(f"{BOLD}" if colour else "") + f"drone-brain system check ({rep['mode']}, {rep['duration_s']} s)"
             + (RESET if colour else "")]
    gw = max(7, max((len(x["group"]) for x in rep["checks"]), default=7))
    nw = min(28, max(10, max((len(x["name"]) for x in rep["checks"]), default=10)))
    fixes: list[str] = []
    last = None
    for x in rep["checks"]:
        head = f"{ICON[x['status']]} {x['group'] if x['group'] != last else '':<{gw}} {x['name'][:nw]:<{nw}} "
        last = x["group"]
        via = f" [{x['via']}]" if x.get("via") else ""
        room = max(30, w - len(head) + (2 if colour else 0) - 2)
        detail = (x["detail"] + via)
        if len(detail) > room:
            detail = detail[: room - 1] + "…"
        lines.append(head + paint(x["status"], detail))
        if x["status"] in ("warn", "fail") and x.get("fix"):
            fixes.append(f"{x['group']}/{x['name']}: {x['fix']}")
    sm = rep["summary"]
    lines.append("")
    verdict = "ALL GOOD" if sm["fail"] == 0 and sm["warn"] == 0 else ("OK with warnings" if sm["fail"] == 0 else "PROBLEMS FOUND")
    st = "ok" if sm["fail"] == 0 and sm["warn"] == 0 else ("warn" if sm["fail"] == 0 else "fail")
    lines.append(paint(st, f"{verdict}: {sm['ok']} ok, {sm['warn']} warnings, {sm['fail']} failed, {sm['skip']} skipped"))
    if fixes:
        lines.append("")
        lines.append("What to do:")
        lines += [f"  {i}. {f}" for i, f in enumerate(fixes, 1)]
    return "\n".join(lines)


class _quiet_native_output:
    """argus/GStreamer/CUDA print straight to fd 1/2 from C; send that noise to data/system_check.log for the duration."""

    def __init__(self, s: Settings) -> None:
        self.path = Path(s.data_dir) / "system_check.log"

    def __enter__(self):
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(self.path, "w")
            self.saved = (os.dup(1), os.dup(2))
            os.dup2(self.f.fileno(), 1)
            os.dup2(self.f.fileno(), 2)
        except OSError:
            self.saved = None
        return self

    def __exit__(self, *exc):
        """fd 1/2 stay pointed at the log until exit (argus prints "Cleaning up" from C at teardown);
        Python's stdout/stderr go back to the real terminal through the saved descriptors."""
        import sys
        if self.saved:
            sys.stdout = os.fdopen(self.saved[0], "w", buffering=1)
            sys.stderr = os.fdopen(self.saved[1], "w", buffering=1)
        return False


def cli(argv: Optional[list[str]] = None, settings: Optional[Settings] = None) -> int:
    import argparse
    import sys

    from config import get_settings

    ap = argparse.ArgumentParser(description="check the whole drone-brain system (read-only)")
    ap.add_argument("--quick", action="store_true", help="skip camera, thermal, YOLO, CLIP")
    ap.add_argument("--json", action="store_true", help="print the JSON report only")
    ap.add_argument("--only", default="", help=f"comma list of: {','.join(GROUPS)}")
    ap.add_argument("--mavlink-url", default=None)
    ap.add_argument("--baud", type=int, default=None)
    ap.add_argument("--standalone", action="store_true", help="do not use a running main.py even if one answers")
    a = ap.parse_args(argv)
    s = settings or get_settings()
    upd: dict[str, Any] = {}
    if a.mavlink_url:
        upd["mavlink_url"] = a.mavlink_url
    if a.baud:
        upd["mavlink_baud"] = a.baud
    s = s.model_copy(update=upd) if upd else s
    only = [x.strip() for x in a.only.split(",") if x.strip()] or None
    if only and set(only) - set(GROUPS):
        ap.error(f"unknown group(s) {sorted(set(only) - set(GROUPS))}; choose from {GROUPS}")
    logging.getLogger().setLevel(logging.ERROR)
    rep = None
    if not a.standalone and find_running_app(s) is not None:
        if not a.json:
            print(f"main.py is already running on port {s.web_port}: asking the running app (devices stay untouched)...", file=sys.stderr)
        try:
            rep = fetch_from_app(s, a.quick, only)
        except Exception as exc:  # noqa: BLE001
            print(f"running app did not answer /api/system/check ({exc}); falling back to a standalone check", file=sys.stderr)
    if rep is None:
        with _quiet_native_output(s):
            rep = run_all(s, quick=a.quick, only=only)
    path = save_report(rep, s)
    if a.json:
        print(json.dumps(rep, indent=1, default=str))
    else:
        print(render(rep, colour=sys.stdout.isatty()))
        if path:
            print(f"\nsaved {path}")
    return 0 if rep["summary"]["fail"] == 0 else 1
