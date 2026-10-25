"""Detection -> track -> crop -> VLM match -> alerts, without ever blocking the producer.

Public API (what main.py uses):
    m = Matcher(settings, backend, storage, on_alert)    # backend: MatchBackend (ClipBackend/VlmBackend) or an
                                                          # LLMClient (auto-wrapped in VlmBackend)
                                                          # on_alert(Alert) runs in a worker/submit thread
    m.update_target(Target(...)); m.start()
    m.submit(frame_bgr, detections)                       # non-blocking
    m.stats(); m.stop()

Scoring (see ``score``):
    C_match = match_confidence if (is_human and matches) else 0
    C_final = C_yolo * C_match                      (C_yolo of the crop that was judged)
    C_avg   = mean(C_final of the track's last AVG_WINDOW VLM results for the current target)
    target_match alert  <=>  C_avg >= match_threshold and (n_results >= min_frames or (n_results >= 2 and C_avg >= match_strong_threshold))
A "person" alert is emitted at submit time the first time a track is seen (no VLM wait).
Per-track rate limit: at most one alert per level per alert_min_interval_s; target_match is
sent once per track per target description, again only if C_avg improved by >= IMPROVE_DELTA.
Changing the target clears every track's match history and drops pending VLM jobs.
"""
from __future__ import annotations

import collections
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import cv2
import numpy as np
from pydantic import BaseModel

from config import Settings
from llm_client import LLMClient, LLMError, LLMTimeout, LLMUnavailable, image_message, text_message
from models import Alert, BBox, ClipPrompts, Detection, MatchResult, Target
from storage import FlightStorage
from tracker import Track, Tracker

log = logging.getLogger(__name__)

AVG_WINDOW = 5          # last N VLM results averaged into C_avg
IMPROVE_DELTA = 0.10    # C_avg gain needed to re-alert target_match on the same track
VLM_MAX_TOKENS = 200
MAX_BACKOFF_S = 15.0


def _conf(det: Detection) -> float:
    """Fused person confidence (Phase 5) when present, else raw YOLO."""
    return det.c_person if det.c_person is not None else det.c_yolo


# ---------- crops --------------------------------------------------------

def make_crop(frame: np.ndarray, bbox: BBox, settings: Settings) -> np.ndarray:
    """Pad bbox by settings.crop_padding on each side, clamp to the image, upscale
    (INTER_CUBIC, aspect kept) if the short side is below settings.crop_min_px."""
    h, w = frame.shape[:2]
    bw, bh = bbox.x2 - bbox.x1, bbox.y2 - bbox.y1
    if bw <= 0 or bh <= 0:
        raise ValueError("empty bbox")
    px, py = bw * settings.crop_padding, bh * settings.crop_padding
    x1 = max(0, int(np.floor(bbox.x1 - px)))
    y1 = max(0, int(np.floor(bbox.y1 - py)))
    x2 = min(w, int(np.ceil(bbox.x2 + px)))
    y2 = min(h, int(np.ceil(bbox.y2 + py)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox outside image")
    crop = frame[y1:y2, x1:x2]
    short = min(crop.shape[:2])
    if short < settings.crop_min_px:
        scale = settings.crop_min_px / short
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                          interpolation=cv2.INTER_CUBIC)
    return crop


def encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


# ---------- prompt -------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the vision module of a search-and-rescue drone looking down from altitude. "
    "You receive one cropped image of a possible person and a description of the person being searched for. "
    "Answer ONLY with a JSON object: "
    '{"is_human": bool, "matches": bool, "match_confidence": number 0-1, "reason": string}. '
    "is_human: the crop shows a real person (not a statue, shadow, bag, tree...). "
    "matches: the person fits the description; judge only what is visible (clothing colour, items), "
    "and if a detail cannot be seen do not count it against the match. "
    "match_confidence: how sure you are of your 'matches' answer, 0 to 1. "
    "reason: at most 25 words."
)


def build_messages(target: Target, jpeg: bytes) -> list[dict[str, Any]]:
    attrs = "; ".join(target.attributes) if target.attributes else "none listed"
    text = (f"Target person: {target.description}\nKey attributes: {attrs}\n"
            "Does the person in this image match the target? Reply with the JSON object only.")
    return [text_message("system", SYSTEM_PROMPT), image_message(jpeg, text)]


# ---------- backends -----------------------------------------------------

class MatchBackend(Protocol):
    def prepare(self, target: Target) -> None:
        """Called on every target update (e.g. pre-encode CLIP text prompts)."""

    def match(self, crop_bgr: np.ndarray, target: Target) -> MatchResult: ...


class VlmBackend:
    """Gemma via llama-server."""

    def __init__(self, client: LLMClient, max_tokens: int = VLM_MAX_TOKENS) -> None:
        self.client = client
        self.max_tokens = max_tokens

    def prepare(self, target: Target) -> None:
        return None

    def match(self, crop_bgr: np.ndarray, target: Target) -> MatchResult:
        return self.client.chat_json(build_messages(target, encode_jpeg(crop_bgr)), MatchResult,
                                     max_tokens=self.max_tokens)


class ImageTextEncoder(Protocol):
    def encode_image(self, images: list[np.ndarray]) -> np.ndarray: ...   # (N, D), L2-normalised
    def encode_text(self, texts: list[str]) -> np.ndarray: ...            # (M, D), L2-normalised


DEFAULT_NEGATIVES = ["a photo of a rock", "a photo of a bush", "a photo of a shadow on the ground"]
PERSON_NEGATIVE = "a photo of a person in different clothes"
GENERIC_TARGETS = ("any person", "person", "anyone", "a person", "people", "someone", "anybody", "everyone", "any people")
COLOURS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black", "grey", "gray", "brown", "beige")
_CONTRAST_COLOURS = ("blue", "black", "white", "green", "red", "grey")
MAX_PROMPT_CHARS = 160


def is_generic_target(desc: str) -> bool:
    return " ".join(desc.lower().split()) in GENERIC_TARGETS


def contrast_negatives(desc: str) -> list[str]:
    """Deterministic 'same person, different attribute' prompts: other colours, other sleeve length, no hat.

    Hardening D: contrastive negatives are what separate the target from OTHER people; generic person negatives
    ("a photo of a person") absorb the probability of every person, including the target (AUC 0.92 -> 0.87)."""
    d = desc.lower()
    out: list[str] = []
    cols = [c for c in COLOURS if re.search(rf"\b{c}\b", d)]
    for c in cols[:1]:
        others = [o for o in _CONTRAST_COLOURS if o != c and o not in cols][:3]
        out += [re.sub(rf"\b{c}\b", o, desc, flags=re.I) for o in others]
    if re.search(r"long[- ]sleeve", d):
        out += [re.sub(r"long[- ]sleeve", "short-sleeve", desc, flags=re.I), re.sub(r"long[- ]sleeve", "sleeveless", desc, flags=re.I)]
    elif re.search(r"short[- ]sleeve", d):
        out.append(re.sub(r"short[- ]sleeve", "long-sleeve", desc, flags=re.I))
    if re.search(r"\b(hat|cap|beanie|helmet)\b", d):
        out += ["person with a bare head and no hat", "person wearing a jacket and no hat"]
    seen: set[str] = set()
    uniq = []
    for x in out:
        if x.lower() not in seen and x.lower() != d:
            seen.add(x.lower())
            uniq.append(x)
    return uniq[:6]


def default_prompts(target: Target) -> ClipPrompts:
    """Deterministic CLIP prompts (also the fallback when the LLM is down). Hardening D tuning: an aerial and a plain
    positive, contrastive negatives (other colours / sleeves / no hat) plus rock and bush."""
    desc = " ".join(target.description.split())[:120] or "person"
    if is_generic_target(desc):       # any person: only non-person things are negatives
        return ClipPrompts(target_description=target.description, positive=["a photo of a person", "a photo from above of a person"],
                           negative=DEFAULT_NEGATIVES + ["a photo of the ground"])
    pos = [f"a photo from above of {desc}", f"a photo of {desc}"] + [f"a photo of a person with {a}" for a in target.attributes[:2]]
    contrast = [f"a photo of {c}" for c in contrast_negatives(desc)]
    neg = contrast + (["a photo of a rock", "a photo of a bush"] if contrast else [PERSON_NEGATIVE] + DEFAULT_NEGATIVES[:2])
    return ClipPrompts(target_description=target.description, positive=pos, negative=neg)


class ClipBackend:
    """CLIP-style zero-shot scoring with an injected encoder (no model is loaded here).

    probs = softmax(logit_scale * cos_sim(image, [positive + negative prompts]));
    match_confidence = sum of positive probabilities; matches = confidence >= cutoff (settings.clip_match_cutoff);
    is_human is always True (YOLO already filtered the crop).
    Robust to garbage: prompt lists are sanitised (non-strings, empty, over-long dropped), a failing or empty
    prompt builder falls back to ``default_prompts``, a non-finite score counts as 0 (never poisons a track's average).
    """

    def __init__(self, encoder: ImageTextEncoder, prompt_builder: Callable[[Target], ClipPrompts] = default_prompts,
                 logit_scale: float = 100.0, cutoff: float = 0.6) -> None:
        self.encoder = encoder
        self.cutoff = cutoff      # settings.clip_match_cutoff
        self.prompt_builder = prompt_builder
        self.logit_scale = logit_scale
        self.prompts: Optional[ClipPrompts] = None
        self._text: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._prep_lock = threading.Lock()      # one prompt build at a time (LLM call ~4 s); a second caller reuses the result

    @staticmethod
    def sanitize(prompts: ClipPrompts) -> ClipPrompts:
        def clean(xs: Any) -> list[str]:
            out: list[str] = []
            for x in (xs or []):
                if isinstance(x, str):
                    x = " ".join(x.split())[:MAX_PROMPT_CHARS]
                    if x and x not in out:
                        out.append(x)
            return out
        pos, neg = clean(prompts.positive), clean(prompts.negative)
        if not pos:
            raise ValueError("no usable positive CLIP prompts")
        if not neg:
            neg = list(DEFAULT_NEGATIVES)
        return ClipPrompts(target_description=prompts.target_description, positive=pos, negative=neg)

    def set_prompts(self, prompts: ClipPrompts) -> None:
        prompts = self.sanitize(prompts)
        text = np.asarray(self.encoder.encode_text(prompts.positive + prompts.negative), dtype=np.float32)
        if text.ndim != 2 or len(text) != len(prompts.positive) + len(prompts.negative) or not np.isfinite(text).all():
            raise ValueError("CLIP text encoder returned unusable embeddings")
        with self._lock:
            self.prompts, self._text = prompts, text

    def prepare(self, target: Target, *, only_if_stale: bool = False) -> None:
        with self._prep_lock:
            if only_if_stale:
                with self._lock:
                    if self.prompts is not None and self.prompts.target_description == target.description:
                        return      # another thread just built them while we waited for the lock
            try:
                self.set_prompts(self.prompt_builder(target))
            except Exception as exc:  # noqa: BLE001 - builder (LLM) failed or produced garbage: deterministic prompts
                log.warning("CLIP prompt builder failed (%s); using default prompts", exc)
                self.set_prompts(default_prompts(target))

    def match(self, crop_bgr: np.ndarray, target: Target) -> MatchResult:
        with self._lock:
            prompts, text = self.prompts, self._text
        if prompts is None or text is None or prompts.target_description != target.description:
            self.prepare(target, only_if_stale=True)
            with self._lock:
                prompts, text = self.prompts, self._text
        assert prompts is not None and text is not None
        img = np.asarray(self.encoder.encode_image([crop_bgr]), dtype=np.float32)[0]
        logits = self.logit_scale * (text @ img)
        if not np.isfinite(logits).all():
            return MatchResult(is_human=True, matches=False, match_confidence=0.0, reason="CLIP: non-finite score")
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        n_pos = len(prompts.positive)
        conf = float(np.clip(p[:n_pos].sum(), 0.0, 1.0))
        top_pos = int(np.argmax(p[:n_pos]))
        reason = f"CLIP: {prompts.positive[top_pos]} {p[top_pos]:.2f}"
        if len(p) > n_pos:
            top_neg = n_pos + int(np.argmax(p[n_pos:]))
            reason += f" vs {prompts.negative[top_neg - n_pos]} {p[top_neg]:.2f}"
        return MatchResult(is_human=True, matches=conf >= self.cutoff, match_confidence=conf, reason=reason[:200])


def make_backend(settings: Settings, *, client: Optional[LLMClient] = None,
                 encoder: Optional[ImageTextEncoder] = None) -> MatchBackend:
    """settings.match_backend -> backend ('clip' needs ``encoder``, 'vlm' needs ``client``)."""
    if settings.match_backend == "vlm":
        return VlmBackend(client or LLMClient(settings=settings))
    if settings.match_backend == "clip":
        if encoder is None:
            raise ValueError("match_backend=clip requires an encoder")
        return ClipBackend(encoder, cutoff=settings.clip_match_cutoff)
    raise ValueError(f"unknown match_backend {settings.match_backend!r}")


# ---------- scoring ------------------------------------------------------

def score(c_yolo: float, res: MatchResult) -> tuple[float, float]:
    """Return (C_match, C_final) for one VLM verdict."""
    c_match = res.match_confidence if (res.is_human and res.matches) else 0.0
    return c_match, c_yolo * c_match


# ---------- bounded job queue -------------------------------------------

@dataclass
class Job:
    track_id: int
    score: float          # c_yolo * bbox area
    crop: np.ndarray
    c_yolo: float
    ts: float
    crop_path: Optional[str]
    det: Detection


class JobQueue:
    """At most one pending job per track (the best), at most ``maxsize`` jobs overall;
    when full the oldest pending job is dropped. ``offer`` never blocks."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = max(1, maxsize)
        self._jobs: "collections.OrderedDict[int, Job]" = collections.OrderedDict()
        self._cond = threading.Condition()
        self.dropped = 0

    def __len__(self) -> int:
        with self._cond:
            return len(self._jobs)

    def offer(self, job: Job) -> bool:
        """True if the job is now pending."""
        with self._cond:
            old = self._jobs.get(job.track_id)
            if old is not None:
                if job.score > old.score:
                    self._jobs[job.track_id] = job   # keep queue position, better crop
                    self.dropped += 1                # the replaced crop
                    return True
                self.dropped += 1
                return False
            while len(self._jobs) >= self.maxsize:
                self._jobs.popitem(last=False)
                self.dropped += 1
            self._jobs[job.track_id] = job
            self._cond.notify()
            return True

    def get(self, timeout: float) -> Optional[Job]:
        with self._cond:
            if not self._jobs:
                self._cond.wait(timeout)
            if not self._jobs:
                return None
            return self._jobs.popitem(last=False)[1]

    def clear(self) -> int:
        with self._cond:
            n = len(self._jobs)
            self._jobs.clear()
            return n

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()


# ---------- matcher ------------------------------------------------------

class Matcher:
    def __init__(self, settings: Settings, backend: Any, storage: FlightStorage,
                 on_alert: Callable[[Alert], None], *, clock: Callable[[], float] = time.time,
                 confirm_backend: Optional[MatchBackend] = None) -> None:
        self.s = settings
        # an LLMClient (has chat_json) is wrapped; anything else must implement MatchBackend
        self.backend: MatchBackend = VlmBackend(backend) if hasattr(backend, "chat_json") else backend
        # optional VLM used once per track/target after a target_match (settings.vlm_confirm_top)
        self.confirm_backend = confirm_backend if settings.vlm_confirm_top else None
        self.storage = storage
        self.on_alert = on_alert
        self.clock = clock
        self.tracker = Tracker(settings)
        self.queue = JobQueue(settings.vlm_queue_size)
        self._lock = threading.RLock()
        self._target: Optional[Target] = None
        self._target_ver = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._processed = 0
        self._errors = 0
        self._stale = 0
        self._alerts = 0
        self._latency_sum = 0.0
        self._consec_err = 0
        self._gen_err = 0

    # ---- lifecycle ----
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="matcher-vlm", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.queue.wake()
        if self._thread:
            self._thread.join(timeout)
            if self._thread.is_alive():
                log.warning("matcher worker still busy after %.1fs (in-flight VLM call)", timeout)

    def update_target(self, target: Optional[Target]) -> None:
        """Later jobs use the new description; per-track match history and target_match
        alert state of the old description are reset; pending (old) jobs are dropped."""
        with self._lock:
            self._target = target
            self._target_ver += 1
            self.tracker.reset_matches()
            self.queue.clear()
        if target is not None:
            try:
                self.backend.prepare(target)
            except Exception:  # noqa: BLE001
                log.exception("backend.prepare failed")
            self.storage.record_target(target)
        log.info("target set: %s", target.description if target else None)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n = self._processed
            return {"queue_depth": len(self.queue), "dropped": self.queue.dropped,
                    "processed": n, "errors": self._errors, "stale": self._stale,
                    "alerts": self._alerts, "tracks": len(self.tracker.tracks),
                    "mean_latency_s": (self._latency_sum / n) if n else None}

    # ---- producer side (camera thread) ----
    def submit(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        """Track, crop, store, enqueue. Never calls the VLM and never raises."""
        try:
            return self._submit(frame, detections)
        except Exception:  # noqa: BLE001 - camera loop must survive anything
            log.exception("matcher.submit failed")
            return detections

    def _submit(self, frame: np.ndarray, detections: list[Detection]) -> list[Detection]:
        dets = [d for d in detections if _conf(d) >= self.s.yolo_conf_threshold]
        if not dets:
            return []
        img_h, img_w = frame.shape[:2]
        frame_path = self.storage.save_frame(frame, f"{dets[0].frame_id:06d}")
        with self._lock:
            target = self._target
            assigned = self.tracker.update(dets, img_w, img_h)
        pending: list[Job] = []
        for det, track, is_new in assigned:
            det.frame_path = frame_path
            self.storage.append("detections", det)
            try:
                crop = make_crop(frame, det.bbox, self.s)
            except ValueError as exc:
                log.warning("bad crop for track %d: %s", track.id, exc)
                continue
            b = det.bbox
            job = Job(track.id, _conf(det) * (b.x2 - b.x1) * (b.y2 - b.y1), crop, _conf(det), det.ts, None, det)
            if is_new or track.best_crop_path is None or job.score > track.best_score:
                job.crop_path = self.storage.save_crop(crop, f"t{track.id}_f{det.frame_id:06d}")
                track.best_crop_path = job.crop_path or track.best_crop_path
                track.best_score = job.score
            if not track.person_alerted:
                self._emit_person(track, det, job.crop_path or track.best_crop_path)
            if target is not None:
                if max(b.x2 - b.x1, b.y2 - b.y1) < self.s.match_min_px:
                    continue        # too small to judge clothing: person alert only, no CLIP job
                if job.crop_path is None:
                    job.crop_path = track.best_crop_path
                pending.append(job)
        # ascending score: when the queue overflows the OLDEST job is dropped, so the best crops are offered last and kept
        for job in sorted(pending, key=lambda j: j.score):
            self.queue.offer(job)
        return [d for d, _, _ in assigned]

    def _emit_person(self, track: Track, det: Detection, crop_path: Optional[str]) -> None:
        now = self.clock()
        track.person_alerted = True
        track.last_person_alert_ts = now
        self._emit(self._alert("person", track, _conf(det), crop_path, now=now, reason="person detected"))

    # ---- alert helpers ----
    def _alert(self, level: str, track: Track, c_yolo: float, crop_path: Optional[str], *, now: float,
               c_match: Optional[float] = None, c_final: Optional[float] = None, c_avg: Optional[float] = None,
               n_frames: int = 1, reason: str = "", target: Optional[Target] = None) -> Alert:
        det = track.last_det
        tel = det.telemetry if det else None
        lat, lon = (track.last_ground if track.last_ground else (None, None))
        if lat is None and tel is not None:
            lat, lon = tel.lat, tel.lon
        return Alert(level=level, track_id=track.id, ts=now, c_yolo=c_yolo, c_match=c_match,  # type: ignore[arg-type]
                     c_final=c_final, c_avg=c_avg, n_frames=n_frames, reason=reason,
                     image_url=self.storage.url_for(crop_path) if crop_path else None,
                     lat=lat, lon=lon, alt_rel_m=tel.alt_rel_m if tel else None,
                     target_description=target.description if target else None)

    def _emit(self, alert: Alert) -> None:
        self.storage.append("alerts", alert)
        with self._lock:
            self._alerts += 1
        try:
            self.on_alert(alert)
        except Exception:  # noqa: BLE001
            log.exception("on_alert callback failed")

    # ---- worker ----
    def _run(self) -> None:
        while not self._stop.is_set():
            job = self.queue.get(0.2)
            if job is None:
                continue
            try:
                self._process(job)
                self._gen_err = 0
            except Exception:  # noqa: BLE001 - e.g. CLIP/CUDA failure: drop the job, back off if it keeps failing
                log.exception("matcher worker error")
                with self._lock:
                    self._errors += 1
                self._gen_err += 1
                if self._gen_err >= 3:
                    self._stop.wait(min(MAX_BACKOFF_S, 0.2 * 2 ** min(self._gen_err - 3, 6)))

    def _backoff(self) -> None:
        delay = min(MAX_BACKOFF_S, 0.5 * 2 ** min(self._consec_err, 6))
        log.warning("VLM unavailable (%d consecutive failures); backing off %.1fs", self._consec_err, delay)
        self._stop.wait(delay)

    def _process(self, job: Job) -> None:
        with self._lock:
            target, ver = self._target, self._target_ver
        if target is None:
            return
        t0 = time.monotonic()
        try:
            res = self.backend.match(job.crop, target)
        except (LLMUnavailable, LLMTimeout) as exc:
            with self._lock:
                self._errors += 1
                self._consec_err += 1
            log.warning("match backend call failed: %s", exc)
            self._backoff()
            return
        except LLMError as exc:
            with self._lock:
                self._errors += 1
            log.warning("match backend error: %s", exc)
            return
        latency = time.monotonic() - t0
        self._consec_err = 0
        with self._lock:
            if ver != self._target_ver:
                self._stale += 1       # target changed while the call was in flight
                return
            self._processed += 1
            self._latency_sum += latency
            track = self.tracker.get(job.track_id)
            c_match, c_final = score(job.c_yolo, res)
            if track is None:
                return
            track.results.append(res)
            track.match_hist.append(c_final)
            del track.results[:-AVG_WINDOW], track.match_hist[:-AVG_WINDOW]
            n = len(track.match_hist)
            c_avg = sum(track.match_hist) / n
            alert = self._decide_target_alert(track, job, res, c_match, c_final, c_avg, n, target)
        self.storage.append("matches", {
            "track_id": job.track_id, "ts": job.ts, "target": target.description, "c_yolo": job.c_yolo,
            "c_match": c_match, "c_final": c_final, "c_avg": c_avg, "n": n, "latency_s": latency,
            "crop": job.crop_path, "result": res.model_dump()})
        if alert is not None:
            self._emit(alert)
            self._maybe_confirm(job, alert, target, ver)

    def _maybe_confirm(self, job: Job, alert: Alert, target: Target, ver: int) -> None:
        """Hybrid mode: once per track/target, ask the VLM for a human-readable reason."""
        if self.confirm_backend is None:
            return
        with self._lock:
            track = self.tracker.get(job.track_id)
            if track is None or track.vlm_confirmed_target == target.description:
                return
            track.vlm_confirmed_target = target.description
        try:
            res = self.confirm_backend.match(job.crop, target)
        except Exception as exc:  # noqa: BLE001
            log.warning("VLM confirm failed: %s", exc)
            return
        with self._lock:
            if ver != self._target_ver:
                return
        self.storage.append("matches", {"track_id": job.track_id, "ts": job.ts, "target": target.description,
                                        "confirm": True, "result": res.model_dump()})
        if res.is_human and res.matches:
            self._emit(alert.model_copy(update={"ts": self.clock(), "reason": res.reason}))

    def _decide_target_alert(self, track: Track, job: Job, res: MatchResult, c_match: float, c_final: float,
                             c_avg: float, n: int, target: Target) -> Optional[Alert]:
        strong = n >= 2 and c_avg >= self.s.match_strong_threshold      # very confident: no need to wait for min_frames
        if (n < self.s.min_frames and not strong) or c_avg < self.s.match_threshold:
            return None
        now = self.clock()
        if track.target_alert_avg is not None:
            if c_avg < track.target_alert_avg + IMPROVE_DELTA:
                return None
        if track.last_target_alert_ts is not None and now - track.last_target_alert_ts < self.s.alert_min_interval_s:
            return None
        track.target_alert_avg = c_avg
        track.last_target_alert_ts = now
        return self._alert("target_match", track, job.c_yolo, job.crop_path, now=now, c_match=c_match,
                           c_final=c_final, c_avg=c_avg, n_frames=n, reason=res.reason, target=target)
