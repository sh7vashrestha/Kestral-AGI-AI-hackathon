"""CLIP image+text encoder for matcher.ClipBackend (OpenCLIP ViT-B/32 laion2b; image tower TensorRT FP16).

    enc = ClipEncoder(settings)           # loads weights once (offline after first download into models/clip/)
    enc.encode_image([crop_bgr, ...]) -> (N, 512) float32, L2-normalised
    enc.encode_text(["a person in a red jacket", ...]) -> (M, 512) float32, L2-normalised (cached per string)

Image tower: models/clip_image.engine (scripts/export_models.py clip) else torch FP16 on CUDA (else CPU fp32).
Text tower: torch FP16 on CUDA (runs only when the target changes).
Crops are padded to a square with neutral grey (not centre-cropped, which would cut heads/feet off tall person crops).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from config import ROOT, Settings

log = logging.getLogger(__name__)
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
MODELS_DIR = ROOT / "models"
ENGINE_PATH = MODELS_DIR / "clip_image.engine"
ONNX_PATH = MODELS_DIR / "clip_image.onnx"
CACHE_DIR = MODELS_DIR / "clip"
EXTRA_CACHE_DIR = MODELS_DIR / "clip_extra"      # other CLIP_MODEL weights (kept apart from the default model's cache)
RES = 224
MAX_BATCH = 4


def _load_open_clip(device: str, name: str = MODEL_NAME, pretrained: str = PRETRAINED):
    import open_clip

    cache = CACHE_DIR if (name, pretrained) == (MODEL_NAME, PRETRAINED) else EXTRA_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained, cache_dir=str(cache), device="cpu")
    model.eval()
    tok = open_clip.get_tokenizer(name)
    mean = np.array(model.visual.preprocess_cfg["mean"] if hasattr(model.visual, "preprocess_cfg")
                    else (0.48145466, 0.4578275, 0.40821073), np.float32)
    std = np.array(model.visual.preprocess_cfg["std"] if hasattr(model.visual, "preprocess_cfg")
                   else (0.26862954, 0.26130258, 0.27577711), np.float32)
    return model, tok, mean, std


def preprocess_u8(crop_bgr: np.ndarray, fit: str = "pad", res: int = RES) -> np.ndarray:
    """BGR uint8 crop -> (res,res,3) uint8 RGB (resize/pad only; normalisation happens on the GPU)."""
    RES = res  # noqa: N806 - the model's input size (224 for ViT-B-32)
    h, w = crop_bgr.shape[:2]
    if fit == "crop":
        s = RES / min(h, w)
        im = cv2.resize(crop_bgr, (max(RES, round(w * s)), max(RES, round(h * s))), interpolation=cv2.INTER_CUBIC)
        y0, x0 = (im.shape[0] - RES) // 2, (im.shape[1] - RES) // 2
        im = im[y0:y0 + RES, x0:x0 + RES]
    elif fit == "squash":
        im = cv2.resize(crop_bgr, (RES, RES), interpolation=cv2.INTER_AREA if max(h, w) > RES else cv2.INTER_LINEAR)
    else:
        s = RES / max(h, w)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        r = cv2.resize(crop_bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        im = np.full((RES, RES, 3), 124, np.uint8)
        y0, x0 = (RES - nh) // 2, (RES - nw) // 2
        im[y0:y0 + nh, x0:x0 + nw] = r
    return np.ascontiguousarray(im[:, :, ::-1])


def preprocess(crop_bgr: np.ndarray, mean: np.ndarray, std: np.ndarray, fit: str = "pad") -> np.ndarray:
    """CPU reference: BGR uint8 crop -> (3,224,224) float32 normalised."""
    x = preprocess_u8(crop_bgr, fit).astype(np.float32) / 255.0
    return ((x - mean) / std).transpose(2, 0, 1).copy()


class _TrtRunner:
    """Minimal TensorRT 10 runner: dynamic batch, torch CUDA tensors as buffers."""

    def __init__(self, path: Path) -> None:
        import tensorrt as trt

        self.trt = trt
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = rt.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize {path} (GPU memory exhausted - NvMap error 12 - or built with another "
                               "TensorRT; re-export with scripts/export_models.py clip)")
        self.ctx = self.engine.create_execution_context()
        self.inp, self.out = self.engine.get_tensor_name(0), self.engine.get_tensor_name(1)
        self.stream = torch.cuda.Stream()
        # allocate the largest activation memory once (shape changes later never re-allocate; matters on a
        # memory-starved Orin where a late allocation fails with NvMap error 12 / CUDA 700)
        for b in (MAX_BATCH, 1):
            self(torch.zeros((b, 3, RES, RES), dtype=torch.float16, device="cuda"))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        x = x.contiguous()
        self.ctx.set_input_shape(self.inp, tuple(x.shape))
        out = torch.empty((n, *tuple(self.engine.get_tensor_shape(self.out))[1:]), dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address(self.inp, x.data_ptr())
        self.ctx.set_tensor_address(self.out, out.data_ptr())
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(torch.cuda.current_stream())
            ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        if not ok or not bool(torch.isfinite(out).all()):
            raise RuntimeError("TensorRT CLIP execution failed (GPU memory exhausted?)")
        return out


class ClipEncoder:
    def __init__(self, settings: Optional[Settings] = None, *, use_trt: Optional[bool] = None, fit: str = "pad",
                 engine_path: Path = ENGINE_PATH) -> None:
        if use_trt is None:      # torch by default: the FP16 engine ranked test images 0/3 on the Jetson (2026-10-20)
            use_trt = bool(getattr(settings, "clip_use_trt", False))
        self.cuda = torch.cuda.is_available()
        self.dev = "cuda" if self.cuda else "cpu"
        self.fit = fit
        self._lock = threading.Lock()
        self._text_cache: dict[str, np.ndarray] = {}
        self.name = getattr(settings, "clip_model", MODEL_NAME) or MODEL_NAME
        self.pretrained = getattr(settings, "clip_pretrained", PRETRAINED) or PRETRAINED
        if (self.name, self.pretrained) != (MODEL_NAME, PRETRAINED):
            use_trt = False                              # the TensorRT engine is built for the default model only
        model, self.tok, self.mean, self.std = _load_open_clip(self.dev, self.name, self.pretrained)
        size = getattr(model.visual, "image_size", RES)
        self.res = int(size[0] if isinstance(size, (tuple, list)) else size)
        self.dim = int(getattr(model.visual, "output_dim", 512) or 512)
        dtype = torch.float16 if self.cuda else torch.float32
        self.dtype = dtype
        self._mean = self._retry(lambda: torch.tensor(self.mean, device=self.dev).view(1, 3, 1, 1))
        self._std = self._retry(lambda: torch.tensor(self.std, device=self.dev).view(1, 3, 1, 1))
        visual = model.visual
        model.visual = torch.nn.Identity()
        self._engine_path = engine_path
        self.model = self._retry(lambda: model.to(self.dev, dtype))   # text tower on the GPU FIRST: creating the TensorRT
        self.trt: Optional[_TrtRunner] = None           # context before this corrupts later TRT launches on Jetson
        self.visual = None
        if use_trt and self.cuda and engine_path.exists():
            for attempt in range(4):        # GPU memory is shared with llama-server: allocation can fail transiently
                try:
                    self.trt = _TrtRunner(engine_path)
                    log.info("CLIP image encoder: TensorRT FP16 (%s)", engine_path.name)
                    break
                except Exception as exc:  # noqa: BLE001
                    log.warning("CLIP TensorRT engine load failed (attempt %d: %s)", attempt + 1, exc)
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(1.5)
        if self.trt is None:
            log.info("CLIP image encoder: torch %s", dtype)
            self.visual = self._retry(lambda: visual.to(self.dev, dtype))
        del visual

    @staticmethod
    def _retry(fn, attempts: int = 5):
        """GPU memory is shared with llama-server and can be momentarily exhausted/fragmented."""
        for i in range(attempts):
            try:
                return fn()
            except RuntimeError as exc:      # torch 2.8 raises AcceleratorError (a RuntimeError) for CUDA OOM
                log.warning("CUDA allocation failed (attempt %d/%d): %s", i + 1, attempts, str(exc)[:80])
                torch.cuda.empty_cache()
                time.sleep(2.0)
                if i == attempts - 1:
                    raise

    @torch.no_grad()
    def encode_image(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), np.float32)
        out = []
        for i in range(0, len(images), MAX_BATCH):
            batch = np.stack([preprocess_u8(im, self.fit, self.res) for im in images[i:i + MAX_BATCH]])
            x = torch.from_numpy(batch).to(self.dev).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - self._mean) / self._std
            with self._lock:
                if self.trt is not None:
                    try:
                        f = self.trt(x.half())
                    except RuntimeError as exc:
                        log.error("%s -> permanently falling back to torch FP16 image encoder", exc)
                        self.trt = None
                        torch.cuda.empty_cache()
                        self.visual = self._retry(lambda: _load_open_clip("cpu", self.name, self.pretrained)[0].visual.to(self.dev, self.dtype))
                        f = self.visual(x.to(self.dtype)).float()
                else:
                    f = self.visual(x.to(self.dtype)).float()
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._text_cache]
        if missing:
            tokens = self.tok(missing).to(self.dev)
            with self._lock:
                f = self.model.encode_text(tokens).float()
            f = torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float32)
            for t, v in zip(missing, f):
                self._text_cache[t] = v
        return np.stack([self._text_cache[t] for t in texts]).astype(np.float32)


# ---------- export (called by scripts/export_models.py clip) -------------------

def export_image_engine(workspace_gb: float = 0.5, max_batch: int = MAX_BATCH, validate: bool = True) -> Path:
    import tensorrt as trt

    MODELS_DIR.mkdir(exist_ok=True)
    model, _, mean, std = _load_open_clip("cpu")
    visual = model.visual.eval().float()

    class Wrap(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v

        def forward(self, x):
            return self.v(x)

    dummy = torch.randn(2, 3, RES, RES)
    torch.onnx.export(Wrap(visual), dummy, str(ONNX_PATH), input_names=["image"], output_names=["feat"],
                      dynamic_axes={"image": {0: "b"}, "feat": {0: "b"}}, opset_version=17, do_constant_folding=True,
                      dynamo=False)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(ONNX_PATH.read_bytes()):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    prof = builder.create_optimization_profile()
    prof.set_shape("image", (1, 3, RES, RES), (2, 3, RES, RES), (max_batch, 3, RES, RES))
    cfg.add_optimization_profile(prof)
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    ENGINE_PATH.write_bytes(bytes(blob))
    log.info("wrote %s (%.1f MB)", ENGINE_PATH, ENGINE_PATH.stat().st_size / 1e6)
    if validate:
        import subprocess
        import sys
        code = ("import numpy as np, logging; from clip_encoder import ClipEncoder; "
                "x = list(np.random.default_rng(0).integers(0, 255, (3, 200, 90, 3), dtype=np.uint8)); "
                "a = ClipEncoder(use_trt=True).encode_image(x); b = ClipEncoder(use_trt=False).encode_image(x); "
                "c = (a * b).sum(1); print('TRT vs torch cosine', c); raise SystemExit(0 if c.min() > 0.98 else 3)")
        r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), timeout=300)
        if r.returncode != 0:
            raise RuntimeError("TensorRT FP16 CLIP validation failed")
    return ENGINE_PATH
