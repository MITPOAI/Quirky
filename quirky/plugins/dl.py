"""
Quirky neural plugin (optional `quirky[dl]` extra).

Opt-in ONNX Runtime backend for heavier touch-up the pure-math core can't do:
image upscale/repaint/face-restore and zero-shot voice conversion. Everything here
is guarded -- importing quirky without the extra never triggers these code paths, and
calling a DL function without the extra raises a clear, actionable error.

Design goals:
  * CPU by default, CUDA auto-detected when present. Zero GPU requirement.
  * Weights are downloaded + cached on first use (never bundled in the wheel).
  * Only commercial-safe (Apache/MIT/BSD) checkpoints are registered.

Status in this build: the runtime, registry, download/cache, and provider selection
are functional. `upscale()` (Real-ESRGAN) is implemented end-to-end. `repaint()`,
`restore_face()` and `clone_voice()` are wired to the registry with documented
inference paths; they require their model weights on first run.
"""
from __future__ import annotations
import os
from typing import Optional, Dict, Any

import numpy as np

ATTRIBUTION = "Powered by Quirky by MITPO"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "quirky", "models")

# name -> (hf_repo, filename, license, task). Commercial-safe licenses only.
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "realesrgan_x4": {
        "repo": "Xenova/real-esrgan",
        "file": "onnx/RealESRGAN_x4plus.onnx",
        "license": "BSD-3-Clause",
        "task": "upscale",
    },
    "lama_inpaint": {
        "repo": "Carve/LaMa-ONNX",
        "file": "lama_fp32.onnx",
        "license": "Apache-2.0",
        "task": "inpaint",
    },
    "gfpgan_face": {
        "repo": "Xenova/gfpgan",
        "file": "onnx/gfpgan.onnx",
        "license": "Apache-2.0",
        "task": "face_restore",
    },
    "rvc_contentvec": {
        "repo": "lengyue233/content-vec-best",
        "file": "content-vec-best.onnx",
        "license": "MIT",
        "task": "voice_content",
    },
}


class DLNotInstalled(RuntimeError):
    """Raised when a [dl] feature is used without the optional extra installed."""


def dl_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def require_dl() -> None:
    if not dl_available():
        raise DLNotInstalled(
            "This feature needs the optional neural plugin.\n"
            "    Install it with:  uv pip install -e \".[dl]\"   (or: pip install quirky[dl])\n"
            "The pure-math Quirky core keeps working without it."
        )


def _providers():
    import onnxruntime as ort
    avail = ort.get_available_providers()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail \
        else ["CPUExecutionProvider"]


def _download(name: str) -> str:
    """Download+cache a registered model, returning the local path."""
    require_dl()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[name]
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise DLNotInstalled("huggingface-hub missing; reinstall with .[dl]") from e
    os.makedirs(CACHE_DIR, exist_ok=True)
    return hf_hub_download(repo_id=entry["repo"], filename=entry["file"], cache_dir=CACHE_DIR)


_SESSIONS: Dict[str, Any] = {}


def _session(name: str):
    if name in _SESSIONS:
        return _SESSIONS[name]
    import onnxruntime as ort
    path = _download(name)
    sess = ort.InferenceSession(path, providers=_providers())
    _SESSIONS[name] = sess
    return sess


# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #
def upscale(image_rgb: np.ndarray, model: str = "realesrgan_x4") -> np.ndarray:
    """
    Neural super-resolution (Real-ESRGAN). Adds real detail an interpolation upscale
    cannot. image_rgb: uint8 or float[0,1] HxWx3. Returns uint8 RGB, ~4x larger.
    """
    require_dl()
    sess = _session(model)
    x = image_rgb.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    inp = np.transpose(x, (2, 0, 1))[None]  # NCHW
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: inp})[0][0]
    out = np.clip(np.transpose(out, (1, 2, 0)), 0, 1)
    return (out * 255).astype(np.uint8)


def repaint(image_rgb: np.ndarray, mask: np.ndarray, model: str = "lama_inpaint") -> np.ndarray:
    """
    Neural inpaint / object+spot repaint (LaMa). mask: uint8 (255 = fill). Reconstructs
    masked regions with generated content (beyond cv2.inpaint's reach for large holes).
    """
    require_dl()
    sess = _session(model)
    h, w = image_rgb.shape[:2]
    H, W = (h // 8) * 8, (w // 8) * 8  # LaMa needs multiple-of-8 dims
    import cv2
    img = cv2.resize(image_rgb, (W, H)).astype(np.float32) / 255.0
    m = (cv2.resize(mask, (W, H)) > 127).astype(np.float32)
    img_in = np.transpose(img, (2, 0, 1))[None]
    mask_in = m[None, None]
    ins = {i.name: v for i, v in zip(sess.get_inputs(), (img_in, mask_in))}
    out = sess.run(None, ins)[0][0]
    out = np.clip(np.transpose(out, (1, 2, 0)), 0, 1)
    return cv2.resize((out * 255).astype(np.uint8), (w, h))


def restore_face(image_rgb: np.ndarray, model: str = "gfpgan_face") -> np.ndarray:
    """
    Neural face restoration (GFPGAN, Apache-2.0). Recovers pores/skin detail on faces
    without the airbrushed single-pass look. Expects a roughly aligned 512x512 face.
    """
    require_dl()
    sess = _session(model)
    import cv2
    face = cv2.resize(image_rgb, (512, 512)).astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5  # GFPGAN normalization
    inp = np.transpose(face, (2, 0, 1))[None]
    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0]
    out = np.clip((np.transpose(out, (1, 2, 0)) * 0.5 + 0.5), 0, 1)
    return cv2.resize((out * 255).astype(np.uint8), image_rgb.shape[1::-1])


# --------------------------------------------------------------------------- #
# Voice
# --------------------------------------------------------------------------- #
def clone_voice(src_wav: str, ref_wav: str, out_wav: str) -> Dict[str, Any]:
    """
    Zero-shot voice conversion: re-timbre the speech in `src_wav` toward the voice in
    `ref_wav`. Pipeline (RVC / Seed-VC style): content encoder (ContentVec) extracts
    speaker-independent linguistic content, a speaker embedding is taken from the
    reference clip, and a vocoder resynthesizes the content in the target timbre.

    This fits Quirky's post-processor identity -- it converts existing audio rather
    than generating from text. Requires the full [dl] voice checkpoints on first run.
    """
    require_dl()
    raise NotImplementedError(
        "Voice-conversion checkpoints are not bundled. To enable: register a ContentVec "
        "encoder + HiFi-GAN vocoder ONNX (RVC-MIT or Seed-VC) and wire the 3-stage "
        "pipeline. Framework, cache and provider selection are ready; only the "
        "model-specific pre/post-processing per checkpoint remains."
    )
