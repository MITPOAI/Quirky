"""
Spatial "tell maps": turn Quirky's scalar detector signals into per-pixel maps
that show *where* an image reads as synthetic, then blend them into a single
Slop X-ray heatmap suitable for overlaying on the original.

All maps are float32 in [0, 1] at the input resolution. Higher = more suspicious.
Pure NumPy + OpenCV (+ SciPy uniform_filter). No weights, no network.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import uniform_filter

# Reuse the classical saliency already shipped in the image pipeline so the heatmap
# is weighted toward the actual subject rather than flat background.
from quirky.image.pipeline import spectral_residual_saliency

ATTRIBUTION = "Powered by Quirky (MITPO)"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_gray_rgb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (gray[0,1] HxW float32, rgb[0,1] HxWx3 float32)."""
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return gray.astype(np.float32), rgb


def _norm01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Individual tell maps  (all HxW float32 in [0,1], higher = more "AI")
# --------------------------------------------------------------------------- #
def plastic_map(gray: np.ndarray, win: int = 15, norm: float = 0.06) -> np.ndarray:
    """
    Local micro-texture deficit. Real skin/fabric/foliage carries dense fine
    gradients; diffusion output is airbrushed. We measure local gradient density
    and invert it: smooth regions -> high plastic score.
    """
    gy, gx = np.gradient(gray.astype(np.float32))
    gmag = np.sqrt(gx * gx + gy * gy)
    density = uniform_filter(gmag, size=win)
    plastic = 1.0 - np.clip(density / max(norm, 1e-6), 0.0, 1.0)
    return plastic.astype(np.float32)


def flatness_map(gray: np.ndarray, win: int = 15, var_thresh: float = 1e-3) -> np.ndarray:
    """
    Variance-dead lighting. Fraction-style local-variance map; regions whose local
    variance sits near zero (impossible flat illumination) light up.
    """
    g = gray.astype(np.float32)
    mean = uniform_filter(g, size=win)
    var = uniform_filter((g - mean) ** 2, size=win)
    # Soft indicator: fully lit when var << thresh, fades out above it.
    flat = np.clip(1.0 - var / max(var_thresh, 1e-8), 0.0, 1.0)
    return flat.astype(np.float32)


def symmetry_map(gray: np.ndarray, win: int = 15) -> np.ndarray:
    """
    Suspicious bilateral symmetry. Compares the image to its horizontal mirror and
    highlights regions that match their mirror too closely (near pixel-perfect
    reflection is a diffusion-face tell). Weighted to fade toward the frame edges,
    where mirror comparison is meaningless.
    """
    g = gray.astype(np.float32)
    diff = np.abs(g - np.fliplr(g))
    local = uniform_filter(diff, size=win)
    # low diff -> high symmetry score
    sym = np.clip(1.0 - local / 0.15, 0.0, 1.0)
    h, w = g.shape
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    axis_weight = np.clip(1.0 - np.abs(xx), 0.0, 1.0)  # strongest on the vertical midline
    return (sym * axis_weight).astype(np.float32)


def highfreq_grid_map(gray: np.ndarray) -> np.ndarray:
    """
    Upsampling / VAE grid residue. High-pass the image, then look at the periodic
    residual: real sensor noise is broadband and incoherent, generator grids leave
    a regular tiled residual. We surface the rectified high-pass energy, normalized,
    which concentrates on the tiled seams AI leaves behind.
    """
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.2)
    hp = g - blur
    energy = uniform_filter(np.abs(hp), size=7)
    # Emphasize structured (coherent) high-frequency: subtract a broad median floor.
    floor = float(np.median(energy))
    struct = np.clip(energy - floor, 0.0, None)
    return _norm01(struct)


# --------------------------------------------------------------------------- #
# Composite  +  rendering
# --------------------------------------------------------------------------- #
def tell_maps(gray: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute every individual tell map. Returns a dict of HxW float32 [0,1]."""
    return {
        "plastic": plastic_map(gray),
        "flatness": flatness_map(gray),
        "symmetry": symmetry_map(gray),
        "grid": highfreq_grid_map(gray),
    }


def composite_slop_map(
    gray: np.ndarray,
    weights: Dict[str, float] | None = None,
    subject_weighted: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Blend the individual tell maps into one Slop X-ray heatmap in [0,1].

    The composite is weighted toward the salient subject (via the classical
    spectral-residual saliency already used by the humanizer), so a busy but
    photographic background does not dominate the picture.

    Returns (composite HxW float32, {individual maps}).
    """
    weights = weights or {"plastic": 0.5, "flatness": 0.2, "symmetry": 0.15, "grid": 0.15}
    maps = tell_maps(gray)
    total_w = sum(weights.get(k, 0.0) for k in maps) or 1.0
    comp = np.zeros_like(gray, dtype=np.float32)
    for k, m in maps.items():
        comp += weights.get(k, 0.0) * m
    comp /= total_w

    if subject_weighted:
        sal = spectral_residual_saliency(gray).astype(np.float32)
        comp = comp * (0.55 + 0.45 * sal)

    comp = cv2.GaussianBlur(comp, (0, 0), sigmaX=max(gray.shape) / 400.0 + 0.5)
    return np.clip(comp, 0.0, 1.0).astype(np.float32), maps


def render_heatmap_overlay(
    rgb: np.ndarray,
    heat: np.ndarray,
    out_path: str,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_TURBO,
) -> str:
    """
    Blend a heatmap over a desaturated copy of the original and write a PNG.
    `rgb` is float[0,1] HxWx3; `heat` is float[0,1] HxW. Returns out_path.
    """
    base = (rgb * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)

    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    cmap_bgr = cv2.applyColorMap(heat_u8, colormap)
    cmap_rgb = cv2.cvtColor(cmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Only tint where the heat is meaningful, so cold regions keep the photo.
    a = (alpha * heat)[:, :, None]
    out = gray3 * (1.0 - a) + cmap_rgb * a
    out = np.clip(out, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    Image.fromarray(out).save(out_path)
    return out_path


def heatmap_png_bytes(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> bytes:
    """Same as render_heatmap_overlay but returns PNG bytes (for API/base64)."""
    import io

    base = (rgb * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    cmap_rgb = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB).astype(np.float32)
    a = (alpha * heat)[:, :, None]
    out = np.clip(gray3 * (1.0 - a) + cmap_rgb * a, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue()


def region_scores(comp: np.ndarray, grid: int = 3) -> Dict[str, float]:
    """
    Coarse grid summary of where the slop concentrates (for text explanations like
    "worst in the upper-center"). Returns {cell_name: mean_score}.
    """
    h, w = comp.shape
    rows = ["top", "mid", "bottom"]
    cols = ["left", "center", "right"]
    out: Dict[str, float] = {}
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    for i in range(grid):
        for j in range(grid):
            cell = comp[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            out[f"{rows[i]}-{cols[j]}"] = round(float(cell.mean()), 4)
    return out
