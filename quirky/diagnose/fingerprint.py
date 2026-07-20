"""
Generator fingerprinting (weight-free, heuristic).

Different synthetic-image sources leave different statistical residue:

  * Latent-diffusion VAE decoders (SD 1.5 / SDXL) tile an 8-px latent grid, so the
    high-pass residual auto-correlates at a fixed small stride.
  * GAN upsamplers leave checkerboard / spectral-spike energy in the HF ring.
  * "Over-smoothed" render styles (some Midjourney-like pipelines) show very low
    micro-texture density and low inter-channel correlation with little grid.
  * Real camera capture has broadband incoherent noise, ~ -2 spectral slope, and
    strong Bayer/demosaic channel correlation.

This module scores those signatures and returns a ranked guess plus the specific
inverse corrections that signature calls for. It is explicitly a *heuristic* prior,
not a trained classifier — it says "consistent with", never "certainly".
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import cv2

from quirky.diagnose.maps import load_gray_rgb
from quirky.detector.formulas import (
    compute_image_spectral_slope,
    compute_image_channel_correlation,
    compute_image_plastic_score,
    compute_image_symmetry_score,
)


def _highpass(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    return gray - blur


def grid_stride_energy(gray: np.ndarray, strides=(2, 4, 8, 16)) -> Dict[int, float]:
    """
    Auto-correlation of the high-pass residual at candidate pixel strides.
    A strong, isolated peak at stride 8 is the classic latent-VAE tile signature.
    Returns {stride: normalized_correlation}.
    """
    hp = _highpass(gray.astype(np.float32))
    hp = hp - hp.mean()
    denom = float((hp * hp).sum()) + 1e-8
    out: Dict[int, float] = {}
    for s in strides:
        if s >= min(hp.shape):
            out[s] = 0.0
            continue
        # correlation of the residual with itself shifted by s px (both axes averaged)
        cx = float((hp[:, :-s] * hp[:, s:]).sum()) / denom
        cy = float((hp[:-s, :] * hp[s:, :]).sum()) / denom
        out[s] = round((abs(cx) + abs(cy)) / 2.0, 4)
    return out


def hf_spike_concentration(gray: np.ndarray) -> float:
    """Fraction-of-energy in isolated high-frequency spikes (GAN/upsample tell)."""
    g = gray.astype(np.float32)
    if max(g.shape) > 256:
        s = 256.0 / max(g.shape)
        g = cv2.resize(g, (max(int(g.shape[1] * s), 16), max(int(g.shape[0] * s), 16)),
                       interpolation=cv2.INTER_AREA)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[-cy:h - cy, -cx:w - cx]
    r = np.sqrt(xx * xx + yy * yy)
    ring = mag[r > 0.6 * np.sqrt(cy * cy + cx * cx)]
    if ring.size == 0:
        return 0.0
    thr = np.percentile(ring, 99.5)
    spikes = ring[ring > thr]
    return float((ring > thr).mean() * (spikes.sum() / (ring.sum() + 1e-8)))


def fingerprint_image(path: str) -> Dict[str, Any]:
    """
    Returns:
        {
          "asset": path,
          "signals": {...raw measurements...},
          "ranking": [ {source, confidence, why, inverse_fixes:[...]}, ... ],
          "top": {source, confidence, ...},
          "attribution": ...,
        }
    """
    gray, rgb = load_gray_rgb(path)

    strides = grid_stride_energy(gray)
    grid8 = strides.get(8, 0.0)
    # "isolation": grid at 8 that stands out above its neighbours 4 and 16.
    neigh = 0.5 * (strides.get(4, 0.0) + strides.get(16, 0.0)) + 1e-6
    grid8_isolation = float(grid8 / neigh)

    spikes = hf_spike_concentration(gray)
    slope = compute_image_spectral_slope(gray)
    slope_dev = abs(slope + 2.0)
    channel_corr = compute_image_channel_correlation((rgb * 255.0).astype(np.uint8))
    plastic = compute_image_plastic_score((gray * 255.0).astype(np.uint8))
    symmetry = compute_image_symmetry_score((gray * 255.0).astype(np.uint8))

    signals = {
        "grid_stride_energy": strides,
        "grid8_isolation": round(grid8_isolation, 3),
        "hf_spike_concentration": round(spikes, 4),
        "spectral_slope": round(float(slope), 3),
        "spectral_slope_dev": round(float(slope_dev), 3),
        "channel_corr": round(float(channel_corr), 3),
        "plastic_score": round(float(plastic), 3),
        "symmetry_score": round(float(symmetry), 3),
    }

    # --- Score each candidate source in [0,1] from the signals ------------- #
    sd_vae = np.clip((grid8_isolation - 1.0) / 2.0, 0, 1) * 0.6 + np.clip(grid8 * 8.0, 0, 1) * 0.4
    gan = np.clip(spikes * 30.0, 0, 1) * 0.7 + np.clip(slope_dev / 2.0, 0, 1) * 0.3
    oversmooth = (
        np.clip(plastic, 0, 1) * 0.5
        + np.clip(1.0 - channel_corr / 0.4, 0, 1) * 0.3
        + np.clip(symmetry, 0, 1) * 0.2
    ) * float(np.clip(1.0 - grid8_isolation / 3.0, 0, 1))  # low grid → not VAE-tiled
    generic_diffusion = (
        np.clip(slope_dev / 2.0, 0, 1) * 0.5
        + np.clip(1.0 - channel_corr / 0.4, 0, 1) * 0.5
    )
    real = (
        np.clip(channel_corr / 0.45, 0, 1) * 0.5
        + np.clip(1.0 - slope_dev / 1.0, 0, 1) * 0.3
        + np.clip(1.0 - plastic, 0, 1) * 0.2
    )

    candidates = [
        {
            "source": "Latent diffusion (SD/SDXL VAE 8px grid)",
            "confidence": round(float(sd_vae), 3),
            "why": f"grid-8 auto-correlation isolation {grid8_isolation:.2f} (energy {grid8:.3f}).",
            "inverse_fixes": ["grid_notch@8px", "channel_corr", "plastic_texture"],
        },
        {
            "source": "GAN / upsample checkerboard",
            "confidence": round(float(gan), 3),
            "why": f"HF spike concentration {spikes:.4f}, spectrum dev {slope_dev:.2f}.",
            "inverse_fixes": ["spectrum", "plastic_texture"],
        },
        {
            "source": "Over-smoothed render (Midjourney-like)",
            "confidence": round(float(oversmooth), 3),
            "why": f"plastic {plastic:.2f}, channel_corr {channel_corr:.2f}, low grid.",
            "inverse_fixes": ["plastic_texture", "channel_corr", "face_relight"],
        },
        {
            "source": "Generic diffusion",
            "confidence": round(float(generic_diffusion), 3),
            "why": f"spectrum dev {slope_dev:.2f}, channel_corr {channel_corr:.2f}.",
            "inverse_fixes": ["spectrum", "channel_corr", "plastic_texture"],
        },
        {
            "source": "Likely real / camera capture",
            "confidence": round(float(real), 3),
            "why": f"channel_corr {channel_corr:.2f}, near-natural slope, textured.",
            "inverse_fixes": [],
        },
    ]

    ranking = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
    return {
        "asset": path,
        "signals": signals,
        "ranking": ranking,
        "top": ranking[0],
        "attribution": "Powered by Quirky (MITPO)",
    }
