"""
Prescriptive diagnosis -> "fix cards".

Instead of a single opaque ai_score, measure the specific, named defects an image
has and emit one card per defect: what it is, how bad it is, where it is, and the
exact correction Quirky would apply (with the parameter it would use). This is the
data the dashboard renders as accept/reject cards, and it maps 1:1 onto the fix ids
ImageHumanizer.humanize() understands, so a user's selections drive the humanizer.

Pure classical CV. Reuses the *same* measurement functions the humanizer uses, so
the diagnosis and the fix never disagree.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from quirky.detector.engine import DetectorEngine
from quirky.image.pipeline import (
    estimate_color_cast,
    estimate_lighting_flatness,
)
from quirky.image.transforms import detect_face_regions, detect_blemishes
from quirky.diagnose.maps import load_gray_rgb, composite_slop_map, region_scores


# Catalogue of fixes the humanizer can apply, keyed by the id used everywhere
# (CLI --fixes, API `fixes`, and ImageHumanizer.humanize(enabled_fixes=...)).
DEFECT_CATALOG: Dict[str, Dict[str, str]] = {
    "white_balance": {
        "title": "Color cast",
        "explains": "Diffusion renders skew cold/teal. Gray-world white balance neutralizes it.",
        "region": "global",
    },
    "clahe_lighting": {
        "title": "Flat lighting",
        "explains": "Large variance-dead areas (impossible even illumination). CLAHE restores local contrast.",
        "region": "global",
    },
    "spot_removal": {
        "title": "Over-rendered specks / blemishes",
        "explains": "Stray dots and over-rendered pores. Content-aware inpaint reconstructs them from neighbours.",
        "region": "subject",
    },
    "face_relight": {
        "title": "Plastic / flat face lighting",
        "explains": "Airbrushed HDR glow on the face. Retinex re-split re-injects micro-shadow.",
        "region": "face",
    },
    "plastic_texture": {
        "title": "Plastic skin (no micro-texture)",
        "explains": "Missing pores/grain. Pore recovery + physically-based sensor grain restores density.",
        "region": "subject",
    },
    "spectrum": {
        "title": "Unnatural frequency spectrum",
        "explains": "Power spectrum deviates from the natural 1/f (slope != -2). 1/f grain shaping corrects it.",
        "region": "global",
    },
    "channel_corr": {
        "title": "Missing camera color correlation",
        "explains": "No Bayer/demosaic cross-channel detail. A demosaic round-trip reinstates it.",
        "region": "global",
    },
    "symmetry": {
        "title": "Suspicious bilateral symmetry",
        "explains": "Near pixel-perfect mirror symmetry. Flagged; asymmetric micro-grain reduces it.",
        "region": "subject",
    },
    "prompt_leak": {
        "title": "Generator metadata leak",
        "explains": "Embedded prompt/parameters found in the file metadata (EXIF/PNG chunks). Strip before sharing.",
        "region": "metadata",
    },
}


def _sev(score: float) -> str:
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


def diagnose_image(path: str, intensity: float = 0.6) -> Dict[str, Any]:
    """
    Full prescriptive diagnosis of an image.

    Returns:
        {
          "asset": path,
          "overall": {ai_score, plastic_score, spectral_slope, channel_corr, symmetry_score, ...},
          "defects": [ {id, title, severity, score, detail, region, recommended, param}, ... ],
          "heatmap_regions": {cell: mean_slop},
          "recommended_fixes": [ids...],
          "attribution": ...,
        }
    """
    gray, rgb = load_gray_rgb(path)
    report = DetectorEngine.analyze_asset(path)
    scores = report.get("metadata", {})

    comp, _maps = composite_slop_map(gray)

    defects: List[Dict[str, Any]] = []

    def add(fix_id: str, score: float, detail: str, param: Any = None, recommended: bool | None = None):
        score = float(np.clip(score, 0.0, 1.0))
        meta = DEFECT_CATALOG[fix_id]
        defects.append({
            "id": fix_id,
            "title": meta["title"],
            "region": meta["region"],
            "severity": _sev(score),
            "score": round(score, 3),
            "detail": detail,
            "explains": meta["explains"],
            "recommended": bool(score >= 0.33) if recommended is None else recommended,
            "param": param,
        })

    # 1. Color cast (same threshold the humanizer uses: >0.04 acts).
    cast = estimate_color_cast(rgb)
    if cast > 0.02:
        wb = float(np.clip(cast * 4.0, 0.0, 0.8)) * intensity
        add("white_balance", min(cast * 4.0, 1.0),
            f"Channel-mean skew {cast:.3f}.", param={"white_balance": round(wb, 3)},
            recommended=cast > 0.04)

    # 2. Flat lighting (>0.4 acts).
    flat = estimate_lighting_flatness(gray)
    if flat > 0.2:
        cl = float(np.clip((flat - 0.4) * 1.2, 0.0, 0.6)) * intensity
        add("clahe_lighting", flat,
            f"{flat*100:.0f}% of the frame has variance-dead lighting.",
            param={"clahe_lighting": round(cl, 3)}, recommended=flat > 0.4)

    # 3. Blemishes / specks (face-scoped when a face is found).
    face_mask = detect_face_regions((rgb * 255.0).astype(np.uint8))
    blem = detect_blemishes((gray * 255.0).astype(np.uint8), region_mask=face_mask)
    blem_px = int((blem > 0).sum())
    if blem_px > 0:
        frac = blem_px / gray.size
        add("spot_removal", min(frac * 400.0, 1.0),
            f"{blem_px} speck pixels detected"
            + (" on the face region." if face_mask is not None else " (skin/saliency region)."),
            param={"spot_removal_px": blem_px}, recommended=blem_px > 20)

    # 4. Face relight (only when a face is present).
    if face_mask is not None:
        add("face_relight", 0.5,
            "Face detected; portrait relighting available to break flat HDR glow.",
            param={"intensity": round(intensity, 3)}, recommended=True)

    # 5. Plastic skin / missing micro-texture.
    plastic = float(scores.get("plastic_score", 0.0))
    if plastic > 0.4:
        add("plastic_texture", plastic,
            f"plastic_score {plastic:.3f} (higher = more airbrushed).",
            param={"delta_grain": round(0.03 * intensity, 4)}, recommended=plastic > 0.6)

    # 6. Spectrum deviation from natural -2.
    slope = float(scores.get("spectral_slope", -2.0))
    slope_dev = abs(slope + 2.0)
    if slope_dev > 0.3:
        add("spectrum", min(slope_dev / 2.0, 1.0),
            f"spectral_slope {slope:.2f} (natural is -2.0; deviation {slope_dev:.2f}).",
            param={"spectral_beta": 0.5}, recommended=slope_dev > 0.5)

    # 7. Channel correlation deficit (camera-likeness).
    cc = float(scores.get("channel_corr", 0.0))
    if cc < 0.35:
        add("channel_corr", 1.0 - cc,
            f"channel_corr {cc:.3f} (real cameras ~0.4+).",
            param={"bayer_strength": round(0.18 * intensity, 3)}, recommended=cc < 0.25)

    # 8. Symmetry flag.
    sym = float(scores.get("symmetry_score", 0.0))
    if sym > 0.6:
        add("symmetry", sym,
            f"symmetry_score {sym:.3f} (near-perfect mirror is a diffusion tell).",
            param=None, recommended=False)

    # 9. Prompt/metadata leak.
    leak = float(scores.get("prompt_leak_score", 0.0))
    if leak > 0.5:
        add("prompt_leak", leak,
            "Generation parameters found embedded in the file metadata.",
            param=None, recommended=True)

    defects.sort(key=lambda d: d["score"], reverse=True)
    recommended = [d["id"] for d in defects if d["recommended"]]

    overall = {
        "ai_score": scores.get("ai_score"),
        "plastic_score": scores.get("plastic_score"),
        "spectral_slope": scores.get("spectral_slope"),
        "channel_corr": scores.get("channel_corr"),
        "symmetry_score": scores.get("symmetry_score"),
        "prompt_leak_score": scores.get("prompt_leak_score"),
        "slop_mean": round(float(comp.mean()), 4),
        "slop_peak": round(float(comp.max()), 4),
    }

    return {
        "asset": path,
        "overall": overall,
        "defects": defects,
        "recommended_fixes": recommended,
        "heatmap_regions": region_scores(comp),
        "attribution": "Powered by Quirky (MITPO)",
    }
