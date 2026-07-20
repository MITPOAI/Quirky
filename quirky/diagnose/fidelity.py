"""
Minimal-edit fidelity lock.

The base humanizer is one-shot at a fixed intensity: too low and it doesn't move
the needle, too high and it over-cooks the image. This wraps it in a closed loop
that climbs intensity in small steps and STOPS at the first setting that meets the
detector target *while* structural similarity to the original stays above a floor.

"Least intervention": we return the smallest edit that does the job, never the
maximal one. If the target can't be met without dropping below the SSIM floor, we
return the best result that still respects the floor and say so.

SSIM here is a compact NumPy implementation (Wang et al. 2004) — no scikit-image
dependency required.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Set

import numpy as np
import cv2
from PIL import Image

from quirky.detector.engine import DetectorEngine
from quirky.image.pipeline import ImageHumanizer


def _gray01(path_or_arr) -> np.ndarray:
    if isinstance(path_or_arr, str):
        with Image.open(path_or_arr) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    else:
        arr = path_or_arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
    return 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]


def ssim(img_a, img_b, win: int = 7) -> float:
    """
    Global mean SSIM between two images (paths or arrays), on luminance.
    Returns a scalar in roughly [-1, 1]; 1.0 = identical.
    """
    a = _gray01(img_a)
    b = _gray01(img_b)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    C1, C2 = (0.01 ** 2), (0.03 ** 2)
    k = (win, win)
    mu_a = cv2.blur(a, k)
    mu_b = cv2.blur(b, k)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = cv2.blur(a * a, k) - mu_a2
    sb = cv2.blur(b * b, k) - mu_b2
    sab = cv2.blur(a * b, k) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sa + sb + C2)
    return float(np.clip(num / (den + 1e-12), -1, 1).mean())


def humanize_locked(
    img_path: str,
    output_path: str,
    target_ai: float = 0.15,
    min_ssim: float = 0.86,
    start: float = 0.2,
    stop: float = 0.9,
    step: float = 0.1,
    gamma: float = 0.6,
    delta: float = 0.03,
    enabled_fixes: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Climb intensity from `start` to `stop`; keep the smallest intensity whose output
    has ai_score <= target_ai AND ssim(original, output) >= min_ssim.

    Returns a dict with the chosen intensity, whether the target was met, the reason
    it stopped, and the full per-step trace. The winning image is written to
    output_path.
    """
    original_score = DetectorEngine.analyze_asset(img_path).get("metadata", {}).get("ai_score", None)
    trace: List[Dict[str, Any]] = []

    tmp_dir = tempfile.mkdtemp(prefix="quirky_lock_")
    ext = os.path.splitext(output_path)[1] or ".png"

    best: Optional[Dict[str, Any]] = None      # best (lowest ai) result respecting ssim floor
    winner: Optional[Dict[str, Any]] = None    # first result meeting target + floor
    reason = "exhausted intensity range without meeting target"

    intensity = start
    n = 0
    while intensity <= stop + 1e-9:
        tmp_out = os.path.join(tmp_dir, f"step_{n}{ext}")
        ImageHumanizer.humanize(
            img_path, tmp_out, intensity=round(intensity, 4),
            gamma=gamma, delta=delta, enabled_fixes=enabled_fixes,
        )
        ai = float(DetectorEngine.analyze_asset(tmp_out).get("metadata", {}).get("ai_score", 1.0))
        sim = ssim(img_path, tmp_out)
        step_rec = {"intensity": round(intensity, 3), "ai_score": round(ai, 4), "ssim": round(sim, 4),
                    "path": tmp_out}
        trace.append({k: v for k, v in step_rec.items() if k != "path"})

        floor_ok = sim >= min_ssim
        if floor_ok and (best is None or ai < best["ai_score"]):
            best = step_rec
        if floor_ok and ai <= target_ai:
            winner = step_rec
            reason = "target met at minimal intensity within fidelity floor"
            break
        if not floor_ok:
            # Pushing harder will only lower SSIM further; stop climbing.
            reason = f"fidelity floor {min_ssim} would be breached above intensity {intensity:.2f}"
            break
        intensity += step
        n += 1

    chosen = winner or best
    if chosen is None:
        # Even the gentlest pass broke the floor; fall back to the gentlest result.
        chosen = trace and {"intensity": trace[0]["intensity"], "ai_score": trace[0]["ai_score"],
                            "ssim": trace[0]["ssim"], "path": os.path.join(tmp_dir, "step_0" + ext)} or None
        reason = "no setting satisfied the fidelity floor; returned gentlest pass"

    # Materialize the chosen image at output_path.
    if chosen and os.path.exists(chosen["path"]):
        Image.open(chosen["path"]).save(output_path)

    return {
        "output_path": output_path,
        "target_met": winner is not None,
        "chosen_intensity": chosen["intensity"] if chosen else None,
        "original_ai_score": original_score,
        "final_ai_score": chosen["ai_score"] if chosen else None,
        "final_ssim": chosen["ssim"] if chosen else None,
        "target_ai": target_ai,
        "min_ssim": min_ssim,
        "reason": reason,
        "trace": trace,
        "attribution": "Powered by Quirky (MITPO)",
    }
