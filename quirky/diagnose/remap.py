"""
Re-map: after any edit, look again.

Formalizes "map it again" as an explicit step. `humanize_locked()` already loops
intensity within a single call; this module closes the loop *around* a full edit by
re-running the same diagnosis on the output that ran on the input, then reporting what
actually changed: which defects were resolved, which remain, and -- importantly --
whether the edit introduced anything NEW that wasn't flagged before, which is the
signature of over-cooking an image. `remap_loop()` chains diagnose -> humanize -> remap
across multiple rounds and stops on convergence, a fidelity-floor breach, or a round cap.

Zero new dependencies: built entirely from quirky.diagnose.maps and
quirky.diagnose.report, which already exist.
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
from typing import Any, Dict, List, Set

import numpy as np
import cv2
from PIL import Image

from quirky.diagnose.maps import load_gray_rgb, composite_slop_map
from quirky.diagnose.report import diagnose_image
from quirky.diagnose.fidelity import ssim
from quirky.image.pipeline import ImageHumanizer

ATTRIBUTION = "Powered by Quirky (MITPO)"


def _defect_ids(defects: List[Dict[str, Any]], min_score: float = 0.33) -> Set[str]:
    return {d["id"] for d in defects if d["score"] >= min_score}


def _render_delta_overlay(rgb: np.ndarray, delta: np.ndarray, alpha: float = 0.55) -> str:
    """Green where slop improved, red where it got worse. Returns a base64 PNG data URI."""
    base = (rgb * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)

    mag = np.clip(np.abs(delta), 0.0, 1.0)
    tint = np.zeros_like(gray3)
    improved = delta > 0
    worsened = ~improved
    tint[improved, 1] = 255.0   # green: slop went down here
    tint[worsened, 0] = 255.0   # red: slop went up here
    a = (alpha * mag)[:, :, None]
    out = np.clip(gray3 * (1.0 - a) + tint * a, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def remap_image(before_path: str, after_path: str, intensity: float = 0.6) -> Dict[str, Any]:
    """
    Re-diagnose `after_path` and compare it against a fresh diagnosis of `before_path`.
    Returns resolved/remaining/new defect ids, the slop-score delta, a red/green delta
    heatmap (base64 PNG), and a plain recommendation: clean, keep_going,
    diminishing_returns, or over_cooked.
    """
    before = diagnose_image(before_path, intensity=intensity)
    after = diagnose_image(after_path, intensity=intensity)

    before_ids = _defect_ids(before["defects"])
    after_ids = _defect_ids(after["defects"])

    resolved = sorted(before_ids - after_ids)
    remaining = sorted(before_ids & after_ids)
    new = sorted(after_ids - before_ids)

    b_gray, b_rgb = load_gray_rgb(before_path)
    a_gray, a_rgb = load_gray_rgb(after_path)
    b_comp, _ = composite_slop_map(b_gray)
    a_comp, _ = composite_slop_map(a_gray)
    if a_comp.shape != b_comp.shape:
        b_comp = cv2.resize(b_comp, (a_comp.shape[1], a_comp.shape[0]))

    delta = b_comp - a_comp  # positive = improved (less slop) at that pixel
    delta_png = _render_delta_overlay(a_rgb, delta)

    b_mean = float(b_comp.mean())
    a_mean = float(a_comp.mean())
    improvement_pct = round(100.0 * (b_mean - a_mean) / (b_mean + 1e-8), 1)

    if new:
        recommendation = "over_cooked"
        note = f"{len(new)} new defect(s) appeared that weren't in the original — consider a lower intensity."
    elif not after_ids:
        recommendation = "clean"
        note = "No flagged defects remain."
    elif remaining and improvement_pct < 5.0:
        recommendation = "diminishing_returns"
        note = "Little improvement from this pass — another round is unlikely to help much."
    else:
        recommendation = "keep_going"
        note = f"{len(remaining)} defect(s) still flagged — another pass could help."

    return {
        "before": before_path,
        "after": after_path,
        "resolved": resolved,
        "remaining": remaining,
        "new": new,
        "slop_before_mean": round(b_mean, 4),
        "slop_after_mean": round(a_mean, 4),
        "improvement_pct": improvement_pct,
        "recommendation": recommendation,
        "note": note,
        "delta_heatmap": delta_png,
        "before_overall": before["overall"],
        "after_overall": after["overall"],
        "attribution": ATTRIBUTION,
    }


def remap_loop(
    img_path: str,
    output_path: str,
    min_ssim: float = 0.80,
    max_rounds: int = 3,
    start_intensity: float = 0.4,
    step: float = 0.2,
) -> Dict[str, Any]:
    """
    Diagnose -> humanize -> re-map, repeated until clean, diminishing returns, the
    fidelity floor would be breached, or max_rounds is hit. Each round's output feeds
    the next round's input. Returns the full round-by-round trace; writes the final
    result to output_path.
    """
    rounds: List[Dict[str, Any]] = []
    current_in = img_path
    intensity = start_intensity
    tmp_dir = tempfile.mkdtemp(prefix="quirky_remap_")
    final_out = img_path

    for i in range(max_rounds):
        d = diagnose_image(current_in, intensity=intensity)
        if not d["recommended_fixes"]:
            rounds.append({"round": i, "action": "none", "reason": "already clean"})
            final_out = current_in
            break

        round_out = os.path.join(tmp_dir, f"round_{i}.png")
        ImageHumanizer.humanize(current_in, round_out, intensity=intensity,
                                enabled_fixes=set(d["recommended_fixes"]))

        rm = remap_image(current_in, round_out, intensity=intensity)
        sim_to_original = ssim(img_path, round_out)
        round_rec = {
            "round": i,
            "intensity": round(intensity, 3),
            "fixes_applied": d["recommended_fixes"],
            "resolved": rm["resolved"],
            "remaining": rm["remaining"],
            "new": rm["new"],
            "improvement_pct": rm["improvement_pct"],
            "recommendation": rm["recommendation"],
            "ssim_to_original": round(sim_to_original, 4),
        }
        final_out = round_out
        current_in = round_out

        if sim_to_original < min_ssim:
            round_rec["stopped_reason"] = "fidelity floor reached"
            rounds.append(round_rec)
            break
        rounds.append(round_rec)
        if rm["recommendation"] in ("clean", "diminishing_returns", "over_cooked"):
            round_rec["stopped_reason"] = rm["recommendation"]
            break
        intensity = min(intensity + step, 0.9)

    Image.open(final_out).save(output_path)

    return {
        "output_path": output_path,
        "rounds": rounds,
        "total_rounds": len(rounds),
        "attribution": ATTRIBUTION,
    }
