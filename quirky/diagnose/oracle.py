"""
Detector-honesty harness.

A humanizer that grades itself with its own metric is grading its own homework: it
can learn to defeat *that one number* without becoming more real. This module lets
before/after be scored by an oracle that is deliberately *separate* from the score
the humanizer optimizes.

Two oracles ship:

  EnsembleHeuristicOracle   Self-contained, weight-free. Combines several independent
                            natural-image statistics into one AI-probability. It is a
                            proxy, not ground truth — but it is a *different* function
                            from `ai_score`, so improving against it is not circular.

  NeuralOracle              Opt-in. Loads a real ONNX AI-image detector via the
                            quirky[dl] path. This is the gold standard; it raises
                            OracleUnavailable if the extra/model isn't present so the
                            core never hard-depends on it.

`audit(before, after)` reports the true before/after transfer against the chosen
oracle, which is what benchmarks should quote.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from quirky.diagnose.maps import load_gray_rgb
from quirky.detector.formulas import (
    compute_image_spectral_slope,
    compute_image_channel_correlation,
    compute_image_plastic_score,
    compute_image_symmetry_score,
    compute_image_texture_score,
)
from quirky.image.transforms import detect_synthetic_frequency_signature


class OracleUnavailable(RuntimeError):
    """Raised when a requested oracle (e.g. the neural one) cannot be loaded."""


class DetectorOracle:
    """Base class. An oracle maps an image path to P(AI-generated) in [0, 1]."""

    name = "base"
    kind = "abstract"

    def score(self, path: str) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def score_verbose(self, path: str) -> Dict[str, Any]:
        return {"oracle": self.name, "kind": self.kind, "ai_probability": round(self.score(path), 4)}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class EnsembleHeuristicOracle(DetectorOracle):
    """
    Weight-free ensemble of natural-image statistics, combined via a fixed logistic.
    Independent of Quirky's `ai_score` composite by construction.
    """

    name = "ensemble-heuristic"
    kind = "weight-free"

    # Fixed, documented coefficients (not fit against Quirky output). Each feature is
    # oriented so that a larger value pushes toward "AI".
    def score_verbose(self, path: str) -> Dict[str, Any]:
        gray, rgb = load_gray_rgb(path)
        g8 = (gray * 255.0).astype(np.uint8)

        slope = compute_image_spectral_slope(gray)
        slope_dev = abs(slope + 2.0)                                  # 0 natural, grows AI
        channel_corr = compute_image_channel_correlation((rgb * 255.0).astype(np.uint8))
        plastic = compute_image_plastic_score(g8)
        symmetry = compute_image_symmetry_score(g8)
        texture = compute_image_texture_score(g8)                     # low = AI
        grid = detect_synthetic_frequency_signature((rgb * 255.0).astype(np.uint8))

        feats = {
            "spectral_slope_dev": float(slope_dev),
            "channel_corr_deficit": float(np.clip(1.0 - channel_corr / 0.4, 0, 1)),
            "plastic": float(plastic),
            "symmetry": float(symmetry),
            "texture_deficit": float(np.clip(1.0 - texture, 0, 1)),
            "grid_spike": float(grid),
        }
        z = (
            -2.2
            + 1.6 * feats["spectral_slope_dev"]
            + 1.8 * feats["channel_corr_deficit"]
            + 1.5 * feats["plastic"]
            + 0.8 * feats["symmetry"]
            + 1.2 * feats["texture_deficit"]
            + 1.4 * feats["grid_spike"]
        )
        prob = float(_sigmoid(z))
        return {
            "oracle": self.name,
            "kind": self.kind,
            "ai_probability": round(prob, 4),
            "features": {k: round(v, 4) for k, v in feats.items()},
        }

    def score(self, path: str) -> float:
        return float(self.score_verbose(path)["ai_probability"])


class NeuralOracle(DetectorOracle):
    """
    Opt-in real detector via ONNX Runtime (quirky[dl]). Expects a model that takes an
    RGB image and returns a P(AI) logit/probability. Raises OracleUnavailable when the
    runtime or model is missing so nothing in core hard-depends on it.
    """

    name = "neural-onnx"
    kind = "neural"

    def __init__(self, model_path: Optional[str] = None):
        try:
            import onnxruntime  # noqa: F401  (quirky[dl])
        except Exception as e:  # pragma: no cover - depends on optional extra
            raise OracleUnavailable(
                "NeuralOracle needs the optional extra:  pip install -e '.[dl]'  and an "
                "ONNX AI-image detector model (set QUIRKY_ORACLE_MODEL or pass model_path)."
            ) from e
        self.model_path = model_path or os.environ.get("QUIRKY_ORACLE_MODEL")
        if not self.model_path or not os.path.exists(self.model_path):
            raise OracleUnavailable(
                "No ONNX detector model found. Point QUIRKY_ORACLE_MODEL at a local "
                "AI-image detector .onnx to enable the neural oracle."
            )
        import onnxruntime as ort
        self._sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])

    def score(self, path: str) -> float:  # pragma: no cover - requires a model file
        import cv2
        _, rgb = load_gray_rgb(path)
        inp = cv2.resize((rgb * 255.0).astype(np.float32), (224, 224))
        inp = np.transpose(inp / 255.0, (2, 0, 1))[None].astype(np.float32)
        name = self._sess.get_inputs()[0].name
        out = self._sess.run(None, {name: inp})[0].ravel()
        val = float(out[-1])
        return float(val if 0.0 <= val <= 1.0 else _sigmoid(val))


def get_oracle(kind: str = "auto", model_path: Optional[str] = None) -> DetectorOracle:
    """
    Factory. kind in {"auto","ensemble","neural"}.
    "auto" prefers a neural oracle if one is configured, else the ensemble.
    """
    if kind in ("neural", "auto"):
        try:
            return NeuralOracle(model_path=model_path)
        except OracleUnavailable:
            if kind == "neural":
                raise
    return EnsembleHeuristicOracle()


def audit(before: str, after: str, oracle: Optional[DetectorOracle] = None) -> Dict[str, Any]:
    """
    Score before/after against an *external* oracle and report the true transfer.
    """
    oracle = oracle or get_oracle("auto")
    b = oracle.score_verbose(before)
    a = oracle.score_verbose(after)
    pb, pa = b["ai_probability"], a["ai_probability"]
    reduction = pb - pa
    return {
        "oracle": oracle.name,
        "oracle_kind": oracle.kind,
        "before": b,
        "after": a,
        "ai_probability_before": pb,
        "ai_probability_after": pa,
        "absolute_reduction": round(reduction, 4),
        "relative_reduction_pct": round(100.0 * reduction / (pb + 1e-8), 1),
        "note": (
            "Scored against an oracle separate from Quirky's own ai_score. "
            "For publishable claims, configure the neural oracle (quirky[dl])."
        ),
        "attribution": "Powered by Quirky (MITPO)",
    }
