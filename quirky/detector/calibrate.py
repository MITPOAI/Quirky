"""
Calibrated slop probability for text.

Turns the per-feature heuristics into a single probability via a small
logistic model fit on the labeled seed set (see fit_calibrator.py). The fitted
coefficients live in data/calibrator.json and are committed, so runtime never
needs a training dependency. If the artifact is missing or invalid, scoring
silently falls back to the raw heuristic (formulas.compute_text_ai_score).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from quirky.detector.formulas import (
    compute_text_ai_score,
    compute_text_prompt_leak_score,
    compute_text_repetition_score,
)
from quirky.detector.lexicons import AI_CLICHES, FILLERS, HEDGES

FEATURE_ORDER: Tuple[str, ...] = (
    "burstiness",
    "cliche_density",
    "repetition",
    "prompt_leak",
    "hedge_density",
    "filler_density",
    "contraction_deficit",
    "sent_len_cv_inv",
    "ai_punct_density",
    "type_token_ratio_inv",
)

DEFAULT_CALIBRATOR_PATH = Path(__file__).parent / "data" / "calibrator.json"

# Expanded forms whose contracted twin exists (mirrors TextHumanizer.CONTRACTIONS
# keys); a text full of "do not"/"it is" with zero apostrophes reads machine-like.
_EXPANDED = (
    "do not", "cannot", "it is", "should not", "we are", "they are",
    "would not", "i am", "that is", "there is", "you are", "does not",
    "will not",
)


def _phrase_pattern(phrases) -> re.Pattern:
    # Longest-first so "delve into" wins over "delve" in the alternation.
    ordered = sorted(phrases, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in ordered) + r")\b",
                      re.IGNORECASE)


_CLICHE_RE = _phrase_pattern(AI_CLICHES)
_HEDGE_RE = _phrase_pattern(HEDGES)
_FILLER_RE = _phrase_pattern(FILLERS)
_EXPANDED_RE = _phrase_pattern(_EXPANDED)
_AI_PUNCT_RE = re.compile(r"[—–…“”‘’]|\.{3}")


def extract_features(text: str) -> Dict[str, float]:
    """Per-text feature dict, keys = FEATURE_ORDER. All values roughly in [0, 1+]."""
    words = re.findall(r"\b\w+\b", text)
    n_words = max(len(words), 1)
    n_chars = max(len(text), 1)

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) > 1:
        arr = np.array(lengths, dtype=float)
        variance = float(arr.var())
        cv = float(arr.std() / (arr.mean() + 1e-8))
    else:
        variance, cv = 0.0, 0.0

    expanded = len(_EXPANDED_RE.findall(text))
    contracted = text.count("'")

    return {
        "burstiness": 1.0 - min(variance / 100.0, 1.0),
        "cliche_density": len(_CLICHE_RE.findall(text)) / n_words * 100.0,
        "repetition": compute_text_repetition_score(text),
        "prompt_leak": compute_text_prompt_leak_score(text),
        "hedge_density": len(_HEDGE_RE.findall(text)) / n_words * 100.0,
        "filler_density": len(_FILLER_RE.findall(text)) / n_words * 100.0,
        "contraction_deficit": expanded / (expanded + contracted + 1.0),
        "sent_len_cv_inv": 1.0 - float(np.clip(cv, 0.0, 1.0)),
        "ai_punct_density": len(_AI_PUNCT_RE.findall(text)) / n_chars * 100.0,
        "type_token_ratio_inv": 1.0 - (len({w.lower() for w in words}) / n_words),
    }


def feature_vector(text: str) -> np.ndarray:
    feats = extract_features(text)
    return np.array([feats[k] for k in FEATURE_ORDER], dtype=float)


class Calibrator:
    """Standardized logistic model loaded from a committed JSON artifact."""

    def __init__(self, mean: np.ndarray, std: np.ndarray,
                 coef: np.ndarray, intercept: float):
        self.mean = mean
        self.std = std
        self.coef = coef
        self.intercept = intercept

    @classmethod
    def load(cls, path: "str | os.PathLike | None" = None) -> "Optional[Calibrator]":
        """Return a Calibrator, or None on any problem (missing/corrupt/mismatch)."""
        candidate = path or os.environ.get("QUIRKY_CALIBRATOR_PATH") or DEFAULT_CALIBRATOR_PATH
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != 1 or data.get("kind") != "logistic":
                return None
            if list(data.get("features", [])) != list(FEATURE_ORDER):
                return None
            mean = np.asarray(data["mean"], dtype=float)
            std = np.asarray(data["std"], dtype=float)
            coef = np.asarray(data["coef"], dtype=float)
            if not (len(mean) == len(std) == len(coef) == len(FEATURE_ORDER)):
                return None
            return cls(mean, std, coef, float(data["intercept"]))
        except Exception:
            return None

    def predict_proba(self, features: Dict[str, float]) -> float:
        x = np.array([features[k] for k in FEATURE_ORDER], dtype=float)
        z = float(self.coef @ ((x - self.mean) / (self.std + 1e-8)) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-z)))


_DEFAULT: "Optional[Calibrator]" = None
_DEFAULT_LOADED = False


def _default_calibrator() -> "Optional[Calibrator]":
    global _DEFAULT, _DEFAULT_LOADED
    if not _DEFAULT_LOADED:
        _DEFAULT = Calibrator.load()
        _DEFAULT_LOADED = True
    return _DEFAULT


def calibrated_text_score(text: str,
                          calibrator: "Optional[Calibrator]" = None) -> Tuple[float, str]:
    """
    (probability, source). source == "calibrated" when the fitted model was used,
    "heuristic" when it fell back to formulas.compute_text_ai_score.
    """
    cal = calibrator if calibrator is not None else _default_calibrator()
    if cal is not None:
        try:
            return cal.predict_proba(extract_features(text)), "calibrated"
        except Exception:
            pass
    return compute_text_ai_score(text), "heuristic"
