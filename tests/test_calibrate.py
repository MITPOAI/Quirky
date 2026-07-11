import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from quirky.detector.calibrate import Calibrator, calibrated_text_score, extract_features
from quirky.detector.fit_calibrator import fit_logistic


def test_fit_and_accuracy():
    # Load seeds to verify fit stats
    seed_path = Path(__file__).parents[1] / "quirky" / "detector" / "data" / "seed_texts.jsonl"
    if not seed_path.exists():
        pytest.skip("seed_texts.jsonl not found")
        
    texts, labels = [], []
    with open(seed_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(int(row["label"]))
            
    X = np.stack([np.array(list(extract_features(t).values())) for t in texts])
    y = np.array(labels, dtype=float)
    
    coef, intercept, mean, std = fit_logistic(X, y)
    Xs = (X - mean) / (std + 1e-8)
    p = 1.0 / (1.0 + np.exp(-(Xs @ coef + intercept)))
    
    acc = np.mean((p >= 0.5) == (y == 1.0))
    brier = np.mean((p - y) ** 2)
    
    assert acc >= 0.8, f"Accuracy too low: {acc}"
    assert brier <= 0.15, f"Brier score too high: {brier}"
    
    # Assert probability bounds
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_calibrator_load_fallbacks():
    # 1. Missing file -> None
    assert Calibrator.load("non_existent_file.json") is None
    
    # 2. Invalid version -> None
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json", encoding="utf-8") as tmp:
        json.dump({"version": 999, "kind": "logistic"}, tmp)
        tmp_name = tmp.name
        
    try:
        assert Calibrator.load(tmp_name) is None
    finally:
        os.remove(tmp_name)
        
    # 3. Bad features -> None
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json", encoding="utf-8") as tmp:
        json.dump({"version": 1, "kind": "logistic", "features": ["bad"]}, tmp)
        tmp_name = tmp.name
        
    try:
        assert Calibrator.load(tmp_name) is None
    finally:
        os.remove(tmp_name)


def test_calibrated_text_score_fallbacks():
    # Create a corrupt calibrator
    prob, source = calibrated_text_score("Hello world", calibrator=None)
    # If default calibrator loads, source is 'calibrated', otherwise 'heuristic'
    assert source in ("calibrated", "heuristic")
    assert 0.0 <= prob <= 1.0


def test_cliche_monotonicity():
    # Cliché injection should monotonically increase slop score
    clean_text = "Let's check the numbers first and run a build."
    slop_text = clean_text + " Furthermore, it is important to note that this facilitates optimization."
    
    score_clean, _ = calibrated_text_score(clean_text)
    score_slop, _ = calibrated_text_score(slop_text)
    
    assert score_slop >= score_clean
