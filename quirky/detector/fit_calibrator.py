"""
Fit the logistic slop calibrator on the labeled seed set.

Pure numpy (no sklearn): full-batch gradient descent with L2 regularization is
plenty for ~100 samples x 10 features. The fitted artifact is committed at
data/calibrator.json so runtime installs never need this script.

Run:
    uv run python -m quirky.detector.fit_calibrator
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np

from quirky.detector.calibrate import FEATURE_ORDER, feature_vector

DATA_DIR = Path(__file__).parent / "data"
SEED_PATH = DATA_DIR / "seed_texts.jsonl"
OUT_PATH = DATA_DIR / "calibrator.json"


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2,
                 lr: float = 0.5, iters: int = 3000) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Returns (coef, intercept, mean, std) for standardized-input logistic regression."""
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    Xs = (X - mu) / (sigma + 1e-8)
    n, d = Xs.shape
    w, b = np.zeros(d), 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        gw = Xs.T @ (p - y) / n + l2 * w
        gb = float(np.mean(p - y))
        w -= lr * gw
        b -= lr * gb
    return w, b, mu, sigma


def main() -> None:
    texts, labels = [], []
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(int(row["label"]))

    X = np.stack([feature_vector(t) for t in texts])
    y = np.array(labels, dtype=float)

    w, b, mu, sigma = fit_logistic(X, y)

    Xs = (X - mu) / (sigma + 1e-8)
    p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
    acc = float(np.mean((p >= 0.5) == (y == 1.0)))
    brier = float(np.mean((p - y) ** 2))

    artifact = {
        "version": 1,
        "kind": "logistic",
        "features": list(FEATURE_ORDER),
        "mean": [round(float(v), 6) for v in mu],
        "std": [round(float(v), 6) for v in sigma],
        "coef": [round(float(v), 6) for v in w],
        "intercept": round(float(b), 6),
        "n_samples": len(y),
        "metrics": {"train_acc": round(acc, 4), "brier": round(brier, 4)},
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"fit {len(y)} samples  train_acc={acc:.3f}  brier={brier:.3f}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
