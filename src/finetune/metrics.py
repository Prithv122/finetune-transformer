"""Metrics for the learning curve: macro-F1, accuracy, calibration error and per-intent F1.

numpy and scikit-learn only, so the fast CI job (no torch) can import it. Every function takes
integer label ids; ``n_classes`` is passed explicitly so an intent that a model never predicts
(or that is missing from a small test slice) still counts as an F1 of 0 instead of vanishing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> float:
    """Unweighted mean F1 over all ``n_classes`` labels; a class never predicted scores 0."""
    return float(
        f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)
    )


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(accuracy_score(y_true, y_pred))


def ece(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    confidence: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected calibration error over the top-class confidence.

    ``n_bins`` equal-width bins on [0, 1]; a confidence of exactly 1.0 goes in the last bin.
    Each non-empty bin adds ``|accuracy - mean confidence| * count / n``; empty bins add 0.
    """
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    conf = np.asarray(confidence, dtype=float)
    if len(correct) != len(conf):
        raise ValueError(f"{len(correct)} predictions but {len(conf)} confidences")
    if len(conf) == 0:
        raise ValueError("ece of an empty set is undefined")
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += abs(correct[mask].mean() - conf[mask].mean()) * mask.sum() / len(conf)
    return float(total)


def per_class_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    """F1 for each label id 0..n_classes-1 (length ``n_classes``)."""
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)


def worst_classes(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
    n: int = 10,
) -> list[tuple[str, float]]:
    """The ``n`` intents with the lowest F1, lowest first; ties are broken by name."""
    scores = per_class_f1(y_true, y_pred, len(label_names))
    ranked = sorted(zip(label_names, scores.tolist(), strict=True), key=lambda p: (p[1], p[0]))
    return ranked[:n]


def top_k(proba: np.ndarray, k: int = 3) -> list[list[tuple[int, float]]]:
    """Per row, the ``k`` most probable ``(label id, probability)`` pairs, best first."""
    proba = np.asarray(proba)
    k = min(k, proba.shape[1])
    order = np.argsort(-proba, axis=1, kind="stable")[:, :k]
    return [[(int(j), float(row[j])) for j in idx] for row, idx in zip(proba, order, strict=True)]
