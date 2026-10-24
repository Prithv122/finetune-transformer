"""Metrics: sklearn agreement on random inputs and hand-computed calibration cases."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from finetune.metrics import accuracy, ece, macro_f1, per_class_f1, top_k, worst_classes


@pytest.mark.parametrize("seed", range(5))
def test_macro_f1_and_accuracy_match_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 8, 200)
    y_pred = rng.integers(0, 8, 200)
    assert macro_f1(y_true, y_pred, 8) == pytest.approx(
        f1_score(y_true, y_pred, average="macro", labels=range(8), zero_division=0)
    )
    assert accuracy(y_true, y_pred) == pytest.approx(accuracy_score(y_true, y_pred))


def test_macro_f1_counts_a_class_that_is_never_predicted() -> None:
    # Class 2 exists in the label space but appears nowhere: it must pull the mean down.
    assert macro_f1([0, 1], [0, 1], 2) == pytest.approx(1.0)
    assert macro_f1([0, 1], [0, 1], 3) == pytest.approx(2 / 3)


def test_per_class_f1_has_one_entry_per_class() -> None:
    scores = per_class_f1([0, 0, 1], [0, 1, 1], 4)
    assert scores.shape == (4,)
    assert scores[0] == pytest.approx(2 / 3)
    assert scores[1] == pytest.approx(2 / 3)
    assert scores[2] == 0.0
    assert scores[3] == 0.0


def test_ece_perfectly_calibrated_is_zero() -> None:
    # Four rows at confidence 0.75, three right: accuracy equals confidence.
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 1]
    assert ece(y_true, y_pred, [0.75] * 4) == pytest.approx(0.0)


def test_ece_confidently_wrong_is_near_one() -> None:
    assert ece([0, 0, 0], [1, 1, 1], [0.99, 0.99, 0.99]) == pytest.approx(0.99)


def test_ece_confidence_exactly_one_lands_in_last_bin() -> None:
    # Without the clamp 1.0 would index bin 10 and be dropped, giving 0 here.
    assert ece([0], [1], [1.0]) == pytest.approx(1.0)
    assert ece([0], [0], [1.0]) == pytest.approx(0.0)


def test_ece_two_bin_worked_example() -> None:
    # Bin 0 ([0, 0.5)): conf 0.2 and 0.4, one right of two -> |0.5 - 0.3| * 2/5 = 0.08
    # Bin 1 ([0.5, 1]): conf 0.6, 0.8, 1.0, all right      -> |1.0 - 0.8| * 3/5 = 0.12
    y_true = [0, 0, 0, 0, 0]
    y_pred = [0, 1, 0, 0, 0]
    conf = [0.2, 0.4, 0.6, 0.8, 1.0]
    assert ece(y_true, y_pred, conf, n_bins=2) == pytest.approx(0.20)


def test_ece_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="confidences"):
        ece([0, 1], [0, 1], [0.5])
    with pytest.raises(ValueError, match="empty"):
        ece([], [], [])


def test_worst_classes_orders_by_f1_then_name() -> None:
    names = ["delta", "alpha", "charlie", "bravo"]
    # alpha(1) and delta(0) never right; bravo(3) half right; charlie(2) perfect.
    y_true = [0, 1, 2, 3, 3]
    y_pred = [1, 0, 2, 3, 2]
    result = worst_classes(y_true, y_pred, names, n=3)
    assert [name for name, _ in result] == ["alpha", "delta", "bravo"]
    assert [score for _, score in result][:2] == [0.0, 0.0]
    assert len(worst_classes(y_true, y_pred, names, n=10)) == 4


def test_top_k_returns_best_first() -> None:
    proba = np.array([[0.1, 0.6, 0.3], [0.5, 0.25, 0.25]])
    out = top_k(proba, k=2)
    assert [j for j, _ in out[0]] == [1, 2]
    assert [j for j, _ in out[1]] == [0, 1]  # tie between 1 and 2 keeps the lower id first
    assert out[0][0][1] == pytest.approx(0.6)
    assert len(top_k(proba, k=9)[0]) == 3
