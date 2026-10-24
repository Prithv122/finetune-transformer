"""Tests for the report: the three decision rules, the bootstrap and the written files.

Every number here is synthetic and written by the test; none says anything about BANKING77.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from finetune.report import (
    HEADER,
    RunRow,
    crossover,
    git_shas,
    headline,
    load_runs,
    macro_f1,
    null_result,
    paired_bootstrap,
    per_class_f1,
    summarise,
    write_report,
)

SEEDS = (1, 2, 3)


def write_runs(path: Path, table: dict[tuple[str, str], list[float]], sha: str = "abc1234") -> Path:
    """``table[(arm, k)]`` is the macro-F1 per seed; the other columns are filler."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER.split(","))
        for (arm, k), values in table.items():
            for seed, f1 in zip(SEEDS, values, strict=False):
                selection = "epoch=7" if arm == "finetune" else "C=1"
                n_train = 385 * (1 if k == "5" else 2) if k != "all" else 9233
                writer.writerow(
                    [arm, k, seed, f1, f1 + 0.01, 0.05, n_train, 1.5, sha, f1 - 0.01, selection]
                )
    return path


def load(tmp_path: Path, table, sha: str = "abc1234") -> list[RunRow]:
    return load_runs(write_runs(tmp_path / "runs.csv", table, sha))


def full_table(ft: dict[str, list[float]], tfidf=None, embed=None):
    """All five k for every arm; each arm defaults to a flat, easy-to-beat series."""
    tfidf = tfidf or {k: [0.50, 0.51, 0.52] for k in ("5", "10", "25", "50", "all")}
    embed = embed or {k: [0.60, 0.61, 0.62] for k in ("5", "10", "25", "50", "all")}
    table: dict[tuple[str, str], list[float]] = {}
    for arm, series in (("tfidf", tfidf), ("embed", embed), ("finetune", ft)):
        for k, values in series.items():
            table[(arm, k)] = values
    return table


# ------------------------------------------------------------------------ crossover


def test_crossover_found_at_a_middle_k(tmp_path):
    ft = {
        "5": [0.10, 0.11, 0.12],
        "10": [0.30, 0.31, 0.32],
        "25": [0.70, 0.71, 0.72],  # clear of embed's 0.60-0.62
        "50": [0.80, 0.81, 0.82],
        "all": [0.90, 0.91, 0.92],
    }
    result = crossover(load(tmp_path, full_table(ft)))
    assert result.status == "found"
    assert result.k == "25"
    assert result.text.startswith("k = 25")
    row = next(r for r in result.rows if r.k == "25")
    assert row.baseline == "embed"  # the better baseline at that k
    assert row.finetune_min == pytest.approx(0.70)
    assert row.baseline_max == pytest.approx(0.62)


def test_crossover_blocked_by_overlapping_seed_ranges(tmp_path):
    # At 25 the finetune mean (0.64) beats embed's (0.61) but its worst seed (0.58) is below
    # embed's best (0.62), so the ranges overlap and 25 does not qualify; 50 does.
    ft = {
        "5": [0.10, 0.11, 0.12],
        "10": [0.30, 0.31, 0.32],
        "25": [0.58, 0.65, 0.69],
        "50": [0.80, 0.81, 0.82],
        "all": [0.90, 0.91, 0.92],
    }
    result = crossover(load(tmp_path, full_table(ft)))
    row = next(r for r in result.rows if r.k == "25")
    assert row.mean_beats and not row.ranges_separate and not row.qualifies
    assert result.k == "50"


def test_crossover_uses_the_better_baseline_not_the_weaker_one(tmp_path):
    # finetune beats tfidf (0.5x) everywhere from 5, but embed (0.6x) is the better baseline.
    ft = {k: [0.55, 0.56, 0.57] for k in ("5", "10", "25", "50", "all")}
    result = crossover(load(tmp_path, full_table(ft)))
    assert result.status == "not_reached"
    assert all(r.baseline == "embed" for r in result.rows)


def test_crossover_not_reached(tmp_path):
    ft = {k: [0.20, 0.21, 0.22] for k in ("5", "10", "25", "50", "all")}
    result = crossover(load(tmp_path, full_table(ft)))
    assert result.status == "not_reached"
    assert result.k is None
    assert result.text == "not reached up to k = all"


def test_crossover_incomplete_when_an_arm_is_missing(tmp_path):
    table = full_table({k: [0.20, 0.21, 0.22] for k in ("5", "10", "25", "50", "all")})
    del table[("embed", "25")]
    result = crossover(load(tmp_path, table))
    assert result.status == "incomplete"
    assert "embed@25" in result.missing
    assert result.text.startswith("incomplete: missing")


def test_crossover_incomplete_when_a_smaller_k_is_missing(tmp_path):
    table = full_table({k: [0.90, 0.91, 0.92] for k in ("5", "10", "25", "50", "all")})
    del table[("finetune", "5")]  # 5 could have been the crossover
    result = crossover(load(tmp_path, table))
    assert result.status == "incomplete"
    assert "finetune@5" in result.missing
    assert "smallest qualifying k among those present is 10" in result.text


def test_crossover_found_even_if_only_a_later_k_is_missing(tmp_path):
    table = full_table({k: [0.90, 0.91, 0.92] for k in ("5", "10", "25", "50", "all")})
    del table[("tfidf", "all")]
    result = crossover(load(tmp_path, table))
    assert result.status == "found"
    assert result.k == "5"
    assert "tfidf@all" in result.missing


# ------------------------------------------------------------------------ headline


def _headline_rows(tmp_path, ft_all: float, tfidf_all: float):
    table = {
        ("finetune", "all"): [ft_all] * 3,
        ("tfidf", "all"): [tfidf_all] * 3,
    }
    return load(tmp_path, table)


def test_headline_paid_off(tmp_path):
    result = headline(_headline_rows(tmp_path, 0.90, 0.80))
    assert result.status == "paid_off"
    assert result.gap_points == pytest.approx(10.0)
    assert "paid off" in result.text and "+10.0 points" in result.text


def test_headline_did_not_pay_off(tmp_path):
    result = headline(_headline_rows(tmp_path, 0.83, 0.80))
    assert result.status == "did_not_pay_off"
    assert result.gap_points == pytest.approx(3.0)
    assert "did not pay off here" in result.text


def test_headline_exactly_five_points_counts(tmp_path):
    # 0.85 - 0.80 is 0.04999999999999993 in floating point; the rule says "at least 5".
    result = headline(_headline_rows(tmp_path, 0.85, 0.80))
    assert result.status == "paid_off"


def test_headline_just_under_five_points_does_not_count(tmp_path):
    result = headline(_headline_rows(tmp_path, 0.849, 0.80))
    assert result.status == "did_not_pay_off"


def test_headline_incomplete_without_tfidf(tmp_path):
    result = headline(load(tmp_path, {("finetune", "all"): [0.9, 0.9, 0.9]}))
    assert result.status == "incomplete"
    assert result.gap_points is None
    assert result.text == "incomplete: missing tfidf@all"


# ------------------------------------------------------------------------ null result


def test_null_result_true_when_finetune_never_beats_embed(tmp_path):
    ft = {k: [0.40, 0.41, 0.42] for k in ("5", "10", "25", "50", "all")}
    embed = {k: [0.50, 0.51, 0.52] for k in ("5", "10", "25", "50", "all")}
    result = null_result(load(tmp_path, full_table(ft, embed=embed)))
    assert result.is_null is True
    assert result.status == "null"


def test_null_result_equal_means_still_null(tmp_path):
    same = {k: [0.50, 0.51, 0.52] for k in ("5", "10", "25", "50", "all")}
    result = null_result(load(tmp_path, full_table(same, embed=same)))
    assert result.is_null is True


def test_null_result_false_when_finetune_wins_somewhere(tmp_path):
    ft = {
        "5": [0.40, 0.41, 0.42],
        "10": [0.40, 0.41, 0.42],
        "25": [0.40, 0.41, 0.42],
        "50": [0.40, 0.41, 0.42],
        "all": [0.90, 0.91, 0.92],
    }
    result = null_result(load(tmp_path, full_table(ft)))
    assert result.is_null is False
    assert result.status == "not_null"
    assert "k = all" in result.text


def test_null_result_incomplete_when_embed_missing(tmp_path):
    table = {("finetune", k): [0.4, 0.4, 0.4] for k in ("5", "10", "25", "50", "all")}
    table[("embed", "5")] = [0.5, 0.5, 0.5]
    result = null_result(load(tmp_path, table))
    assert result.is_null is None
    assert result.status == "incomplete"
    assert result.text.startswith("incomplete: missing embed@10")


def test_null_result_false_is_decided_despite_missing_cells(tmp_path):
    table = {("finetune", "5"): [0.9] * 3, ("embed", "5"): [0.5] * 3}
    result = null_result(load(tmp_path, table))
    assert result.is_null is False
    assert result.missing  # still reported


# ------------------------------------------------------------------------ loading


def test_duplicate_run_is_an_error(tmp_path):
    path = write_runs(tmp_path / "runs.csv", {("tfidf", "5"): [0.5, 0.5, 0.5]})
    with open(path, "a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            ["tfidf", "5", 1, 0.5, 0.5, 0.05, 385, 1.5, "abc1234", 0.5, "C=1"]
        )
    with pytest.raises(ValueError, match="duplicate run"):
        load_runs(path)


def test_wrong_header_is_an_error(tmp_path):
    path = tmp_path / "runs.csv"
    path.write_text("arm,k,seed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_runs(path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [(0, "svm", "unknown arm"), (1, "7", "unknown k_per_class"), (3, "1.5", "above")],
)
def test_bad_values_are_an_error(tmp_path, column, value, message):
    path = write_runs(tmp_path / "runs.csv", {("tfidf", "5"): [0.5]})
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[column] = value
    path.write_text("\n".join([lines[0], ",".join(fields)]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_runs(path)


def test_k_all_sorts_last_in_the_summary(tmp_path):
    table = {("tfidf", k): [0.5, 0.6, 0.7] for k in ("all", "50", "5", "25", "10")}
    cells = summarise(load(tmp_path, table))
    assert [c.k for c in cells] == ["5", "10", "25", "50", "all"]
    cell = cells[0]
    assert cell.n_seeds == 3
    assert cell.macro_f1.mean == pytest.approx(0.6)
    assert (cell.macro_f1.min, cell.macro_f1.max) == (0.5, 0.7)


def test_git_shas_counts_each_sha(tmp_path):
    rows = load(tmp_path, {("tfidf", "5"): [0.5] * 3})
    assert git_shas(rows) == {"abc1234": 3}


# ------------------------------------------------------------------------ bootstrap


def _labels_and_preds(n: int = 400, n_classes: int = 5, accuracy: float = 0.8, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    wrong = rng.random(n) > accuracy
    pred = np.where(wrong, (y + rng.integers(1, n_classes, n)) % n_classes, y)
    return y, pred


def test_macro_f1_matches_sklearn_including_absent_classes():
    y, pred = _labels_and_preds()
    y[y == 4] = 0  # class 4 has no support; it can still be predicted
    labels = list(range(6))
    expected = f1_score(y, pred, labels=labels, average="macro", zero_division=0)
    assert macro_f1(y, pred, 6) == pytest.approx(expected)
    expected_each = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    assert per_class_f1(y, pred, 6) == pytest.approx(expected_each)


def test_bootstrap_identical_predictions_give_zero():
    y, pred = _labels_and_preds()
    assert paired_bootstrap(y, pred, pred.copy(), 5, n_resamples=200) == (0.0, 0.0, 0.0)


def test_bootstrap_clearly_better_arm_is_above_zero():
    y, good = _labels_and_preds(accuracy=0.95, seed=1)
    _, poor = _labels_and_preds(accuracy=0.60, seed=2)
    point, lo, hi = paired_bootstrap(y, good, poor, 5, n_resamples=300)
    assert lo > 0 and lo <= point <= hi
    # swapping the arms mirrors the interval
    point2, _lo2, hi2 = paired_bootstrap(y, poor, good, 5, n_resamples=300)
    assert hi2 < 0 and point2 == pytest.approx(-point)


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    y, a = _labels_and_preds(seed=1)
    _, b = _labels_and_preds(accuracy=0.7, seed=2)
    first = paired_bootstrap(y, a, b, 5, n_resamples=100, seed=7)
    assert first == paired_bootstrap(y, a, b, 5, n_resamples=100, seed=7)
    assert first != paired_bootstrap(y, a, b, 5, n_resamples=100, seed=8)


def test_bootstrap_rejects_out_of_range_labels():
    y, pred = _labels_and_preds()
    with pytest.raises(ValueError, match="outside"):
        paired_bootstrap(y, pred, pred, 3, n_resamples=10)


# ------------------------------------------------------------------------ write_report


def write_preds(results: Path, arm: str, seed: int, y, pred) -> None:
    folder = results / "preds"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{arm}_kall_s{seed}.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "label", "pred", "confidence"])
        for i, (label, p) in enumerate(zip(y, pred, strict=True)):
            writer.writerow([i, int(label), int(p), 0.9])


def _results_dir(tmp_path: Path) -> tuple[Path, list[str]]:
    results = tmp_path / "results"
    ft = {
        "5": [0.10, 0.11, 0.12],
        "10": [0.30, 0.31, 0.32],
        "25": [0.70, 0.71, 0.72],
        "50": [0.80, 0.81, 0.82],
        "all": [0.90, 0.91, 0.92],
    }
    write_runs(results / "runs.csv", full_table(ft))
    names = ["alpha", "bravo", "charlie", "delta", "echo"]
    y, _ = _labels_and_preds(n=300)
    for seed in SEEDS:
        # accuracies are arbitrary: runs.csv and the preds do not agree, which gets flagged
        write_preds(
            results, "finetune", seed, y, _labels_and_preds(n=300, accuracy=0.9, seed=seed)[1]
        )
        write_preds(
            results, "embed", seed, y, _labels_and_preds(n=300, accuracy=0.6, seed=seed + 9)[1]
        )
    return results, names


def test_write_report_produces_summary_and_png(tmp_path):
    results, names = _results_dir(tmp_path)
    summary = write_report(results, label_names=names)
    assert summary == results / "summary.md"
    text = summary.read_text(encoding="utf-8")

    # rule answers, from the synthetic numbers above
    assert "1. **Crossover:** k = 25 labelled examples per intent" in text
    assert "2. **Headline:** fine-tuning paid off" in text
    assert "+40.0 points" in text  # finetune 0.910 vs tfidf 0.510 at k = all
    assert "3. **Null result:** no null result" in text
    # tables and formats
    assert "0.910 [0.900, 0.920]" in text
    assert "Git SHA: `abc1234`" in text
    assert "9,233" in text  # n_train at k = all
    assert "### Worst 10 intents by F1, finetune, k = all, seed 1" in text
    assert "### Worst 10 intents by F1, embed, k = all, seed 1" in text
    assert any(name in text for name in names)
    assert text.count("| 1 | embed |") == 1  # the bootstrap row of run seed 1
    # the synthetic runs.csv does not match its preds, so the consistency flag fires
    assert "WARNING" in text and "recomputed from its predictions file" in text

    png = (results / "learning_curve.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_write_report_flags_mixed_git_shas(tmp_path):
    results = tmp_path / "results"
    table = {("tfidf", "5"): [0.5, 0.5, 0.5]}
    path = write_runs(results / "runs.csv", table, sha="aaaaaaa")
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1].replace("aaaaaaa", "bbbbbbb")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    text = write_report(results).read_text(encoding="utf-8")
    assert "WARNING: runs come from 2 different git SHAs" in text
    assert "aaaaaaa (2 runs)" in text and "bbbbbbb (1 runs)" in text


def test_write_report_on_a_partial_sweep(tmp_path):
    results = tmp_path / "results"
    write_runs(results / "runs.csv", {("finetune", "5"): [0.3] * 3, ("embed", "5"): [0.5] * 3})
    text = write_report(results).read_text(encoding="utf-8")
    assert "1. **Crossover:** incomplete: missing" in text
    assert "2. **Headline:** incomplete: missing finetune@all, tfidf@all" in text
    assert "Not computed: incomplete: missing" in text
    assert (results / "learning_curve.png").stat().st_size > 0


def test_write_report_notes_missing_predictions(tmp_path):
    results = tmp_path / "results"
    ft = {"all": [0.9, 0.9, 0.9]}
    write_runs(
        results / "runs.csv", full_table(ft, tfidf={"all": [0.5] * 3}, embed={"all": [0.6] * 3})
    )
    text = write_report(results, n_classes=5).read_text(encoding="utf-8")
    assert "predictions file missing" in text
    assert "predictions files not found" in text
