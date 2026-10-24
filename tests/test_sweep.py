"""The sweep loop with fake arms on the fixture: rows, prediction files, resume, one test look."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from finetune import data, sweep


def _splits(per_class_test: int = 2) -> tuple[data.Split, data.Split]:
    """Fixture rows split into a 'train' (8 per intent) and a 'test' (2 per intent)."""
    fixture = data.load_fixture()
    train_idx, test_idx = [], []
    for rows in fixture.by_class().values():
        test_idx += rows[:per_class_test]
        train_idx += rows[per_class_test:]
    return fixture.take(train_idx), fixture.take(test_idx)


class _Prior:
    """Predicts each class in proportion to its training count."""

    def __init__(self, train: data.Split) -> None:
        counts = np.bincount(train.labels, minlength=len(train.label_names)).astype(float)
        self.p = counts / counts.sum()
        self.calls = 0

    def predict_proba(self, texts):
        self.calls += 1
        return np.tile(self.p, (len(texts), 1))


def _fake_arm(log: list):
    def fit(train, val, seed, name):
        log.append((name, len(train), len(val)))
        model = _Prior(train)
        log.append(model)
        return sweep.Fitted(model, "C=1", 0.5, cleanup=lambda: log.append(f"clean {name}"))

    return fit


@pytest.fixture(autouse=True)
def small_val(monkeypatch):
    # The fixture has 8 train rows per intent; carve 2 per intent for val.
    original = data.carve_validation
    monkeypatch.setattr(data, "carve_validation", lambda train: original(train, per_class=2))


def test_rows_and_prediction_files(tmp_path):
    train, test = _splits()
    log: list = []
    rows = sweep.run_sweep(
        train, test, tmp_path, {"tfidf": _fake_arm(log)}, ks=(2, None), seeds=(1, 2), sha="abc"
    )
    assert [(r["k_per_class"], r["seed"]) for r in rows] == [
        ("2", 1),
        ("2", 2),
        ("all", 1),
        ("all", 2),
    ]
    with open(tmp_path / "runs.csv", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == sweep.RUNS_HEADER
        written = list(reader)
    assert len(written) == 4
    assert {r["git_sha"] for r in written} == {"abc"}
    assert written[0]["n_train"] == "12"  # 2 per intent x 6 intents
    assert written[2]["n_train"] == "36"  # the whole pool: 6 per intent after val
    with open(tmp_path / "preds" / "tfidf_k2_s1.csv", encoding="utf-8") as handle:
        preds = list(csv.DictReader(handle))
    assert list(preds[0]) == list(sweep.PREDS_HEADER)
    assert len(preds) == len(test)
    assert [int(p["label"]) for p in preds] == list(test.labels)


def test_val_is_shared_and_test_scored_once_per_run(tmp_path):
    train, test = _splits()
    log: list = []
    sweep.run_sweep(train, test, tmp_path, {"a": _fake_arm(log)}, ks=(2,), seeds=(1, 2, 3))
    models = [m for m in log if isinstance(m, _Prior)]
    assert [m.calls for m in models] == [1, 1, 1]
    assert {entry[2] for entry in log if isinstance(entry, tuple)} == {12}  # same val size
    assert sum(1 for entry in log if isinstance(entry, str)) == 3  # cleanup ran each time


def test_resume_skips_completed_runs(tmp_path):
    train, test = _splits()
    first: list = []
    sweep.run_sweep(train, test, tmp_path, {"a": _fake_arm(first)}, ks=(2,), seeds=(1,))
    # A row whose predictions file is missing is not trusted: it runs again.
    (tmp_path / "preds" / "a_k2_s1.csv").unlink()
    again: list = []
    rows = sweep.run_sweep(train, test, tmp_path, {"a": _fake_arm(again)}, ks=(2,), seeds=(1, 2))
    assert [r["seed"] for r in rows] == [1, 2]
    third: list = []
    rows = sweep.run_sweep(train, test, tmp_path, {"a": _fake_arm(third)}, ks=(2,), seeds=(1, 2))
    assert rows == [] and third == []


def test_wrong_proba_shape_raises(tmp_path):
    train, test = _splits()

    class Bad:
        def predict_proba(self, texts):
            return np.ones((len(texts), 2))

    def arm(train, val, seed, name):
        return sweep.Fitted(Bad(), "C=1", 0.0)

    with pytest.raises(ValueError, match="proba shape"):
        sweep.run_sweep(train, test, tmp_path, {"a": arm}, ks=(2,), seeds=(1,))
    assert not (tmp_path / "runs.csv").exists()


def test_bad_header_refuses_to_resume(tmp_path):
    (tmp_path / "runs.csv").write_text("arm,seed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        sweep.completed_runs(tmp_path)


def test_parsers():
    assert sweep.parse_ks("5,10,all") == (5, 10, None)
    with pytest.raises(ValueError):
        sweep.parse_ks("0")
    assert sweep.parse_keep("all:1, 50:2") == {("all", 1), ("50", 2)}
    assert sweep.parse_keep(None) == set()
    assert sweep.run_name("finetune", None, 3) == "finetune_kall_s3"


def test_unknown_arm_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown arm"):
        sweep.default_arms(["bert"], tmp_path)


def test_tfidf_arm_runs_end_to_end(tmp_path):
    train, test = _splits()
    arms = sweep.default_arms(["tfidf"], tmp_path)
    rows = sweep.run_sweep(train, test, tmp_path, arms, ks=(2,), seeds=(1,))
    assert rows[0]["selection"].startswith("C=")
    assert 0.0 <= float(rows[0]["macro_f1"]) <= 1.0
