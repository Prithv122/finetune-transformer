"""Training loop and classifier, on a tiny random DistilBERT and the 60-row fixture.

Nothing here says anything about BANKING77: the model is random and the data is a fixture. The
tests check the loop's mechanics (it runs, saves, reloads, stops early, repeats under a seed).
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")

from finetune import train as train_module
from finetune.data import Split, load_fixture
from finetune.train import Classifier, TrainConfig, TrainResult, train

pytestmark = pytest.mark.slow

TINY = "hf-internal-testing/tiny-random-distilbert"
CONFIG = TrainConfig(model_name=TINY, max_epochs=2, patience=1, batch_size=8, eval_batch_size=16)


def _split_fixture() -> tuple[Split, Split]:
    """7 rows per intent to train on, the other 3 to validate on (file order, no randomness)."""
    fixture = load_fixture()
    train_idx: list[int] = []
    val_idx: list[int] = []
    for _, rows in sorted(fixture.by_class().items()):
        train_idx.extend(rows[:7])
        val_idx.extend(rows[7:])
    return fixture.take(sorted(train_idx)), fixture.take(sorted(val_idx))


@pytest.fixture(scope="module")
def data() -> tuple[Split, Split]:
    return _split_fixture()


@pytest.fixture(scope="module")
def run(data, tmp_path_factory) -> tuple[TrainResult, Split]:
    out = tmp_path_factory.mktemp("run")
    result = train(CONFIG, data[0], data[1], out, device="cpu", log=lambda _: None)
    return result, data[1]


def test_run_finishes_with_finite_loss(run) -> None:
    result, _ = run
    assert result.epochs_run == len(result.history) <= CONFIG.max_epochs
    assert 1 <= result.best_epoch <= result.epochs_run
    assert result.selection == f"epoch={result.best_epoch}"
    assert 0.0 <= result.best_val_macro_f1 <= 1.0
    assert result.train_seconds > 0
    assert all(np.isfinite(h["train_loss"]) for h in result.history)


def test_checkpoint_reloads_and_predicts(run, data) -> None:
    result, val = run
    clf = Classifier.load(result.checkpoint_dir, device="cpu")
    assert clf.label_names == val.label_names

    proba = clf.predict_proba(val.texts, batch_size=7)
    assert proba.shape == (len(val), len(val.label_names))
    assert proba.dtype == np.float32
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    out = clf.predict("How do I top up my card?")
    assert out["label"] in val.label_names
    assert out["label"] == out["top3"][0]["label"]
    assert out["confidence"] == out["top3"][0]["p"]
    probs = [t["p"] for t in out["top3"]]
    assert len(probs) == 3
    assert probs == sorted(probs, reverse=True)
    assert {t["label"] for t in out["top3"]} <= set(val.label_names)


def test_batching_does_not_change_probabilities(run) -> None:
    result, val = run
    clf = Classifier.load(result.checkpoint_dir, device="cpu")
    small = clf.predict_proba(val.texts, batch_size=1)
    large = clf.predict_proba(val.texts, batch_size=64)
    np.testing.assert_allclose(small, large, atol=1e-5)


def test_run_record_has_expected_keys(run, data) -> None:
    result, _ = run
    record = json.loads((result.checkpoint_dir / "run_record.json").read_text(encoding="utf-8"))
    for key in (
        "config",
        "best_epoch",
        "best_val_macro_f1",
        "epochs_run",
        "train_seconds",
        "n_train",
        "history",
        "versions",
        "git_sha",
        "train_data_sha256",
    ):
        assert key in record
    assert record["config"] == CONFIG.to_dict()
    assert record["n_train"] == len(data[0])
    assert len(record["train_data_sha256"]) == 64
    assert {"python", "torch", "transformers", "cuda_device"} <= set(record["versions"])
    assert set(record["history"][0]) == {"epoch", "train_loss", "val_macro_f1"}


def test_same_seed_gives_identical_history(data, tmp_path) -> None:
    cfg = replace(CONFIG, seed=7)
    a = train(cfg, data[0], data[1], tmp_path / "a", device="cpu", log=lambda _: None)
    b = train(cfg, data[0], data[1], tmp_path / "b", device="cpu", log=lambda _: None)
    assert a.history == b.history


def test_early_stopping_stops_at_patience(data, tmp_path, monkeypatch) -> None:
    # val macro-F1 is forced to fall after epoch 1, so epoch 1 stays the best and training
    # stops once `patience` epochs in a row fail to beat it.
    scores = iter([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    monkeypatch.setattr(train_module, "_val_macro_f1", lambda *a, **k: next(scores))
    cfg = replace(CONFIG, max_epochs=6, patience=2)
    result = train(cfg, data[0], data[1], tmp_path, device="cpu", log=lambda _: None)
    assert result.epochs_run == 3
    assert result.best_epoch == 1
    assert result.best_val_macro_f1 == 0.5
    assert [h["val_macro_f1"] for h in result.history] == [0.5, 0.4, 0.3]


def test_ties_do_not_replace_the_best_epoch(data, tmp_path, monkeypatch) -> None:
    scores = iter([0.5, 0.5, 0.6, 0.6])
    monkeypatch.setattr(train_module, "_val_macro_f1", lambda *a, **k: next(scores))
    cfg = replace(CONFIG, max_epochs=4, patience=3)
    result = train(cfg, data[0], data[1], tmp_path, device="cpu", log=lambda _: None)
    assert result.best_epoch == 3


def test_mismatched_label_names_are_rejected(data, tmp_path) -> None:
    train_split, val = data
    other = Split(val.texts, val.labels, tuple(reversed(val.label_names)))
    with pytest.raises(ValueError, match="label_names"):
        train(CONFIG, train_split, other, tmp_path, device="cpu", log=lambda _: None)


def test_no_decay_group_covers_bias_and_layernorm() -> None:
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        TINY, num_labels=3, ignore_mismatched_sizes=True
    )
    skip = train_module._no_decay_names(model)
    names = dict(model.named_parameters())
    assert skip <= set(names)
    assert all(n.endswith("bias") or "LayerNorm" in n or "layer_norm" in n for n in skip)
    assert any(n.endswith("bias") for n in skip)
    assert any("LayerNorm" in n or "layer_norm" in n for n in skip)
    decayed = set(names) - skip
    assert decayed
    assert all(not n.endswith("bias") for n in decayed)
