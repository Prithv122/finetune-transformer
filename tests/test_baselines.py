"""Baseline arms on the 60-row fixture. Shapes and rules only: no number here says anything
about BANKING77."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from finetune import baselines
from finetune.baselines import C_GRID, CachedEncoder, fit_embed, fit_tfidf
from finetune.data import Split, load_fixture


class FakeEncoder:
    """Bag of hashed tokens, L2-normalised: deterministic, numpy only, records its calls."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode()).digest()
                out[row, int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


@pytest.fixture(scope="module")
def parts() -> tuple[Split, Split, Split]:
    """Fixture rows come 10 per intent: positions 0-5 train, 6-7 val, 8-9 test."""
    fixture = load_fixture()

    def pick(positions: tuple[int, ...]) -> Split:
        return fixture.take([i for i in range(len(fixture)) if i % 10 in positions])

    return pick((0, 1, 2, 3, 4, 5)), pick((6, 7)), pick((8, 9))


def test_tfidf_shapes_probabilities_and_grid(parts: tuple[Split, Split, Split]) -> None:
    train, val, test = parts
    arm = fit_tfidf(train, val)
    proba = arm.predict_proba(test.texts)
    assert proba.shape == (len(test), len(train.label_names))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert arm.C in C_GRID
    assert 0.0 <= arm.val_macro_f1 <= 1.0
    assert arm.selection == f"C={arm.C:g}"


def test_embed_shapes_probabilities_and_grid(parts: tuple[Split, Split, Split]) -> None:
    train, val, test = parts
    arm = fit_embed(train, val, FakeEncoder())
    proba = arm.predict_proba(test.texts)
    assert proba.shape == (len(test), len(train.label_names))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert arm.C in C_GRID
    # Hashed tokens are a weak encoder and val has 12 rows; the bound only needs to beat the
    # 1/6 chance level clearly, to catch a broken pipeline.
    assert arm.val_macro_f1 > 0.3


def test_single_value_grid_is_its_own_pick(parts: tuple[Split, Split, Split]) -> None:
    train, val, _ = parts
    arm = fit_tfidf(train, val, c_grid=(10.0,))
    assert arm.C == 10.0
    assert arm.selection == "C=10"


def test_tie_goes_to_the_smaller_c(parts: tuple[Split, Split, Split]) -> None:
    train, val, _ = parts
    # Scoring on the training rows themselves: every C in the grid separates them perfectly,
    # so the scores tie and the smallest C must win whatever order the grid is given in.
    scores = {c: fit_tfidf(train, train, c_grid=(c,)).val_macro_f1 for c in C_GRID}
    assert len(set(scores.values())) == 1
    for grid in (C_GRID, tuple(reversed(C_GRID))):
        chosen = fit_tfidf(train, train, c_grid=grid).C
        assert chosen == min(C_GRID)

    # An encoder that returns zeros leaves only the intercept, so every C scores the same.
    class ZeroEncoder:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            return np.zeros((len(texts), 4), dtype=np.float32)

    for grid in (C_GRID, tuple(reversed(C_GRID))):
        chosen = fit_embed(train, val, ZeroEncoder(), c_grid=grid).C
        assert chosen == min(C_GRID)


def test_missing_class_columns_are_padded_with_zero(parts: tuple[Split, Split, Split]) -> None:
    train, val, test = parts
    keep = [i for i, y in enumerate(train.labels) if y != 2]
    sparse = train.take(keep)  # label_names still lists all six intents
    for arm in (fit_tfidf(sparse, val), fit_embed(sparse, val, FakeEncoder())):
        proba = arm.predict_proba(test.texts)
        assert proba.shape == (len(test), 6)
        assert np.all(proba[:, 2] == 0.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_cached_encoder_embeds_each_text_once() -> None:
    inner = FakeEncoder()
    cached = CachedEncoder(inner)
    first = cached.encode(["a b", "c d", "a b"])
    assert first.shape == (3, inner.dim)
    np.testing.assert_array_equal(first[0], first[2])
    assert inner.calls == [["a b", "c d"]]  # the duplicate inside one call is sent once

    second = cached.encode(["c d", "e f", "a b"])
    assert inner.calls[1:] == [["e f"]]  # only the uncached text reaches the inner encoder
    np.testing.assert_array_equal(second[0], first[1])  # order preserved
    np.testing.assert_array_equal(second[2], first[0])

    cached.encode(["a b", "e f"])
    assert len(inner.calls) == 2  # fully cached: no call at all
    assert cached.encode([]).shape == (0, inner.dim)


def test_cached_encoder_gives_the_same_fit(parts: tuple[Split, Split, Split]) -> None:
    train, val, test = parts
    plain = fit_embed(train, val, FakeEncoder())
    cached = fit_embed(train, val, CachedEncoder(FakeEncoder()))
    np.testing.assert_allclose(plain.predict_proba(test.texts), cached.predict_proba(test.texts))
    assert plain.C == cached.C


def test_importing_baselines_does_not_import_torch() -> None:
    src = str(Path(baselines.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    code = "import sys, finetune.baselines; assert 'torch' not in sys.modules, 'torch imported'"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


@pytest.mark.slow
def test_sentence_encoder_on_a_tiny_model(parts: tuple[Split, Split, Split]) -> None:
    pytest.importorskip("sentence_transformers")
    encoder = baselines.SentenceEncoder(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors", batch_size=8, device="cpu"
    )
    vectors = encoder.encode(["my card has not arrived", "how do I top up"])
    assert vectors.dtype == np.float32
    assert vectors.shape[0] == 2
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    train, val, test = parts
    arm = fit_embed(train, val, CachedEncoder(encoder))
    assert arm.predict_proba(test.texts).shape == (len(test), 6)
