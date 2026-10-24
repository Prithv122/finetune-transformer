"""The two non-fine-tuned arms: TF-IDF + logistic regression and frozen embeddings + the same LR.

Both follow ``EVAL_PLAN.md``: C is picked from :data:`C_GRID` by macro-F1 on the shared
validation set (ties go to the smaller C), and the model fitted on the k-shot train rows with that
C is the one scored. The encoder sits behind a small protocol so tests can swap in a numpy-only
fake, and ``sentence_transformers`` is imported lazily so this module loads without torch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from finetune.data import Split
from finetune.metrics import macro_f1

#: Regularisation strengths tried for both baselines (inverse strength, as in scikit-learn).
C_GRID = (0.1, 1.0, 10.0)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return float32 embeddings of shape ``(len(texts), d)``."""
        ...


class SentenceEncoder:
    """A frozen sentence-transformers model with L2-normalised output."""

    def __init__(
        self, model_name: str = EMBED_MODEL, batch_size: int = 64, device: str | None = None
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: Any = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            # Imported here so `import finetune.baselines` works where torch is not installed.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class CachedEncoder:
    """Memoise ``inner`` per text, so the sweep embeds each pool/test text once in total."""

    def __init__(self, inner: Encoder) -> None:
        self.inner = inner
        self._cache: dict[str, np.ndarray] = {}
        self._dim = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # dict.fromkeys keeps first-seen order and drops duplicates inside one call.
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for text, vector in zip(missing, self.inner.encode(missing), strict=True):
                self._cache[text] = vector
                self._dim = len(vector)
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self._cache[t] for t in texts])


@dataclass(frozen=True)
class LinearArm:
    """A fitted logistic regression plus the featuriser that feeds it."""

    model: LogisticRegression
    featurize: Callable[[Sequence[str]], Any]
    n_classes: int
    C: float
    val_macro_f1: float

    @property
    def selection(self) -> str:
        """The choice made on validation, for the results CSV (e.g. ``"C=1"``)."""
        return f"C={self.C:g}"

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Shape ``(n, n_classes)``, column j = label id j; classes unseen in training get 0."""
        return _padded_proba(self.model, self.featurize(texts), self.n_classes)


def _padded_proba(model: LogisticRegression, features: Any, n_classes: int) -> np.ndarray:
    raw = model.predict_proba(features)
    proba = np.zeros((raw.shape[0], n_classes), dtype=np.float64)
    proba[:, model.classes_] = raw
    return proba


def _select_c(
    x_train: Any,
    train: Split,
    x_val: Any,
    val: Split,
    c_grid: Sequence[float],
) -> tuple[LogisticRegression, float, float]:
    n_classes = len(train.label_names)
    best: tuple[LogisticRegression, float, float] | None = None
    for c in sorted(c_grid):
        model = LogisticRegression(C=c, max_iter=2000).fit(x_train, train.labels)
        pred = _padded_proba(model, x_val, n_classes).argmax(axis=1)
        score = macro_f1(val.labels, pred, n_classes)
        # Strictly greater: on a tie the smaller C (visited first) stays.
        if best is None or score > best[2]:
            best = (model, c, score)
    if best is None:
        raise ValueError("c_grid is empty")
    return best


def fit_tfidf(train: Split, val: Split, c_grid: Sequence[float] = C_GRID) -> LinearArm:
    """Word 1-2 gram TF-IDF + logistic regression, C chosen on ``val`` macro-F1."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    x_train = vectorizer.fit_transform(train.texts)
    model, c, score = _select_c(x_train, train, vectorizer.transform(val.texts), val, c_grid)
    return LinearArm(model, vectorizer.transform, len(train.label_names), c, score)


def fit_embed(
    train: Split, val: Split, encoder: Encoder, c_grid: Sequence[float] = C_GRID
) -> LinearArm:
    """Frozen ``encoder`` embeddings + the same logistic regression, C chosen on ``val``."""
    model, c, score = _select_c(
        encoder.encode(train.texts), train, encoder.encode(val.texts), val, c_grid
    )
    return LinearArm(model, encoder.encode, len(train.label_names), c, score)
