"""BANKING77: pinned download, loader, fixed validation carve-out and the k-shot sampler.

BANKING77 (PolyAI) is 13,083 short banking customer-service queries, each labelled with one
of 77 intents; the official split is 10,003 train / 3,080 test (40 per intent).

* **Licence:** CC-BY-4.0. **Citation:** Casanueva et al. 2020, "Efficient Intent Detection with
  Dual Sentence Encoders", arXiv:2003.04807.
* The CSVs come from the upstream repository pinned to one commit; each file is checked
  against a SHA-256 digest and a row count, so a moved branch or a truncated download fails
  loudly instead of quietly changing the data under a result.

Every arm in the sweep sees exactly the same rows: :func:`carve_validation` removes a fixed
validation set from train once, and :func:`sample_k_shot` draws k rows per intent from what is
left. For one seed the draws are **nested** (the k=5 rows are the first 5 of the k=10 rows), so
a step up the learning curve only ever adds data.
"""

from __future__ import annotations

import csv
import hashlib
import os
import random
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

LICENCE = "CC-BY-4.0"
SOURCE = "https://github.com/PolyAI-LDN/task-specific-datasets (banking_data)"
_PINNED_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
_PINNED = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{_PINNED_COMMIT}/banking_data"
)
URLS = {"train": f"{_PINNED}/train.csv", "test": f"{_PINNED}/test.csv"}
SHA256 = {
    "train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
ROWS = {"train": 10003, "test": 3080}
N_INTENTS = 77

DEFAULT_DIRECTORY = Path("data/raw/banking77")

#: Validation rows per intent, carved out of train once and shared by every arm and k.
VAL_PER_CLASS = 10
#: Seed of the validation carve-out. Fixed for the whole study, independent of the run seed.
VAL_SEED = 20260930

_FIXTURE = "fixtures/mini_intents.csv"


@dataclass(frozen=True)
class Split:
    """Parallel ``texts`` and integer ``labels``; labels index into ``label_names``."""

    texts: tuple[str, ...]
    labels: tuple[int, ...]
    label_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.texts) != len(self.labels):
            raise ValueError(f"{len(self.texts)} texts but {len(self.labels)} labels")
        bad = [y for y in self.labels if not 0 <= y < len(self.label_names)]
        if bad:
            raise ValueError(f"label id {bad[0]} outside 0..{len(self.label_names) - 1}")

    def __len__(self) -> int:
        return len(self.texts)

    def take(self, indices: Sequence[int]) -> Split:
        return Split(
            texts=tuple(self.texts[i] for i in indices),
            labels=tuple(self.labels[i] for i in indices),
            label_names=self.label_names,
        )

    def by_class(self) -> dict[int, list[int]]:
        """Row indices per label id, in file order."""
        groups: dict[int, list[int]] = {}
        for i, y in enumerate(self.labels):
            groups.setdefault(y, []).append(i)
        return groups


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(handle, origin: str) -> list[tuple[str, str]]:
    reader = csv.DictReader(handle)
    for column in ("text", "category"):
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{origin}: missing required column {column!r}")
    return [(row["text"], row["category"]) for row in reader]


def _to_split(rows: list[tuple[str, str]], names: tuple[str, ...], origin: str) -> Split:
    index = {name: i for i, name in enumerate(names)}
    unknown = sorted({c for _, c in rows if c not in index})
    if unknown:
        raise ValueError(f"{origin}: unknown category {unknown[0]!r}")
    return Split(
        texts=tuple(t for t, _ in rows),
        labels=tuple(index[c] for _, c in rows),
        label_names=names,
    )


def read_csv(path: str | Path, label_names: tuple[str, ...] | None = None) -> Split:
    """Read a ``text,category`` CSV (the csv module: many texts contain commas).

    Label names default to the sorted set of categories in the file.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        rows = _read_rows(handle, str(path))
    names = label_names or tuple(sorted({c for _, c in rows}))
    return _to_split(rows, names, str(path))


def load_banking77(directory: str | Path = DEFAULT_DIRECTORY) -> tuple[Split, Split]:
    """Load ``(train, test)``, verifying digest and row count first. Both share label ids."""
    directory = Path(directory)
    fix = "run: finetune data-download"
    for split in ("train", "test"):
        path = directory / f"{split}.csv"
        if not path.is_file():
            raise ValueError(f"{path} not found; {fix}")
        if sha256_file(path) != SHA256[split]:
            raise ValueError(f"{path}: sha256 does not match the pinned {split} file; {fix}")
    train = read_csv(directory / "train.csv")
    if len(train.label_names) != N_INTENTS:
        raise ValueError(f"train has {len(train.label_names)} intents, expected {N_INTENTS}")
    test = read_csv(directory / "test.csv", label_names=train.label_names)
    for name, split in (("train", train), ("test", test)):
        if len(split) != ROWS[name]:
            raise ValueError(f"{name}: {len(split)} rows, expected {ROWS[name]}; {fix}")
    return train, test


def download_banking77(
    directory: str | Path = DEFAULT_DIRECTORY,
    *,
    opener: Callable = urllib.request.urlopen,
    sha256: dict[str, str] = SHA256,
) -> dict[str, Path]:
    """Fetch the pinned CSVs into ``directory``; return ``{split: path}``.

    A file already present with the right digest is skipped without touching the network.
    Bytes are hashed in memory and written (temp name, then rename) only when they match.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, url in URLS.items():
        path = directory / f"{split}.csv"
        paths[split] = path
        if path.is_file() and sha256_file(path) == sha256[split]:
            continue
        with opener(url) as response:
            payload = response.read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != sha256[split]:
            raise ValueError(f"{url}: sha256 {actual} does not match the pinned {sha256[split]}")
        temporary = path.with_name(path.name + ".part")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return paths


def load_fixture() -> Split:
    """60 real BANKING77 train rows, 10 for each of 6 intents (CC-BY-4.0, attributed above).

    For tests only: no number computed from it says anything about BANKING77.
    """
    resource = resources.files("finetune").joinpath(_FIXTURE)
    with resource.open("r", encoding="utf-8", newline="") as handle:
        rows = _read_rows(handle, _FIXTURE)
    return _to_split(rows, tuple(sorted({c for _, c in rows})), _FIXTURE)


def carve_validation(
    train: Split, per_class: int = VAL_PER_CLASS, seed: int = VAL_SEED
) -> tuple[Split, Split]:
    """Split train into ``(pool, val)`` with exactly ``per_class`` val rows per intent.

    The carve-out is the same for every arm, k and run seed. Rows keep their file order.
    """
    rng = random.Random(seed)
    val_idx: set[int] = set()
    for label, rows in sorted(train.by_class().items()):
        if len(rows) <= per_class:
            raise ValueError(f"intent {train.label_names[label]!r} has only {len(rows)} rows")
        val_idx.update(rng.sample(rows, per_class))
    pool = [i for i in range(len(train)) if i not in val_idx]
    return train.take(pool), train.take(sorted(val_idx))


def sample_k_shot(pool: Split, k: int | None, seed: int) -> Split:
    """Up to ``k`` rows per intent from ``pool`` (``None`` = the whole pool).

    Each intent's rows are shuffled once with ``seed`` and the first ``k`` are kept, so for a
    fixed seed the draws are nested across k. An intent with fewer than ``k`` rows contributes
    all of them; check ``len()`` of the result rather than assuming ``k * n_intents``.
    """
    if k is None:
        return pool
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    chosen: list[int] = []
    for label, rows in sorted(pool.by_class().items()):
        shuffled = list(rows)
        random.Random(f"{seed}:{label}").shuffle(shuffled)
        chosen.extend(shuffled[:k])
    return pool.take(sorted(chosen))
