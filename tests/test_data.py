"""data.py without the network: validation, the carve-out, the k-shot sampler, the downloader."""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from pathlib import Path

import pytest

from finetune import data
from finetune.data import (
    Split,
    carve_validation,
    download_banking77,
    load_banking77,
    load_fixture,
    sample_k_shot,
)

NAMES = ("a", "b", "c")


def _split(per_class: dict[int, int]) -> Split:
    """Rows t0, t1, ... with the given number of rows per label id (grouped, in file order)."""
    labels = [y for y, n in sorted(per_class.items()) for _ in range(n)]
    return Split(tuple(f"t{i}" for i in range(len(labels))), tuple(labels), NAMES)


# --- Split -------------------------------------------------------------------------------------


def test_split_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="2 texts but 1 labels"):
        Split(("x", "y"), (0,), NAMES)


@pytest.mark.parametrize("bad", [3, -1])
def test_split_rejects_label_outside_names(bad: int) -> None:
    with pytest.raises(ValueError, match=r"outside 0\.\.2"):
        Split(("x", "y"), (0, bad), NAMES)


def test_split_take_and_by_class() -> None:
    split = _split({0: 2, 1: 1, 2: 2})
    assert split.by_class() == {0: [0, 1], 1: [2], 2: [3, 4]}
    sub = split.take([4, 0])
    assert sub.texts == ("t4", "t0")
    assert sub.labels == (2, 0)
    assert sub.label_names == NAMES
    assert len(sub) == 2


# --- carve_validation --------------------------------------------------------------------------


def test_carve_validation_exact_count_disjoint_and_complete() -> None:
    train = _split({0: 20, 1: 15, 2: 12})
    pool, val = carve_validation(train, per_class=4, seed=1)
    assert Counter(val.labels) == {0: 4, 1: 4, 2: 4}
    assert set(pool.texts).isdisjoint(val.texts)
    assert sorted(pool.texts + val.texts) == sorted(train.texts)
    assert len(pool) == len(train) - 12


def test_carve_validation_is_deterministic_and_seed_dependent() -> None:
    train = _split({0: 20, 1: 15, 2: 12})
    assert carve_validation(train, 4, 1) == carve_validation(train, 4, 1)
    assert carve_validation(train, 4, 1)[1] != carve_validation(train, 4, 2)[1]


def test_carve_validation_keeps_file_order() -> None:
    train = _split({0: 10, 1: 10})
    pool, val = carve_validation(train, per_class=3, seed=0)
    index = {t: i for i, t in enumerate(train.texts)}
    for part in (pool, val):
        positions = [index[t] for t in part.texts]
        assert positions == sorted(positions)


def test_carve_validation_needs_rows_left_in_every_intent() -> None:
    with pytest.raises(ValueError, match="only 3 rows"):
        carve_validation(_split({0: 10, 1: 3}), per_class=3)


# --- sample_k_shot -----------------------------------------------------------------------------


def test_sample_k_shot_k_per_class() -> None:
    pool = _split({0: 10, 1: 12, 2: 9})
    sample = sample_k_shot(pool, 4, seed=1)
    assert Counter(sample.labels) == {0: 4, 1: 4, 2: 4}
    assert set(sample.texts) <= set(pool.texts)


def test_sample_k_shot_is_nested_across_k_for_one_seed() -> None:
    pool = _split({0: 10, 1: 12, 2: 9})
    small = set(sample_k_shot(pool, 2, seed=7).texts)
    large = set(sample_k_shot(pool, 4, seed=7).texts)
    assert small < large


def test_sample_k_shot_seeds_differ_and_repeat() -> None:
    pool = _split({0: 30, 1: 30, 2: 30})
    assert sample_k_shot(pool, 5, 1) == sample_k_shot(pool, 5, 1)
    assert sample_k_shot(pool, 5, 1) != sample_k_shot(pool, 5, 2)


def test_sample_k_shot_none_returns_the_pool() -> None:
    pool = _split({0: 5, 1: 5})
    assert sample_k_shot(pool, None, seed=1) is pool


@pytest.mark.parametrize("k", [0, -3])
def test_sample_k_shot_rejects_k_below_one(k: int) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        sample_k_shot(_split({0: 5}), k, seed=1)


def test_sample_k_shot_short_class_contributes_all_its_rows() -> None:
    pool = _split({0: 10, 1: 2, 2: 10})
    sample = sample_k_shot(pool, 5, seed=1)
    assert Counter(sample.labels) == {0: 5, 1: 2, 2: 5}
    assert len(sample) == 12


# --- download_banking77 ------------------------------------------------------------------------

PAYLOADS = {"train": b"text,category\nhello,greeting\n", "test": b"text,category\nbye,farewell\n"}
DIGESTS = {name: hashlib.sha256(body).hexdigest() for name, body in PAYLOADS.items()}


class FakeOpener:
    """Stands in for urllib.request.urlopen: returns a file-like object with read()."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.urls: list[str] = []

    def __call__(self, url: str) -> io.BytesIO:
        self.urls.append(url)
        split = "train" if url.endswith("train.csv") else "test"
        return io.BytesIO(self.payloads[split])


def test_download_writes_verified_bytes(tmp_path: Path) -> None:
    opener = FakeOpener(PAYLOADS)
    paths = download_banking77(tmp_path / "raw", opener=opener, sha256=DIGESTS)
    assert set(paths) == {"train", "test"}
    assert paths["train"].read_bytes() == PAYLOADS["train"]
    assert paths["test"].read_bytes() == PAYLOADS["test"]
    assert len(opener.urls) == 2
    assert not list((tmp_path / "raw").glob("*.part"))


def test_download_bad_digest_raises_and_writes_nothing(tmp_path: Path) -> None:
    # The train file is fetched first, so a bad train payload stops before anything is written.
    opener = FakeOpener({"train": b"tampered", "test": PAYLOADS["test"]})
    with pytest.raises(ValueError, match="does not match the pinned"):
        download_banking77(tmp_path, opener=opener, sha256=DIGESTS)
    assert list(tmp_path.iterdir()) == []


def test_download_skips_a_file_that_is_already_good(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_bytes(PAYLOADS["train"])
    (tmp_path / "test.csv").write_bytes(PAYLOADS["test"])
    opener = FakeOpener({})  # any network call would raise KeyError
    download_banking77(tmp_path, opener=opener, sha256=DIGESTS)
    assert opener.urls == []


def test_download_refetches_a_corrupt_existing_file(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_bytes(b"corrupt")
    (tmp_path / "test.csv").write_bytes(PAYLOADS["test"])
    opener = FakeOpener(PAYLOADS)
    download_banking77(tmp_path, opener=opener, sha256=DIGESTS)
    assert len(opener.urls) == 1
    assert (tmp_path / "train.csv").read_bytes() == PAYLOADS["train"]


# --- load_banking77 ----------------------------------------------------------------------------


def test_load_banking77_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found; run: finetune data-download"):
        load_banking77(tmp_path)


def test_load_banking77_wrong_digest(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_bytes(b"text,category\nx,y\n")
    (tmp_path / "test.csv").write_bytes(b"text,category\nx,y\n")
    with pytest.raises(ValueError, match="sha256 does not match"):
        load_banking77(tmp_path)


# --- fixture and the real data -----------------------------------------------------------------


def test_fixture_is_ten_rows_for_each_of_six_intents() -> None:
    fixture = load_fixture()
    assert len(fixture) == 60
    assert len(fixture.label_names) == 6
    assert set(Counter(fixture.labels).values()) == {10}


_REAL = data.DEFAULT_DIRECTORY
_REAL_PRESENT = (_REAL / "train.csv").is_file() and (_REAL / "test.csv").is_file()


@pytest.mark.skipif(not _REAL_PRESENT, reason="data/raw/banking77 not downloaded")
def test_real_banking77_counts_match_the_plan() -> None:
    train, test = load_banking77(_REAL)
    assert (len(train), len(test)) == (10003, 3080)
    pool, val = carve_validation(train)
    assert (len(pool), len(val)) == (9233, 770)
    assert len(sample_k_shot(pool, 50, seed=1)) == 3805
    assert len(sample_k_shot(pool, None, seed=1)) == 9233
