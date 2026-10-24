"""The sweep: every arm x k x seed, each scored on the test split exactly once.

One row per run goes to ``results/runs.csv`` and one file per run to ``results/preds/``. Rows are
appended and flushed as each run finishes, and a run whose row and predictions file both exist is
skipped, so a sweep that dies halfway (a Kaggle session timing out) resumes where it stopped.

Arms are plain callables, so tests swap in fakes and the real arms import torch only when a
fine-tuning or embedding run actually starts.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from . import data, metrics

ARMS = ("tfidf", "embed", "finetune")
KS: tuple[int | None, ...] = (5, 10, 25, 50, None)
SEEDS = (1, 2, 3)

RUNS_HEADER = [
    "arm",
    "k_per_class",
    "seed",
    "macro_f1",
    "accuracy",
    "ece",
    "n_train",
    "train_seconds",
    "git_sha",
    "val_macro_f1",
    "selection",
]
PREDS_HEADER = ("row", "label", "pred", "confidence")


class Predictor(Protocol):
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class Fitted:
    """What an arm hands back: something to score, and how it chose itself on val."""

    predictor: Predictor
    selection: str
    val_macro_f1: float
    cleanup: Callable[[], None] | None = None


#: ``(train, val, seed, run_name) -> Fitted``. Must not look at the test split.
Arm = Callable[[data.Split, data.Split, int, str], Fitted]


def k_label(k: int | None) -> str:
    return "all" if k is None else str(k)


def parse_ks(text: str) -> tuple[int | None, ...]:
    ks = tuple(None if part.strip() == "all" else int(part) for part in text.split(","))
    for k in ks:
        if k is not None and k < 1:
            raise ValueError(f"k must be >= 1 or 'all', got {k}")
    return ks


def run_name(arm: str, k: int | None, seed: int) -> str:
    return f"{arm}_k{k_label(k)}_s{seed}"


def git_sha(cwd: Path | None = None) -> str:
    """HEAD, with ``-dirty`` appended when tracked files differ from it; ``unknown`` off git."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def completed_runs(out_dir: Path) -> set[tuple[str, str, int]]:
    """``(arm, k_label, seed)`` for runs whose row and predictions file are both on disk."""
    runs_csv = out_dir / "runs.csv"
    if not runs_csv.is_file():
        return set()
    with open(runs_csv, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RUNS_HEADER:
            raise ValueError(f"{runs_csv}: header {reader.fieldnames} != {RUNS_HEADER}")
        done = set()
        for row in reader:
            key = (row["arm"], row["k_per_class"], int(row["seed"]))
            name = f"{key[0]}_k{key[1]}_s{key[2]}"
            if (out_dir / "preds" / f"{name}.csv").is_file():
                done.add(key)
    return done


def _drop_stale_rows(out_dir: Path, done: set[tuple[str, str, int]]) -> int:
    """Remove rows whose predictions file is missing, so their re-run does not duplicate them."""
    runs_csv = out_dir / "runs.csv"
    if not runs_csv.is_file():
        return 0
    with open(runs_csv, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    kept = [r for r in rows if (r["arm"], r["k_per_class"], int(r["seed"])) in done]
    if len(kept) == len(rows):
        return 0
    temporary = runs_csv.with_name(runs_csv.name + ".part")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNS_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept)
    temporary.replace(runs_csv)
    return len(rows) - len(kept)


def _append_row(path: Path, row: dict) -> None:
    new = not path.is_file()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNS_HEADER, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def _write_preds(path: Path, labels: Sequence[int], pred: np.ndarray, conf: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(PREDS_HEADER)
        for i, (y, p, c) in enumerate(zip(labels, pred, conf, strict=True)):
            writer.writerow((i, y, int(p), f"{float(c):.6f}"))
    temporary.replace(path)


def run_sweep(
    train_full: data.Split,
    test: data.Split,
    out_dir: Path,
    arms: dict[str, Arm],
    *,
    ks: Iterable[int | None] = KS,
    seeds: Iterable[int] = SEEDS,
    sha: str | None = None,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Run every missing (arm, k, seed); return the rows written by this call.

    The validation carve-out happens here, once, so every arm gets the same val rows.
    """
    if train_full.label_names != test.label_names:
        raise ValueError("train and test disagree on label ids")
    out_dir.mkdir(parents=True, exist_ok=True)
    pool, val = data.carve_validation(train_full)
    n_classes = len(test.label_names)
    sha = sha or git_sha()
    done = completed_runs(out_dir)
    dropped = _drop_stale_rows(out_dir, done)
    if dropped:
        log(f"dropped {dropped} row(s) without a predictions file; they run again")
    ks, seeds = tuple(ks), tuple(seeds)
    written = []
    for arm_name, arm in arms.items():
        for k in ks:
            for seed in seeds:
                name = run_name(arm_name, k, seed)
                if (arm_name, k_label(k), seed) in done:
                    log(f"skip {name}: already recorded")
                    continue
                subset = data.sample_k_shot(pool, k, seed)
                started = time.perf_counter()
                fitted = arm(subset, val, seed, name)
                seconds = time.perf_counter() - started
                # The one and only look at the test split for this run.
                proba = np.asarray(fitted.predictor.predict_proba(list(test.texts)))
                if proba.shape != (len(test), n_classes):
                    raise ValueError(f"{name}: proba shape {proba.shape}")
                pred, conf = proba.argmax(axis=1), proba.max(axis=1)
                row = {
                    "arm": arm_name,
                    "k_per_class": k_label(k),
                    "seed": seed,
                    "macro_f1": f"{metrics.macro_f1(test.labels, pred, n_classes):.6f}",
                    "accuracy": f"{metrics.accuracy(test.labels, pred):.6f}",
                    "ece": f"{metrics.ece(test.labels, pred, conf):.6f}",
                    "n_train": len(subset),
                    "train_seconds": f"{seconds:.1f}",
                    "git_sha": sha,
                    "val_macro_f1": f"{fitted.val_macro_f1:.6f}",
                    "selection": fitted.selection,
                }
                # Predictions first: a row without its file would be re-run, never trusted.
                _write_preds(out_dir / "preds" / f"{name}.csv", test.labels, pred, conf)
                _append_row(out_dir / "runs.csv", row)
                if fitted.cleanup:
                    fitted.cleanup()
                log(f"{name}: macro-F1 {row['macro_f1']} n_train {len(subset)} {seconds:.0f}s")
                written.append(row)
    return written


# --- the real arms -------------------------------------------------------------------------


def tfidf_arm() -> Arm:
    from . import baselines

    def fit(train: data.Split, val: data.Split, seed: int, name: str) -> Fitted:
        model = baselines.fit_tfidf(train, val)
        return Fitted(model, model.selection, model.val_macro_f1)

    return fit


def embed_arm(model_name: str = "BAAI/bge-small-en-v1.5") -> Arm:
    from . import baselines

    encoder = None  # built on first use, then shared so each text is embedded once

    def fit(train: data.Split, val: data.Split, seed: int, name: str) -> Fitted:
        nonlocal encoder
        if encoder is None:
            encoder = baselines.CachedEncoder(baselines.SentenceEncoder(model_name))
        model = baselines.fit_embed(train, val, encoder)
        # LinearArm scores raw texts through the same cached encoder.
        return Fitted(model, model.selection, model.val_macro_f1)

    return fit


def parse_keep(spec: str | None) -> set[tuple[str, int]]:
    """``"all:1,50:2"`` -> {("all", 1), ("50", 2)}: fine-tuned checkpoints to keep on disk."""
    if not spec:
        return set()
    keep = set()
    for part in spec.split(","):
        k, seed = part.split(":")
        keep.add((k.strip(), int(seed)))
    return keep


def finetune_arm(models_dir: Path, keep: set[tuple[str, int]] = frozenset()) -> Arm:
    from . import train as training

    def fit(train_split: data.Split, val: data.Split, seed: int, name: str) -> Fitted:
        checkpoint = models_dir / name
        result = training.train(training.TrainConfig(seed=seed), train_split, val, checkpoint)
        classifier = training.Classifier.load(checkpoint)
        k, _, s = name.removeprefix("finetune_k").rpartition("_s")

        def cleanup() -> None:
            # Checkpoints are ~260 MB each; Kaggle's working disk holds 20 GB.
            if (k, int(s)) not in keep:
                shutil.rmtree(checkpoint, ignore_errors=True)

        return Fitted(classifier, result.selection, result.best_val_macro_f1, cleanup)

    return fit


def default_arms(names: Iterable[str], models_dir: Path, keep: str | None = None) -> dict[str, Arm]:
    factories = {
        "tfidf": tfidf_arm,
        "embed": embed_arm,
        "finetune": lambda: finetune_arm(models_dir, parse_keep(keep)),
    }
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise ValueError(f"unknown arm {unknown[0]!r}; choose from {', '.join(ARMS)}")
    return {n: factories[n]() for n in names}
