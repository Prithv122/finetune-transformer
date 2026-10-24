"""CPU latency per query, batch 1, for each arm: informative only, not part of any decision rule.

Queries are validation texts, never test texts. The baselines are refitted on the whole pool with
one fixed C (the one the sweep chose at k = all, seed 1, when ``runs.csv`` has it) because only
their prediction speed is measured here; the fine-tuned arm loads a saved checkpoint.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from . import data

DEFAULT_C = 10.0


def time_per_query(
    predict_proba: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    *,
    warmup: int = 10,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, float]:
    """Median and p95 milliseconds for one ``predict_proba([text])`` call."""
    for text in texts[:warmup]:
        predict_proba([text])
    times = []
    for text in texts:
        started = clock()
        predict_proba([text])
        times.append((clock() - started) * 1000)
    return {
        "n": len(times),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def chosen_c(results_dir: Path, arm: str) -> float | None:
    """The C the sweep selected for ``arm`` at k = all, seed 1, if it has been recorded."""
    path = results_dir / "runs.csv"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["arm"], row["k_per_class"], row["seed"]) == (arm, "all", "1"):
                return float(row["selection"].removeprefix("C="))
    return None


def measure(
    data_dir: Path,
    results_dir: Path,
    *,
    arms: Sequence[str] = ("tfidf", "embed", "finetune"),
    model_dir: Path | None = None,
    n: int = 200,
    log: Callable[[str], None] = print,
) -> Path:
    """Time each arm on ``n`` validation queries and write ``results/latency.json``."""
    from . import baselines

    train, _ = data.load_banking77(data_dir)
    pool, val = data.carve_validation(train)
    queries = list(val.texts[:n])
    out: dict[str, object] = {
        "queries": f"first {len(queries)} validation texts, batch 1",
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "arms": {},
    }
    for arm in arms:
        if arm == "finetune":
            if model_dir is None or not Path(model_dir).is_dir():
                log("finetune: no checkpoint given, skipped")
                continue
            from .train import Classifier

            predictor = Classifier.load(model_dir, device="cpu")
            detail = {"checkpoint": str(model_dir)}
        else:
            c = chosen_c(results_dir, arm) or DEFAULT_C
            if arm == "tfidf":
                predictor = baselines.fit_tfidf(pool, val, c_grid=(c,))
            else:
                encoder = baselines.CachedEncoder(baselines.SentenceEncoder())
                fitted = baselines.fit_embed(pool, val, encoder, c_grid=(c,))
                # Time the encoder itself, not the cache that already holds every val text.
                predictor = baselines.LinearArm(
                    fitted.model, encoder.inner.encode, fitted.n_classes, c, fitted.val_macro_f1
                )
            detail = {"C": c, "fitted_on": f"whole pool, {len(pool)} rows"}
        stats = time_per_query(predictor.predict_proba, queries)
        out["arms"][arm] = {**stats, **detail}
        log(f"{arm}: p50 {stats['p50_ms']:.2f} ms, p95 {stats['p95_ms']:.2f} ms")
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "latency.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return path
