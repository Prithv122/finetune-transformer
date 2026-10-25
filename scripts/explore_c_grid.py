"""Exploratory, not pre-registered: would C = 1000 have been chosen on validation?

The sweep picked C = 100, the top of the grid, in 22 of 30 baseline runs. This refits both baselines
at C in {10, 100, 1000} on exactly the sweep's rows and scores the **validation split only**: the
test split is not read. The output says how much validation macro-F1 the grid edge left on the
table; it cannot change any pre-registered answer.

    uv run --extra embed python scripts/explore_c_grid.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from finetune import baselines, data

CS = (10.0, 100.0, 1000.0)
KS = (5, 10, 25, 50, None)
SEEDS = (1, 2, 3)
OUT = Path("results/exploratory/c_grid_val.csv")


def main() -> None:
    train, _ = data.load_banking77()  # the test split is returned and discarded unread
    pool, val = data.carve_validation(train)
    encoder = baselines.CachedEncoder(baselines.SentenceEncoder())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["arm", "k_per_class", "seed", "C", "val_macro_f1"])
        for arm in ("tfidf", "embed"):
            for k in KS:
                for seed in SEEDS:
                    subset = data.sample_k_shot(pool, k, seed)
                    for c in CS:
                        started = time.perf_counter()
                        if arm == "tfidf":
                            fitted = baselines.fit_tfidf(subset, val, c_grid=(c,))
                        else:
                            fitted = baselines.fit_embed(subset, val, encoder, c_grid=(c,))
                        label = "all" if k is None else str(k)
                        writer.writerow([arm, label, seed, c, f"{fitted.val_macro_f1:.6f}"])
                        handle.flush()
                        took = time.perf_counter() - started
                        print(f"{arm} k={label} s={seed} C={c:g}: {took:.0f}s", flush=True)


if __name__ == "__main__":
    main()
