"""Latency timing with a fake clock, and reading the chosen C back from runs.csv."""

from __future__ import annotations

import itertools

import numpy as np

from finetune import latency, sweep


def test_time_per_query_uses_one_call_per_text_after_warmup():
    calls = []
    ticks = itertools.count(step=0.002)  # every clock read advances 2 ms

    def predict(texts):
        calls.append(list(texts))
        return np.ones((len(texts), 3)) / 3

    stats = latency.time_per_query(
        predict, ["a", "b", "c", "d"], warmup=2, clock=lambda: next(ticks)
    )
    assert calls == [["a"], ["b"], ["a"], ["b"], ["c"], ["d"]]
    assert stats["n"] == 4
    assert np.isclose(stats["p50_ms"], 2.0) and np.isclose(stats["p95_ms"], 2.0)


def test_chosen_c_reads_k_all_seed_1(tmp_path):
    assert latency.chosen_c(tmp_path, "tfidf") is None
    row = dict.fromkeys(sweep.RUNS_HEADER, "0")
    with open(tmp_path / "runs.csv", "w", encoding="utf-8") as handle:
        handle.write(",".join(sweep.RUNS_HEADER) + "\n")
        for arm, k, seed, sel in (("tfidf", "all", "2", "C=1"), ("tfidf", "all", "1", "C=100")):
            row.update(arm=arm, k_per_class=k, seed=seed, selection=sel)
            handle.write(",".join(row[c] for c in sweep.RUNS_HEADER) + "\n")
    assert latency.chosen_c(tmp_path, "tfidf") == 100.0
    assert latency.chosen_c(tmp_path, "embed") is None
