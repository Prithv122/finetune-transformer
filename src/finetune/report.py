"""Report: the pre-registered decision rules, a paired bootstrap, and ``summary.md`` + the curve.

Reads what the sweep wrote (``results/runs.csv`` and ``results/preds/{arm}_k{k}_s{seed}.csv``)
and applies ``EVAL_PLAN.md`` exactly as written. Every number in the output is computed from those
files; the three rules are pure functions over the parsed rows, each returning its answer together
with the numbers behind it. A partial sweep is not an error: a rule that lacks an arm or a k
answers ``incomplete: missing ...`` instead of guessing.

Imports numpy and matplotlib only (no torch), so the fast test job can run it.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARMS = ("tfidf", "embed", "finetune")
BASELINES = ("tfidf", "embed")
K_GRID = ("5", "10", "25", "50", "all")
HEADER = (
    "arm,k_per_class,seed,macro_f1,accuracy,ece,n_train,train_seconds,git_sha,"
    "val_macro_f1,selection"
)
PREDS_HEADER = ("row", "label", "pred", "confidence")

#: Headline threshold from the plan: 5 macro-F1 points at k = all.
HEADLINE_POINTS = 5.0
#: Float slack so that "exactly 5.0 points" and "exactly equal means" are not decided by rounding.
_EPS = 1e-9
#: runs.csv vs macro-F1 recomputed from the predictions file; larger gaps are flagged.
_CONSISTENCY_TOL = 1e-4

#: Okabe-Ito colours: distinguishable under the common forms of colour blindness.
_COLOURS = {"tfidf": "#E69F00", "embed": "#009E73", "finetune": "#0072B2"}
_MARKERS = {"tfidf": "s", "embed": "^", "finetune": "o"}
_STYLES = {"tfidf": "--", "embed": "-.", "finetune": "-"}


@dataclass(frozen=True)
class RunRow:
    arm: str
    k: str
    seed: int
    macro_f1: float
    accuracy: float
    ece: float
    n_train: int
    train_seconds: float
    git_sha: str
    val_macro_f1: float
    selection: str


@dataclass(frozen=True)
class Stat:
    """Mean, min and max of one metric over the seeds of a cell."""

    mean: float
    min: float
    max: float


@dataclass(frozen=True)
class CellSummary:
    arm: str
    k: str
    n_seeds: int
    macro_f1: Stat
    accuracy: Stat
    ece: Stat
    n_train_min: int
    n_train_max: int


@dataclass(frozen=True)
class CrossoverRow:
    """One k of rule 1: finetune against the better baseline at that k."""

    k: str
    finetune_mean: float
    finetune_min: float
    baseline: str
    baseline_mean: float
    baseline_max: float
    mean_beats: bool
    ranges_separate: bool

    @property
    def qualifies(self) -> bool:
        return self.mean_beats and self.ranges_separate


@dataclass(frozen=True)
class CrossoverResult:
    status: str  # "found" | "not_reached" | "incomplete"
    k: str | None
    missing: tuple[str, ...]
    rows: tuple[CrossoverRow, ...]
    text: str


@dataclass(frozen=True)
class HeadlineResult:
    status: str  # "paid_off" | "did_not_pay_off" | "incomplete"
    gap_points: float | None
    finetune_mean: float | None
    tfidf_mean: float | None
    missing: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class NullRow:
    k: str
    finetune_mean: float
    embed_mean: float

    @property
    def finetune_not_better(self) -> bool:
        return self.finetune_mean <= self.embed_mean + _EPS


@dataclass(frozen=True)
class NullResult:
    status: str  # "null" | "not_null" | "incomplete"
    is_null: bool | None
    missing: tuple[str, ...]
    rows: tuple[NullRow, ...]
    text: str


@dataclass(frozen=True)
class Preds:
    row: np.ndarray
    label: np.ndarray
    pred: np.ndarray
    confidence: np.ndarray


# ----------------------------------------------------------------------------- loading


def k_sort_key(k: str) -> tuple[int, int]:
    """Numeric k ascending, ``all`` last."""
    return (1, 0) if k == "all" else (0, int(k))


def _parse_row(line_no: int, raw: dict[str, str]) -> RunRow:
    def fail(message: str) -> ValueError:
        return ValueError(f"runs.csv line {line_no}: {message}")

    if raw["arm"] not in ARMS:
        raise fail(f"unknown arm {raw['arm']!r} (expected one of {', '.join(ARMS)})")
    if raw["k_per_class"] not in K_GRID:
        raise fail(f"unknown k_per_class {raw['k_per_class']!r} (expected one of {K_GRID})")

    def number(name: str, cast, low: float | None = None, high: float | None = None):
        try:
            value = cast(raw[name])
        except ValueError:
            raise fail(f"{name} is not a valid {cast.__name__}: {raw[name]!r}") from None
        if cast is float and not math.isfinite(value):
            raise fail(f"{name} is not finite")
        if low is not None and value < low:
            raise fail(f"{name} = {value} is below {low}")
        if high is not None and value > high:
            raise fail(f"{name} = {value} is above {high}")
        return value

    return RunRow(
        arm=raw["arm"],
        k=raw["k_per_class"],
        seed=number("seed", int),
        macro_f1=number("macro_f1", float, 0.0, 1.0),
        accuracy=number("accuracy", float, 0.0, 1.0),
        ece=number("ece", float, 0.0, 1.0),
        n_train=number("n_train", int, 1),
        train_seconds=number("train_seconds", float, 0.0),
        git_sha=raw["git_sha"],
        val_macro_f1=number("val_macro_f1", float, 0.0, 1.0),
        selection=raw["selection"],
    )


def load_runs(path: str | Path) -> list[RunRow]:
    """Parse and validate ``runs.csv``: exact header, allowed arm/k, no duplicate (arm, k, seed)."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path} not found; run the sweep first")
    expected = HEADER.split(",")
    rows: list[RunRow] = []
    seen: dict[tuple[str, str, int], int] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != expected:
            raise ValueError(f"{path}: header is {header}, expected {expected}")
        for line_no, fields in enumerate(reader, start=2):
            if not fields:
                continue
            if len(fields) != len(expected):
                raise ValueError(
                    f"runs.csv line {line_no}: {len(fields)} fields, expected {len(expected)}"
                )
            row = _parse_row(line_no, dict(zip(expected, fields, strict=True)))
            key = (row.arm, row.k, row.seed)
            if key in seen:
                raise ValueError(
                    f"runs.csv line {line_no}: duplicate run {key} (first seen on line {seen[key]})"
                )
            seen[key] = line_no
            rows.append(row)
    return rows


def git_shas(rows: Sequence[RunRow]) -> dict[str, int]:
    """Each git SHA found in the rows with the number of runs that carry it."""
    return dict(Counter(r.git_sha for r in rows))


def load_preds(path: str | Path) -> Preds:
    """Read one ``row,label,pred,confidence`` predictions file."""
    path = Path(path)
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != list(PREDS_HEADER):
            raise ValueError(f"{path}: header is {header}, expected {list(PREDS_HEADER)}")
        data = [line for line in reader if line]
    if not data:
        raise ValueError(f"{path}: no prediction rows")
    try:
        return Preds(
            row=np.array([int(d[0]) for d in data]),
            label=np.array([int(d[1]) for d in data]),
            pred=np.array([int(d[2]) for d in data]),
            confidence=np.array([float(d[3]) for d in data]),
        )
    except (ValueError, IndexError) as error:
        raise ValueError(f"{path}: malformed prediction row ({error})") from None


def preds_path(results_dir: str | Path, arm: str, k: str, seed: int) -> Path:
    return Path(results_dir) / "preds" / f"{arm}_k{k}_s{seed}.csv"


# ----------------------------------------------------------------------------- summary


def _values(rows: Sequence[RunRow], arm: str, k: str, field: str = "macro_f1") -> list[float]:
    return [getattr(r, field) for r in rows if r.arm == arm and r.k == k]


def _stat(values: Sequence[float]) -> Stat:
    return Stat(mean=float(np.mean(values)), min=float(min(values)), max=float(max(values)))


def summarise(rows: Sequence[RunRow]) -> list[CellSummary]:
    """Mean, min, max over seeds for every (arm, k) present, ordered arm then k."""
    cells: list[CellSummary] = []
    for arm in ARMS:
        for k in sorted({r.k for r in rows if r.arm == arm}, key=k_sort_key):
            group = [r for r in rows if r.arm == arm and r.k == k]
            sizes = [r.n_train for r in group]
            cells.append(
                CellSummary(
                    arm=arm,
                    k=k,
                    n_seeds=len(group),
                    macro_f1=_stat([r.macro_f1 for r in group]),
                    accuracy=_stat([r.accuracy for r in group]),
                    ece=_stat([r.ece for r in group]),
                    n_train_min=min(sizes),
                    n_train_max=max(sizes),
                )
            )
    return cells


# ----------------------------------------------------------------------------- the rules


def _f3(x: float) -> str:
    return f"{x:.3f}"


def _points(x: float) -> str:
    """A macro-F1 difference as signed points with one decimal."""
    return f"{x * 100:+.1f}"


def _missing_text(missing: Sequence[str]) -> str:
    return "incomplete: missing " + ", ".join(missing)


def better_baseline(rows: Sequence[RunRow], k: str) -> str | None:
    """The baseline with the higher mean macro-F1 at ``k`` (``None`` unless both are present)."""
    means = {}
    for arm in BASELINES:
        values = _values(rows, arm, k)
        if not values:
            return None
        means[arm] = float(np.mean(values))
    return max(BASELINES, key=lambda arm: means[arm])


def crossover(rows: Sequence[RunRow]) -> CrossoverResult:
    """Rule 1: the smallest k where finetune beats the better baseline on mean and on ranges.

    "Beats on the ranges" means the worst finetune seed is above the best seed of that baseline.
    A k missing any of the three arms cannot be judged; if that could change the answer (nothing
    qualified, or a smaller k is missing) the result is ``incomplete``.
    """
    evaluated: list[CrossoverRow] = []
    missing: list[str] = []
    found: str | None = None
    missing_before_found = False
    for k in K_GRID:
        absent = [f"{arm}@{k}" for arm in ARMS if not _values(rows, arm, k)]
        if absent:
            missing.extend(absent)
            continue
        baseline = better_baseline(rows, k)
        assert baseline is not None
        ft = _values(rows, "finetune", k)
        base = _values(rows, baseline, k)
        row = CrossoverRow(
            k=k,
            finetune_mean=float(np.mean(ft)),
            finetune_min=min(ft),
            baseline=baseline,
            baseline_mean=float(np.mean(base)),
            baseline_max=max(base),
            mean_beats=float(np.mean(ft)) > float(np.mean(base)) + _EPS,
            ranges_separate=min(ft) > max(base),
        )
        evaluated.append(row)
        if found is None and row.qualifies:
            found = k
            missing_before_found = bool(missing)
    gone = tuple(missing)
    if found is not None and not missing_before_found:
        row = next(r for r in evaluated if r.k == found)
        text = (
            f"k = {found} labelled examples per intent: finetune mean {_f3(row.finetune_mean)} "
            f"(worst seed {_f3(row.finetune_min)}) vs {row.baseline} mean "
            f"{_f3(row.baseline_mean)} (best seed {_f3(row.baseline_max)})"
        )
        return CrossoverResult("found", found, gone, tuple(evaluated), text)
    if gone:
        suffix = f"; the smallest qualifying k among those present is {found}" if found else ""
        return CrossoverResult(
            "incomplete", None, gone, tuple(evaluated), _missing_text(gone) + suffix
        )
    return CrossoverResult("not_reached", None, gone, tuple(evaluated), "not reached up to k = all")


def headline(rows: Sequence[RunRow]) -> HeadlineResult:
    """Rule 2: at k = all, finetune must beat tfidf by at least 5 macro-F1 points (seed means)."""
    missing = tuple(f"{arm}@all" for arm in ("finetune", "tfidf") if not _values(rows, arm, "all"))
    if missing:
        return HeadlineResult("incomplete", None, None, None, missing, _missing_text(missing))
    ft = float(np.mean(_values(rows, "finetune", "all")))
    tf = float(np.mean(_values(rows, "tfidf", "all")))
    gap = (ft - tf) * 100
    detail = (
        f"finetune {_f3(ft)} vs tfidf {_f3(tf)} at k = all: {_points(ft - tf)} points "
        f"(threshold {HEADLINE_POINTS:.1f})"
    )
    if gap >= HEADLINE_POINTS - _EPS:
        return HeadlineResult("paid_off", gap, ft, tf, (), f"fine-tuning paid off; {detail}")
    return HeadlineResult(
        "did_not_pay_off", gap, ft, tf, (), f"fine-tuning did not pay off here; {detail}"
    )


def null_result(rows: Sequence[RunRow]) -> NullResult:
    """Rule 3: finetune mean <= embed mean at every k. One k where finetune is ahead settles it."""
    evaluated: list[NullRow] = []
    missing: list[str] = []
    for k in K_GRID:
        absent = [f"{arm}@{k}" for arm in ("finetune", "embed") if not _values(rows, arm, k)]
        if absent:
            missing.extend(absent)
            continue
        evaluated.append(
            NullRow(
                k=k,
                finetune_mean=float(np.mean(_values(rows, "finetune", k))),
                embed_mean=float(np.mean(_values(rows, "embed", k))),
            )
        )
    gone = tuple(missing)
    ahead = [r for r in evaluated if not r.finetune_not_better]
    if ahead:
        r = ahead[0]
        text = (
            f"no null result: finetune {_f3(r.finetune_mean)} beats embed "
            f"{_f3(r.embed_mean)} at k = {r.k}"
        )
        return NullResult("not_null", False, gone, tuple(evaluated), text)
    if gone:
        return NullResult("incomplete", None, gone, tuple(evaluated), _missing_text(gone))
    return NullResult(
        "null",
        True,
        gone,
        tuple(evaluated),
        "null result: finetune mean <= embed mean at every k "
        "(do not fine-tune below the largest k here)",
    )


# ----------------------------------------------------------------------------- bootstrap


def _f1_per_class(y: np.ndarray, p: np.ndarray, n_classes: int) -> np.ndarray:
    """F1 for every label in ``0..n_classes-1``; a class with no support and no predictions is 0."""
    tp = np.bincount(y[y == p], minlength=n_classes)
    denominator = np.bincount(y, minlength=n_classes) + np.bincount(p, minlength=n_classes)
    f1 = np.zeros(n_classes)
    np.divide(2 * tp, denominator, out=f1, where=denominator > 0)
    return f1


def per_class_f1(y_true, y_pred, n_classes: int) -> np.ndarray:
    """F1 per label id (zero_division = 0), the same numbers as sklearn with ``labels=range(n)``."""
    y, p = _as_labels(y_true, y_pred, n_classes)
    return _f1_per_class(y, p, n_classes)


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    """Macro-F1 over all ``n_classes`` labels, zero_division = 0."""
    return float(per_class_f1(y_true, y_pred, n_classes).mean())


def _as_labels(y_true, y_pred, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(f"label and prediction arrays differ in shape: {y.shape} vs {p.shape}")
    if len(y) == 0:
        raise ValueError("no rows to score")
    if min(y.min(), p.min()) < 0 or max(y.max(), p.max()) >= n_classes:
        raise ValueError(f"label id outside 0..{n_classes - 1}")
    return y, p


def paired_bootstrap(
    y_true,
    pred_a,
    pred_b,
    n_classes: int,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Macro-F1(a) minus macro-F1(b): point estimate and a 95% percentile interval.

    Both arms are scored on the same resampled rows each time (that is the pairing), so shared
    difficulty cancels. Returns ``(point, lo, hi)`` as fractions, not points.
    """
    y, a = _as_labels(y_true, pred_a, n_classes)
    _, b = _as_labels(y_true, pred_b, n_classes)
    point = macro_f1(y, a, n_classes) - macro_f1(y, b, n_classes)
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        diffs[i] = (
            _f1_per_class(yt, a[idx], n_classes).mean()
            - _f1_per_class(yt, b[idx], n_classes).mean()
        )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ----------------------------------------------------------------------------- the report


def _name(label: int, label_names: Sequence[str] | None) -> str:
    if label_names is not None and 0 <= label < len(label_names):
        return label_names[label]
    return str(label)


def _range_cell(stat: Stat | None) -> str:
    if stat is None:
        return "-"
    return f"{_f3(stat.mean)} [{_f3(stat.min)}, {_f3(stat.max)}]"


def _arm_by_k_table(cells: Sequence[CellSummary], metric: str) -> list[str]:
    index = {(c.arm, c.k): c for c in cells}
    lines = [
        "| arm | " + " | ".join(f"k={k}" for k in K_GRID) + " |",
        "|---|" + "---|" * len(K_GRID),
    ]
    for arm in ARMS:
        cols = []
        for k in K_GRID:
            cell = index.get((arm, k))
            cols.append(_range_cell(getattr(cell, metric) if cell else None))
        lines.append(f"| {arm} | " + " | ".join(cols) + " |")
    return lines


def _n_train_table(cells: Sequence[CellSummary]) -> list[str]:
    index = {(c.arm, c.k): c for c in cells}

    def size(arm: str, k: str) -> str:
        cell = index.get((arm, k))
        if cell is None:
            return "-"
        if cell.n_train_min == cell.n_train_max:
            return f"{cell.n_train_min:,}"
        return f"{cell.n_train_min:,}-{cell.n_train_max:,}"

    lines = [
        "| k per intent | " + " | ".join(ARMS) + " | seeds (min-max over arms) |",
        "|---|" + "---|" * (len(ARMS) + 1),
    ]
    for k in K_GRID:
        counts = [index[(a, k)].n_seeds for a in ARMS if (a, k) in index]
        seeds = "-" if not counts else f"{min(counts)}-{max(counts)}"
        lines.append(f"| {k} | " + " | ".join(size(a, k) for a in ARMS) + f" | {seeds} |")
    return lines


def _crossover_table(result: CrossoverResult) -> list[str]:
    lines = [
        "| k | finetune mean (worst seed) | better baseline | its mean (best seed) "
        "| mean gap (points) | ranges separate | qualifies |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in result.rows:
        gap = _points(r.finetune_mean - r.baseline_mean)
        lines.append(
            f"| {r.k} | {_f3(r.finetune_mean)} ({_f3(r.finetune_min)}) | {r.baseline} "
            f"| {_f3(r.baseline_mean)} ({_f3(r.baseline_max)}) | {gap} "
            f"| {'yes' if r.ranges_separate else 'no'} | {'yes' if r.qualifies else 'no'} |"
        )
    return lines


def _null_table(result: NullResult) -> list[str]:
    lines = [
        "| k | finetune mean | embed mean | gap (points) | finetune <= embed |",
        "|---|---|---|---|---|",
    ]
    for r in result.rows:
        gap = _points(r.finetune_mean - r.embed_mean)
        lines.append(
            f"| {r.k} | {_f3(r.finetune_mean)} | {_f3(r.embed_mean)} | {gap} "
            f"| {'yes' if r.finetune_not_better else 'no'} |"
        )
    return lines


def _worst_table(
    y: np.ndarray, p: np.ndarray, n_classes: int, label_names: Sequence[str] | None, n: int = 10
) -> list[str]:
    f1 = _f1_per_class(y, p, n_classes)
    support = np.bincount(y, minlength=n_classes)
    order = sorted(range(n_classes), key=lambda c: (f1[c], c))[:n]
    lines = ["| rank | intent | F1 | test rows |", "|---|---|---|---|"]
    for rank, c in enumerate(order, start=1):
        lines.append(f"| {rank} | {_name(c, label_names)} | {_f3(f1[c])} | {support[c]} |")
    return lines


def _check_aligned(a: Preds, b: Preds, what: str) -> None:
    if not (np.array_equal(a.row, b.row) and np.array_equal(a.label, b.label)):
        raise ValueError(f"{what}: predictions files do not cover the same test rows and labels")


def _full_data_section(
    rows: Sequence[RunRow],
    results_dir: Path,
    n_classes: int | None,
    label_names: Sequence[str] | None,
    warnings: list[str],
) -> list[str]:
    """Bootstrap intervals and worst-10 at k = all; skipped sections say why."""
    baseline = better_baseline(rows, "all")
    seeds = sorted({r.seed for r in rows if r.arm == "finetune" and r.k == "all"})
    if baseline is None or not seeds:
        gone = [f"{a}@all" for a in ("finetune", *BASELINES) if not _values(rows, a, "all")]
        return [
            "## Full-data intervals and worst intents (k = all)",
            "",
            f"Not computed: {_missing_text(gone)}.",
            "",
        ]
    loaded: dict[tuple[str, int], Preds] = {}
    absent: list[str] = []
    for arm in ("finetune", baseline):
        for seed in seeds:
            path = preds_path(results_dir, arm, "all", seed)
            if path.is_file():
                loaded[(arm, seed)] = load_preds(path)
            else:
                absent.append(path.name)
    if n_classes is None:
        if label_names is not None:
            n_classes = len(label_names)
        elif loaded:
            n_classes = 1 + max(int(max(p.label.max(), p.pred.max())) for p in loaded.values())

    for (arm, seed), preds in loaded.items():
        assert n_classes is not None
        reported = next(r.macro_f1 for r in rows if (r.arm, r.k, r.seed) == (arm, "all", seed))
        recomputed = macro_f1(preds.label, preds.pred, n_classes)
        if abs(reported - recomputed) > _CONSISTENCY_TOL:
            warnings.append(
                f"{arm} k=all seed {seed}: runs.csv macro-F1 {_f3(reported)} differs from "
                f"{_f3(recomputed)} recomputed from its predictions file"
            )

    out = [
        "## Full-data intervals and worst intents (k = all)",
        "",
        f"Paired bootstrap, 2,000 resamples of the test rows (seed 0), macro over "
        f"{n_classes or 'all'} labels, 95% percentile interval: macro-F1(finetune) minus "
        f"macro-F1({baseline}), the better baseline at k = all. Same run seed for both arms.",
        "",
        "| run seed | baseline | difference (points) | 95% interval (points) | excludes 0 |",
        "|---|---|---|---|---|",
    ]
    for seed in seeds:
        ft, base = loaded.get(("finetune", seed)), loaded.get((baseline, seed))
        if ft is None or base is None:
            out.append(f"| {seed} | {baseline} | not computed: predictions file missing | | |")
            continue
        _check_aligned(ft, base, f"k=all seed {seed}")
        point, lo, hi = paired_bootstrap(ft.label, ft.pred, base.pred, n_classes)
        out.append(
            f"| {seed} | {baseline} | {_points(point)} | [{_points(lo)}, {_points(hi)}] "
            f"| {'yes' if lo > 0 or hi < 0 else 'no'} |"
        )
    out.append("")
    if absent:
        warnings.append("predictions files not found: " + ", ".join(sorted(absent)))

    for arm in ("finetune", baseline):
        out.append(f"### Worst 10 intents by F1, {arm}, k = all, seed 1")
        out.append("")
        preds = loaded.get((arm, 1))
        if preds is None:
            out.append(f"Not computed: {preds_path(results_dir, arm, 'all', 1).name} missing.")
        else:
            out.extend(_worst_table(preds.label, preds.pred, n_classes, label_names))
        out.append("")
    return out


def build_summary(
    rows: Sequence[RunRow],
    results_dir: Path,
    label_names: Sequence[str] | None = None,
    n_classes: int | None = None,
) -> str:
    """The text of ``summary.md``."""
    cells = summarise(rows)
    cross, head, null = crossover(rows), headline(rows), null_result(rows)
    shas = git_shas(rows)
    warnings: list[str] = []
    if len(shas) > 1:
        listing = ", ".join(f"{sha or '(empty)'} ({n} runs)" for sha, n in sorted(shas.items()))
        warnings.append(f"runs come from {len(shas)} different git SHAs: {listing}")
    per_cell = {c.n_seeds for c in cells}
    if len(per_cell) > 1:
        warnings.append(f"cells have different seed counts ({sorted(per_cell)})")

    body = _full_data_section(rows, Path(results_dir), n_classes, label_names, warnings)

    out = [
        "# Results summary",
        "",
        f"{len(rows)} runs. Every number below is computed from `runs.csv` and `preds/`.",
        "",
        "## Source",
        "",
    ]
    if shas:
        out.append("Git SHA: " + ", ".join(f"`{sha}`" for sha in sorted(shas)))
    else:
        out.append("No runs found.")
    out.append("")
    if warnings:
        out += ["## Warnings", ""] + [f"- WARNING: {w}" for w in warnings] + [""]
    out += [
        "## Decision rules (as written in `EVAL_PLAN.md`)",
        "",
        f"1. **Crossover:** {cross.text}",
        f"2. **Headline:** {head.text}",
        f"3. **Null result:** {null.text}",
        "",
        "### Rule 1 detail: finetune vs the better baseline at each k",
        "",
        *_crossover_table(cross),
        "",
        "### Rule 3 detail: finetune vs embed at each k",
        "",
        *_null_table(null),
        "",
        "## Macro-F1, mean [min, max] over seeds",
        "",
        *_arm_by_k_table(cells, "macro_f1"),
        "",
        "## Accuracy, mean [min, max] over seeds",
        "",
        *_arm_by_k_table(cells, "accuracy"),
        "",
        "## Expected calibration error, mean [min, max] over seeds",
        "",
        *_arm_by_k_table(cells, "ece"),
        "",
        "## Training set size (rows) and seeds per k",
        "",
        *_n_train_table(cells),
        "",
        *body,
        "![Learning curve](learning_curve.png)",
        "",
    ]
    return "\n".join(out)


def plot_learning_curve(rows: Sequence[RunRow], path: str | Path) -> Path:
    """Mean line with a min-max band per arm; ``all`` is the last tick, spaced like the others."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7.5, 4.8), dpi=150)
    ax = fig.subplots()
    cells = {(c.arm, c.k): c for c in summarise(rows)}
    for arm in ARMS:
        xs = [i for i, k in enumerate(K_GRID) if (arm, k) in cells]
        if not xs:
            continue
        stats = [cells[(arm, K_GRID[i])].macro_f1 for i in xs]
        ax.fill_between(
            xs, [s.min for s in stats], [s.max for s in stats], color=_COLOURS[arm], alpha=0.2
        )
        ax.plot(
            xs,
            [s.mean for s in stats],
            color=_COLOURS[arm],
            marker=_MARKERS[arm],
            linestyle=_STYLES[arm],
            label=arm,
        )
    ax.set_xticks(range(len(K_GRID)), list(K_GRID))
    ax.set_xlabel("labelled examples per intent (all = the whole pool)")
    ax.set_ylabel("macro-F1 on the test split")
    ax.set_title("Learning curve: mean over seeds, band = min to max")
    ax.grid(alpha=0.3)
    ax.legend(title="arm")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, format="png")
    return path


def write_report(
    results_dir: str | Path,
    label_names: Sequence[str] | None = None,
    n_classes: int | None = None,
) -> Path:
    """Write ``summary.md`` and ``learning_curve.png`` into ``results_dir``; return the summary."""
    results_dir = Path(results_dir)
    rows = load_runs(results_dir / "runs.csv")
    summary = results_dir / "summary.md"
    summary.write_text(build_summary(rows, results_dir, label_names, n_classes), encoding="utf-8")
    plot_learning_curve(rows, results_dir / "learning_curve.png")
    return summary
