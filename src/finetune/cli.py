"""``finetune`` command line: data-download, data-check, sweep, report, predict."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import data


def _data_download(args: argparse.Namespace) -> int:
    for split, path in data.download_banking77(args.dest).items():
        print(f"{split}: {path} ({data.sha256_file(path)[:12]})")
    return 0


def _data_check(args: argparse.Namespace) -> int:
    train, test = data.load_banking77(args.data_dir)
    pool, val = data.carve_validation(train)
    print(f"train {len(train)}  test {len(test)}  intents {len(train.label_names)}")
    print(f"pool {len(pool)}  val {len(val)} ({data.VAL_PER_CLASS} per intent)")
    for k in (5, 10, 25, 50):
        subset = data.sample_k_shot(pool, k, seed=1)
        short = sum(1 for rows in subset.by_class().values() if len(rows) < k)
        print(f"k={k:<3} rows {len(subset):>5}  intents with fewer than k: {short}")
    return 0


def _sweep(args: argparse.Namespace) -> int:
    from . import sweep

    train, test = data.load_banking77(args.data_dir)
    arms = sweep.default_arms(args.arms.split(","), Path(args.models), args.keep_checkpoint)
    sweep.run_sweep(
        train,
        test,
        Path(args.out),
        arms,
        ks=sweep.parse_ks(args.ks),
        seeds=tuple(int(s) for s in args.seeds.split(",")),
    )
    return 0


def _report(args: argparse.Namespace) -> int:
    from . import report

    label_names = None
    if args.data_dir and Path(args.data_dir).is_dir():
        label_names = data.load_banking77(args.data_dir)[1].label_names
    path = report.write_report(Path(args.results), label_names=label_names)
    print(f"wrote {path}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    from .train import Classifier

    classifier = Classifier.load(args.model)
    for text in args.text:
        print(json.dumps({"text": text, **classifier.predict(text)}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="finetune", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    default_dir = str(data.DEFAULT_DIRECTORY)

    p = sub.add_parser("data-download", help="fetch the pinned BANKING77 CSVs")
    p.add_argument("--dest", default=default_dir)
    p.set_defaults(func=_data_download)

    p = sub.add_parser("data-check", help="verify the data and print split and k-shot sizes")
    p.add_argument("--data-dir", default=default_dir)
    p.set_defaults(func=_data_check)

    p = sub.add_parser("sweep", help="run arms x k x seeds; resumable")
    p.add_argument("--data-dir", default=default_dir)
    p.add_argument("--out", default="results")
    p.add_argument("--models", default="models", help="where fine-tuned checkpoints go")
    p.add_argument("--arms", default="tfidf,embed,finetune")
    p.add_argument("--ks", default="5,10,25,50,all")
    p.add_argument("--seeds", default="1,2,3")
    p.add_argument("--keep-checkpoint", help='checkpoints to keep, e.g. "all:1"')
    p.set_defaults(func=_sweep)

    p = sub.add_parser("report", help="summary.md and learning_curve.png from runs.csv")
    p.add_argument("--results", default="results")
    p.add_argument("--data-dir", default=default_dir, help="for intent names, if present")
    p.set_defaults(func=_report)

    p = sub.add_parser("predict", help="classify texts with a local or Hub checkpoint")
    p.add_argument("--model", required=True, help="checkpoint dir or Hub repo id")
    p.add_argument("text", nargs="+")
    p.set_defaults(func=_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
