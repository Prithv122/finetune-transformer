# Fine-tune or not? A learning curve on BANKING77 — G2

**Tier:** 2 · **Category:** AI engineering (NLP) · **Wave:** 4

The project's own engineering guide. Keep it short and current.

## What this is

A paired learning curve: at 5, 10, 25, 50 and all labelled examples per intent, does a fine-tuned
DistilBERT beat TF-IDF + LR and frozen `bge-small` embeddings + LR on BANKING77? The protocol is
fixed in `EVAL_PLAN.md` before any test score exists.

## Stack

Python 3.12, uv. Core: numpy, scikit-learn, matplotlib. Extras: `embed` (sentence-transformers),
`train` (torch, transformers), `demo` (Streamlit). torch comes from the CPU wheel index locally and in
CI; the real sweep runs on a Kaggle GPU through `notebooks/train_kaggle.ipynb`.

## Acceptance criteria

- [ ] `EVAL_PLAN.md` committed before the first test score
- [ ] 45 runs (3 arms x 5 k x 3 seeds) from one sweep at one SHA in `results/runs.csv`
- [ ] Crossover, headline and null rules answered from `finetune report` output only
- [ ] Fine-tuned weights on the Hugging Face Hub with a model card; demo loads them
- [ ] Runs from a clean clone using only the documented steps
- [ ] Tests, lint and CI green; README results come from real runs

## Conventions

- `uv` for dependencies; `ruff check` + `ruff format`; `pytest`.
- The fast test job never installs torch; anything needing it is marked `slow`.
- No number about BANKING77 is computed from the 60-row fixture.
- No hyperparameter is chosen by looking at the test split.

## Project-specific notes

- Data: `uv run finetune data-download` fetches the pinned CSVs into `data/raw/banking77/` (ignored).
- Local pytest needs `--basetemp` pointed somewhere writable on this machine; not in `pyproject.toml`.
- transformers is 5.x: check API names against the installed version, not memory.
