# Build Notes — Fine-tune or not? A learning curve on BANKING77

Working notes: what broke, what was tried, why X over Y.

---

## Log

### 2026-10-25 — pipeline built, no test score yet

- **Order of work:** data module and `EVAL_PLAN.md` first, committed before any model existed, so the
  rules could not be shaped by a result. Then baselines + metrics, the training loop, the report and
  the demo were built against fixed interfaces (`Split`, the `runs.csv` and `preds/*.csv` headers).
- **Data:** the Hub's BANKING77 entry is a loading script that current `datasets` refuses to run, so
  the upstream CSVs are pinned by commit and SHA-256 (same approach as #16). 77 intents, train
  10,003 (35 to 187 per intent), test 3,080 (40 each).
- **k = 50 is not 3,850 rows.** After the 10-per-intent validation carve-out the smallest intents have
  25, 31 and 49 pool rows; three intents are short and the k = 50 set is 3,805 rows (the plan first said four; corrected by amendment). Kept the intent's
  whole pool rather than dropping k = 50 or the short intents; `n_train` is recorded per run.
- **Smoke run at real sizes, not on test:** both baselines at k = 5, 50, all (seed 1), scored on 15
  rows per intent cut from train (`scratchpad`, never committed, never quoted). Mechanics worked;
  the C chosen on val was **10, the top of the grid, in all six runs.** A selection on the grid's
  edge means the grid is too narrow, which under-tunes the baselines and biases the study toward
  fine-tuning. Added C = 100 as a dated amendment, before any test score.
- **Gradient clipping** at 1.0 was added while writing the loop (the GradScaler unscales before the
  step anyway). Not named in the original plan, so it is the first amendment.
- **transformers 5.x:** `from_pretrained` raises on a classifier-head size mismatch unless
  `ignore_mismatched_sizes=True` (the tiny test model ships a 2-label head); it drops nothing for
  `distilbert-base-uncased`, which has no head.
- **Kaggle's Python may not be 3.12.** `pip install -e .` would refuse on the `requires-python` pin,
  so the notebook runs the code from `src/` on `PYTHONPATH` and the source is checked to compile on
  3.11 (same lesson as #26).
- **CPU contention:** with other projects' jobs running, the slow tests took 25+ minutes until torch
  threads were capped (`OMP_NUM_THREADS=4`). Not a code problem; worth knowing when timing anything.

## Decisions

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| Training loop | plain PyTorch | HF `Trainer` | Short enough to explain and test on a tiny CPU model; nothing hidden in callbacks |
| Tuning budget | one choice per arm on the same val | a grid for fine-tuning only | Equal treatment; otherwise the answer is biased toward the arm that got tuned |
| k-shot draws | nested per seed, shared by all arms | independent draws | Paired comparison; the curve only ever adds data |
| Val set | fixed 10 per intent, shared | a fraction of each k-shot subset | At k = 5 there is nothing to spare; a shared set keeps selection comparable across k |
| Demo host | Streamlit Community Cloud | HF Spaces (needs PRO), Cloud Run | Free, no billing account; DistilBERT fits in the RAM limit |

## Open questions

- [ ] Kaggle runtime for the 15 fine-tuning runs (unknown until the first run).
- [ ] Whether the embedding arm already closes most of the gap at small k (the smoke run suggests it
      is strong, but those numbers are on a train slice and do not count).

### 2026-10-25 — fresh-context review, before the GPU run

A read-only review against `EVAL_PLAN.md` found no leakage and no pairing or training-loop bug, and
five real problems, all fixed:

1. The report's rules accepted a cell with any seeds: baselines on seed 1 and fine-tuning on seed 2
   at every k gave "crossover found". Now a cell counts only with exactly seeds {1, 2, 3}, else the
   rule says "incomplete" and names the missing seeds.
2. Re-running a row whose predictions file was lost appended a duplicate, which the report then
   refused to load. The sweep now drops stale rows before re-running them.
3. "Four short intents" was three (amendment).
4. The notebook could stay on a stale local `main` after `fetch`, and could not resume in a new
   Kaggle session (`/kaggle/working` is wiped). Fixed: detached checkout of `origin/REF`, and an
   optional `RESUME_FROM` input.
5. Latency was promised but not implemented (`finetune latency` now, on validation texts); weight
   decay split now stated; `run_record.json` carries the same `-dirty` SHA as `runs.csv`.

### 2026-10-25 — the Kaggle sweep (the one look at test)

- Private kernel `prithvraj02/finetune-transformer-banking77-sweep` v1, pushed with the Kaggle CLI
  (`uvx --from kaggle kaggle` 2.2.5), GPU T4, internet on, `REF` = `dd2e52a`. Ran 03:54 to 04:40 IST
  (46 min; estimate was 1.5-2.5 h). Kaggle: Python 3.13.15, torch 2.11.0+cu128, transformers 5.16.1,
  sentence-transformers 5.7.0 (the code ran from `src/`, so the 3.12 pin did not matter).
- **Output validated before any number was read:** 45 rows, one SHA (`dd2e52a`, not dirty), exactly
  seeds {1, 2, 3} in all 15 cells, no duplicates, `n_train` equal to the local sampler for every run,
  45 prediction files of 3,080 rows whose labels equal the local test labels and whose macro-F1
  recomputes to `runs.csv`. One checkpoint kept (`finetune_kall_s1`: best epoch 5 of 8).
- **C hit the grid's top again: C = 100 in 22 of 30 baseline runs** (all 15 TF-IDF runs but two; all
  embedding cells up to k = 25; embeddings at k = all chose C = 10). The grid cannot change now that
  test is scored. Reported as a limitation: TF-IDF may be under-tuned, which favours fine-tuning. A
  sensitivity check with C = 1000 on validation only is exploratory and labelled as such.

### 2026-10-25 — reading the results, write-up

- **Answer: a null result.** Frozen bge-small + LR beat fine-tuned DistilBERT at every k (gap -14.8 points at k = 5,
  -1.4 at all); crossover not reached; headline "did not pay off" (+2.7 over TF-IDF, bar 5). Full-data paired bootstrap
  excludes zero for all three seeds. `finetune report` regenerates `summary.md` byte for byte locally.
- **Grid edge, quantified (exploratory, val only, `scripts/explore_c_grid.py`):** C = 1000 beat the better of 10 and 100
  in 9 of 30 cells, by at most 0.32 points; worse elsewhere (TF-IDF median -0.96). The edge cost the baselines well under a
  point, and a better-tuned baseline only strengthens the null.
- **Latency deviation:** `finetune latency` ran with the CPU at 100% from other projects' jobs (embedding arm 1.1 s p50,
  13.9 s p95, obviously inflated). Not committed and not quoted; README says latency is not reported yet. Re-run on an idle
  machine before quoting anything.
- Fact-check caught one wrong claim before commit: the embedding arm's 48 s includes GPU encoding on Kaggle, so "under a
  minute on CPU" was wrong; rewritten as "no GPU training step".

### 2026-11-01 — local CI replay of `941af97`, then pushed on the owner's yes

Hosted Actions was expected to be unavailable until about 2026-11-02 (account limit), so the local replay is the CI
record for this push. Fresh clones of `941af9733c020628918a184299f854286ad00293` (86 tracked files), checked out
detached, fresh installs (`UV_NO_CACHE=1`), `SECRETS_NO_KEYRING=1`, `OMP_NUM_THREADS=4`, run 00:54-01:04 IST.

| Workflow step | Command | Result |
|---|---|---|
| `test`: install | `uv sync --all-groups` | ok |
| `test`: assert torch absent | `uv run python -c "import torch"` must fail | torch absent |
| `test`: lint | `uv run ruff check .` ; `uv run ruff format --check .` | clean; 29 files formatted |
| `test`: tests | `uv run pytest -m "not slow" --cov=src --cov-report=term-missing` | **93 passed, 2 skipped**, coverage 73% |
| `slow`: install | `uv sync --all-groups --extra train --extra embed` | ok |
| `slow`: tests | `uv run pytest -m slow` | **10 passed** |

Differences from the hosted runner: Windows 11 instead of ubuntu-latest; `--basetemp` in a scratch folder and
`-p no:cacheprovider` (the sandbox blocks `%TEMP%`); torch threads capped at 4 because other jobs shared the CPU.
The 2 skips are expected without the data download (the real-data test) and without torch (the training module).

Hygiene at the same SHA: no weights, caches, `.env` or raw data tracked (largest file `uv.lock`, 160 KB; prediction
files carry ids and confidences only, no query text); no secret patterns in `git log --all -p`; one author and
committer; no assistant-named file in history.

Pushed on the owner's yes: `dd2e52a..941af97` to `origin/main`. Hosted run 41234893110 started on `941af97`.
