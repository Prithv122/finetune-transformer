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
