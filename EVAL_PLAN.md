# Evaluation plan (pre-registered)

Committed before any model is scored on the BANKING77 test split. Nothing below changes after the
first test score; a deviation is recorded as a dated amendment at the end, with its reason, and the
original text stays.

## Question

On BANKING77 (77 intents), at how many labelled examples per intent does a fine-tuned DistilBERT
overtake (a) TF-IDF + logistic regression and (b) frozen sentence embeddings + logistic regression,
measured by macro-F1 on the official test split?

## Data

- Pinned upstream CSVs (`src/finetune/data.py`: commit, SHA-256, row counts). Train 10,003, test 3,080
  (40 per intent).
- **Validation:** 10 rows per intent (770) carved from train once (`VAL_SEED = 20260930`), shared by
  every arm, k and seed. The remaining 9,233 rows are the pool. These 770 rows are extra labels that a
  true k-shot user would not have; the README says so.
- **k grid:** 5, 10, 25, 50 per intent, and `all` (the whole pool). Intents with fewer than k pool rows
  contribute all of theirs: at k = 50 four intents are short and the set is 3,805 rows, not 3,850.
- **Seeds:** 1, 2, 3. A seed fixes the k-shot draw (nested across k) and the training randomness. All
  three arms train on the same rows for a given (k, seed), so the comparison is paired.

## Arms (the recipe is fixed here; no search on test)

| Arm | Model | Fit | Selection on val |
|---|---|---|---|
| `tfidf` | word 1-2 gram TF-IDF, sublinear tf | multinomial logistic regression, `max_iter=2000` | C in {0.1, 1, 10} by val macro-F1, ties to the smaller C |
| `embed` | frozen `BAAI/bge-small-en-v1.5`, normalised embeddings | same LR | same grid |
| `finetune` | `distilbert-base-uncased`, full fine-tune, plain PyTorch loop | AdamW lr 5e-5, weight decay 0.01, batch 16, max length 64, linear schedule with 10% warmup over 30 epochs, mixed precision on GPU | early stop: val macro-F1 each epoch, patience 3; the best-on-val checkpoint is the one scored |

Each arm gets exactly one selection on the shared val set (C for the baselines, the epoch for
fine-tuning). The test split is scored once per (arm, k, seed): 45 runs.

## Metrics

- **Primary:** macro-F1 on test.
- Also: accuracy, expected calibration error (10 equal-width bins on the top-class probability), the
  10 worst intents by F1 at k = all, mean and min-max range over the 3 seeds.
- **Full-data interval:** at k = all, a paired bootstrap (2,000 resamples of the 3,080 test rows, seed 0)
  of macro-F1(`finetune`) minus macro-F1(best baseline), one interval per run seed.
- **Latency** per query on CPU, batch 1, on this machine: informative only, not part of any rule.

## Decision rules

1. **Crossover** = the smallest k where the `finetune` mean macro-F1 exceeds the better baseline's mean
   at that k **and** the seed ranges do not overlap (worst `finetune` seed > best baseline seed). If no
   k qualifies, the crossover is "not reached up to k = all".
2. **Headline:** at k = all, `finetune` must beat `tfidf` by at least 5 macro-F1 points (mean over seeds).
   Otherwise the headline is "fine-tuning did not pay off here".
3. **Null result:** `finetune` mean <= `embed` mean at every k. This is reported as a finding ("do not
   fine-tune below N labels per intent"), not hidden.

## Where the numbers come from

All 45 runs come from one sweep (`finetune sweep`) at one git SHA, on a Kaggle GPU, writing
`results/runs.csv` and one predictions file per run. The README, NOTES and report quote only numbers
regenerated from those files by `finetune report`.

## Amendments

- **2026-10-25, before any test score:** the fine-tune recipe also clips the gradient norm at 1.0
  (`TrainConfig.max_grad_norm`). The original row did not name clipping; the training loop has it
  because mixed precision unscales gradients before the step anyway, and 1.0 is the usual default.
  Decided while writing the loop, not from any result.
- **2026-10-25, before any test score:** the logistic-regression grid becomes C in {0.1, 1, 10, 100}.
  A smoke run of both baselines at real sizes (k = 5, 50, all; seed 1), scored on 15 rows per intent cut
  from train and never on the test split, picked C = 10, the top of the old grid, in all six runs.
  A choice that sits on the grid's edge means the grid is too narrow, which under-tunes the baselines and
  would bias the comparison toward fine-tuning. Selection is still by val macro-F1 only.
