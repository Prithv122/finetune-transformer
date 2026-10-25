# Interview Prep — Fine-tune or not? A learning curve on BANKING77

**Five questions, five answers.** An unanswered question means this project is not shipped.

---

### Q1. Walk me through the architecture in 90 seconds.

_A:_ The question is "how many labels per intent before fine-tuning a small transformer beats cheaper options?" on
BANKING77, 77 banking intents. The data comes from the upstream CSVs pinned by commit and SHA-256. I carve 10 rows per
intent out of train once as a shared validation set; the rest is the pool. A seeded sampler draws k rows per intent
(5, 10, 25, 50, all), nested so a seed's k = 5 rows are inside its k = 10 rows, and **every arm trains on the same rows**.
Three arms: TF-IDF + logistic regression, frozen `bge-small` embeddings + logistic regression, and a fully fine-tuned
DistilBERT in a plain PyTorch loop with early stopping on validation macro-F1. Each arm makes exactly one choice on
validation (C, or the epoch). The sweep runs 3 arms x 5 k x 3 seeds = 45 runs, scores the official test split once per run,
and writes `runs.csv` plus one predictions file per run. A report applies the rules I wrote in `EVAL_PLAN.md` before any
test score, and draws the learning curve. The GPU part ran as one Kaggle T4 job pinned to one commit, about 45 minutes.

### Q2. Why did you choose a plain PyTorch loop over the Hugging Face `Trainer`?

_A:_ Because the training loop is the thing being compared, so I wanted to see and test every part of it: AdamW with no
decay on biases and LayerNorm, a linear warmup schedule, mixed precision with the GradScaler unscaling before gradient
clipping, early stopping that only saves on a strict improvement. It's about a hundred lines, and CI runs it end to end on
a tiny random DistilBERT on CPU, including a test that a forced non-improving validation stops at the patience limit.
With `Trainer` those behaviours sit behind callbacks and arguments, and its defaults change between versions. I hit that
anyway: transformers 5 raises on a classifier-head size mismatch unless `ignore_mismatched_sizes=True`.

### Q3. What's the weakest part of this, and what would break first under load?

_A:_ Two weaknesses in the study, one in the system. Study: the logistic-regression grid was too narrow. C = 100, the top,
won 22 of 30 runs even after I widened it before scoring. I couldn't change it after the test split was scored, so I said
so in the README and ran a validation-only check at C = 1000: it helped in 9 of 30 cells by at most 0.32 points. And the
fine-tuning recipe was fixed with no learning-rate search, so the claim is about this recipe. System: the winning arm's
probabilities are the least calibrated at full data (ECE 0.110), and nothing rejects out-of-scope queries, so in production
it would confidently route "what's the weather" to a banking intent. Under load the encoder is the cost, not the linear
head, so first I'd batch, quantise and cache embeddings.

### Q4. How do you know it works? What did you measure, and against what baseline?

_A:_ Macro-F1 on BANKING77's official 3,080-query test split, 3 seeds per point, against two baselines on identical rows.
Frozen embeddings + LR won at every k: 0.826 vs 0.678 for fine-tuning at 5 labels per intent, 0.932 vs 0.918 with all
labels. At full data the paired bootstrap of fine-tuned minus embeddings is -1.1 [-2.1, -0.2], -1.8 [-2.8, -0.9] and
-1.2 [-2.1, -0.4] points for the three seeds, all excluding zero. Fine-tuning beat TF-IDF by only 2.7 points at full data,
under the 5-point bar I'd set. For trusting the pipeline itself: I checked the Kaggle output before reading any number
(45 rows, one clean commit, all seeds in every cell, training sizes equal to my local sampler, test labels identical, every
macro-F1 recomputed from its predictions file), and the report regenerates byte for byte from the committed files.

### Q5. You pre-registered the protocol, then changed it twice. Isn't that p-hacking?

_A:_ The test is whether a change could have been informed by the answer. Both amendments were dated and committed before
the test split was ever scored, and the original text stayed. One added gradient clipping while I was writing the loop.
The other widened the C grid because a smoke run, scored on rows cut from train and never on test, picked the grid's top
in six of six runs, and an under-tuned baseline biases the study *toward* fine-tuning. So the change made the comparison
harder for the arm I might have hoped would win. After the test score, when C still sat on the edge, I didn't change
anything: I reported it, explained why it can't flip either answer (a better-tuned baseline only strengthens the null), and
quantified it on validation as an exploratory check labelled outside the plan.

---

## 30-second pitch

Teams reach for fine-tuning to classify customer messages, so I measured when it actually pays off. On BANKING77's 77
banking intents I ran a pre-registered, paired learning curve: TF-IDF, frozen `bge-small` embeddings with a linear head, and
a fully fine-tuned DistilBERT, at 5 to all labels per intent, three seeds, 45 runs on a Kaggle GPU. Fine-tuning never won:
the frozen encoder beat it at every size, by 14.8 points with 5 labels and 1.4 points with all of them, with the full-data
gap's bootstrap interval excluding zero. And its training step is a linear-head fit over frozen embeddings, with no GPU
fine-tune to run, version or host.
