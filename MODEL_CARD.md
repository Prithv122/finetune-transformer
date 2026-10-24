---
license: apache-2.0
language: en
base_model: distilbert-base-uncased
datasets:
  - PolyAI/banking77
pipeline_tag: text-classification
tags:
  - intent-classification
  - banking77
---

# DistilBERT fine-tuned on BANKING77 (all labels, seed 1)

`distilbert-base-uncased` fully fine-tuned to classify short banking customer-service queries into the
77 BANKING77 intents. It is one run of a pre-registered learning-curve study that asks when fine-tuning
beats TF-IDF and frozen-embedding baselines:
[github.com/Prithv122/finetune-transformer](https://github.com/Prithv122/finetune-transformer).

## Training

- Data: BANKING77 train (CC-BY-4.0), minus 770 rows (10 per intent) held out for validation;
  9,233 training rows. The upstream CSVs are pinned by commit and SHA-256.
- Recipe (fixed before any test score, see `EVAL_PLAN.md` in the repo): AdamW, lr 5e-5, weight decay
  0.01, batch 16, max length 64, linear schedule with 10% warmup over 30 epochs, gradient clipping at
  1.0, mixed precision; early stopping on validation macro-F1 with patience 3, best epoch kept.
- Plain PyTorch training loop; the exact git commit and library versions are in the repo's results.

## Evaluation

Macro-F1, accuracy and calibration on the official 3,080-row test split, with the seed range and a
paired bootstrap interval against the best baseline, are in the repository README (section 5) and
`results/summary.md`. This card deliberately repeats no numbers so it cannot drift from them.

## Use

```python
from transformers import pipeline

classify = pipeline("text-classification", model="Prithv122/distilbert-banking77", top_k=3)
classify("my card still hasn't arrived")
```

## Limitations

English only, banking-domain queries only, one intent per query. Out-of-scope queries still get one
of the 77 labels: there is no "none of these" class, so a confidence threshold is needed in practice.
