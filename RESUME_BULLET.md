# Resume Bullets — Fine-tune or not? A learning curve on BANKING77

Form: **action → technical specifics → measured outcome.** Numbers or it doesn't go on the resume.

---

## Bullets

- Ran a pre-registered, paired learning-curve study on BANKING77 (77 intents, 45 runs on a Kaggle T4) comparing a fully
  fine-tuned DistilBERT (plain PyTorch loop, AMP, early stopping) with TF-IDF and frozen bge-small embeddings + logistic
  regression; the frozen encoder won at every training size, 0.826 vs 0.678 macro-F1 at 5 labels per intent and 0.932 vs
  0.918 at full data (paired bootstrap interval excluding zero for all 3 seeds).
- Built the evaluation harness so the comparison could not drift: SHA-pinned data, one shared validation set, nested k-shot
  sampling identical across arms, the test split scored once per run, decision rules committed before any score, a
  resumable sweep validated on arrival (45 rows, one commit, all seeds, predictions recomputed), and CI that trains a tiny
  model on CPU.

Shorter variant (one line):

- Measured when fine-tuning pays off for intent classification: on BANKING77, frozen embeddings + logistic regression beat
  fine-tuned DistilBERT at every label budget (0.932 vs 0.918 macro-F1 at full data) in a pre-registered 45-run study.

## Which roles this supports

- [x] Data Scientist / ML
- [x] AI Engineer (LLM/NLP/CV)
- [ ] Data Engineer
- [ ] Data Analyst / Python Developer

## Keywords this project earns

PyTorch, Hugging Face transformers, DistilBERT, fine-tuning, sentence-transformers, embeddings, scikit-learn, logistic
regression, TF-IDF, macro-F1, expected calibration error, paired bootstrap, pre-registration, learning curves, early
stopping, mixed precision, Kaggle GPU, reproducible evaluation, GitHub Actions.
