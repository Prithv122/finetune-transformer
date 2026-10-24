# Demo

A Streamlit app that loads the fine-tuned DistilBERT from the Hugging Face Hub and classifies a
banking query into one of the 77 BANKING77 intents, showing the top-3 intents with probabilities.

## Run locally

```bash
uv run --extra demo streamlit run demo/app.py
```

The model repo defaults to `Prithv122/distilbert-banking77`. To use another one:

```bash
MODEL_REPO=<user>/<repo> uv run --extra demo streamlit run demo/app.py
```

The first start downloads the weights (about 270 MB) and caches them. If the repo does not exist
or the Hub is unreachable, the page shows an error message instead of crashing.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub (the weights live on the Hub, not in the repo).
2. On <https://share.streamlit.io>, choose **New app**, pick the repository and branch, and set the
   main file path to `demo/app.py`.
3. Streamlit installs from the root `requirements.txt`, which pulls the CPU-only torch wheel so
   the build stays small.
4. Under **Advanced settings**, add a secret or environment variable `MODEL_REPO` only if the model
   is not at the default repo id. A private repo also needs `HF_TOKEN`.
