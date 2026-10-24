"""Streamlit demo: type a banking query, see the fine-tuned DistilBERT's predicted intent.

The model is pulled from the Hugging Face Hub once at boot (``MODEL_REPO``, default below) and
runs on CPU. Locally: ``uv run --extra demo streamlit run demo/app.py``.
"""

from __future__ import annotations

import os

import streamlit as st

MODEL_REPO = os.environ.get("MODEL_REPO", "Prithv122/distilbert-banking77")
MAX_LENGTH = 64
TOP_N = 3
EXAMPLES = (
    "My card has not arrived yet, where is it?",
    "Why was I charged a fee for withdrawing cash abroad?",
    "I want to top up my account with a transfer from another bank",
)


@st.cache_resource(show_spinner="Loading the model from the Hugging Face Hub ...")
def load_model(repo: str):
    """Load tokenizer and classifier once per process; a failure is not cached and is retried."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model, top_n: int = TOP_N) -> list[tuple[str, float]]:
    """The ``top_n`` (intent, probability) pairs for one query, most likely first."""
    import torch

    encoded = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits[0], dim=-1)
    top = torch.topk(probabilities, k=min(top_n, probabilities.shape[-1]))
    id2label = model.config.id2label
    return [
        (str(id2label[int(i)]).replace("_", " "), float(p))
        for p, i in zip(top.values, top.indices, strict=True)
    ]


def _use_example(example: str) -> None:
    st.session_state["query"] = example


def main() -> None:
    st.set_page_config(page_title="BANKING77 intent classifier", page_icon=None)
    st.title("BANKING77 intent classifier")
    st.caption(f"Fine-tuned DistilBERT, running on CPU. Model: `{MODEL_REPO}`")

    try:
        tokenizer, model = load_model(MODEL_REPO)
    except Exception as error:  # a missing repo, no network or a bad checkpoint: say so plainly
        st.error(
            f"Could not load the model `{MODEL_REPO}` from the Hugging Face Hub. "
            "It may not be published yet, or the Hub may be unreachable. "
            "Set the `MODEL_REPO` environment variable to try a different model repo."
        )
        st.caption(f"Details: {type(error).__name__}: {str(error)[:300]}")
        st.stop()

    st.session_state.setdefault("query", EXAMPLES[0])
    st.write("Try an example, or type your own query:")
    for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES, strict=True):
        column.button(
            example,
            key=f"example-{example}",
            on_click=_use_example,
            args=(example,),
        )
    text = st.text_area("Customer query", key="query", height=100).strip()
    if not text:
        st.info("Enter a query to classify.")
        return

    ranked = predict(text, tokenizer, model)
    label, confidence = ranked[0]
    left, right = st.columns(2)
    left.metric("Predicted intent", label)
    right.metric("Confidence", f"{confidence:.1%}")
    st.dataframe(
        {
            "Rank": list(range(1, len(ranked) + 1)),
            "Intent": [name for name, _ in ranked],
            "Probability": [f"{p:.1%}" for _, p in ranked],
        },
        hide_index=True,
    )
    st.caption(
        "Confidence is the model's own softmax probability for the top intent; the model was "
        "fine-tuned on BANKING77 (77 banking intents), so queries outside that domain still get "
        "a banking label. Accuracy and macro-F1 on the held-out test split are in the repository "
        "README."
    )


main()
