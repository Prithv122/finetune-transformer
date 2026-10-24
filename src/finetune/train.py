"""Fine-tune DistilBERT with a plain PyTorch loop, and the classifier that loads the result.

The recipe is the ``finetune`` row of ``EVAL_PLAN.md`` and is the :class:`TrainConfig` default:
AdamW (lr 5e-5, weight decay 0.01), batch 16, max length 64, a linear schedule with 10% warmup
over ``max_epochs`` epochs, mixed precision on a GPU. After every epoch the model is scored on
the shared validation split by macro-F1; the best epoch (strictly greater than all before it)
is saved, and training stops after ``patience`` epochs without improvement. That one choice, the
epoch, is the only selection the fine-tuned arm gets, and it never sees the test split.

``import finetune.train`` does not import torch or transformers: they are imported inside the
functions that need them, so the fast CI job (which has no torch) can import this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from finetune.data import Split

if TYPE_CHECKING:
    import torch

DEFAULT_MAX_LENGTH = 64


@dataclass(frozen=True)
class TrainConfig:
    """The fine-tune recipe pinned in ``EVAL_PLAN.md``. Changing a default changes the study."""

    model_name: str = "distilbert-base-uncased"
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 30
    patience: int = 3
    seed: int = 1
    #: Only takes effect on CUDA (fp16 autocast + loss scaling); CPU runs are fp32.
    mixed_precision: bool = True
    eval_batch_size: int = 128
    #: Gradient-norm clip. Not named in the plan; 1.0 is the usual default for BERT fine-tuning.
    max_grad_norm: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    best_epoch: int
    best_val_macro_f1: float
    epochs_run: int
    train_seconds: float
    checkpoint_dir: Path
    history: list[dict[str, float]] = field(default_factory=list)

    @property
    def selection(self) -> str:
        """The one thing chosen on validation, as a short label for the results table."""
        return f"epoch={self.best_epoch}"


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pick_device(device: str | torch.device | None) -> torch.device:
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _no_decay_names(model: torch.nn.Module) -> set[str]:
    """Names of parameters that skip weight decay: every bias and every LayerNorm parameter."""
    import torch

    names: set[str] = set()
    for module_name, module in model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full = f"{module_name}.{param_name}" if module_name else param_name
            if param_name == "bias" or isinstance(module, torch.nn.LayerNorm):
                names.add(full)
    return names


def _make_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with the standard split: decay on weight matrices, none on biases and LayerNorm."""
    import torch

    skip = _no_decay_names(model)
    decay = [p for n, p in model.named_parameters() if n not in skip]
    plain = [p for n, p in model.named_parameters() if n in skip]
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.lr)


def _tokenize(tokenizer: Any, texts: Sequence[str], max_length: int, device: torch.device) -> dict:
    batch = tokenizer(
        list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return {key: value.to(device) for key, value in batch.items()}


def _predict_proba(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Softmax probabilities, shape ``(len(texts), n_labels)``, float32; leaves train mode alone."""
    import torch

    was_training = model.training
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = _tokenize(tokenizer, texts[start : start + batch_size], max_length, device)
            logits = model(**batch).logits.float()
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    model.train(was_training)
    if not chunks:
        return np.zeros((0, model.config.num_labels), dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _val_macro_f1(
    model: torch.nn.Module,
    tokenizer: Any,
    val: Split,
    *,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> float:
    from sklearn.metrics import f1_score

    proba = _predict_proba(
        model, tokenizer, val.texts, device=device, max_length=max_length, batch_size=batch_size
    )
    n = len(val.label_names)
    return float(
        f1_score(
            val.labels, proba.argmax(axis=1), average="macro", labels=range(n), zero_division=0
        )
    )


def _data_digest(split: Split) -> str:
    """SHA-256 of the training texts and labels, so a run record names the exact rows used."""
    payload = json.dumps([list(split.texts), list(split.labels)], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_sha() -> str | None:
    """HEAD with ``-dirty`` when tracked files differ, as the sweep records it; None off git."""
    from .sweep import git_sha

    sha = git_sha(Path(__file__).resolve().parent)
    return None if sha == "unknown" else sha


def _library_versions() -> dict[str, str | None]:
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def train(
    config: TrainConfig,
    train: Split,
    val: Split,
    out_dir: Path,
    *,
    device: str | torch.device | None = None,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Fine-tune on ``train``, select the epoch on ``val``, save the best to ``out_dir``.

    ``out_dir`` ends up holding the best-on-val model and tokenizer plus ``run_record.json``.
    ``train`` and ``val`` must share ``label_names``.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    if train.label_names != val.label_names:
        raise ValueError("train and val must share label_names")
    if len(train) == 0:
        raise ValueError("train split is empty")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _pick_device(device)
    use_amp = config.mixed_precision and dev.type == "cuda"

    _seed_everything(config.seed)
    names = train.label_names
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(names),
        id2label=dict(enumerate(names)),
        label2id={name: i for i, name in enumerate(names)},
        # distilbert-base-uncased ships no classification head, so nothing real is dropped; a
        # checkpoint that does ship one of another size (the tiny test model) gets a fresh head.
        ignore_mismatched_sizes=True,
    ).to(dev)

    optimizer = _make_optimizer(model, config)
    steps_per_epoch = math.ceil(len(train) / config.batch_size)
    total_steps = config.max_epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config.warmup_ratio * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    shuffler = torch.Generator().manual_seed(config.seed)
    labels = torch.tensor(train.labels)

    history: list[dict[str, float]] = []
    best_f1, best_epoch, waited = -1.0, 0, 0
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=shuffler).tolist()
        losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            idx = order[start : start + config.batch_size]
            batch = _tokenize(tokenizer, [train.texts[i] for i in idx], config.max_length, dev)
            target = labels[idx].to(dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

        f1 = _val_macro_f1(
            model,
            tokenizer,
            val,
            device=dev,
            max_length=config.max_length,
            batch_size=config.eval_batch_size,
        )
        train_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_macro_f1": f1})
        log(f"epoch {epoch}/{config.max_epochs}  loss {train_loss:.4f}  val macro-F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1, best_epoch, waited = f1, epoch, 0
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
        else:
            waited += 1
            if waited >= config.patience:
                log(f"no val improvement for {config.patience} epochs; stopping")
                break
    train_seconds = time.perf_counter() - started

    result = TrainResult(
        best_epoch=best_epoch,
        best_val_macro_f1=best_f1,
        epochs_run=len(history),
        train_seconds=train_seconds,
        checkpoint_dir=out_dir,
        history=history,
    )
    record = {
        "config": config.to_dict(),
        "best_epoch": result.best_epoch,
        "best_val_macro_f1": result.best_val_macro_f1,
        "epochs_run": result.epochs_run,
        "train_seconds": result.train_seconds,
        "n_train": len(train),
        "history": history,
        "versions": _library_versions(),
        "git_sha": _git_sha(),
        "train_data_sha256": _data_digest(train),
    }
    (out_dir / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return result


class Classifier:
    """A saved sequence classifier: batched probabilities, and a one-text ``predict``."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        id2label = model.config.id2label
        self.label_names: tuple[str, ...] = tuple(id2label[i] for i in range(len(id2label)))

    @classmethod
    def load(
        cls,
        path_or_hub_id: str | Path,
        device: str | torch.device | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> Classifier:
        """Load from a local directory or a Hugging Face Hub repo id."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        source = str(path_or_hub_id)
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source)
        return cls(model, tokenizer, _pick_device(device), max_length)

    def predict_proba(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        """Probabilities, shape ``(len(texts), n_labels)``, columns in ``label_names`` order."""
        return _predict_proba(
            self.model,
            self.tokenizer,
            list(texts),
            device=self.device,
            max_length=self.max_length,
            batch_size=batch_size,
        )

    def predict(self, text: str) -> dict[str, Any]:
        """``{"label", "confidence", "top3": [{"label", "p"}, ...]}`` for one text."""
        proba = self.predict_proba([text])[0]
        order = np.argsort(-proba, kind="stable")[:3]
        top3 = [{"label": self.label_names[i], "p": float(proba[i])} for i in order]
        return {"label": top3[0]["label"], "confidence": top3[0]["p"], "top3": top3}
