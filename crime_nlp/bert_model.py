"""Fine-tune real, pretrained two-layer BERT for narrative sequence classification."""
import time
import numpy as np

from .config import RUNTIME, MODELS, SEED, CATEGORIES


def fit_bert(train_text, train_y, val_text, val_y, test_text, epochs=3, batch_size=32,
             learning_rate=5e-4, scheduled=False, max_tokens=160, target_dir=None, windowed=False):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
    from transformers.utils import logging
    logging.set_verbosity_error()
    from sklearn.metrics import f1_score
    set_seed(SEED)
    torch.set_num_threads(4)
    source = RUNTIME / "bert"
    if not (source / "config.json").exists():
        raise RuntimeError("Pretrained BERT is missing. Run python scripts/download_resources.py first.")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    options = {"hidden_dropout_prob": .2, "attention_probs_dropout_prob": .2, "classifier_dropout": .2} if scheduled else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=len(CATEGORIES), id2label=dict(enumerate(CATEGORIES)),
        label2id={v: k for k, v in enumerate(CATEGORIES)}, local_files_only=True,
        attn_implementation="eager", **options)
    model.config.narrative_max_tokens = max_tokens
    model.config.narrative_windowed = windowed
    train_encoding = tokenizer(list(train_text), padding=False, truncation=True, max_length=max_tokens,
                               **({"return_overflowing_tokens": True, "stride": min(64, max_tokens // 4)} if windowed else {}))
    mapping = train_encoding.pop("overflow_to_sample_mapping", list(range(len(train_y))))
    labels = torch.tensor([CATEGORIES.index(train_y[index]) for index in mapping])
    counts = np.bincount(mapping, minlength=len(train_y))
    sample_weights = torch.tensor([1. / counts[index] for index in mapping], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.05 if scheduled else .01)
    total_steps = epochs * ((len(labels) + batch_size - 1) // batch_size)
    warmup = max(1, int(total_steps * .1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / warmup, (total_steps - step) / max(1, total_steps - warmup))) if scheduled else None
    generator = torch.Generator().manual_seed(SEED)
    history, best = [], -1.
    started = time.perf_counter()
    dest = target_dir or MODELS / "bert_classifier"
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(labels), generator=generator)
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = tokenizer.pad([{key: value[int(i)] for key, value in train_encoding.items()} for i in indices],
                                  padding=True, return_tensors="pt")
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            losses_by_window = torch.nn.functional.cross_entropy(out.logits, labels[indices], label_smoothing=.03 if scheduled else 0., reduction="none")
            loss = (losses_by_window * sample_weights[indices]).sum() / sample_weights[indices].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(loss.detach()))
        predicted = predict_bert(model, tokenizer, val_text)
        f1 = f1_score(val_y, predicted, labels=CATEGORIES, average="macro", zero_division=0)
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_macro_f1": float(f1)})
        print(f"BERT epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.4f}, validation macro-F1={f1:.4f}", flush=True)
        if f1 > best:
            best = f1
            model.save_pretrained(dest, safe_serialization=True)
            tokenizer.save_pretrained(dest)
    model = AutoModelForSequenceClassification.from_pretrained(dest, local_files_only=True)
    prediction = predict_bert(model, tokenizer, test_text)
    return prediction, {"history": history, "best_validation_macro_f1": best, "train_seconds": round(time.perf_counter() - started, 2), "architecture": "Pretrained BERT: 2 layers, 128 hidden units, 2 attention heads; all layers fine-tuned", "max_tokens": max_tokens,
                       "recipe": {"windowed_full_document": windowed, "learning_rate": learning_rate, "warmup_ratio": .1 if scheduled else 0.,
                                  "linear_decay": scheduled, "dropout": .2 if scheduled else .1,
                                  "label_smoothing": .03 if scheduled else 0., "epochs": epochs}}


def predict_bert(model, tokenizer, texts, batch_size=64, full_text=None):
    import torch
    model.eval()
    result = []
    windowed = getattr(model.config, "narrative_windowed", False) if full_text is None else full_text
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start:start + batch_size])
            limit = getattr(model.config, "narrative_max_tokens", 160)
            inputs = tokenizer(batch_texts, padding=True, truncation=True,
                               max_length=limit, return_tensors="pt", **({"return_overflowing_tokens": True, "stride": min(64, limit // 4)} if windowed else {}))
            mapping = inputs.pop("overflow_to_sample_mapping", None)
            # Bound inference memory even when one report produces many windows.
            windows = torch.cat([model(**{key: value[offset:offset + batch_size] for key, value in inputs.items()}).logits.softmax(-1)
                                 for offset in range(0, len(inputs["input_ids"]), batch_size)])
            if mapping is not None:
                pooled = torch.zeros((len(batch_texts), len(CATEGORIES)))
                pooled.index_add_(0, mapping, windows)
                pooled /= torch.bincount(mapping, minlength=len(batch_texts)).clamp(min=1)[:, None]
                ids = pooled.argmax(dim=-1).tolist()
            else:
                ids = windows.argmax(dim=-1).tolist()
            result.extend(CATEGORIES[i] for i in ids)
    return result
