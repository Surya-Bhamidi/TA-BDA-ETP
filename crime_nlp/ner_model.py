"""Real BERT token classification with word/subword alignment and long-text windows."""
import json
import numpy as np
from .config import ENTITY_LABELS, RUNTIME, MODELS, SEED, write_json
from .text import tokens_with_offsets, bio_labels, spans_from_bio
from .evaluation import entity_metrics, repair_bio

BIO_TAGS = ["O"] + [prefix + name for name in ENTITY_LABELS for prefix in ("B-", "I-")]


def align_labels(encoding, word_labels, bio_tags=None):
    """Only each word's first subtoken contributes to loss; others use -100."""
    bio_tags = bio_tags or BIO_TAGS
    aligned = []
    for i, labels in enumerate(word_labels):
        previous, row = None, []
        for word_id in encoding.word_ids(batch_index=i):
            row.append(bio_tags.index(labels[word_id]) if word_id is not None and word_id != previous else -100)
            previous = word_id
        aligned.append(row)
    return aligned


def sequences(part):
    token_lists, gold = [], []
    for row in part.itertuples():
        tokens = tokens_with_offsets(row.narrative)
        token_lists.append(tokens)
        gold.append(bio_labels(tokens, json.loads(row.entities_json)))
    return token_lists, gold


def bounded_windows(words, tokenizer, max_tokens, offset=0):
    """Split exceptional subword-heavy inputs while preserving every word offset."""
    size = len(tokenizer(words, is_split_into_words=True, add_special_tokens=True)["input_ids"])
    if size <= max_tokens:
        return [(offset, words)]
    if len(words) <= 1:
        raise ValueError("A token exceeds the entity model's supported input length")
    middle = len(words) // 2
    return bounded_windows(words[:middle], tokenizer, max_tokens, offset) + bounded_windows(words[middle:], tokenizer, max_tokens, offset + middle)


def predict_tags(model, tokenizer, token_lists, batch_size=32):
    """Use overlapping 48-word windows; no part of a long case is silently dropped."""
    import torch
    windows, mapping = [], []
    for doc_id, tokens in enumerate(token_lists):
        for start in range(0, len(tokens), 32):
            chunk = tokens[start:start + 48]
            words = [x["text"] for x in chunk]
            for offset, bounded in bounded_windows(words, tokenizer, model.config.max_position_embeddings, start):
                windows.append(bounded)
                mapping.append((doc_id, offset, len(bounded)))
    bio_tags = [model.config.id2label[index] for index in range(model.config.num_labels)]
    sums = [np.zeros((len(tokens), len(bio_tags)), dtype=np.float32) for tokens in token_lists]
    counts = [np.zeros(len(tokens), dtype=np.float32) for tokens in token_lists]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            batch = tokenizer(windows[start:start + batch_size], is_split_into_words=True, truncation=False, padding=True, return_tensors="pt")
            logits = model(**batch).logits.softmax(-1).cpu().numpy()
            for row in range(len(logits)):
                doc_id, offset, _ = mapping[start + row]
                previous = None
                for position, word_id in enumerate(batch.word_ids(batch_index=row)):
                    if word_id is not None and word_id != previous:
                        sums[doc_id][offset + word_id] += logits[row, position]
                        counts[doc_id][offset + word_id] += 1
                    previous = word_id
    predictions, confidence = [], []
    for scores, count in zip(sums, counts):
        averages = scores / np.maximum(count[:, None], 1)
        predictions.append(repair_bio([bio_tags[i] for i in averages.argmax(axis=1)]))
        confidence.append(averages.max(axis=1).tolist())
    return predictions, confidence


def train_ner(train, validation, epochs=5, scheduled=False, target_dir=None, entity_labels=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, set_seed
    from transformers.utils import logging
    logging.set_verbosity_error()
    set_seed(SEED)
    torch.set_num_threads(4)
    bio_tags = ["O"] + [prefix + label for label in (entity_labels or ENTITY_LABELS) for prefix in ("B-", "I-")]
    tokenizer = AutoTokenizer.from_pretrained(RUNTIME / "bert", local_files_only=True)
    options = {"hidden_dropout_prob": .15, "attention_probs_dropout_prob": .15, "classifier_dropout": .15} if scheduled else {}
    model = AutoModelForTokenClassification.from_pretrained(RUNTIME / "bert", num_labels=len(bio_tags), id2label=dict(enumerate(bio_tags)), label2id={v: k for k, v in enumerate(bio_tags)}, local_files_only=True, attn_implementation="eager", **options)
    train_tokens, train_gold = sequences(train)
    words, tags = [], []
    for tokens, labels in zip(train_tokens, train_gold):
        window, stride = (48, 32) if scheduled else (64, 40)
        for start in range(0, len(tokens), stride):
            chunk = [t["text"] for t in tokens[start:start + window]]
            for offset, bounded in bounded_windows(chunk, tokenizer, model.config.max_position_embeddings):
                words.append(bounded)
                tags.append(repair_bio(labels[start + offset:start + offset + len(bounded)]))
    encoded = tokenizer(words, is_split_into_words=True, padding=False, truncation=False)
    labels = align_labels(encoded, tags, bio_tags)
    val_tokens, val_gold = sequences(validation)
    learning_rate = 1.5e-4 if scheduled else 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.03 if scheduled else .01)
    total_steps = epochs * ((len(labels) + 31) // 32)
    warmup = max(1, int(total_steps * .1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / warmup, (total_steps - step) / max(1, total_steps - warmup))) if scheduled else None
    class_weights = torch.ones(len(bio_tags))
    if scheduled:
        class_weights[0] = .7
    random = torch.Generator().manual_seed(SEED)
    history, best = [], -1.
    target = target_dir or MODELS / "bert_ner"
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(labels), generator=random)
        losses = []
        for start in range(0, len(order), 32):
            indices = order[start:start + 32].tolist()
            batch = tokenizer.pad([{key: value[i] for key, value in encoded.items()} for i in indices], padding=True, return_tensors="pt")
            batch_labels = torch.full(batch["input_ids"].shape, -100, dtype=torch.long)
            for row, index in enumerate(indices):
                batch_labels[row, :len(labels[index])] = torch.tensor(labels[index])
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = torch.nn.functional.cross_entropy(output.logits.reshape(-1, len(bio_tags)), batch_labels.reshape(-1),
                                                     weight=class_weights, label_smoothing=.02 if scheduled else 0.)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(loss.detach()))
        guesses, _ = predict_tags(model, tokenizer, val_tokens)
        score = entity_metrics(val_gold, guesses)["strict_entity_f1"]
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_entity_f1": score})
        print(f"BERT NER epoch {epoch + 1}/{epochs}: validation span F1={score:.4f}", flush=True)
        if score > best:
            best = score
            model.save_pretrained(target, safe_serialization=True)
            tokenizer.save_pretrained(target)
    write_json(target / "training.json", {"history": history, "validation_f1": best, "labels": bio_tags, "train_rows": len(train), "validation_rows": len(validation), "window_words": 48 if scheduled else 64, "stride_words": 32 if scheduled else 40,
               "recipe": {"learning_rate": learning_rate, "linear_decay": scheduled, "warmup_ratio": .1 if scheduled else 0.,
                          "dropout": .15 if scheduled else .1, "label_smoothing": .02 if scheduled else 0., "outside_class_weight": .7 if scheduled else 1.}})
    return {"history": history, "validation_f1": best}


def load_ner(path=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    torch.set_num_threads(4)
    path = path or MODELS / "bert_ner"
    return AutoModelForTokenClassification.from_pretrained(path, local_files_only=True), AutoTokenizer.from_pretrained(path, local_files_only=True)


def extract_bert(text, model, tokenizer):
    tokens = tokens_with_offsets(text)
    tags, confidence = predict_tags(model, tokenizer, [tokens])
    entities = spans_from_bio(text, tokens, tags[0])
    for entity in entities:
        values = [p for t, p in zip(tokens, confidence[0]) if entity["start"] <= t["start"] < entity["end"]]
        entity["confidence"] = float(np.mean(values))
    return entities
