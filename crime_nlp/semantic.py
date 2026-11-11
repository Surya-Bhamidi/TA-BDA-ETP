"""Local sentence embeddings, hybrid retrieval, and evidence-grounded answers."""
from functools import lru_cache
import json
import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .config import RUNTIME, ARTIFACTS, SETTINGS
from .text import sentences
from .windowing import source_windows


@lru_cache(maxsize=1)
def encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(4)
    path = RUNTIME / "semantic_encoder"
    return AutoModel.from_pretrained(path, local_files_only=True).eval(), AutoTokenizer.from_pretrained(path, local_files_only=True)


def encode(texts, batch_size=64, progress=False):
    import torch
    model, tokenizer = encoder()
    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(list(texts[start:start + batch_size]), padding=True, truncation=True, max_length=128, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            output.append(torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy())
            if progress and start % 6400 == 0:
                print(f"Semantic index: {min(start + batch_size, len(texts)):,}/{len(texts):,}", flush=True)
    return np.concatenate(output) if output else np.empty((0, 384), dtype=np.float32)


@lru_cache(maxsize=1)
def embedding_index():
    return np.load(ARTIFACTS / "semantic_embeddings.npy", mmap_mode="r")


def semantic_scores(query):
    vector = encode([query])[0]
    paths = [ARTIFACTS / "semantic_chunk_embeddings.npy", ARTIFACTS / "semantic_chunk_rows.npy"]
    manifest = ARTIFACTS / "semantic_chunk_manifest.json"
    legacy_manifest = ARTIFACTS / "index_manifest.json"
    ready = all(path.exists() for path in paths) and manifest.exists() and legacy_manifest.exists()
    if ready:
        chunk_info = json.loads(manifest.read_text(encoding="utf-8"))
        corpus_info = json.loads(legacy_manifest.read_text(encoding="utf-8"))
        ready = chunk_info["source_ids_sha256"] == corpus_info["report_order_sha256"] and bool(chunk_info.get("source_narratives_sha256")) and chunk_info.get("source_narratives_sha256") == corpus_info.get("report_text_sha256")
    if ready:
        chunks, rows = chunk_index(tuple((str(path), path.stat().st_mtime_ns) for path in paths))
        if len(chunks) != len(rows) or len(rows) and int(rows.max()) >= len(embedding_index()):
            raise RuntimeError("Semantic source-window index is incomplete; rebuild the chunk index")
        scores = np.full(len(embedding_index()), -1., dtype=np.float32)
        for start in range(0, len(chunks), 4096):
            similarities = np.asarray(chunks[start:start + 4096], dtype=np.float32) @ vector
            np.maximum.at(scores, rows[start:start + 4096], similarities)
        return scores
    matrix = embedding_index()
    return np.concatenate([np.asarray(matrix[start:start + 4096], dtype=np.float32) @ vector for start in range(0, len(matrix), 4096)])


@lru_cache(maxsize=1)
def chunk_index(revision):
    return np.load(ARTIFACTS / "semantic_chunk_embeddings.npy", mmap_mode="r"), np.load(ARTIFACTS / "semantic_chunk_rows.npy", mmap_mode="r")


def rank_scores(lexical, semantic, mode="Hybrid"):
    if mode == "Keyword":
        return lexical
    if mode == "Semantic":
        return np.maximum(semantic, 0)
    # Reciprocal-rank fusion avoids pretending lexical and semantic cosine scales match.
    combined = np.zeros(len(lexical), dtype=np.float64)
    weight = SETTINGS["inference"]["semantic_weight"]
    for scores, contribution in [(lexical, 1 - weight), (semantic, weight)]:
        order = np.argsort(-scores, kind="stable")
        valid = scores[order] > (.02 if scores is lexical else .25)
        ranks = np.arange(1, len(order) + 1)
        combined[order[valid]] += contribution / (60 + ranks[valid])
    return combined


def semantic_evidence(question, text, report_id="custom narrative"):
    """Retrieve cited source windows by full keyword coverage or semantic similarity."""
    source = sentences(text)
    if not source or not question.strip():
        return {"answer": "Insufficient evidence in this narrative.", "citations": [], "score": 0.}
    _, tokenizer = encoder()
    chunks = source_windows(source, tokenizer)
    if not chunks:
        return {"answer": "Insufficient evidence in this narrative.", "citations": [], "score": 0.}
    chunk_scores = encode([chunk["text"] for chunk in chunks]) @ encode([question])[0]
    # Short literal queries can have low embedding similarity to a full sentence.
    # Require every content token in the same source window; shared stop words
    # and substrings (e.g. "gun" in "begun") cannot bypass the semantic threshold.
    query_terms = set(re.findall(r"\w+", question.casefold())) - ENGLISH_STOP_WORDS
    keyword_matches = [bool(query_terms) and query_terms <= set(re.findall(r"\w+", chunk["text"].casefold()))
                       for chunk in chunks]
    scores = np.full(len(source), -1., dtype=np.float32)
    best_chunks = {}
    for index, (chunk, score) in enumerate(zip(chunks, chunk_scores)):
        source_id = chunk["source"]
        previous = best_chunks.get(source_id)
        if previous is None or (keyword_matches[index], score) > (keyword_matches[previous], chunk_scores[previous]):
            scores[source_id] = score
            best_chunks[source_id] = index
    order = sorted(best_chunks, key=lambda i: (keyword_matches[best_chunks[i]], scores[i]), reverse=True)
    chosen = [i for i in order if keyword_matches[best_chunks[i]] or scores[i] >= .30][:2]
    if not chosen:
        return {"answer": "Insufficient evidence in this narrative.", "citations": [], "score": float(scores.max())}
    selected_indices = [best_chunks[i] for i in sorted(chosen)]
    selected = [chunks[i] for i in selected_indices]
    keyword_match = any(keyword_matches[i] for i in selected_indices)
    semantic_match = any(chunk_scores[i] >= .30 for i in selected_indices)
    match_type = "Keyword + semantic" if keyword_match and semantic_match else "Keyword" if keyword_match else "Semantic"
    return {"answer": " ".join(chunk["text"] for chunk in selected), "score": float(max(chunk_scores[i] for i in selected_indices)),
            "match_type": match_type,
            "citations": [{"report_id": report_id, "sentence": chunk["source"] + 1, "text": chunk["text"],
                           "sentence_start": chunk["start"], "sentence_end": chunk["end"]} for chunk in selected],
            "method": "Keyword and semantic evidence retrieval; source matches, not an answer-confidence probability"}
