"""Check context validation against the unchanged, viewed entity evaluations."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from crime_nlp.config import ARTIFACTS, MODELS, PROCESSED, write_json
from crime_nlp.evidence_context import validate_evidence_context
from crime_nlp.evaluation import entity_metrics
from crime_nlp.text import tokens_with_offsets, spans_from_bio, bio_labels
from crime_nlp.train import entity_split, sequence_data, file_hash


def compare(texts, gold, raw):
    checked, changes = [], 0
    for text, tags in zip(texts, raw):
        tokens = tokens_with_offsets(text)
        spans = validate_evidence_context(text, spans_from_bio(text, tokens, tags))
        checked_tags = bio_labels(tokens, spans)
        changes += list(tags) != checked_tags
        checked.append(checked_tags)
    before = entity_metrics(gold, raw)["strict_entity_f1"]
    after = entity_metrics(gold, checked)["strict_entity_f1"]
    if after < before - 1e-12:
        raise RuntimeError(f"Context check regressed the viewed evaluation: {before:.6f} -> {after:.6f}")
    return {"rows": len(texts), "raw_span_f1": before, "context_checked_span_f1": after, "changed_reports": int(changes)}


if __name__ == "__main__":
    frame = pd.read_parquet(PROCESSED / "reports")
    validation = entity_split(frame, "validation", 400)
    x, gold = sequence_data(validation)
    raw = list(joblib.load(MODELS / "crf.joblib").predict(x))
    result = {"status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
              "crf_validation": compare(validation.narrative.tolist(), gold, raw), "test": {},
              "correction_source_sha256": file_hash(Path(__file__).resolve().parents[1] / "crime_nlp" / "evidence_context.py"),
              "scope": "Narrow disclosed context-rule correction, not CRF retraining. Existing viewed synthetic cases are diagnostic; reported-example tests are development regressions. Raw model weights and benchmark predictions are retained."}
    rows = pd.read_json(PROCESSED / "entity_evaluation_predictions.jsonl", lines=True)
    for model, column in [("CRF", "crf_bio"), ("BERT", "bert_bio")]:
        result["test"][model] = compare(rows.narrative.tolist(), rows.gold_bio.tolist(), rows[column].tolist())
    write_json(ARTIFACTS / "evidence_context_validation.json", result)
    print(json.dumps(result, indent=2))
