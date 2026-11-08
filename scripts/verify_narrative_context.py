"""Measure deployed context repairs separately from unchanged raw NER scores."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from crime_nlp.config import ARTIFACTS, MODELS, PROCESSED, write_json
from crime_nlp.evaluation import entity_metrics
from crime_nlp.evidence_context import validate_evidence_context
from crime_nlp.inference import language_model
from crime_nlp.narrative_context import resolve_narrative_context
from crime_nlp.text import tokens_with_offsets, spans_from_bio, bio_labels
from crime_nlp.train import entity_split, sequence_data, file_hash


def compare(texts, gold, raw):
    checked, changes = [], []
    for index, (text, tags, doc) in enumerate(zip(texts, raw, language_model().pipe(texts, batch_size=64))):
        tokens = tokens_with_offsets(text)
        spans, _ = resolve_narrative_context(text, spans_from_bio(text, tokens, tags), doc)
        checked_tags = bio_labels(tokens, validate_evidence_context(text, spans))
        if list(tags) != checked_tags:
            changes.append(index)
        checked.append(checked_tags)
    return {"rows": len(texts), "raw_span_f1": entity_metrics(gold, raw)["strict_entity_f1"],
            "context_span_f1": entity_metrics(gold, checked)["strict_entity_f1"],
            "changed_reports": len(changes), "changed_row_indices": changes}


if __name__ == "__main__":
    frame = pd.read_parquet(PROCESSED / "reports")
    validation = entity_split(frame, "validation", 400)
    x, gold = sequence_data(validation)
    raw = list(joblib.load(MODELS / "crf.joblib").predict(x))
    result = {"verified_at": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {name: file_hash(Path(__file__).resolve().parents[1] / "crime_nlp" / name)
                                for name in ["narrative_context.py", "evidence_context.py", "material_context.py"]},
              "scope": "Diagnostic on previously viewed synthetic evaluation; WITNESS is separate from the eight trained labels. Raw models and predictions unchanged.",
              "crf_validation": compare(validation.narrative.tolist(), gold, raw), "test": {}}
    rows = pd.read_json(PROCESSED / "entity_evaluation_predictions.jsonl", lines=True)
    for model, column in [("CRF", "crf_bio"), ("BERT", "bert_bio")]:
        result["test"][model] = compare(rows.narrative.tolist(), rows.gold_bio.tolist(), rows[column].tolist())
    write_json(ARTIFACTS / "narrative_context_validation.json", result)
    print(json.dumps(result, indent=2))
