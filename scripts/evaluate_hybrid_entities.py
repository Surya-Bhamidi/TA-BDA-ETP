"""Compare current context core with/without learned roles on identical annotations.

These synthetic examples have been viewed. This is a frozen diagnostic
comparison, not an independent human or real-source accuracy estimate.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from crime_nlp.config import ARTIFACTS, MODELS, PROCESSED, write_json
from crime_nlp.inference import language_model
from crime_nlp.text import tokens_with_offsets, token_features, spans_from_bio, bio_labels
from crime_nlp.narrative_context import resolve_narrative_context
from crime_nlp.evidence_context import validate_evidence_context
from crime_nlp.trained_roles import predict_roles, merge_roles, revision_paths
from crime_nlp.evaluation import entity_metrics
from crime_nlp.train import file_hash

receipt = {"frozen_at": datetime.now(timezone.utc).isoformat(), "core_sha256": file_hash(MODELS / "crf.joblib"),
           "role_files": {str(path): file_hash(path) for path in revision_paths()},
           "source_sha256": {path.name: file_hash(path) for path in (Path(__file__).resolve().parents[1] / "crime_nlp").glob("*.py")},
           "scope": "Diagnostic comparison on previously viewed synthetic ten-label annotations; same current core/context rules in both arms"}
write_json(ARTIFACTS / "hybrid_evaluation_freeze.json", receipt)
crf = joblib.load(MODELS / "crf.joblib")
results = {}
for split in ["validation", "test"]:
    frame = pd.read_json(PROCESSED / "annotation_v4" / f"{split}.jsonl", lines=True)
    texts = frame.narrative.tolist()
    tokens = [tokens_with_offsets(text) for text in texts]
    raw = crf.predict([token_features(batch) for batch in tokens])
    baseline, hybrid, gold = [], [], []
    for row, batch, tags, doc in zip(frame.itertuples(), tokens, raw, language_model().pipe(texts, batch_size=64)):
        core, roles = resolve_narrative_context(row.narrative, spans_from_bio(row.narrative, batch, tags), doc)
        core = validate_evidence_context(row.narrative, core)
        baseline.append(bio_labels(batch, core + roles))
        enriched, witnesses = merge_roles(row.narrative, core, roles, predict_roles(row.narrative))
        hybrid.append(bio_labels(batch, enriched + witnesses))
        gold.append(bio_labels(batch, json.loads(row.entities_json)))
    results[split] = {"rows": len(frame), "without_trained_roles": entity_metrics(gold, baseline), "with_trained_roles": entity_metrics(gold, hybrid)}
write_json(ARTIFACTS / "hybrid_entity_evaluation.json", {"scope": receipt["scope"], "frozen_at": receipt["frozen_at"], "evaluated_at": datetime.now(timezone.utc).isoformat(), "splits": results})
print({split: {name: result["strict_entity_f1"] for name, result in row.items() if name != "rows"} for split, row in results.items()})
