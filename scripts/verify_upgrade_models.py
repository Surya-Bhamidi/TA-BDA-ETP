"""Verify completed candidate artifacts and recompute their diagnostic metrics."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from crime_nlp.config import ARTIFACTS, PROCESSED, write_json
from crime_nlp.train import file_hash
from crime_nlp.evaluation import entity_metrics
from crime_nlp.text import tokens_with_offsets, bio_labels

reports = [("schema_v4_evaluation.json", PROCESSED / "annotation_v4" / "test_predictions.jsonl"),
           ("role_specialist_evaluation.json", ARTIFACTS / "models" / "v4_roles" / "diagnostic_predictions.jsonl"),
           ("role_specialist_broad_evaluation.json", ARTIFACTS / "models" / "v4_roles_broad" / "diagnostic_predictions.jsonl")]
checked = []
for name, predictions in reports:
    report = json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))
    directory = Path(report["candidate_directory"])
    freeze = json.loads((directory / "frozen.json").read_text(encoding="utf-8"))
    for relative, expected in freeze["models"].items():
        if file_hash(directory / relative) != expected:
            raise RuntimeError(f"Frozen model artifact changed: {directory / relative}")
    frame = pd.read_json(predictions, lines=True)
    for row in frame.itertuples():
        if bio_labels(tokens_with_offsets(row.narrative), json.loads(row.entities_json)) != row.gold_bio:
            raise RuntimeError("Reference spans and saved BIO labels disagree")
    for model, column in [("CRF", "crf_bio"), ("BERT", "bert_bio")]:
        score = entity_metrics(frame.gold_bio.tolist(), frame[column].tolist())["strict_entity_f1"]
        if abs(score - report["models"][model]["strict_entity_f1"]) > 1e-12:
            raise RuntimeError("Saved score does not match row-level predictions")
    checked.append({"report": name, "model_files_verified": len(freeze["models"]), "rows": len(frame), "predictions_sha256": file_hash(predictions)})
write_json(ARTIFACTS / "upgrade_model_verification.json", {"status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
           "experiments": checked, "scope": "Verifies completed model files, annotations and score recomputation; does not establish real-world accuracy"})
print("Verified all completed entity experiments and their saved prediction metrics.")
