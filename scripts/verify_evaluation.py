"""Recompute saved test scores and verify frozen weights without fitting models."""
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from crime_nlp.config import ARTIFACTS, MODELS, PROCESSED, write_json
from crime_nlp.evaluation import entity_metrics
from crime_nlp.text import bio_labels, tokens_with_offsets
from crime_nlp.train import file_hash


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    def read(name):
        return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))

    frozen, receipt = read("frozen_models.json"), read("final_evaluation_receipt.json")
    require(receipt["frozen_at"] == frozen["frozen_at"], "Evaluation does not identify the current freeze")
    require(datetime.fromisoformat(receipt["evaluated_at"]) >= datetime.fromisoformat(frozen["frozen_at"]),
            "Evaluation predates model freezing")
    for name, expected in frozen["model_sha256"].items():
        require(file_hash(MODELS / name) == expected, f"Frozen model changed: {name}")
    for name, expected in receipt["prediction_files"].items():
        require(file_hash(PROCESSED / name) == expected, f"Saved predictions changed: {name}")

    evaluation, ner = read("evaluation.json"), read("ner_evaluation.json")
    predictions = pd.read_parquet(PROCESSED / "evaluation_predictions.parquet")
    ids = pd.read_csv(ARTIFACTS / "test_ids.csv")
    require(predictions.report_id.is_unique, "Duplicate document prediction IDs")
    require(predictions.report_id.tolist() == ids.report_id.tolist(), "Prediction/test ID order differs")
    require(len(predictions) == evaluation["protocol"]["test_rows"], "Wrong test prediction count")
    corpus = pd.read_parquet(PROCESSED / "reports", columns=["report_id", "crime_type", "split"])
    reference = corpus.set_index("report_id").loc[predictions.report_id]
    require(reference.split.eq("test").all(), "Document scoring includes non-test rows")
    require(reference.crime_type.tolist() == predictions.crime_type.tolist(), "Reference categories changed")
    measured = {}
    for name, saved in evaluation["models"].items():
        truth, guesses = predictions.crime_type, predictions[name]
        scores = {"accuracy": accuracy_score(truth, guesses),
                  "macro_f1": f1_score(truth, guesses, labels=saved["labels"], average="macro", zero_division=0),
                  "weighted_f1": f1_score(truth, guesses, average="weighted", zero_division=0)}
        for metric, score in scores.items():
            require(math.isclose(score, saved[metric], abs_tol=1e-12), f"{name} {metric} differs from saved predictions")
        measured[name] = scores

    entities = pd.read_json(PROCESSED / "entity_evaluation_predictions.jsonl", lines=True)
    require(len(entities) == ner["test_rows"], "Wrong entity evaluation count")
    require(entities.report_id.is_unique, "Duplicate entity prediction IDs")
    fitting = pd.read_csv(ARTIFACTS / "train_ids.csv")
    validation = pd.read_csv(ARTIFACTS / "validation_ids.csv")
    require(not set(entities.report_id) & (set(fitting.report_id) | set(validation.report_id)), "Entity/document fitting IDs overlap evaluation")
    for row in entities.itertuples():
        spans = json.loads(row.entities_json)
        require(all(row.narrative[e["start"]:e["end"]] == e["text"] for e in spans), "Invalid reference character span")
        require(bio_labels(tokens_with_offsets(row.narrative), spans) == row.gold_bio, "Gold BIO tags differ from reference spans")
    entity_scores = {}
    for name, column in [("CRF", "crf_bio"), ("BERT", "bert_bio")]:
        require(all(len(a) == len(b) for a, b in zip(entities.gold_bio, entities[column])), "Entity token counts differ")
        score = entity_metrics(entities.gold_bio.tolist(), entities[column].tolist())["strict_entity_f1"]
        require(math.isclose(score, ner["models"][name]["strict_entity_f1"], abs_tol=1e-12), f"{name} span F1 differs")
        entity_scores[name] = score

    result = {"status": "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
              "frozen_at": frozen["frozen_at"], "document_rows": len(predictions), "entity_rows": len(entities),
              "document_metrics": measured, "strict_entity_f1": entity_scores,
              "checks": ["frozen model hashes", "prediction hashes", "freeze/evaluation order", "test IDs and source categories",
                         "recomputed accuracy and macro/weighted F1", "reference spans and BIO alignment", "recomputed strict span F1"],
              "scope": evaluation["protocol"]["evaluation_scope"]}
    write_json(ARTIFACTS / "evaluation_verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
