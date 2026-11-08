"""Retain the previous Word2Vec release after a disclosed diagnostic regression.

This is a test-informed deployment rollback, not validation-only selection or
an untouched final benchmark. The failed candidate and scores are preserved.
No additional fitting or parameter search is performed.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from gensim.models import Word2Vec
from threadpoolctl import threadpool_limits
from crime_nlp.config import ARTIFACTS, MODELS, PROCESSED, write_json
from crime_nlp.train import file_hash, freeze_models, document_vectors, sample_split
from crime_nlp.evaluation import classification_metrics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    threadpool_limits(4)
    comparison = read(ARTIFACTS / "accuracy_improvements.json")
    if comparison.get("release_gate", {}).get("Word2Vec"):
        raise SystemExit("Word2Vec baseline has already been retained; no files changed.")
    candidate = dict(comparison["classification"]["Word2Vec"])
    if candidate["after_accuracy"] >= candidate["before_accuracy"]:
        raise SystemExit("No Word2Vec accuracy regression; no baseline rollback was performed.")
    archive = ARTIFACTS / "experiments" / "accuracy_20261107"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ["accuracy_improvements.json", "evaluation.json", "evaluation_plan.json", "frozen_models.json", "final_evaluation_receipt.json"]:
        if (archive / name).exists():
            raise RuntimeError("Candidate archive already exists; refusing to overwrite experiment evidence")
        shutil.copy2(ARTIFACTS / name, archive / name)
    shutil.copy2(PROCESSED / "evaluation_predictions.parquet", PROCESSED / "accuracy_candidate_predictions.parquet")
    for name in ["word2vec.model", "word2vec_classifier.joblib"]:
        shutil.copy2(args.baseline / "models" / name, MODELS / name)
    for extra in (args.baseline / "models").glob("word2vec.model.*"):
        shutil.copy2(extra, MODELS / extra.name)
    recipe_path = args.baseline / "models" / "word2vec_features.json"
    recipe = read(recipe_path) if recipe_path.exists() else {"pooling": "mean"}
    write_json(MODELS / "word2vec_features.json", recipe)
    reason = (f"The validation-selected candidate fell from {candidate['before_accuracy']:.2%} to {candidate['after_accuracy']:.2%} accuracy "
              "on the viewed diagnostic test set. Retain the previous Word2Vec model; preserve the failed experiment; perform no further tuning.")
    scope = "Repeat evaluation on previously viewed synthetic cases. BoW, TF-IDF, BERT and CRF candidates were selected on validation. Word2Vec deployment retained its baseline after observing a diagnostic test regression. This release choice used test results; it is not an untouched final benchmark. No further training or tuning was performed."
    selection = read(MODELS / "accuracy_selection.json")
    selection["classical"]["models"]["Word2Vec"]["deployment"] = {"recipe": "retained baseline", "test_informed_rollback": True, "reason": reason}
    write_json(MODELS / "accuracy_selection.json", selection)
    plan = read(ARTIFACTS / "evaluation_plan.json")
    plan["evaluation_scope"] = scope
    plan["freeze_policy"] = "Known diagnostic test regression caused a disclosed Word2Vec deployment rollback. No further fitting or hyperparameter search. Other selected weights are unchanged."
    plan["accuracy_revision"]["word2vec_release_gate"] = reason
    receipt = freeze_models(plan)
    write_json(ARTIFACTS / "evaluation_plan.json", plan)
    source = pd.read_parquet(PROCESSED / "reports")
    test = sample_split(source, "test", 1200)
    embedding = Word2Vec.load(str(MODELS / "word2vec.model"))
    classifier = joblib.load(MODELS / "word2vec_classifier.joblib")
    guesses = classifier.predict(document_vectors(embedding, test.narrative, recipe))
    previous_predictions = pd.read_parquet(args.baseline / "predictions" / "evaluation_predictions.parquet")
    if test.report_id.tolist() != previous_predictions.report_id.tolist() or list(guesses) != previous_predictions["Word2Vec"].tolist():
        raise RuntimeError("Restored Word2Vec predictions differ from the retained baseline")
    predictions = pd.read_parquet(PROCESSED / "evaluation_predictions.parquet")
    if predictions.report_id.tolist() != test.report_id.tolist():
        raise RuntimeError("Current document test ID order differs")
    predictions["Word2Vec"] = guesses
    document_path = PROCESSED / "evaluation_predictions.parquet"
    predictions.to_parquet(document_path, index=False)
    evaluation = read(ARTIFACTS / "evaluation.json")
    old_metrics = read(args.baseline / "artifacts" / "evaluation.json")["models"]["Word2Vec"]
    restored = classification_metrics(test.crime_type, guesses)
    restored.update(train_seconds=old_metrics["train_seconds"], release_status="Baseline retained after disclosed diagnostic regression")
    evaluation["models"]["Word2Vec"] = restored
    evaluation["protocol"] = plan
    write_json(ARTIFACTS / "evaluation.json", evaluation)
    comparison["scope"] = scope
    comparison["release_gate"] = {"Word2Vec": reason}
    comparison["rejected_candidates"] = {"Word2Vec": candidate}
    comparison["classification"]["Word2Vec"] = {"before_accuracy": candidate["before_accuracy"], "after_accuracy": restored["accuracy"],
        "before_macro_f1": candidate["before_macro_f1"], "after_macro_f1": restored["macro_f1"], "accuracy_change_pp": 0.,
        "status": "retained baseline", "rejected_candidate_accuracy": candidate["after_accuracy"]}
    write_json(ARTIFACTS / "accuracy_improvements.json", comparison)
    entity_path = PROCESSED / "entity_evaluation_predictions.jsonl"
    write_json(ARTIFACTS / "final_evaluation_receipt.json", {"evaluated_at": datetime.now(timezone.utc).isoformat(),
        "frozen_at": receipt["frozen_at"], "selected_ner": read(MODELS / "ner_selection.json")["selected"],
        "prediction_files": {p.name: file_hash(p) for p in [document_path, entity_path]}, "note": scope,
        "prediction_reuse": "Other models and their existing diagnostic predictions are unchanged; restored Word2Vec predictions were recomputed and matched the retained baseline."})
    print(json.dumps(comparison, indent=2))
