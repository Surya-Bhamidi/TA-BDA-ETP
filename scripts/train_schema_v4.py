"""Train ten-label candidates without replacing legacy models or benchmark data."""
import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
import sklearn_crfsuite
from crime_nlp.annotation_schema import LABELS, validate_record
from crime_nlp.config import ARTIFACTS, MODELS, PROCESSED, write_json
from crime_nlp.entity_corpus_v4 import generate
from crime_nlp.evaluation import entity_metrics, split_integrity
from crime_nlp.train import entity_split, sequence_data, file_hash
from crime_nlp.ner_model import train_ner, load_ner, sequences, predict_tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "models" / "v4_entities")
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Candidate output already exists; use a fresh directory to preserve provenance.")
    args.output.mkdir(parents=True)
    data_dir = PROCESSED / "annotation_v4"
    data_dir.mkdir(exist_ok=True)
    frame = pd.read_parquet(PROCESSED / "reports")
    legacy = entity_split(frame, "train", 8000)
    # Versioned, explicitly disclosed weak-supervision migration of old training
    # annotations. Existing validation/test references are never overwritten.
    from crime_nlp.inference import language_model
    from crime_nlp.narrative_context import resolve_narrative_context
    from crime_nlp.evidence_context import validate_evidence_context
    annotations = []
    for row, doc in zip(legacy.itertuples(), language_model().pipe(legacy.narrative.tolist(), batch_size=64)):
        core, roles = resolve_narrative_context(row.narrative, json.loads(row.entities_json), doc)
        spans = validate_evidence_context(row.narrative, core) + roles
        spans = [{key: entity[key] for key in ("start", "end", "text", "label")} for entity in sorted(spans, key=lambda entity: entity["start"])]
        validate_record({"narrative": row.narrative, "entities": spans})
        annotations.append(json.dumps(spans, ensure_ascii=False))
    legacy["entities_json"] = annotations
    legacy["annotation_provenance"] = "Rule-assisted migration of synthetic training references; not human reviewed"
    legacy["annotation_schema"] = "4.0"
    train = pd.concat([legacy, generate("train", 4000)], ignore_index=True)
    validation = generate("validation", 800)
    development = generate("development", 800)
    integrity = split_integrity({"train": train, "validation": validation, "development": development})
    for name, part in [("train", train), ("validation", validation), ("development", development)]:
        part.to_json(data_dir / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    plan = {"schema": "4.0", "labels": LABELS, "train_rows": len(train), "validation_rows": len(validation),
            "development_rows": len(development), "planned_test_rows": 800, "integrity": integrity,
            "scope": "Synthetic experiment; explicit new templates plus rule-assisted migration of old training references. No independent real-data or human-annotation claim.",
            "selection": "Checkpoint and tagger selection on validation only; new generated test opened after freeze; no test-informed retraining.",
            "legacy_models_preserved": str(MODELS)}
    write_json(args.output / "plan.json", plan)
    x, y = sequence_data(train)
    vx, vy = sequence_data(validation)
    baseline = joblib.load(MODELS / "crf.joblib")
    baseline_metrics = entity_metrics(vy, baseline.predict(vx))
    print("Training CRF on versioned ten-label annotations...", flush=True)
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=.01, c2=.1, max_iterations=180, all_possible_transitions=True).fit(x, y)
    joblib.dump(crf, args.output / "crf.joblib")
    crf_validation = entity_metrics(vy, crf.predict(vx))
    write_json(args.output / "crf_validation.json", {"legacy_eight_label_baseline_on_new_schema": baseline_metrics, "candidate": crf_validation})
    print(f"CRF validation strict Span-F1: {crf_validation['strict_entity_f1']:.4f}", flush=True)
    del x, y
    gc.collect()
    details = train_ner(train, validation, epochs=args.epochs, scheduled=True, target_dir=args.output / "bert_ner", entity_labels=LABELS)
    chosen = "BERT" if details["validation_f1"] > crf_validation["strict_entity_f1"] else "CRF"
    write_json(args.output / "ner_selection.json", {"selected": chosen, "labels": LABELS,
               "crf_validation_f1": crf_validation["strict_entity_f1"], "bert_validation_f1": details["validation_f1"], "criterion": "Validation strict Span-F1, ties prefer CRF"})
    freeze = {"frozen_at": datetime.now(timezone.utc).isoformat(), "models": {
        path.relative_to(args.output).as_posix(): file_hash(path) for path in sorted(args.output.rglob("*")) if path.is_file()},
        "training_data_sha256": file_hash(data_dir / "train.jsonl"), "validation_data_sha256": file_hash(data_dir / "validation.jsonl"), "plan": plan}
    write_json(args.output / "frozen.json", freeze)
    print("Candidates frozen; opening new synthetic test once...", flush=True)
    test = generate("test", 800)
    split_integrity({"train": train, "validation": validation, "development": development, "test": test})
    test.to_json(data_dir / "test.jsonl", orient="records", lines=True, force_ascii=False)
    tx, ty = sequence_data(test)
    crf_tags = list(crf.predict(tx))
    bert, tokenizer = load_ner(args.output / "bert_ner")
    tokens, _ = sequences(test)
    bert_tags, _ = predict_tags(bert, tokenizer, tokens)
    scores = {"CRF": entity_metrics(ty, crf_tags), "BERT": entity_metrics(ty, bert_tags)}
    predictions = test.copy()
    predictions["gold_bio"], predictions["crf_bio"], predictions["bert_bio"] = ty, crf_tags, bert_tags
    predictions.to_json(data_dir / "test_predictions.jsonl", orient="records", lines=True, force_ascii=False)
    write_json(ARTIFACTS / "schema_v4_evaluation.json", {"plan": plan, "selected": chosen,
               "validation": {"CRF": crf_validation["strict_entity_f1"], "BERT": details["validation_f1"]}, "models": scores,
               "frozen_at": freeze["frozen_at"], "evaluated_at": datetime.now(timezone.utc).isoformat(),
               "test_data_sha256": file_hash(data_dir / "test.jsonl"), "predictions_sha256": file_hash(data_dir / "test_predictions.jsonl"),
               "candidate_directory": str(args.output), "bert_history": details["history"]})
    print({name: score["strict_entity_f1"] for name, score in scores.items()}, flush=True)


if __name__ == "__main__":
    main()
