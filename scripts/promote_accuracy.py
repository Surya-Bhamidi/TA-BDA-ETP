"""Validate candidates, freeze selected weights, then score the existing test splits.

Stop the dashboard before running this command. No candidate is selected from
test results, and a worse repeat-test score is reported without retuning.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.model_selection import GroupKFold, cross_val_score
from threadpoolctl import threadpool_limits
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

from crime_nlp.config import ARTIFACTS, MODELS, PROCESSED, ENTITY_LABELS, SETTINGS, write_json
from crime_nlp.train import sample_split, entity_split, sequence_data, document_vectors, file_hash, freeze_models
from crime_nlp.bert_model import predict_bert
from crime_nlp.ner_model import predict_tags, sequences
from crime_nlp.evaluation import (classification_metrics, entity_metrics, split_integrity,
                                  fit_temperature, temperature_scale, calibration_metrics)
from crime_nlp.tuning import validation_score


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_score(actual, expected, name):
    if not np.isclose(actual, expected, atol=1e-12, rtol=0.):
        raise ValueError(f"Validation score changed for {name}: {actual} != {expected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--transformers", type=Path, required=True)
    parser.add_argument("--alternate-classification", type=Path)
    args = parser.parse_args()
    threadpool_limits(4)
    torch.set_num_threads(4)
    baseline = read(args.baseline / "artifacts" / "evaluation.json")
    baseline_ner = read(args.baseline / "artifacts" / "ner_evaluation.json")
    for name, expected in read(args.baseline / "artifacts" / "frozen_models.json")["model_sha256"].items():
        if file_hash(MODELS / name) != expected:
            raise ValueError(f"Active model changed since the backup: {name}")
    df = pd.read_parquet(PROCESSED / "reports")
    train, val, development = (sample_split(df, split, size) for split, size in
                              [("train", 12000), ("validation", 800), ("development", 800)])
    for split, frame in [("train", train), ("validation", val), ("development", development)]:
        old_ids = pd.read_csv(args.baseline / "artifacts" / f"{split}_ids.csv")
        if frame.report_id.tolist() != old_ids.report_id.tolist():
            raise ValueError(f"The {split} sample changed; this comparison would not match the baseline")
    ner_train, ner_val = entity_split(df, "train", 8000), entity_split(df, "validation", 400)
    vx, vy = sequence_data(ner_val)
    selection = read(args.classical / "validation_selection.json")
    crf_selection = read(args.classical / "crf_selection.json")
    bert_selection = read(args.transformers / "classification_selection.json")
    bert_candidate_source = args.transformers
    if args.alternate_classification:
        alternate = read(args.alternate_classification / "classification_selection.json")
        trials = [{"source": str(args.transformers), **bert_selection},
                  {"source": str(args.alternate_classification), **alternate}]
        if alternate["candidate"]["macro_f1"] > bert_selection["candidate"]["macro_f1"] + 1e-12:
            bert_selection, bert_candidate_source = alternate, args.alternate_classification
        bert_selection = {**bert_selection, "trials": trials}
    entity_selection = read(args.transformers / "entity_selection.json")
    entity_selection = {**entity_selection, "minimum_gain": .002,
                        "use_candidate": entity_selection["candidate_f1"] >= entity_selection["baseline_f1"] + .002,
                        "criterion": "Require at least 0.2 percentage points validation Span-F1 gain to replace the retained BERT entity model"}

    models = {"Bag of Words": joblib.load(args.classical / "bow.joblib"),
              "TF-IDF": joblib.load(args.classical / "tfidf.joblib")}
    for name, model in models.items():
        check_score(validation_score(val.crime_type, model.predict(val.narrative))["macro_f1"],
                    selection["models"][name]["selected"]["macro_f1"], name)
    w2v = Word2Vec.load(str(args.classical / "word2vec.model"))
    w2v_classifier = joblib.load(args.classical / "word2vec_classifier.joblib")
    recipe = read(args.classical / "word2vec_features.json")
    check_score(validation_score(val.crime_type, w2v_classifier.predict(document_vectors(w2v, val.narrative, recipe)))["macro_f1"],
                selection["models"]["Word2Vec"]["selected"]["macro_f1"], "Word2Vec")
    crf = joblib.load(args.classical / "crf.joblib")
    crf_val = entity_metrics(vy, crf.predict(vx))["strict_entity_f1"]
    check_score(crf_val, crf_selection["selected"]["validation_f1"], "CRF")
    bert_source = bert_candidate_source if bert_selection["use_candidate"] else MODELS
    bert = AutoModelForSequenceClassification.from_pretrained(bert_source / "bert_classifier", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(bert_source / "bert_classifier", local_files_only=True)
    expected = bert_selection["candidate"] if bert_selection["use_candidate"] else bert_selection["baseline"]
    check_score(validation_score(val.crime_type, predict_bert(bert, tokenizer, val.narrative.tolist()))["macro_f1"], expected["macro_f1"], "BERT")
    entity_source = args.transformers if entity_selection["use_candidate"] else MODELS
    ner_bert = AutoModelForTokenClassification.from_pretrained(entity_source / "bert_ner", local_files_only=True)
    ner_tokenizer = AutoTokenizer.from_pretrained(entity_source / "bert_ner", local_files_only=True)
    tokens, gold = sequences(ner_val)
    bert_val = entity_metrics(gold, predict_tags(ner_bert, ner_tokenizer, tokens)[0])["strict_entity_f1"]
    check_score(bert_val, entity_selection["candidate_f1"] if entity_selection["use_candidate"] else entity_selection["baseline_f1"], "BERT entities")

    # Selection is complete before any test prediction is computed.
    for name in ["bow.joblib", "tfidf.joblib", "word2vec.model", "word2vec_classifier.joblib", "word2vec_features.json", "crf.joblib"]:
        shutil.copy2(args.classical / name, MODELS / name)
    for extra in args.classical.glob("word2vec.model.*"):
        shutil.copy2(extra, MODELS / extra.name)
    if bert_selection["use_candidate"]:
        shutil.copytree(bert_source / "bert_classifier", MODELS / "bert_classifier", dirs_exist_ok=True)
    if entity_selection["use_candidate"]:
        shutil.copytree(entity_source / "bert_ner", MODELS / "bert_ner", dirs_exist_ok=True)
    selected_ner = "BERT" if bert_val > crf_val else "CRF"
    write_json(MODELS / "ner_selection.json", {"selected": selected_ner, "crf_validation_f1": crf_val,
               "bert_validation_f1": bert_val, "criterion": "strict validation span F1; ties retain faster CRF"})
    tfidf = models["TF-IDF"]
    temperature = fit_temperature(tfidf.predict_proba(val.narrative), val.crime_type, tfidf.classes_)
    write_json(MODELS / "calibration.json", {"temperature": temperature, "threshold": SETTINGS["inference"]["confidence_threshold"], "fit_split": "validation"})
    write_json(MODELS / "accuracy_selection.json", {"classical": selection, "crf": crf_selection,
               "bert_classification": bert_selection, "bert_entities": entity_selection})
    plan = dict(baseline["protocol"])
    plan["integrity"] = split_integrity({name: df[df.split == name] for name in ["train", "validation", "development", "test"]})
    ner_test = entity_split(df, "test", 600)
    plan["entity_integrity"] = split_integrity({"train": ner_train, "validation": ner_val, "test": ner_test})
    plan["accuracy_revision"] = {"selection": "Bounded feature/regularization and BERT schedule comparisons on validation only",
                                 "training_sizes_unchanged": True, "baseline": str(args.baseline),
                                 "selected_document_recipes": {k: v["selected"] for k, v in selection["models"].items()}}
    plan["evaluation_scope"] = "Repeat evaluation on the previously viewed synthetic test split. All candidates and checkpoints were selected on validation, before test prediction. No retuning from this run's test results."
    plan["entity_evaluation_scope"] = "Previously viewed synthetic test subset plus generated role/context cases sharing grammar; diagnostic evaluation, not untouched real-source validation."
    cv = cross_val_score(tfidf, train.narrative.tolist(), train.crime_type, groups=train.template_group.str.replace(r"-layout\d+$", "", regex=True),
                         cv=GroupKFold(3), scoring="f1_macro", n_jobs=1)
    receipt = freeze_models(plan)
    write_json(ARTIFACTS / "evaluation_plan.json", plan)
    print("Validation-selected models frozen. Evaluating existing test splits once...", flush=True)
    test = sample_split(df, "test", 1200)
    if test.report_id.tolist() != pd.read_csv(args.baseline / "artifacts" / "test_ids.csv").report_id.tolist():
        raise ValueError("The document test IDs changed; before/after comparison is invalid")
    old_entity_rows = pd.read_json(args.baseline / "predictions" / "entity_evaluation_predictions.jsonl", lines=True)
    if ner_test.report_id.tolist() != old_entity_rows.report_id.tolist() or ner_test.narrative.tolist() != old_entity_rows.narrative.tolist():
        raise ValueError("The entity evaluation cases changed; before/after comparison is invalid")
    guesses = {name: model.predict(test.narrative) for name, model in models.items()}
    guesses["Word2Vec"] = w2v_classifier.predict(document_vectors(w2v, test.narrative, recipe))
    guesses["BERT"] = predict_bert(bert, tokenizer, test.narrative.tolist())
    predictions = test[["report_id", "crime_type", "template_group"]].copy()
    for name, guessed in guesses.items():
        predictions[name] = guessed
    document_file = PROCESSED / "evaluation_predictions.parquet"
    predictions.to_parquet(document_file, index=False)
    results = {"protocol": plan, "models": {}, "grouped_cv": {"model": "TF-IDF", "fold_macro_f1": cv.tolist(), "mean": float(cv.mean()), "groups": "incident variants within training only"}}
    for name, guessed in guesses.items():
        results["models"][name] = classification_metrics(test.crime_type, guessed)
        if name in selection["models"]:
            results["models"][name]["train_seconds"] = selection["models"][name].get("search_seconds", baseline["models"][name]["train_seconds"])
            results["models"][name]["timing_note"] = "Includes the full bounded candidate search when search_seconds is recorded."
    bert_details = bert_selection["details"] if bert_selection["use_candidate"] else baseline["models"]["BERT"]
    results["models"]["BERT"].update({k: v for k, v in bert_details.items() if k in {"history", "best_validation_macro_f1", "train_seconds", "architecture", "max_tokens", "recipe"}})
    for split, frame in [("development", development), ("test", test)]:
        probabilities = tfidf.predict_proba(frame.narrative)
        results[f"calibration_{split}"] = {"before": calibration_metrics(frame.crime_type, probabilities, tfidf.classes_),
            "after": calibration_metrics(frame.crime_type, temperature_scale(probabilities, temperature), tfidf.classes_), "temperature": temperature}
    write_json(ARTIFACTS / "evaluation.json", results)
    errors = test[["report_id", "crime_type", "template_group"]].copy()
    errors["prediction"], errors["characters"] = guesses["TF-IDF"], test.narrative.str.len()
    errors[errors.crime_type != errors.prediction].to_csv(ARTIFACTS / "classification_errors.csv", index=False)
    tx, ty = sequence_data(ner_test)
    crf_guesses = crf.predict(tx)
    tokens, gold = sequences(ner_test)
    bert_guesses = predict_tags(ner_bert, ner_tokenizer, tokens)[0]
    entity_scores = {"CRF": entity_metrics(ty, crf_guesses), "BERT": entity_metrics(gold, bert_guesses)}
    entity_rows = ner_test[["report_id", "narrative", "entities_json", "template_group"]].copy()
    entity_rows["gold_bio"], entity_rows["crf_bio"], entity_rows["bert_bio"] = ty, list(crf_guesses), bert_guesses
    entity_file = PROCESSED / "entity_evaluation_predictions.jsonl"
    entity_rows.to_json(entity_file, orient="records", lines=True, force_ascii=False)
    training = read(MODELS / "bert_ner" / "training.json")
    write_json(ARTIFACTS / "ner_evaluation.json", {**entity_scores[selected_ner], "model": selected_ner, "labels": ENTITY_LABELS,
               "train_rows": len(ner_train), "validation_rows": len(ner_val), "test_rows": len(ner_test), "models": entity_scores,
               "validation": {"CRF": crf_val, "BERT": bert_val}, "bert_history": training["history"], "split": plan["entity_evaluation_scope"]})
    write_json(ARTIFACTS / "final_evaluation_receipt.json", {"evaluated_at": datetime.now(timezone.utc).isoformat(),
               "frozen_at": receipt["frozen_at"], "selected_ner": selected_ner,
               "prediction_files": {str(p.relative_to(PROCESSED)): file_hash(p) for p in [document_file, entity_file]}, "note": plan["evaluation_scope"]})
    comparison = {"scope": plan["evaluation_scope"], "same_document_test_ids": True, "document_rows": len(test), "entity_rows": len(ner_test),
                  "classification": {}, "entities": {}}
    for name, current in results["models"].items():
        previous = baseline["models"][name]
        comparison["classification"][name] = {"before_accuracy": previous["accuracy"], "after_accuracy": current["accuracy"],
            "before_macro_f1": previous["macro_f1"], "after_macro_f1": current["macro_f1"], "accuracy_change_pp": 100 * (current["accuracy"] - previous["accuracy"])}
    for name, current in entity_scores.items():
        previous = baseline_ner["models"][name]["strict_entity_f1"]
        comparison["entities"][name] = {"before_span_f1": previous, "after_span_f1": current["strict_entity_f1"], "change_pp": 100 * (current["strict_entity_f1"] - previous)}
    write_json(ARTIFACTS / "accuracy_improvements.json", comparison)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
