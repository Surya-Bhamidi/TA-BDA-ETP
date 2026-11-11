"""Compare Scala and PySpark accuracy, predictions, output integrity and local runtime.

All rebuilt Spark outputs and fresh predictions stay under .runtime. No NLP
weights, production data or historical benchmark predictions are replaced.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crime_nlp.config import ARTIFACTS, MODELS, PROCESSED, RAW, SETTINGS, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def worker(engine, output):
    from crime_nlp import spark_pipeline
    spark_pipeline.PROCESSED = output / "processed"
    spark_pipeline.ARTIFACTS = output / "artifacts"
    spark_pipeline.PROCESSED.mkdir(parents=True)
    spark_pipeline.ARTIFACTS.mkdir(parents=True)
    spark_pipeline.run_spark(master="local[2]", raw_path=RAW, engine=engine)


def compare_spark_predictions(pairs):
    import numpy as np
    from pyspark.ml import PipelineModel
    from crime_nlp.spark_pipeline import create_spark
    spark = create_spark("local[2]")
    results = []
    try:
        for scala_dir, python_dir in pairs:
            frame = spark.read.parquet(str(scala_dir / "processed/reports")).filter("split = 'test'")
            predictions = []
            for output in (scala_dir, python_dir):
                java_model = spark._jvm.org.apache.spark.ml.PipelineModel.load(str(output / "artifacts/spark_mllib_model"))
                model = PipelineModel._from_java(java_model)
                scored = model.transform(frame).select("report_id", "label", "prediction", "probability").orderBy("report_id").collect()
                predictions.append(scored)
            left, right = predictions
            assert [r.report_id for r in left] == [r.report_id for r in right]
            assert [r.label for r in left] == [r.label for r in right]
            differing = sum(a.prediction != b.prediction for a, b in zip(left, right))
            delta = float(np.max(np.abs(np.asarray([list(r.probability) for r in left]) -
                                        np.asarray([list(r.probability) for r in right]))))
            results.append({"test_rows": len(left), "prediction_disagreements": differing,
                            "max_absolute_probability_difference": delta})
            assert differing == 0, f"MLlib labels differ: {results[-1]}"
            assert delta < 1e-8, f"MLlib probability drift: {results[-1]}"
    finally:
        spark.stop()
    return results


def fresh_nlp_evaluation(scala_reports, python_reports, output):
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from gensim.models import Word2Vec
    from sklearn.metrics import accuracy_score, f1_score
    from threadpoolctl import threadpool_limits
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from crime_nlp.bert_model import predict_bert
    from crime_nlp.evaluation import entity_metrics
    from crime_nlp.ner_model import load_ner, predict_tags, sequences
    from crime_nlp.train import document_vectors, file_hash, sequence_data
    threadpool_limits(4)
    torch.set_num_threads(4)
    frozen = read(ARTIFACTS / "frozen_models.json")["model_sha256"]
    for name, expected in frozen.items():
        assert file_hash(MODELS / name) == expected, f"Frozen NLP model changed: {name}"
    ids = pd.read_csv(ARTIFACTS / "test_ids.csv").report_id.tolist()
    frames = [pd.read_parquet(path).set_index("report_id").loc[ids].reset_index()
              for path in (scala_reports, python_reports)]
    pd.testing.assert_frame_equal(frames[0], frames[1], check_exact=True)
    texts = [frame.narrative.tolist() for frame in frames]
    truth = frames[0].crime_type.tolist()
    saved_predictions = pd.read_parquet(PROCESSED / "evaluation_predictions.parquet").set_index("report_id").loc[ids]
    baseline = read(ARTIFACTS / "evaluation.json")["models"]
    metrics = {}
    fresh = pd.DataFrame({"report_id": ids, "crime_type": truth})

    def score(name, predictor, expected, saved_column=None):
        print(f"Fresh NLP inference: {name}, {len(ids)} reports per backend", flush=True)
        predicted = [list(predictor(part)) for part in texts]
        disagreements = sum(a != b for a, b in zip(*predicted))
        assert disagreements == 0, f"{name}: Scala/PySpark prediction mismatch"
        actual = {"accuracy": float(accuracy_score(truth, predicted[0])),
                  "macro_f1": float(f1_score(truth, predicted[0], average="macro", zero_division=0)),
                  "weighted_f1": float(f1_score(truth, predicted[0], average="weighted", zero_division=0))}
        delta = {key: actual[key] - expected[key] for key in actual}
        metrics[name] = {"rows": len(ids), "baseline": {key: expected[key] for key in actual},
                         "scala": actual, "pyspark": actual.copy(), "change_from_baseline": delta,
                         "backend_prediction_disagreements": disagreements}
        if saved_column:
            metrics[name]["disagreements_with_saved_predictions"] = int(np.sum(np.asarray(predicted[0]) != saved_predictions[saved_column].to_numpy()))
        fresh[name] = predicted[0]

    for name, filename in [("Bag of Words", "bow.joblib"), ("TF-IDF", "tfidf.joblib")]:
        model = joblib.load(MODELS / filename)
        score(name, model.predict, baseline[name], name)
    vectors = Word2Vec.load(str(MODELS / "word2vec.model"))
    classifier = joblib.load(MODELS / "word2vec_classifier.joblib")
    recipe = read(MODELS / "word2vec_features.json")
    score("Word2Vec", lambda part: classifier.predict(document_vectors(vectors, part, recipe)), baseline["Word2Vec"], "Word2Vec")
    model = AutoModelForSequenceClassification.from_pretrained(MODELS / "bert_classifier", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODELS / "bert_classifier", local_files_only=True)
    score("BERT (benchmark)", lambda part: predict_bert(model, tokenizer, part, full_text=False), baseline["BERT"], "BERT")
    live = read(ARTIFACTS / "active_document_model.json")
    live_path = ARTIFACTS / live["directory"]
    if live_path.resolve() != (MODELS / "bert_classifier").resolve():
        model = AutoModelForSequenceClassification.from_pretrained(live_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(live_path, local_files_only=True)
    score("BERT (full document)", lambda part: predict_bert(model, tokenizer, part, full_text=live.get("full_text", True)),
          read(ARTIFACTS / "windowed_classification_evaluation.json")["full_document"])
    fresh.to_parquet(output / "fresh_document_predictions.parquet", index=False)

    print("Fresh NER inference: CRF and BERT, saved 600-case evaluation", flush=True)
    entities = pd.read_json(PROCESSED / "entity_evaluation_predictions.jsonl", lines=True)
    features, gold = sequence_data(entities)
    token_lists, bert_gold = sequences(entities)
    assert gold == bert_gold == entities.gold_bio.tolist()
    crf = joblib.load(MODELS / "crf.joblib")
    bert, entity_tokenizer = load_ner()
    ner_predictions = {"CRF": list(crf.predict(features)), "BERT": predict_tags(bert, entity_tokenizer, token_lists)[0]}
    ner_baseline = read(ARTIFACTS / "ner_evaluation.json")["models"]
    ner = {}
    for name, predicted in ner_predictions.items():
        value = entity_metrics(gold, predicted)["strict_entity_f1"]
        expected = ner_baseline[name]["strict_entity_f1"]
        prior = entities["crf_bio" if name == "CRF" else "bert_bio"].tolist()
        ner[name] = {"rows": len(entities), "strict_entity_f1": value, "baseline_strict_entity_f1": expected,
                     "change_from_baseline": value - expected,
                     "sequences_different_from_saved": sum(list(a) != list(b) for a, b in zip(predicted, prior)),
                     "scope": "Fresh inference on unchanged saved entity cases, including augmented synthetic cases; not two separate NER training runs."}
        entities[name + "_fresh"] = predicted
    entities.to_json(output / "fresh_entity_predictions.jsonl", orient="records", lines=True, force_ascii=False)
    for name, expected in frozen.items():
        assert file_hash(MODELS / name) == expected, f"NLP model changed during evaluation: {name}"
    return {"documents": metrics, "entities": ner, "frozen_model_files_verified": len(frozen),
            "scope": "Fresh inference with retained weights, not NLP retraining. Previously viewed synthetic diagnostic test sets."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker-engine", choices=["scala", "pyspark"], help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_engine:
        worker(args.worker_engine, args.output)
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    from scripts.verify_scala_pipeline import compare_reports, compare_trends
    target = ROOT / ".runtime" / "spark-comparison" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    target.mkdir(parents=True)
    receipt_path = ARTIFACTS / "spark_backend_comparison.json"
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "directory": target.relative_to(ROOT).as_posix(), "spark_version": "3.5.5", "master": "local[2]",
              "partitions": SETTINGS["spark"]["partitions"], "driver_memory": SETTINGS["spark"]["driver_memory"],
              "repeats": args.repeats, "runs": [], "checks": []}
    try:
        pairs = []
        for index in range(args.repeats):
            outputs = {}
            order = ("pyspark", "scala") if index % 2 == 0 else ("scala", "pyspark")
            for engine in order:
                output = target / f"round-{index + 1}-{engine}"
                output.mkdir()
                print(f"Full-corpus round {index + 1}/{args.repeats}: {engine}", flush=True)
                started = time.perf_counter()
                with (output / "process.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-engine", engine,
                                             "--output", str(output)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                elapsed = time.perf_counter() - started
                if result.returncode:
                    raise RuntimeError(f"{engine} failed (exit {result.returncode}); see {output / 'process.log'}")
                metrics = read(output / "artifacts/spark_metrics.json")
                report["runs"].append({"round": index + 1, "backend": engine, "wall_seconds": round(elapsed, 3),
                                       "pipeline_metrics": metrics, "directory": output.relative_to(ROOT).as_posix()})
                write_json(receipt_path, report)
                outputs[engine] = output
            compare_reports(outputs["scala"] / "processed/reports", outputs["pyspark"] / "processed/reports")
            compare_reports(outputs["scala"] / "processed/reports", PROCESSED / "reports")
            compare_trends(outputs["scala"] / "processed/trends.parquet", outputs["pyspark"] / "processed/trends.parquet")
            assert read(outputs["scala"] / "artifacts/data_quality.json") == read(outputs["pyspark"] / "artifacts/data_quality.json")
            left, right = [read(outputs[engine] / "artifacts/spark_metrics.json") for engine in ("scala", "pyspark")]
            for key in ("raw_rows", "clean_rows", "invalid_rows_removed", "duplicates_removed", "input_partitions", "shuffle_partitions"):
                assert left[key] == right[key], key
            for key in ("accuracy", "weighted_f1"):
                assert abs(left["mllib"][key] - right["mllib"][key]) < 1e-12, key
            pairs.append((outputs["scala"], outputs["pyspark"]))
        report["checks"] += ["all cleaned columns match between both backends and deployed corpus in every round",
                              "SQL trends and data quality match", "full-corpus MLlib accuracy and weighted F1 parity"]
        report["spark_predictions"] = compare_spark_predictions(pairs)
        report["checks"].append("all held-out MLlib labels and probability vectors compared in every round")
        report["nlp"] = fresh_nlp_evaluation(pairs[-1][0] / "processed/reports", pairs[-1][1] / "processed/reports", target)
        report["timing"] = {}
        for engine in ("pyspark", "scala"):
            times = [r["wall_seconds"] for r in report["runs"] if r["backend"] == engine]
            report["timing"][engine] = {"wall_seconds": times, "median_seconds": statistics.median(times),
                                         "minimum_seconds": min(times), "maximum_seconds": max(times)}
        report["timing"]["scala_wall_time_change_percent"] = (
            report["timing"]["scala"]["median_seconds"] / report["timing"]["pyspark"]["median_seconds"] - 1) * 100
        report["timing"]["scope"] = ("Sequential fresh-process local runs with alternating order; end-to-end time includes JVM startup, "
                                        "Scala compilation on each Scala run, ETL, MLlib fit/evaluation, writing outputs and process exit. "
                                        "OS caches are uncontrolled; this is not a multi-host or statistically powered performance benchmark.")
        nlp_changes = [value for m in report["nlp"]["documents"].values() for value in m["change_from_baseline"].values()]
        nlp_changes += [m["change_from_baseline"] for m in report["nlp"]["entities"].values()]
        report["accuracy_regression_detected"] = any(value < -1e-12 for value in nlp_changes)
        report["accuracy_improvement_detected"] = any(value > 1e-12 for value in nlp_changes)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(receipt_path, report)
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(report["status"] != "passed")


if __name__ == "__main__":
    main()
