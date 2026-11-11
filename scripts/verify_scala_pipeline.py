"""Verify Scala/PySpark equivalence without rebuilding the deployed data or NLP models.

Use --full to additionally run Scala on the full corpus and compare every output
row against the deployed Parquet dataset. Outputs stay in .runtime/scala-verification.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import ROOT, RAW, ARTIFACTS, PROCESSED, write_json
from crime_nlp.generate import make_report
from crime_nlp import spark_pipeline


def compare_reports(left, right):
    import pandas as pd
    def normalized(path):
        frame = pd.read_parquet(path)
        frame["split"] = frame["split"].astype(str)
        return frame.sort_values("report_id").reset_index(drop=True).sort_index(axis=1)
    pd.testing.assert_frame_equal(normalized(left), normalized(right), check_exact=True)


def compare_trends(left, right):
    import pandas as pd
    keys = ["month", "crime_type", "district"]
    def normalized(path):
        return pd.read_parquet(path).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(normalized(left), normalized(right), check_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    target = ROOT / ".runtime" / "scala-verification" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    target.mkdir(parents=True)
    receipt = {"status": "failed", "verified_at": datetime.now(timezone.utc).isoformat(),
               "directory": target.relative_to(ROOT).as_posix(), "checks": []}
    original_processed, original_artifacts = spark_pipeline.PROCESSED, spark_pipeline.ARTIFACTS
    def execute(engine, source, name):
        spark_pipeline.PROCESSED = target / name / "processed"
        spark_pipeline.ARTIFACTS = target / name / "artifacts"
        spark_pipeline.PROCESSED.mkdir(parents=True)
        spark_pipeline.ARTIFACTS.mkdir(parents=True)
        print(f"Verifying {name} with {engine}...", flush=True)
        return spark_pipeline.run_spark(raw_path=source, engine=engine)
    try:
        rng = random.Random(42)
        rows = [make_report(i, rng) for i in range(128)]
        unicode_row = {**rows[0], "report_id": "unicode-case", "narrative": "José García reported a stolen téléphone near Nairobi."}
        dirty = [{**rows[0], "report_id": "000-canonical"},
                 {**rows[1], "report_id": "null", "narrative": None},
                 {**rows[2], "report_id": "blank", "narrative": " \t "},
                 {**rows[3], "report_id": "date", "reported_at": "invalid-date"},
                 {**rows[4], "report_id": "category", "crime_type": "Unknown"},
                 {**rows[5], "report_id": "split", "split": "unknown"},
                 {**rows[6], "report_id": "punctuation", "narrative": "!!!"},
                 {**rows[7], "report_id": "entities", "entities_json": None}]
        source = target / "fixture.jsonl"
        source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows + [unicode_row] + dirty), encoding="utf-8")
        scala = execute("scala", source, "scala")
        python = execute("pyspark", source, "pyspark")
        assert scala["language"] == "Scala" and scala["spark_version"] == "3.5.5"
        assert scala["raw_rows"] == 137 and scala["clean_rows"] == 129
        assert scala["invalid_rows_removed"] == 7 and scala["duplicates_removed"] == 1
        compare_reports(target / "scala/processed/reports", target / "pyspark/processed/reports")
        compare_trends(target / "scala/processed/trends.parquet", target / "pyspark/processed/trends.parquet")
        import pandas as pd
        frame = pd.read_parquet(target / "scala/processed/reports")
        expected = {r["report_id"]: r["narrative"] for r in rows[1:] + [unicode_row, dirty[0]]}
        assert dict(zip(frame.report_id, frame.narrative)) == expected
        left_quality = json.loads((target / "scala/artifacts/data_quality.json").read_text(encoding="utf-8"))
        right_quality = json.loads((target / "pyspark/artifacts/data_quality.json").read_text(encoding="utf-8"))
        assert left_quality == right_quality
        for key in ("accuracy", "weighted_f1"):
            assert abs(scala["mllib"][key] - python["mllib"][key]) < 1e-10
        receipt["checks"] += ["exact Scala/PySpark cleaned-row equivalence", "seven invalid-row cases",
                              "deterministic duplicate survivor", "original Unicode narratives and entity JSON preserved",
                              "identical SQL trends and quality reports", "MLlib held-out metric parity"]
        # Prove a model fitted and saved by Scala can be consumed through PySpark.
        spark = spark_pipeline.create_spark()
        try:
            from pyspark.ml import PipelineModel
            model = PipelineModel._from_java(spark._jvm.org.apache.spark.ml.PipelineModel.load(str(target / "scala/artifacts/spark_mllib_model")))
            predictions = model.transform(spark.read.parquet(str(target / "scala/processed/reports")))
            scored = predictions.select("prediction", "probability").collect()
            assert len(scored) == 129
            assert all(0 <= r.prediction < 8 and abs(sum(r.probability) - 1) < 1e-6 for r in scored)
        finally:
            spark.stop()
        receipt["checks"].append("Scala-saved MLlib model loaded and scored through PySpark")
        receipt["fixture"] = scala
        if args.full:
            full = execute("scala", RAW, "full")
            compare_reports(target / "full/processed/reports", PROCESSED / "reports")
            compare_trends(target / "full/processed/trends.parquet", PROCESSED / "trends.parquet")
            deployed = json.loads((ARTIFACTS / "spark_metrics.json").read_text(encoding="utf-8"))
            for key in ("raw_rows", "invalid_rows_removed", "duplicates_removed", "clean_rows"):
                assert full[key] == deployed[key]
            receipt["full_corpus"] = full
            receipt["checks"].append("every full-corpus row and SQL trend matches deployed data")
        receipt["status"] = "passed"
    except Exception as error:
        receipt.update(error=str(error), traceback=traceback.format_exc())
    finally:
        spark_pipeline.PROCESSED, spark_pipeline.ARTIFACTS = original_processed, original_artifacts
    write_json(ARTIFACTS / "scala_migration_verification.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    raise SystemExit(receipt["status"] != "passed")


if __name__ == "__main__":
    main()
