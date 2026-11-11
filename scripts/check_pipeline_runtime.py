"""Run Spark cleaning, SQL, Parquet and MLlib on an isolated synthetic fixture.

Production reports, models and evaluation outputs are only read. All pipeline
writes are redirected to a fresh directory below .runtime/feature-audit.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crime_nlp.config import ARTIFACTS, PROCESSED, write_json
from crime_nlp.generate import make_report
from crime_nlp import spark_pipeline


def main():
    started = datetime.now(timezone.utc)
    target = ROOT / ".runtime" / "feature-audit" / ("pipeline-" + started.strftime("%Y%m%dT%H%M%S%f"))
    target.mkdir(parents=True)
    for directory in ["artifacts", "processed"]:
        (target / directory).mkdir()
    rng = random.Random(42)
    clean_rows = [make_report(i, rng) for i in range(128)]
    dirty = [dict(clean_rows[0]), {**clean_rows[1], "report_id": "dirty-null", "narrative": None},
             {**clean_rows[2], "report_id": "dirty-blank", "narrative": " \t "},
             {**clean_rows[3], "report_id": "dirty-date", "reported_at": "invalid-date"}]
    source = target / "raw.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in clean_rows + dirty), encoding="utf-8")
    receipt = {"verified_at": started.isoformat(), "status": "failed", "fixture_directory": target.relative_to(ROOT).as_posix()}
    # run_spark uses module-local output roots. Redirect both before executing.
    spark_pipeline.PROCESSED = target / "processed"
    spark_pipeline.ARTIFACTS = target / "artifacts"
    try:
        report = spark_pipeline.run_spark(raw_path=source)
        assert report["raw_rows"] == 132
        assert report["invalid_rows_removed"] == 3
        assert report["duplicates_removed"] == 1
        assert report["clean_rows"] == 128
        import pandas as pd
        saved = pd.read_parquet(target / "processed" / "reports")
        expected = {row["report_id"]: row["narrative"] for row in clean_rows}
        assert dict(zip(saved.report_id, saved.narrative)) == expected
        assert set(saved.split) == {"train", "validation", "development", "test"}
        assert pd.read_parquet(target / "processed" / "trends.parquet").reports.sum() == 128
        assert 0 <= report["mllib"]["accuracy"] <= 1
        spark = spark_pipeline.create_spark()
        try:
            from pyspark.ml import PipelineModel
            # Score with the deployed model as well as the fixture-trained model.
            # Every deployed stage is a Java MLlib stage. Load through its JVM
            # reader, avoiding Python-RDD metadata deserialization on Windows.
            deployed = PipelineModel._from_java(spark._jvm.org.apache.spark.ml.PipelineModel.load(str(ARTIFACTS / "spark_mllib_model")))
            frame = spark.read.parquet(str(PROCESSED / "reports")).filter("split = 'test'").limit(32)
            predictions = deployed.transform(frame).select("report_id", "prediction", "probability").collect()
            assert len(predictions) == 32
            assert all(0 <= row.prediction < 8 and abs(sum(row.probability) - 1) < 1e-6 for row in predictions)
            receipt["deployed_mllib_live_predictions"] = len(predictions)
        finally:
            spark.stop()
        receipt.update(status="passed", fixture_pipeline=report,
                       checks=["raw JSONL ingestion", "invalid-row removal", "deterministic deduplication",
                               "exact original-text preservation", "partitioned Parquet read/write", "SQL aggregates",
                               "MLlib pipeline fit and held-out scoring", "deployed MLlib model loading and scoring"])
    except Exception as error:
        receipt.update(error=str(error), traceback=traceback.format_exc())
    write_json(ARTIFACTS / "feature_audit_pipeline.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    raise SystemExit(receipt["status"] != "passed")


if __name__ == "__main__":
    main()
