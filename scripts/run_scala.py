"""Use the Scala compiler shipped with PySpark; no additional Scala install required."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import ROOT, ARTIFACTS, PROCESSED, write_json
from crime_nlp.scala_runner import submit_scala


def main():
    log_path = ROOT / "logs" / "scala.log"
    submit_scala("CrimeAnalytics", [PROCESSED / "reports"], log_path=log_path)
    match = re.search(r"SCALA_VALID_REPORTS=(\d+)", log_path.read_text(encoding="utf-8", errors="replace"))
    actual = int(match[1]) if match else None
    expected = json.loads((ARTIFACTS / "spark_metrics.json").read_text(encoding="utf-8"))["clean_rows"]
    verified = actual == expected
    write_json(ARTIFACTS / "scala_verification.json", {"exit_code": 0, "records_verified": verified,
               "rows": actual, "expected_rows": expected, "temporary_jar_cleanup_warning": "Exception while deleting Spark temp dir" in log_path.read_text(encoding="utf-8", errors="replace")})
    if not verified:
        raise RuntimeError(f"Scala verification failed: {actual} reports, expected {expected}. See logs/scala.log.")
    print(f"SCALA_VALID_REPORTS={actual}; verified against the current Spark corpus.")


if __name__ == "__main__":
    main()
