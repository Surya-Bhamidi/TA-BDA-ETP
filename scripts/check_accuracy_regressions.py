"""Exercise validation-selected classifiers against existing development regressions.

Only this process's model cache is changed; active model files stay untouched.
These examples are development checks, not an independent benchmark.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pytest
from crime_nlp.inference import models


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical", type=Path, required=True)
    args = parser.parse_args()
    resources = models()
    resources["classifier"] = joblib.load(args.classical / "tfidf.joblib")
    resources["crf"] = joblib.load(args.classical / "crf.joblib")
    raise SystemExit(pytest.main(["tests/test_entity_context.py", "tests/test_robustness.py", "-q"]))
