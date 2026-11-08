"""Run validation-only classical experiments without replacing active models."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from threadpoolctl import threadpool_limits
from crime_nlp.config import MODELS, PROCESSED
from crime_nlp.train import sample_split
from crime_nlp.tuning import tune_documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    threadpool_limits(4)
    frame = pd.read_parquet(PROCESSED / "reports")
    tune_documents(sample_split(frame, "train", 12000), sample_split(frame, "validation", 800), args.output, MODELS)
