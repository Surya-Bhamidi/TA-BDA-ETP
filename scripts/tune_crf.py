"""Compare CRF regularization using only the entity validation split."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
import sklearn_crfsuite
from crime_nlp.config import MODELS, PROCESSED, write_json
from crime_nlp.train import entity_split, sequence_data
from crime_nlp.evaluation import entity_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(PROCESSED / "reports")
    train, val = entity_split(frame, "train", 8000), entity_split(frame, "validation", 400)
    x, y = sequence_data(train)
    vx, vy = sequence_data(val)
    best_model = joblib.load(MODELS / "crf.joblib")
    baseline = entity_metrics(vy, best_model.predict(vx))["strict_entity_f1"]
    best = {"name": "retained baseline", "validation_f1": baseline}
    trials = [dict(best)]
    for c1, c2 in [(.03, .03), (.01, .1), (.1, .01)]:
        model = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=150, all_possible_transitions=True).fit(x, y)
        score = entity_metrics(vy, model.predict(vx))["strict_entity_f1"]
        trial = {"name": f"c1={c1}, c2={c2}, max_iterations=150", "validation_f1": score}
        trials.append(trial)
        if score > best["validation_f1"] + 1e-12:
            best_model, best = model, trial
        print(f"CRF {trial['name']}: validation F1={score:.4f}", flush=True)
    joblib.dump(best_model, args.output / "crf.joblib")
    write_json(args.output / "crf_selection.json", {"baseline_f1": baseline, "selected": best, "trials": trials, "selection_split": "validation"})
