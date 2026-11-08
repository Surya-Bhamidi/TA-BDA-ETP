"""Fit validation-only span reliability, then freeze it before diagnostic scoring."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from crime_nlp.config import ARTIFACTS, PROCESSED, write_json
from crime_nlp.trained_roles import predict_roles
from crime_nlp.train import file_hash

parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=Path, required=True)
args = parser.parse_args()
selection = json.loads((args.directory / "ner_selection.json").read_text(encoding="utf-8"))
settings = {"path": args.directory, "selected": selection["selected"]}
validation = pd.read_json(PROCESSED / "annotation_v4" / "validation.jsonl", lines=True)
examples = {label: [] for label in ["ORGANIZATION", "WITNESS"]}
for row in validation.itertuples():
    gold = {(entity["start"], entity["end"], entity["label"]) for entity in json.loads(row.entities_json)}
    for entity in predict_roles(row.narrative, settings):
        examples[entity["label"]].append((entity["confidence"], int((entity["start"], entity["end"], entity["label"]) in gold)))
models = {}
for label, values in examples.items():
    if not values:
        continue
    scores, truth = np.asarray(values).T
    truth = truth.astype(int)
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    x = np.log(clipped / (1 - clipped))[:, None]
    crossfit = None
    if len(set(truth)) == 2:
        estimator = LogisticRegression(C=1., random_state=42).fit(x, truth)
        coefficient, intercept = float(estimator.coef_[0, 0]), float(estimator.intercept_[0])
        if np.bincount(truth).min() >= 3:
            crossfit = cross_val_predict(LogisticRegression(C=1., random_state=42), x, truth,
                                         cv=StratifiedKFold(3, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    else:
        mean = (truth.sum() + .5) / (len(truth) + 1)
        coefficient, intercept = 0., float(np.log(mean / (1 - mean)))
    models[label] = {"coefficient": coefficient, "intercept": intercept, "validation_spans": len(values),
                     "empirical_precision": float(truth.mean()), "raw_brier": float(brier_score_loss(truth, scores)),
                     "crossfit_brier": float(brier_score_loss(truth, crossfit)) if crossfit is not None else None,
                     "scope": "Span correctness on synthetic validation; cannot estimate missed entities or guarantee real-source calibration"}
path = args.directory / "role_calibration.json"
write_json(path, {"fitted_at": datetime.now(timezone.utc).isoformat(), "fit_split": "synthetic validation only", "models": models})
write_json(ARTIFACTS / "role_calibration_receipt.json", {"path": str(path), "sha256": file_hash(path), "models": models,
           "scope": "Validation-only reliability model; no claim of real-data calibration. One-class labels use a smoothed empirical constant."})
print({label: {key: value for key, value in result.items() if key in {"validation_spans", "empirical_precision", "raw_brier", "crossfit_brier"}} for label, result in models.items()})
