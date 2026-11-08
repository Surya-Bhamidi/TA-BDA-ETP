"""Fit exact-span reliability to synthetic validation, without changing labels."""
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from crime_nlp.config import ARTIFACTS, PROCESSED, MODELS, ENTITY_LABELS, write_json
from crime_nlp.text import tokens_with_offsets, token_features, spans_from_bio
from crime_nlp.ner_model import load_ner, extract_bert
from crime_nlp.train import file_hash

validation = pd.read_json(PROCESSED / "annotation_v4" / "validation.jsonl", lines=True)
results = {}
for name in ["CRF", "BERT"]:
    model = joblib.load(MODELS / "crf.joblib") if name == "CRF" else load_ner()
    examples = {label: [] for label in ENTITY_LABELS}
    for row in validation.itertuples():
        gold = {(entity["start"], entity["end"], entity["label"]) for entity in json.loads(row.entities_json)}
        if name == "CRF":
            tokens = tokens_with_offsets(row.narrative)
            features = token_features(tokens)
            tags = model.predict_single(features)
            marginal = model.predict_marginals_single(features)
            entities = spans_from_bio(row.narrative, tokens, tags)
            for entity in entities:
                entity["confidence"] = float(np.mean([marginal[index].get(tags[index], 0) for index, token in enumerate(tokens) if entity["start"] <= token["start"] < entity["end"]]))
        else:
            entities = extract_bert(row.narrative, *model)
        for entity in entities:
            examples[entity["label"]].append((entity["confidence"], int((entity["start"], entity["end"], entity["label"]) in gold)))
    results[name] = {}
    for label, pairs in examples.items():
        if not pairs:
            continue
        scores, truth = np.asarray(pairs).T
        truth = truth.astype(int)
        raw = np.clip(scores, 1e-6, 1 - 1e-6)
        x = np.log(raw / (1 - raw))[:, None]
        crossfit = None
        if len(set(truth)) > 1:
            fitted = LogisticRegression(C=1., random_state=42).fit(x, truth)
            coefficient, intercept = float(fitted.coef_[0, 0]), float(fitted.intercept_[0])
            if np.bincount(truth).min() >= 3:
                crossfit = cross_val_predict(LogisticRegression(C=1., random_state=42), x, truth, cv=StratifiedKFold(3, shuffle=True, random_state=42), method="predict_proba")[:, 1]
        else:
            mean = (truth.sum() + .5) / (len(truth) + 1)
            coefficient, intercept = 0., float(np.log(mean / (1 - mean)))
        results[name][label] = {"coefficient": coefficient, "intercept": intercept, "validation_spans": len(pairs),
                               "empirical_precision": float(truth.mean()), "raw_brier": float(brier_score_loss(truth, scores)),
                               "crossfit_brier": float(brier_score_loss(truth, crossfit)) if crossfit is not None else None}
    print(f"Calibrated {name} validation spans", flush=True)
write_json(ARTIFACTS / "core_entity_calibration.json", {"fitted_at": datetime.now(timezone.utc).isoformat(), "models": results,
           "fit_split": "Synthetic schema-4 validation only", "legacy_core_sha256": {"CRF": file_hash(MODELS / "crf.joblib"), "BERT": file_hash(MODELS / "bert_ner" / "model.safetensors")},
           "scope": "Raw span correctness on synthetic validation; fitting does not change model weights or labels. Missed entities and real-world calibration are not measured; one-class labels use a smoothed empirical constant."})
