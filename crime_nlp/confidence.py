"""Validation-only reliability for learned core spans; rules receive no score."""
import json
import hashlib
from functools import lru_cache
import numpy as np
from .config import ARTIFACTS, MODELS

PATH = ARTIFACTS / "core_entity_calibration.json"


@lru_cache(maxsize=1)
def _read(stamp):
    return json.loads(PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _weight_hash(path, stamp, size):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate_core(entities, model):
    if not PATH.exists():
        return entities
    calibration = _read(PATH.stat().st_mtime_ns)
    weight = MODELS / "crf.joblib" if model == "CRF" else MODELS / "bert_ner" / "model.safetensors"
    if not weight.exists() or calibration.get("legacy_core_sha256", {}).get(model) != _weight_hash(weight, weight.stat().st_mtime_ns, weight.stat().st_size):
        return entities
    values = calibration["models"].get(model, {})
    for entity in entities:
        recipe = values.get(entity["label"])
        if recipe is None or entity.get("confidence") is None or entity.get("trained_model"):
            continue
        raw = np.clip(entity["confidence"], 1e-6, 1 - 1e-6)
        logit = recipe["coefficient"] * np.log(raw / (1 - raw)) + recipe["intercept"]
        entity["calibrated_confidence"] = float(1 / (1 + np.exp(-np.clip(logit, -40, 40))))
        entity["confidence_status"] = "Calibrated on synthetic validation spans; not a real-world guarantee"
    return entities
