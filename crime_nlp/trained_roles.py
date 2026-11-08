"""Optional validation-selected role specialist, separate from legacy core NER."""
import json
import re
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
from .config import ARTIFACTS
from .text import tokens_with_offsets, token_features, spans_from_bio

ACTIVE = ARTIFACTS / "active_entity_roles.json"
GENERIC = {"a", "an", "the", "witness", "eyewitness", "bystander", "observer", "police", "officer", "incident", "it", "someone", "they", "he", "she"}


def contract():
    if not ACTIVE.exists():
        return None
    settings = json.loads(ACTIVE.read_text(encoding="utf-8"))
    path = (ARTIFACTS / settings["directory"]).resolve()
    if not path.is_relative_to((ARTIFACTS / "models").resolve()):
        raise ValueError("Role model directory must be inside project model artifacts")
    expected = path / "crf.joblib" if settings["selected"] == "CRF" else path / "bert_ner" / "config.json"
    if not expected.exists():
        return None
    return {**settings, "path": path}


def revision_paths():
    settings = contract()
    if settings is None:
        return []
    paths = [ACTIVE]
    if settings["selected"] == "CRF":
        paths.append(settings["path"] / "crf.joblib")
    else:
        paths.extend(path for path in (settings["path"] / "bert_ner").glob("*") if path.is_file())
    calibration = settings["path"] / "role_calibration.json"
    if calibration.exists():
        paths.append(calibration)
    return [path for path in paths if path.exists()]


@lru_cache(maxsize=2)
def _model(path, choice, stamp):
    if choice == "CRF":
        return joblib.load(Path(path) / "crf.joblib")
    from .ner_model import load_ner
    return load_ner(Path(path) / "bert_ner")


def predict_roles(text, settings=None):
    settings = settings or contract()
    if settings is None:
        return []
    path = Path(settings["path"])
    model_file = path / "crf.joblib" if settings["selected"] == "CRF" else path / "bert_ner" / "config.json"
    model = _model(str(path), settings["selected"], model_file.stat().st_mtime_ns)
    if settings["selected"] == "CRF":
        tokens = tokens_with_offsets(text)
        features = token_features(tokens)
        tags = model.predict_single(features)
        marginal = model.predict_marginals_single(features)
        entities = spans_from_bio(text, tokens, tags)
        for entity in entities:
            values = [marginal[i].get(tags[i], 0.) for i, token in enumerate(tokens) if entity["start"] <= token["start"] < entity["end"]]
            entity["confidence"] = float(np.mean(values))
    else:
        from .ner_model import extract_bert
        entities = extract_bert(text, *model)
    calibration_path = path / "role_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {}
    for entity in entities:
        entity["trained_model"] = settings["selected"] + " role specialist v4"
        entity["confidence_status"] = "Uncalibrated model score"
        if calibration:
            values = calibration["models"].get(entity["label"])
            if values:
                raw = np.clip(entity["confidence"], 1e-6, 1 - 1e-6)
                entity["calibrated_confidence"] = float(1 / (1 + np.exp(-np.clip(values["coefficient"] * np.log(raw / (1 - raw)) + values["intercept"], -40, 40))))
                entity["confidence_status"] = "Calibrated on synthetic validation spans; not a real-world guarantee"
    return entities


def merge_roles(text, entities, witnesses, learned):
    entities, witnesses = [dict(entity) for entity in entities], [dict(role) for role in witnesses]
    for entity in learned:
        if entity["label"] not in {"ORGANIZATION", "WITNESS"}:
            continue
        from .material_context import MATERIAL
        if entity["label"] == "ORGANIZATION" and MATERIAL.fullmatch(entity["text"]):
            continue
        if entity["confidence"] < .55:
            continue
        first, last = entity["start"], entity["end"]
        overlaps = lambda other: other["start"] < last and other["end"] > first
        if entity["label"] == "WITNESS":
            words = set(re.findall(r"\w+", entity["text"].lower()))
            if not words - GENERIC:
                continue
            exact = next((role for role in witnesses if role["start"] == first and role["end"] == last), None)
            if exact is not None:
                witnesses[witnesses.index(exact)] = dict(entity)
            elif not any(overlaps(other) for other in entities + witnesses):
                witnesses.append(dict(entity))
        elif not any(overlaps(other) for other in entities + witnesses):
            entities.append(dict(entity))
    return sorted(entities, key=lambda entity: entity["start"]), sorted(witnesses, key=lambda entity: entity["start"])
