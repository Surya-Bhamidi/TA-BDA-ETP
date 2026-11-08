"""Offline narrative analysis, model loading and safe entity highlighting."""
import html
import joblib
import json
import numpy as np
from pathlib import Path
from functools import lru_cache

from .config import MODELS, RUNTIME, SETTINGS, ARTIFACTS
from .text import tokens_with_offsets, token_features, spans_from_bio, summarize, threat_language

COLORS = {"SUSPECT": "#ffd9c9", "VICTIM": "#d9e6ff", "LOCATION": "#c4eee5", "WEAPON": "#ffe8ad"}
COLORS.update(PROPERTY="#eee0fb", DATE="#dfeec9", TIME="#dbeaf2", EVIDENCE="#f9d8e9")
COLORS["WITNESS"] = "#cdebe8"
COLORS["ORGANIZATION"] = "#e2d9fa"


@lru_cache(maxsize=1)
def language_model():
    import spacy
    return spacy.load("en_core_web_sm")


def model_revision():
    names = ["tfidf.joblib", "crf.joblib", "topics.joblib", "calibration.json", "ner_selection.json"]
    for folder in ["bert_classifier", "bert_ner"]:
        names.extend(str(path.relative_to(MODELS)) for path in sorted((MODELS / folder).glob("*"))
                     if path.suffix in {".json", ".safetensors", ".bin"})
    paths = [MODELS / name for name in names] + [Path(__file__).with_name(name) for name in ["evidence_context.py", "narrative_context.py", "material_context.py"]]
    from .trained_roles import revision_paths
    paths.extend(revision_paths())
    paths.extend(Path(__file__).with_name(name) for name in ["trained_roles.py", "assertions.py", "relations.py", "references.py", "confidence.py"] if Path(__file__).with_name(name).exists())
    calibration = ARTIFACTS / "core_entity_calibration.json"
    if calibration.exists():
        paths.append(calibration)
    document_contract = ARTIFACTS / "active_document_model.json"
    if document_contract.exists():
        paths.append(document_contract)
        directory = (ARTIFACTS / json.loads(document_contract.read_text(encoding="utf-8"))["directory"]).resolve()
        if not directory.is_relative_to((ARTIFACTS / "models").resolve()):
            raise ValueError("Classifier must be inside project model artifacts")
        paths.extend(path for path in directory.glob("*") if path.suffix in {".json", ".safetensors", ".bin"})
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)


def model_revision_id():
    import hashlib
    return hashlib.sha256(repr(model_revision()).encode("utf-8")).hexdigest()[:16]


def models():
    return _load_models(model_revision())


@lru_cache(maxsize=1)
def _load_models(revision):
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.data.path.insert(0, str(RUNTIME / "nltk_data"))
    return {"classifier": joblib.load(MODELS / "tfidf.joblib"), "crf": joblib.load(MODELS / "crf.joblib"),
            "topics": joblib.load(MODELS / "topics.joblib"), "sentiment": SentimentIntensityAnalyzer(),
            "calibration": json.loads((MODELS / "calibration.json").read_text()),
            "ner_selection": json.loads((MODELS / "ner_selection.json").read_text())["selected"]}


def bert_model():
    return _load_bert_model(model_revision())


@lru_cache(maxsize=1)
def _load_bert_model(revision):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.set_num_threads(4)
    contract_path = ARTIFACTS / "active_document_model.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else {}
    path = ARTIFACTS / contract["directory"] if contract else MODELS / "bert_classifier"
    return AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True), AutoTokenizer.from_pretrained(path, local_files_only=True)


def entity_transformer():
    return _load_entity_transformer(model_revision())


@lru_cache(maxsize=1)
def _load_entity_transformer(revision):
    from .ner_model import load_ner
    return load_ner()


def explain_prediction(classifier, text, top=8):
    vectorizer = classifier.named_steps["vectorizer"]
    estimator = classifier.named_steps["classifier"]
    row = vectorizer.transform([text])
    class_id = estimator.predict_proba(row)[0].argmax()
    weights = row.toarray()[0] * estimator.coef_[class_id]
    terms = vectorizer.get_feature_names_out()
    return [{"term": terms[i].removeprefix("word__").removeprefix("char__"),
             "feature": "character fragment" if terms[i].startswith("char__") else "word / phrase",
             "contribution": float(weights[i])} for i in weights.argsort()[::-1][:top] if weights[i] > 0]


def analyze(text, use_bert=False, ner_choice="Automatic"):
    if not text.strip():
        raise ValueError("Enter a narrative to analyze.")
    if not any(c.isalpha() for c in text) or "\x00" in text:
        raise ValueError("Enter a plain-text description of what happened, including some words.")
    if len(text) > SETTINGS["inference"]["max_characters"]:
        raise ValueError("Please use a narrative of at most 30,000 characters.")
    resources = models()
    tokens = tokens_with_offsets(text)
    selected_ner = resources["ner_selection"] if ner_choice == "Automatic" else ner_choice
    if selected_ner not in {"CRF", "BERT"}:
        raise ValueError("Entity model must be Automatic, CRF or BERT")
    if selected_ner == "BERT":
        from .ner_model import extract_bert
        model, tokenizer = entity_transformer()
        entities = extract_bert(text, model, tokenizer)
    else:
        features = token_features(tokens)
        labels = resources["crf"].predict_single(features)
        entities = spans_from_bio(text, tokens, labels)
        marginal = resources["crf"].predict_marginals_single(features)
        for ent in entities:
            probabilities = [marginal[i].get(labels[i], 0) for i, token in enumerate(tokens) if ent["start"] <= token["start"] < ent["end"]]
            ent["confidence"] = float(np.mean(probabilities))
    classifier = resources["classifier"]
    from .evaluation import temperature_scale
    from .topics import topic_input
    from .privacy import annotate_assertions, structured_summary, redact
    raw_probabilities = classifier.predict_proba([text])
    probabilities = temperature_scale(raw_probabilities, resources["calibration"]["temperature"])[0]
    topic_distribution = resources["topics"]["model"].transform(resources["topics"]["vectorizer"].transform(topic_input([text])))[0]
    doc = language_model()(text)
    from .narrative_context import resolve_narrative_context
    raw_entities = [dict(entity) for entity in entities]
    entities, contextual_roles = resolve_narrative_context(text, entities, doc)
    from .evidence_context import validate_evidence_context
    entities = validate_evidence_context(text, entities)
    from .trained_roles import predict_roles, merge_roles
    learned_roles = predict_roles(text)
    entities, contextual_roles = merge_roles(text, entities, contextual_roles, learned_roles)
    entities = annotate_assertions(text, entities)
    contextual_roles = annotate_assertions(text, contextual_roles)
    from .assertions import qualify_entities, entity_review
    entities = qualify_entities(text, entities, doc)
    contextual_roles = qualify_entities(text, contextual_roles, doc)
    from .confidence import calibrate_core
    entities = calibrate_core(entities, selected_ner)
    result = {"text": text, "entities": entities, "summary": " ".join(summarize(text, count=1).split()), "ner_model": selected_ner,
              "contextual_roles": contextual_roles, "raw_entities": raw_entities, "raw_role_entities": learned_roles,
              "crime_type": str(classifier.classes_[probabilities.argmax()]), "model_score": float(probabilities.max()),
              "sentiment": resources["sentiment"].polarity_scores(text), "threat": threat_language(text),
              "topic_id": int(topic_distribution.argmax()), "topic_distribution": topic_distribution.tolist(),
              "tokens": [{"token": t.text, "lemma": t.lemma_, "POS": t.pos_, "tag": t.tag_, "morphology": str(t.morph), "dependency": t.dep_, "head": t.head.text} for t in doc],
              "generic_entities": [{"text": e.text, "label": e.label_} for e in doc.ents]}
    from .features import prediction_review
    result.update(prediction_review(classifier, text, probabilities, resources["calibration"]["threshold"], raw_probabilities[0]))
    result["entity_context_corrections"] = [entity for entity in entities + contextual_roles if entity.get("extraction_method")]
    import re
    if re.search(r"\b(kill(?:ed|ing)?|murder(?:ed)?|homicide)\b", text, re.I):
        result["review_required"] = True
        result["review_reasons"].append("The eight-category project has no separate homicide category; the suggested category needs review.")
    result["model_version"] = SETTINGS["project"]["version"]
    result["model_revision"] = model_revision_id()
    result["explanation"] = explain_prediction(classifier, text)
    result["structured_summary"] = structured_summary(text, entities, result["summary"])
    result["structured_summary"]["contextual_roles"] = contextual_roles
    from .relations import action_notes
    from .references import reference_notes
    result["references"] = reference_notes(text, entities + contextual_roles, doc)
    result["relations"] = action_notes(text, entities, doc, result["references"])
    events = result["relations"]["events"]
    if events and all(event["assertion"] in {"negated", "disputed", "uncertain"} for event in events):
        result["review_required"] = True
        result["review_reasons"].append("Identified incident actions are only negated, disputed or uncertain; review the suggested category.")
    result["structured_summary"]["action_notes"] = result["relations"]
    result["structured_summary"]["references"] = result["references"]
    result["entity_review"] = entity_review(entities + contextual_roles)
    result["entity_review_required"] = bool(result["entity_review"])
    people = [{"start": e.start_char, "end": e.end_char, "label": "PERSON"} for e in doc.ents if e.label_ == "PERSON"]
    result["redacted_text"] = redact(text, entities + contextual_roles + people)
    if use_bert:
        from .bert_model import predict_bert
        model, tokenizer = bert_model()
        contract_path = ARTIFACTS / "active_document_model.json"
        document_contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else {}
        result["bert_prediction"] = predict_bert(model, tokenizer, [text], full_text=document_contract.get("full_text"))[0]
        result["bert_full_document"] = document_contract.get("full_text", getattr(model.config, "narrative_windowed", False))
    return result


def highlight_entities(text, entities):
    """Escape all user text and labels before inserting spans into HTML."""
    # Character references prevent Markdown from reinterpreting blank lines,
    # tabs and literal markup within the HTML block. The DOM retains the text.
    def escaped(value):
        return html.escape(value).replace("\r", "&#13;").replace("\n", "&#10;").replace("\t", "&#9;")
    chunks, cursor = [], 0
    for ent in sorted(entities, key=lambda e: (e["start"], e["end"])):
        start, end = int(ent["start"]), int(ent["end"])
        if start < cursor or end <= start or end > len(text):
            continue
        color = COLORS.get(ent["label"], "#e7eaf0")
        chunks.append(escaped(text[cursor:start]))
        wrapping = "white-space:nowrap;" if end - start <= 32 and "\n" not in text[start:end] else "box-decoration-break:clone;"
        chunks.append(f'<mark style="{wrapping}background:{color};padding:2px 3px;border-radius:5px;color:#17283d">{escaped(text[start:end])}<small style="font-size:9px;font-weight:700;margin-left:6px">{html.escape(ent["label"])}</small></mark>')
        cursor = end
    chunks.append(escaped(text[cursor:]))
    return '<div style="line-height:2.4;font-size:16px;white-space:pre-wrap">' + "".join(chunks) + "</div>"
