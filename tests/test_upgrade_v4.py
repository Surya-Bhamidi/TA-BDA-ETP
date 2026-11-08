import json
import sqlite3
import pytest
from crime_nlp.annotation_schema import LABELS, validate_record
from crime_nlp.annotation_workflow import source_split, agreement, validate_adjudicated


def row(name="A", group="a"):
    return {"report_id": name, "narrative": name + " saw it.", "source_group": group,
            "entities": [{"start": 0, "end": len(name), "text": name, "label": "WITNESS"}]}


def test_new_schema_keeps_legacy_contract_and_validates_boundaries():
    from crime_nlp.config import ENTITY_LABELS
    from crime_nlp.ner_model import BIO_TAGS
    assert len(ENTITY_LABELS) == 8 and len(BIO_TAGS) == 17
    assert set(LABELS) - set(ENTITY_LABELS) == {"ORGANIZATION", "WITNESS"}
    bad = row()
    bad["entities"][0]["end"] = 1.0
    with pytest.raises(ValueError):
        validate_record(bad)


def test_source_groups_stay_together_and_text_is_preserved():
    records = [row("Alpha", "a"), row("Beta", "a"), row("Gamma", "b"), row("Delta", "c")]
    result = source_split(records)
    assert len({record["split"] for record in result}) == 3
    assert result[0]["split"] == result[1]["split"]
    assert [record["narrative"] for record in result] == [record["narrative"] for record in records]
    with pytest.raises(ValueError):
        source_split([row("Alpha", "a"), row("Alpha", "b"), row("Gamma", "c")])


def test_disagreements_require_human_adjudication():
    first, second = row(), row()
    second["entities"][0]["label"] = "VICTIM"
    report = agreement([first], [second])
    assert report["reports_requiring_adjudication"] == 1
    assert report["exact_entity_and_assertion_f1"] == 0
    with pytest.raises(ValueError):
        validate_adjudicated(first)


def test_entity_feedback_omits_source_and_entity_names_by_default(tmp_path, monkeypatch):
    from crime_nlp import store
    monkeypatch.setattr(store, "PROCESSED", tmp_path)
    example = row("PrivateName")
    store.save_entity_feedback(example["narrative"], example["entities"], example["entities"], "fixture")
    with sqlite3.connect(tmp_path / "feedback.sqlite") as connection:
        payload = connection.execute("SELECT payload_json FROM entity_feedback").fetchone()[0]
    assert "PrivateName" not in payload and "narrative" not in payload
    assert json.loads(payload)["corrected"][0]["label"] == "WITNESS"


def test_entity_feedback_text_requires_explicit_opt_in(tmp_path, monkeypatch):
    from crime_nlp import store
    monkeypatch.setattr(store, "PROCESSED", tmp_path)
    example = row("ReviewDraft")
    store.save_entity_feedback(example["narrative"], [], example["entities"], "fixture", include_text=True)
    with sqlite3.connect(tmp_path / "feedback.sqlite") as connection:
        payload = json.loads(connection.execute("SELECT payload_json FROM entity_feedback").fetchone()[0])
    assert payload["narrative"] == example["narrative"] and payload["review_status"] == "unreviewed"


def test_ten_label_subword_alignment_does_not_use_legacy_head_size():
    from crime_nlp.ner_model import align_labels
    class Encoding:
        def word_ids(self, batch_index):
            return [None, 0, 0, None]
    tags = ["O"] + [prefix + label for label in LABELS for prefix in ("B-", "I-")]
    result = align_labels(Encoding(), [["B-ORGANIZATION"]], tags)
    assert result == [[-100, tags.index("B-ORGANIZATION"), -100, -100]]


@pytest.fixture(scope="module")
def nlp():
    import spacy
    if not spacy.util.is_package("en_core_web_sm"):
        pytest.skip("Requires English spaCy resource")
    from crime_nlp.inference import language_model
    return language_model()


@pytest.mark.parametrize("text,name,expected", [
    ("Ravi did not hit Asha.", "Asha", "negated"),
    ("Ravi denied hitting Asha.", "Asha", "disputed"),
    ("Neha said Ravi hit Asha.", "Asha", "alleged"),
    ("Ravi might hit Asha.", "Asha", "uncertain"),
    ("Petrol was not used.", "Petrol", "negated"),
    ("Ravi left without taking the wallet.", "wallet", "negated"),
])
def test_assertion_scope_tracks_negation_denial_and_attribution(nlp, text, name, expected):
    from crime_nlp.assertions import qualify_entities
    first = text.index(name)
    result = qualify_entities(text, [{"text": name, "start": first, "end": first + len(name), "label": "VICTIM"}], nlp(text))
    assert result[0]["assertion"] == expected


def test_role_merging_preserves_core_and_full_witness_boundary():
    from crime_nlp.trained_roles import merge_roles
    core = [{"text": "Pune", "start": 0, "end": 4, "label": "LOCATION"}]
    witness = [{"text": "Asha Rao", "start": 6, "end": 14, "label": "WITNESS", "confidence": None}]
    learned = [{"text": "Pune", "start": 0, "end": 4, "label": "ORGANIZATION", "confidence": .99},
               {"text": "Rao", "start": 11, "end": 14, "label": "WITNESS", "confidence": .99}]
    actual_core, actual_witness = merge_roles("Pune, Asha Rao", core, witness, learned)
    assert actual_core == core and actual_witness == witness


def test_quality_metrics_require_real_submitted_review_status():
    from crime_nlp.quality_review import evaluate
    with pytest.raises(ValueError):
        evaluate([{"task": "summary", "review_status": "unreviewed", "reviewer": ""}])
    fixture = {"task": "retrieval", "review_status": "reviewed", "reviewer": "synthetic-unit-fixture",
               "ranked_ids": ["irrelevant", "relevant"], "relevant_ids": ["relevant"], "k": 2}
    result = evaluate([fixture])["retrieval"]["metrics"]
    assert result["precision_at_k"] == .5 and result["reciprocal_rank"] == .5
    assert result["recall_within_judged_set_at_k"] == 1


def test_passive_observation_identifies_observer_not_incident(nlp):
    from crime_nlp.narrative_context import resolve_narrative_context
    text = "The incident was seen by Neha Sharma."
    _, witnesses = resolve_narrative_context(text, [], nlp(text))
    assert [entity["text"] for entity in witnesses] == ["Neha Sharma"]


def test_denied_action_notes_preserve_source_and_qualifier(nlp):
    from crime_nlp.assertions import qualify_entities
    from crime_nlp.relations import action_notes
    text = "Ravi denied hitting Asha."
    entities = [{"text": "Ravi", "start": 0, "end": 4, "label": "SUSPECT"},
                {"text": "Asha", "start": 20, "end": 24, "label": "VICTIM"}]
    doc = nlp(text)
    notes = action_notes(text, qualify_entities(text, entities, doc), doc)
    assert notes["events"][0]["assertion"] == "disputed"
    assert notes["events"][0]["source"]["text"] == text


def test_reference_resolution_does_not_guess_between_two_people(nlp):
    from crime_nlp.references import reference_notes
    text = "Ravi hit Asha. He ran away."
    entities = [{"text": "Ravi", "start": 0, "end": 4, "label": "SUSPECT"},
                {"text": "Asha", "start": 9, "end": 13, "label": "VICTIM"}]
    reference = reference_notes(text, entities, nlp(text))[0]
    assert reference["status"] == "ambiguous" and reference["antecedent"] is None


def test_reported_pronoun_does_not_automatically_refer_to_speaker(nlp):
    from crime_nlp.references import reference_notes
    text = "Neha said he hit someone."
    reference = reference_notes(text, [{"text": "Neha", "start": 0, "end": 4, "label": "WITNESS"}], nlp(text))[0]
    assert reference["antecedent"] is None and reference["status"] == "unresolved"


def test_only_negated_actions_require_category_review():
    from crime_nlp.config import MODELS
    if not (MODELS / "crf.joblib").exists():
        pytest.skip("Requires local model artifacts")
    from crime_nlp.inference import analyze
    result = analyze("Ravi did not hit Asha with a stick.")
    assert result["review_required"]
    assert any("negated" in reason for reason in result["review_reasons"])


def test_lack_of_permission_does_not_negate_the_named_victim(nlp):
    from crime_nlp.privacy import annotate_assertions
    from crime_nlp.assertions import qualify_entities
    text = "Ravi took the wallet without Asha's permission."
    first = text.index("Asha")
    entities = annotate_assertions(text, [{"text": "Asha", "start": first, "end": first + 4, "label": "VICTIM"}])
    assert entities[0]["assertion"] == "negated"
    assert qualify_entities(text, entities, nlp(text))[0]["assertion"] == "mentioned"


def test_calibration_keeps_raw_scores_and_never_scores_rule_repairs(tmp_path, monkeypatch):
    import hashlib
    from crime_nlp import confidence
    weight = tmp_path / "crf.joblib"
    weight.write_bytes(b"synthetic-model-fixture")
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"models": {"CRF": {"SUSPECT": {"coefficient": 0., "intercept": 0.}}},
                                "legacy_core_sha256": {"CRF": hashlib.sha256(weight.read_bytes()).hexdigest()}}), encoding="utf-8")
    monkeypatch.setattr(confidence, "PATH", path)
    monkeypatch.setattr(confidence, "MODELS", tmp_path)
    confidence._read.cache_clear()
    entities = [{"label": "SUSPECT", "confidence": .95}, {"label": "SUSPECT", "confidence": None}]
    result = confidence.calibrate_core(entities, "CRF")
    assert result[0]["confidence"] == .95 and result[0]["calibrated_confidence"] == .5
    assert "calibrated_confidence" not in result[1]
    weight.write_bytes(b"different-model-weights")
    assert "calibrated_confidence" not in confidence.calibrate_core([{"label": "SUSPECT", "confidence": .95}], "CRF")[0]
    confidence._read.cache_clear()
