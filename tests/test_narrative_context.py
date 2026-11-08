"""Annotated development regressions: these are not independent accuracy estimates."""
import json
import sqlite3
import pytest
from crime_nlp.challenges import cases
from crime_nlp.crosscheck_cases import cases as crosscheck_cases
from crime_nlp.config import MODELS


@pytest.mark.parametrize("case", [*cases(), *crosscheck_cases()], ids=lambda case: case["id"])
def test_annotated_narratives(case):
    if not (MODELS / "ner_selection.json").exists():
        pytest.skip("Requires local model artifacts and spaCy model")
    from crime_nlp.inference import analyze
    result = analyze(case["text"])
    if case["category"] is not None:
        assert result["crime_type"] == case["category"]
    actual = result["entities"] + result["contextual_roles"]
    if case["score_entities"]:
        assert {(e["start"], e["end"], e["label"]) for e in actual} == {
            (e["start"], e["end"], e["label"]) for e in case["entities"]}
    for entity in actual:
        assert case["text"][entity["start"]:entity["end"]] == entity["text"]
        if entity.get("extraction_method"):
            assert entity["confidence"] is None
        if entity["label"] == "WITNESS":
            assert entity["text"] not in result["redacted_text"]


def test_annotation_offsets_preserve_unicode():
    for case in [*cases(), *crosscheck_cases()]:
        for entity in case["entities"]:
            assert case["text"][entity["start"]:entity["end"]] == entity["text"]


@pytest.mark.parametrize("text,name", [
    ("sir\nLakshmi\nsaw Sanjay Kumar intentionally starting a fire in Dubai", "Lakshmi"),
    ("yari qaneneba saw deska intentionally starting a fire in São Paulo.", "yari qaneneba"),
    ("R. K. Menon saw Müller intentionally starting a fire.", "R. K. Menon"),
])
def test_overlapping_witness_detectors_keep_one_complete_name(text, name):
    if not (MODELS / "ner_selection.json").exists():
        pytest.skip("Requires local model artifacts")
    from crime_nlp.inference import analyze
    result = analyze(text)
    assert [role["text"] for role in result["contextual_roles"]] == [name]
    mentions = sorted(result["entities"] + result["contextual_roles"], key=lambda entity: entity["start"])
    assert all(a["end"] <= b["start"] for a, b in zip(mentions, mentions[1:]))


def test_entity_filters_search_contextual_roles_without_mixing_labels(tmp_path, monkeypatch):
    from crime_nlp import store
    database = tmp_path / "cases.sqlite"
    monkeypatch.setattr(store, "DB", database)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE reports (row_number INTEGER, predicted_entities_json TEXT, contextual_roles_json TEXT)")
        connection.execute("INSERT INTO reports VALUES (0,?,?)", (
            json.dumps([{"text": "bus stand", "label": "LOCATION"}]),
            json.dumps([{"text": "rahul", "label": "WITNESS"}])))
    assert store.entity_matches("WITNESS", "rahul") == [0]
    assert store.entity_matches("LOCATION", "rahul") == []
    assert store.entity_matches("LOCATION", "stand") == [0]
    assert store.entity_matches("Any", "rahul") == [0]
