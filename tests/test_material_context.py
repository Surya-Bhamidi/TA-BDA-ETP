"""Guard context-dependent material roles and unchanged learned predictions."""
import pytest
from crime_nlp.config import MODELS
from crime_nlp.material_cases import CASES


@pytest.fixture(scope="module")
def nlp():
    import spacy
    if not spacy.util.is_package("en_core_web_sm"):
        pytest.skip("Requires local spaCy English model")
    from crime_nlp.inference import language_model
    return language_model()


@pytest.mark.parametrize("name,text,mention,label", CASES, ids=[case[0] for case in CASES])
def test_material_roles_require_linked_context(nlp, name, text, mention, label):
    from crime_nlp.material_context import validate_material_context, MATERIAL
    result = validate_material_context(text, [], nlp(text))
    if label is None:
        assert result == []
    else:
        materials = [entity for entity in result if MATERIAL.fullmatch(entity["text"])]
        assert [(entity["text"], entity["label"]) for entity in materials] == [(mention, label)]
    for entity in result:
        assert text[entity["start"]:entity["end"]] == entity["text"]
        assert entity["confidence"] is None


def test_already_correct_model_span_keeps_its_probability(nlp):
    from crime_nlp.material_context import validate_material_context
    text = "Ravi set the shop on fire with a petrol can."
    first = text.index("petrol")
    entity = {"text": "petrol can", "start": first, "end": first + len("petrol can"), "label": "WEAPON", "confidence": .91}
    result = validate_material_context(text, [entity], nlp(text))
    assert len(result) == 1
    assert result[0]["confidence"] == .91
    assert not result[0].get("extraction_method")
    assert entity.get("instrument_type") is None


def test_reported_petrol_case_through_live_pipeline():
    if not (MODELS / "ner_selection.json").exists():
        pytest.skip("Requires local model artifacts")
    from crime_nlp.inference import analyze, highlight_entities
    text = CASES[0][1]
    result = analyze(text)
    assert result["crime_type"] == "Arson"
    found = {(entity["text"], entity["label"]) for entity in result["entities"]}
    assert {("Abebe Bekele", "SUSPECT"), ("Olga Ivanova", "VICTIM"), ("Dubai", "LOCATION"), ("petrol", "WEAPON")} <= found
    assert result["structured_summary"]["mentioned_facts"]["WEAPON"] == ["petrol"]
    assert "petrol<small" in highlight_entities(text, result["entities"])


@pytest.mark.parametrize("ner", ["CRF", "BERT"])
def test_supplier_and_reused_fuel_sentence(ner):
    if not (MODELS / "ner_selection.json").exists():
        pytest.skip("Requires local model artifacts")
    from crime_nlp.inference import analyze
    text = "Suresh used petrol from Indian Oil in Vanasthalipuram and used it to burn Ramesh's flat in Hasthinapuram"
    result = analyze(text, ner_choice=ner)
    assert result["crime_type"] == "Arson"
    expected = {
        ("Suresh", "SUSPECT"), ("petrol", "WEAPON"), ("Vanasthalipuram", "LOCATION"),
        ("Ramesh", "VICTIM"), ("flat", "PROPERTY"), ("Hasthinapuram", "LOCATION")}
    from crime_nlp.trained_roles import contract
    if contract():
        expected.add(("Indian Oil", "ORGANIZATION"))
    assert {(entity["text"], entity["label"]) for entity in result["entities"]} == expected
    mentions = sorted(result["entities"] + result["contextual_roles"], key=lambda entity: entity["start"])
    assert all(left["end"] <= right["start"] for left, right in zip(mentions, mentions[1:]))
