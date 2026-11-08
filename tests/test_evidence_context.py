"""Evidence-subject regression and camera/property contrast cases."""
import pytest

from crime_nlp.evidence_context import validate_evidence_context

DEMO = ("Rajiv hit Sneha with a stick near Pune on 29-06-2026. "
        "A witness called the police. Camera footage was collected.")


def analyzer():
    from crime_nlp.config import MODELS
    if not (MODELS / "crf.joblib").exists():
        pytest.skip("Train the evidence-context CRF before model integration checks")
    from crime_nlp.inference import analyze
    return analyze


def test_context_correction_preserves_offsets_and_records_provenance():
    text = "José reported a theft.\n\nCamera footage was reviewed."
    start = text.index("Camera")
    raw = [{"text": "Camera footage", "start": start, "end": start + 14, "label": "VICTIM", "confidence": .4}]
    corrected = validate_evidence_context(text, raw)
    assert corrected[0]["label"] == "EVIDENCE"
    assert corrected[0]["confidence"] is None
    assert corrected[0]["previous_labels"] == ["VICTIM"]
    assert text[corrected[0]["start"]:corrected[0]["end"]] == "Camera footage"
    assert raw[0]["label"] == "VICTIM"


def test_context_check_requires_supported_context_and_preserves_larger_spans():
    text = "Camera Footage Ltd supplied the camera."
    raw = [{"text": "Camera Footage Ltd", "start": 0, "end": 18, "label": "SUSPECT", "confidence": .9}]
    assert validate_evidence_context(text, raw) == raw
    text = "Camera footage is interesting."
    assert validate_evidence_context(text, []) == []


@pytest.mark.parametrize("text, evidence", [
    (DEMO, "Camera footage"),
    ("Camera footage was collected.", "Camera footage"),
    ("camera footage was collected.", "camera footage"),
    ("CAMERA FOOTAGE WAS COLLECTED.", "CAMERA FOOTAGE"),
    ("CCTV footage was reviewed.", "CCTV footage"),
    ("Police collected surveillance footage.", "surveillance footage"),
    ("A witness supplied video recordings.", "video recordings"),
    ("No camera footage was available.", "camera footage"),
])
def test_evidence_mentions_are_not_people(text, evidence):
    result = analyzer()(text, ner_choice="CRF")
    found = {(e["text"], e["label"]) for e in result["entities"]}
    assert (evidence, "EVIDENCE") in found
    assert not any(e["label"] in {"SUSPECT", "VICTIM"} and evidence.lower() in e["text"].lower()
                   for e in result["entities"])
    assert result["text"] == text
    if text.startswith("No "):
        assert next(e for e in result["entities"] if e["text"] == evidence)["assertion"] == "negated"


def test_stolen_camera_remains_property_and_people_keep_roles():
    text = "Mayank stole the camera of Surya on 29-06-2026"
    result = analyzer()(text, ner_choice="CRF")
    found = {(e["text"], e["label"]) for e in result["entities"]}
    assert {("Mayank", "SUSPECT"), ("Surya", "VICTIM"), ("camera", "PROPERTY")} <= found
    assert ("camera", "EVIDENCE") not in found
