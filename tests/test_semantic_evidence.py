"""Keyword recall and semantic abstention for cited narrative evidence."""
import numpy as np
import pytest

from crime_nlp import semantic


NARRATIVE = "Vivek killed Barkha with a gun near Faridabad on 20-10-2026"


@pytest.fixture
def low_similarity(monkeypatch):
    monkeypatch.setattr(semantic, "encoder", lambda: (None, None))
    monkeypatch.setattr(semantic, "source_windows", lambda source, tokenizer: [
        {"source": i, "start": 0, "end": len(text), "text": text} for i, text in enumerate(source)
    ])
    monkeypatch.setattr(semantic, "encode", lambda texts: np.array([[.233] for _ in texts]))


@pytest.mark.parametrize("query", ["gun", "GUN!", "a gun", "Vivek", "Faridabad", "20-10-2026"])
def test_literal_queries_retrieve_user_narrative(low_similarity, query):
    answer = semantic.semantic_evidence(query, NARRATIVE, "case-1")
    assert answer["answer"] == NARRATIVE
    assert answer["match_type"] == "Keyword"
    assert answer["score"] < .30
    assert answer["citations"] == [{"report_id": "case-1", "sentence": 1, "text": NARRATIVE,
                                    "sentence_start": 0, "sentence_end": len(NARRATIVE)}]


@pytest.mark.parametrize("query,text", [
    ("gun", "The investigation has begun."),
    ("gun", "A handgun was reported."),
    ("the with on", NARRATIVE),
    ("knife", NARRATIVE),
    ("gun recovered", NARRATIVE),
    ("gun recovered", "A gun was mentioned. A phone was recovered."),
    ("20-10-2027", NARRATIVE),
    ("!!!", NARRATIVE),
    ("", NARRATIVE),
    ("gun", ""),
])
def test_nonmatches_still_abstain(low_similarity, query, text):
    answer = semantic.semantic_evidence(query, text)
    assert answer["answer"] == "Insufficient evidence in this narrative."
    assert answer["citations"] == []


def test_negated_keyword_preserves_source_wording(low_similarity):
    text = "No gun was found."
    answer = semantic.semantic_evidence("gun", text)
    assert answer["answer"] == text
    assert answer["citations"][0]["text"] == text


def test_semantic_paraphrase_without_keyword_overlap(low_similarity, monkeypatch):
    monkeypatch.setattr(semantic, "encode", lambda texts: np.ones((len(texts), 1)))
    answer = semantic.semantic_evidence("firearm", NARRATIVE)
    assert answer["answer"] == NARRATIVE
    assert answer["match_type"] == "Semantic"


def test_literal_match_preferred_over_high_similarity_window(low_similarity, monkeypatch):
    text = "Unrelated background; a gun was mentioned."
    chunks = [{"source": 0, "start": 0, "end": 20, "text": text[:20]},
              {"source": 0, "start": 22, "end": len(text), "text": text[22:]}]
    monkeypatch.setattr(semantic, "source_windows", lambda source, tokenizer: chunks)
    monkeypatch.setattr(semantic, "encode", lambda texts: np.array([[.9], [.233]]) if len(texts) == 2 else np.array([[1.]]))
    answer = semantic.semantic_evidence("gun", text)
    assert answer["answer"] == chunks[1]["text"]
    citation = answer["citations"][0]
    assert citation["text"] == text[citation["sentence_start"]:citation["sentence_end"]]
    assert answer["score"] == pytest.approx(.233)


def test_literal_match_survives_two_higher_semantic_results(low_similarity, monkeypatch):
    text = "Police arrived. A witness called. A gun was mentioned."
    monkeypatch.setattr(semantic, "encode", lambda texts: np.array([[.9], [.8], [.233]]) if len(texts) == 3 else np.array([[1.]]))
    answer = semantic.semantic_evidence("gun", text)
    assert [citation["sentence"] for citation in answer["citations"]] == [1, 3]
    assert answer["match_type"] == "Keyword + semantic"


def test_user_narrative_with_local_encoder():
    if not (semantic.RUNTIME / "semantic_encoder" / "config.json").exists():
        pytest.skip("Requires local semantic encoder")
    answer = semantic.semantic_evidence("gun", NARRATIVE)
    assert answer["answer"] == NARRATIVE
    assert "Keyword" in answer["match_type"]
    assert answer["citations"][0]["text"] == NARRATIVE
