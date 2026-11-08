"""Integration invariants for the generated end-to-end deliverable."""
import json
import pandas as pd
import pytest
from scipy import sparse

from crime_nlp.config import ARTIFACTS, PROCESSED, CATEGORIES, ENTITY_LABELS
from crime_nlp.annotation_schema import LABELS

pytestmark = pytest.mark.skipif(not (PROCESSED / "dashboard.parquet").exists(), reason="Run the pipeline before artifact integration tests.")


@pytest.fixture(scope="module")
def data():
    return pd.read_parquet(PROCESSED / "dashboard.parquet")


def test_complete_corpus_cleaning_and_search_alignment(data):
    metrics = json.loads((ARTIFACTS / "spark_metrics.json").read_text())
    assert len(data) >= 50000
    assert metrics["raw_rows"] == metrics["clean_rows"] + metrics["invalid_rows_removed"] + metrics["duplicates_removed"]
    assert len(data) == metrics["clean_rows"]
    manifest = json.loads((ARTIFACTS / "dataset_manifest.json").read_text())
    assert len(data) == manifest["valid_unique_reports"]
    assert metrics["duplicates_removed"] == manifest["injected_errors"]["duplicates"]
    assert data.report_id.is_unique and data.text_hash.is_unique
    assert data.narrative.notna().all()
    assert data.event_time.notna().all()
    assert data.clean_text.str.len().min() > 0
    assert sparse.load_npz(ARTIFACTS / "search_matrix.npz").shape[0] == len(data)


def test_split_has_no_template_or_duplicate_leakage(data):
    assert data.groupby("template_group", observed=True).split.nunique().max() == 1
    assert data.groupby("text_hash", observed=True).split.nunique().max() == 1
    for split in ["train", "validation", "development", "test"]:
        assert set(data.loc[data.split == split, "crime_type"]) == set(CATEGORIES)


def test_case_summaries_are_one_source_sentence_and_match_sqlite(data):
    from crime_nlp.text import sentences
    from crime_nlp.store import fetch_reports
    assert data.summary.notna().all()
    assert data.summary.str.strip().ne("").all()
    assert not data.summary.str.contains(r"[\r\n]").any()
    sample = data.sample(100, random_state=42)
    for row in sample.itertuples():
        assert row.summary and "\n" not in row.summary and "\r" not in row.summary
        assert len(sentences(row.summary)) == 1
        assert row.summary in {" ".join(sentence.split()) for sentence in sentences(row.narrative)}
    assert fetch_reports(sample.index.tolist()).summary.tolist() == sample.summary.tolist()


def test_predictions_and_spans_are_valid(data):
    assert set(data.predicted_crime) <= set(CATEGORIES)
    assert data.sentiment.between(-1, 1).all()
    assert data.topic_probability.between(0, 1).all()
    assert data.threat_score.between(0, 12).all()
    for row in data.itertuples():
        entities = json.loads(row.predicted_entities_json)
        roles = json.loads(row.contextual_roles_json)
        for ent in entities:
            assert row.narrative[ent["start"]:ent["end"]] == ent["text"]
            assert ent["label"] in LABELS
        for role in roles:
            assert role["label"] == "WITNESS"
            assert row.narrative[role["start"]:role["end"]] == role["text"]
            assert role["confidence"] is None or role.get("trained_model") and 0 <= role["confidence"] <= 1
        mentions = sorted(entities + roles, key=lambda entity: (entity["start"], entity["end"]))
        assert all(left["end"] <= right["start"] for left, right in zip(mentions, mentions[1:]))


def test_stored_arson_materials_are_searchable(data):
    from crime_nlp.store import entity_matches
    selected = data.loc[data.narrative.str.contains("poured petrol and set", case=False, regex=False)]
    assert len(selected) > 0
    matches = set(entity_matches("WEAPON", "petrol"))
    for position, row in selected.iterrows():
        assert any(entity["text"].lower() == "petrol" and entity["label"] == "WEAPON"
                   for entity in json.loads(row.predicted_entities_json))
        assert position in matches


def test_metrics_are_consistent_with_test_count():
    metrics = json.loads((ARTIFACTS / "evaluation.json").read_text())
    assert set(metrics["models"]) == {"Bag of Words", "TF-IDF", "Word2Vec", "BERT"}
    for item in metrics["models"].values():
        matrix = item["confusion_matrix"]
        assert sum(sum(row) for row in matrix) == metrics["protocol"]["test_rows"]
        true_positives = sum(matrix[i][i] for i in range(len(matrix)))
        assert item["accuracy"] == pytest.approx(true_positives / metrics["protocol"]["test_rows"])
        assert 0 <= item["macro_f1"] <= 1


def test_offline_inference_has_syntax_and_bert(data):
    from crime_nlp.inference import analyze
    result = analyze(data.iloc[0].narrative, use_bert=True)
    assert result["bert_prediction"] in CATEGORIES
    assert any(t["dependency"] == "ROOT" for t in result["tokens"])
    assert any(t["morphology"] for t in result["tokens"])
    assert result["entities"]
    assert result["summary"]


def test_case_store_and_semantic_index_preserve_report_order(data):
    import numpy as np
    from crime_nlp.store import fetch_reports, entity_matches
    positions = [len(data) - 1, 0, 19]
    fetched = fetch_reports(positions)
    assert fetched.report_id.tolist() == data.iloc[positions].report_id.tolist()
    assert np.load(ARTIFACTS / "semantic_embeddings.npy", mmap_mode="r").shape == (len(data), 384)
    assert entity_matches("Any", "' OR 1=1 --") == []


def test_both_entity_models_and_grounded_evidence_on_development(data):
    from crime_nlp.inference import analyze
    from crime_nlp.semantic import semantic_evidence
    text = data.loc[data.split == "development"].iloc[0].narrative
    for name in ("CRF", "BERT"):
        result = analyze(text, ner_choice=name)
        assert result["ner_model"] == name
        assert result["entities"]
        assert all(text[e["start"]:e["end"]] == e["text"] for e in result["entities"])
    answer = semantic_evidence("What evidence was reviewed?", text, "development-fixture")
    assert all(c["text"] in text and c["report_id"] == "development-fixture" for c in answer["citations"])
