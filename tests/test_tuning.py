import numpy as np
import json
import pandas as pd

from crime_nlp.train import document_vectors
from crime_nlp.tuning import choose_candidate, mask_training_mentions


def test_validation_selection_uses_f1_and_retains_ties():
    baseline = {"macro_f1": .9, "accuracy": .9}
    assert not choose_candidate(baseline, {"macro_f1": .9, "accuracy": 1.})
    assert not choose_candidate(baseline, {"macro_f1": .89, "accuracy": 1.})
    assert choose_candidate(baseline, {"macro_f1": .91, "accuracy": .9})


def test_word2vec_pooling_weights_signal_and_handles_unknown_words():
    class Vocabulary(dict):
        def __getitem__(self, key):
            if isinstance(key, list):
                return np.stack([dict.__getitem__(self, word) for word in key])
            return dict.__getitem__(self, key)
    class Embedding:
        vector_size = 2
        wv = Vocabulary({"routine": np.array([1., 0.]), "crime": np.array([0., 1.])})
    recipe = {"pooling": "idf_max", "idf": {"routine": 1., "crime": 3.}}
    rows = document_vectors(Embedding(), ["routine crime", "unseen"], recipe)
    np.testing.assert_allclose(rows[0], [.25, .75, 1., 1.])
    np.testing.assert_array_equal(rows[1], np.zeros(4))
    np.testing.assert_allclose(document_vectors(Embedding(), ["routine crime"])[0], [.5, .5])


def test_training_mention_dropout_preserves_actions_and_original_text():
    original = "José hit Wei near Nairobi."
    spans = [{"start": 0, "end": 4, "label": "SUSPECT"}, {"start": 9, "end": 12, "label": "VICTIM"},
             {"start": 18, "end": 25, "label": "LOCATION"}]
    frame = pd.DataFrame([{"narrative": original, "entities_json": json.dumps(spans)}])
    assert mask_training_mentions(frame, 1.) == ["someone hit someone near somewhere."]
    assert mask_training_mentions(frame, 0.) == [original]
    assert frame.narrative.iloc[0] == original


def test_bert_config_update_invalidates_analysis_revision(tmp_path, monkeypatch):
    from crime_nlp import inference
    monkeypatch.setattr(inference, "MODELS", tmp_path)
    for name in ["tfidf.joblib", "crf.joblib", "topics.joblib", "calibration.json", "ner_selection.json"]:
        (tmp_path / name).write_text("placeholder")
    folder = tmp_path / "bert_classifier"
    folder.mkdir()
    config = folder / "config.json"
    config.write_text('{"narrative_max_tokens":160}')
    original_revision = inference.model_revision_id()
    config.write_text('{"narrative_max_tokens":192,"revision":"new"}')
    assert inference.model_revision_id() != original_revision
