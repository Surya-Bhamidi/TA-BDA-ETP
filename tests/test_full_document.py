from types import SimpleNamespace
import pytest
from crime_nlp.config import RUNTIME


@pytest.fixture(scope="module")
def tokenizer():
    if not (RUNTIME / "bert" / "config.json").exists():
        pytest.skip("Requires local tokenizer resource")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(RUNTIME / "bert", local_files_only=True)


def test_source_windows_cover_late_text_and_preserve_offsets(tokenizer):
    from crime_nlp.windowing import source_windows
    text = "José " + "background words " * 300 + "camera footage was collected."
    windows = source_windows([text], tokenizer, max_tokens=64, overlap=16)
    assert len(windows) > 1 and windows[-1]["end"] == len(text)
    assert "camera footage" in windows[-1]["text"]
    for window in windows:
        assert window["text"] == text[window["start"]:window["end"]]
    assert all(right["start"] <= left["end"] for left, right in zip(windows, windows[1:]))


def test_windowed_classifier_sees_marker_after_first_window(tokenizer):
    import torch
    from crime_nlp.bert_model import predict_bert
    from crime_nlp.config import CATEGORIES
    marker = tokenizer.convert_tokens_to_ids("hack")
    class MarkerModel:
        config = SimpleNamespace(narrative_max_tokens=32, narrative_windowed=True)
        def eval(self):
            return self
        def __call__(self, input_ids, **kwargs):
            scores = torch.zeros((len(input_ids), len(CATEGORIES)))
            scores[:, CATEGORIES.index("Cybercrime")] = (input_ids == marker).any(dim=1).float() * 8
            return SimpleNamespace(logits=scores)
    text = "background information " * 300 + "hack"
    assert predict_bert(MarkerModel(), tokenizer, [text]) == ["Cybercrime"]
    assert predict_bert(MarkerModel(), tokenizer, [text], full_text=False) == ["Arson"]


def test_evidence_search_reads_late_part_of_one_sentence(tokenizer, monkeypatch):
    import numpy as np
    from crime_nlp import semantic
    monkeypatch.setattr(semantic, "encoder", lambda: (None, tokenizer))
    def fixture_encode(texts, **kwargs):
        return np.array([[1., 0.] if "camera footage" in text else [0., 1.] for text in texts])
    monkeypatch.setattr(semantic, "encode", fixture_encode)
    text = "background information " * 300 + "camera footage was collected."
    result = semantic.semantic_evidence("camera footage", text)
    assert "camera footage" in result["answer"]
    citation = result["citations"][0]
    assert citation["sentence_start"] > 0
    assert citation["text"] == text[citation["sentence_start"]:citation["sentence_end"]]
