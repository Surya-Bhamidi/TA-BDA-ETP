"""Evaluation must reject leakage and count entity boundary errors."""
import pandas as pd
import pytest

from crime_nlp.evaluation import split_integrity, entity_metrics


def part(report_id, text, group):
    return pd.DataFrame([{"report_id": report_id, "narrative": text, "template_group": group}])


@pytest.mark.parametrize("test, field", [
    (part("train-1", "A different event", "test-group"), "report_id"),
    (part("test-1", "A different event", "train-group"), "template_group"),
    (part("test-1", "  A STOLEN phone!!! ", "test-group"), "normalized_text"),
])
def test_leakage_is_rejected_even_when_other_fields_differ(test, field):
    with pytest.raises(ValueError, match=field):
        split_integrity({"train": part("train-1", "A stolen phone.", "train-group"), "test": test})


def test_split_fingerprint_is_stable_under_row_reordering_and_tracks_source_text():
    frame = pd.concat([part("one", "A phone was stolen", "first"), part("two", "A window was broken", "second")])
    expected = split_integrity({"train": frame})
    assert split_integrity({"train": frame.iloc[::-1]}) == expected
    changed = frame.copy()
    changed.iloc[0, changed.columns.get_loc("narrative")] += " yesterday"
    assert split_integrity({"train": changed}) != expected


def test_strict_span_f1_penalizes_partial_person_names():
    gold = [["B-SUSPECT", "I-SUSPECT", "O", "B-VICTIM"]]
    partial = [["B-SUSPECT", "O", "O", "B-VICTIM"]]
    assert entity_metrics(gold, gold)["strict_entity_f1"] == 1.
    assert entity_metrics(gold, partial)["strict_entity_f1"] == .5
