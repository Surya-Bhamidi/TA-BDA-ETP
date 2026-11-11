"""Audit interactions beyond the main narrative regression cases."""
import json
import sqlite3
from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from crime_nlp.config import ROOT, PROCESSED

pytestmark = pytest.mark.skipif(not (PROCESSED / "dashboard.parquet").exists(), reason="Requires local dashboard artifacts")


def app_test():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()


def upload(content, name="case.txt"):
    return SimpleNamespace(name=name, size=len(content), getvalue=lambda: content)


def test_replacing_upload_with_same_name_updates_narrative(monkeypatch):
    import streamlit
    first = "Ravi hit Asha near Pune."
    second = "Vivek killed Barkha with a gun near Faridabad on 20-10-2026"
    current = [upload(first.encode())]
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: current[0])
    app = app_test()
    app.sidebar.radio[0].set_value("Narrative lab").run()
    assert app.text_area[0].value == first
    current[0] = upload(second.encode())
    app.run()
    assert not app.exception
    assert app.text_area[0].value == second


@pytest.mark.parametrize("value", [None, [], 1, "text"])
def test_review_upload_rejects_nonobject_json_without_crashing(monkeypatch, value):
    import streamlit
    fixture = upload(json.dumps(value).encode(), "review.jsonl")
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: fixture if kwargs.get("key") == "quality_review_upload" else None)
    app = app_test()
    app.sidebar.radio[0].set_value("Model evaluation").run()
    assert not app.exception
    assert any("JSON object" in item.value for item in app.error)


def test_review_upload_calculates_submitted_judgments(monkeypatch):
    import streamlit
    judgment = {"task": "summary", "review_status": "reviewed", "reviewer": "automated-test-fixture",
                "faithful_to_source": True, "preserves_participants": True, "preserves_negation": False, "covers_main_event": True}
    fixture = upload(json.dumps(judgment).encode(), "review.jsonl")
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: fixture if kwargs.get("key") == "quality_review_upload" else None)
    app = app_test()
    app.sidebar.radio[0].set_value("Model evaluation").run()
    assert not app.exception
    results = [json.loads(item.value) for item in app.json]
    summary = next(result["summaries"] for result in results if "summaries" in result)
    assert summary["judgments"] == 1
    assert summary["metrics"]["preserves_negation"] == 0


def test_password_gate_rejects_wrong_password_and_accepts_correct_one(monkeypatch):
    monkeypatch.setenv("CRIME_DASHBOARD_PASSWORD", "audit-test-only")
    app = app_test()
    assert not app.title
    app.text_input[0].set_value("wrong").run()
    app.button[0].click().run()
    assert any("Password not recognized" in item.value for item in app.error)
    assert not app.title
    app.text_input[0].set_value("audit-test-only").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.title[0].value == "From narratives to insight"


def test_overview_filters_counts_and_empty_date_range():
    frame = pd.read_parquet(PROCESSED / "dashboard.parquet", columns=["crime_type", "district", "event_time"])
    app = app_test()
    next(item for item in app.multiselect if item.label == "Crime categories").set_value(["Assault"]).run()
    assert app.metric[0].value == f"{int(frame.crime_type.eq('Assault').sum()):,}"
    district = frame.loc[frame.crime_type.eq("Assault"), "district"].iloc[0]
    next(item for item in app.multiselect if item.label == "Districts").set_value([district]).run()
    assert app.metric[0].value == f"{int((frame.crime_type.eq('Assault') & frame.district.eq(district)).sum()):,}"
    last = pd.to_datetime(frame.event_time).max().date()
    app.date_input[0].set_value((last + timedelta(days=2), last + timedelta(days=3))).run()
    assert not app.exception
    assert app.metric[0].value == "0"
    assert any("No reports match" in item.value for item in app.info)


def test_case_page_size_pagination_and_exact_report_id():
    app = app_test()
    app.sidebar.radio[0].set_value("Case explorer").run()
    app.slider[0].set_value(5).run()
    first = app.dataframe[0].value.report_id.tolist()
    assert len(first) == 5
    app.number_input[0].set_value(2).run()
    second = app.dataframe[0].value.report_id.tolist()
    assert len(second) == 5 and not set(first) & set(second)
    next(item for item in app.text_input if item.label == "Search narratives").set_value(first[0]).run()
    assert not app.exception
    assert app.number_input[0].value == 1
    assert app.dataframe[0].value.report_id.tolist() == [first[0]]


def test_category_feedback_saves_correction_without_source(tmp_path, monkeypatch):
    from crime_nlp import store
    monkeypatch.setattr(store, "PROCESSED", tmp_path)
    text = "Ravi hit Asha near Pune."
    app = app_test()
    app.sidebar.radio[0].set_value("Narrative lab").run()
    app.checkbox[0].uncheck().run()
    app.text_area[0].set_value(text).run()
    next(item for item in app.button if item.label == "Analyze narrative").click().run()
    next(item for item in app.selectbox if item.label == "Suggest a corrected category").set_value("Fraud").run()
    next(item for item in app.button if item.label == "Save correction locally").click().run()
    assert not app.exception
    with sqlite3.connect(tmp_path / "feedback.sqlite") as connection:
        row = connection.execute("SELECT narrative_sha256,predicted,corrected FROM feedback").fetchone()
    assert row[1:] == ("Assault", "Fraud")
    assert len(row[0]) == 64 and text not in str(row)
