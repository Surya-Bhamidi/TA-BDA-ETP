from pathlib import Path
import pytest
from crime_nlp.config import ROOT, PROCESSED

pytestmark = pytest.mark.skipif(not (PROCESSED / "dashboard.parquet").exists(), reason="Run the full pipeline before UI integration tests.")


def test_every_dashboard_page_and_narrative_analysis():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not app.exception
    assert app.title[0].value == "From narratives to insight"
    for page in ["Case explorer", "Narrative lab", "Model evaluation", "Topics", "Pipeline & syllabus", "Learning guide"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"{page}: {app.exception}"
        if page == "Narrative lab":
            app.button[0].click().run()
            assert not app.exception
            assert app.metric[0].value
            from crime_nlp.text import sentences
            result = app.session_state["lab_result"]
            assert len(sentences(result["summary"])) == 1
            assert "\n" not in result["summary"]
            assert result["structured_summary"]["citations"]


def test_case_search_empty_results():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value("Case explorer").run()
    app.text_input[0].set_value("zzzzzzunmatchabletoken").run()
    assert not app.exception
    assert any("No matching reports" in item.value for item in app.info)


def test_hybrid_search_and_pagination():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value("Case explorer").run()
    app.number_input[0].set_value(2).run()
    assert not app.exception
    app.text_input[0].set_value("stolen mobile phone").run()
    next(radio for radio in app.radio if radio.label == "Search method").set_value("Hybrid").run()
    assert not app.exception
    assert app.number_input[0].value == 1
    assert len(app.dataframe) > 0


def test_informal_input_review_and_stale_result_handling():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value("Narrative lab").run()
    app.checkbox[0].uncheck().run()
    app.text_area[0].set_value("sir my moblie stoln from pocket in bus ystrday").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == "Theft"
    app.text_area[0].set_value("please help").run()
    assert any("text has changed" in item.value for item in app.info)
    app.button[0].click().run()
    assert not app.exception
    assert any("Needs review" in item.value for item in app.warning)
    app.text_area[0].set_value("12345").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("plain-text description" in item.value for item in app.warning)


def test_camera_footage_is_evidence_in_narrative_lab():
    from streamlit.testing.v1 import AppTest
    demo = ("Rajiv hit Sneha with a stick near Pune on 29-06-2026. "
            "A witness called the police. Camera footage was collected.")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value("Narrative lab").run()
    app.checkbox[0].uncheck().run()
    app.text_area[0].set_value(demo).run()
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["lab_result"]
    assert result["text"] == demo
    assert ("Camera footage", "EVIDENCE") in {(e["text"], e["label"]) for e in result["entities"]}


@pytest.mark.parametrize("case_id", ["user_international_witness", "user_informal_witness"])
def test_user_witness_cases_in_dashboard(case_id):
    from streamlit.testing.v1 import AppTest
    from crime_nlp.challenges import cases
    case = next(case for case in cases() if case["id"] == case_id)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value("Narrative lab").run()
    app.checkbox[0].uncheck().run()
    app.text_area[0].set_value(case["text"]).run()
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["lab_result"]
    expected = {(e["text"], e["label"]) for e in case["entities"]}
    assert {(e["text"], e["label"]) for e in result["entities"] + result["contextual_roles"]} == expected
    assert any("WITNESS" in element.value for element in app.markdown)


def test_petrol_is_highlighted_as_an_arson_instrument():
    from streamlit.testing.v1 import AppTest
    from crime_nlp.material_cases import CASES
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value("Narrative lab").run()
    app.checkbox[0].uncheck().run()
    app.text_area[0].set_value(CASES[0][1]).run()
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["lab_result"]
    assert ("petrol", "WEAPON") in {(entity["text"], entity["label"]) for entity in result["entities"]}
    assert any("petrol<small" in item.value and "WEAPON" in item.value for item in app.markdown)
    assert any("accelerants" in item.value for item in app.caption)


def test_supplier_sentence_reuses_fuel_and_preserves_two_locations():
    from streamlit.testing.v1 import AppTest
    text = "Suresh used petrol from Indian Oil in Vanasthalipuram and used it to burn Ramesh's flat in Hasthinapuram"
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value("Narrative lab").run()
    app.checkbox[0].uncheck().run()
    app.text_area[0].set_value(text).run()
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["lab_result"]
    expected = {
        ("Suresh", "SUSPECT"), ("petrol", "WEAPON"), ("Vanasthalipuram", "LOCATION"),
        ("Ramesh", "VICTIM"), ("flat", "PROPERTY"), ("Hasthinapuram", "LOCATION")}
    from crime_nlp.trained_roles import contract
    if contract():
        expected.add(("Indian Oil", "ORGANIZATION"))
    assert {(entity["text"], entity["label"]) for entity in result["entities"]} == expected


def test_entity_editor_saves_private_draft_and_resets_opt_in(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from crime_nlp import store
    import sqlite3
    monkeypatch.setattr(store, "PROCESSED", tmp_path)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    app.sidebar.radio[0].set_value("Narrative lab").run()
    app.checkbox[0].uncheck().run()
    text = "Ravi hit Asha near Pune."
    app.text_area[0].set_value(text).run()
    app.button[0].click().run()
    include = next(item for item in app.checkbox if item.label == "Include this narrative in the local annotation queue")
    assert not include.value
    next(button for button in app.button if button.label == "Save entity corrections").click().run()
    assert not app.exception
    with sqlite3.connect(tmp_path / "feedback.sqlite") as connection:
        payload = connection.execute("SELECT includes_text,payload_json FROM entity_feedback").fetchone()
    assert payload[0] == 0 and text not in payload[1] and "Ravi" not in payload[1]
    next(item for item in app.checkbox if item.label == "Include this narrative in the local annotation queue").check().run()
    app.text_area[0].set_value("Arjun hit Meera near Delhi.").run()
    app.button[0].click().run()
    assert not next(item for item in app.checkbox if item.label == "Include this narrative in the local annotation queue").value
