"""Exercise the running dashboard in Edge; save audit results without changing models.

Run with the project Python. Playwright may be installed in the environment or
in .runtime/feature-audit-tools. Test downloads stay in .runtime/feature-audit.
Feedback persistence is tested separately by pytest using temporary databases.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".runtime" / "feature-audit-tools"))
from playwright.sync_api import sync_playwright, expect

NARRATIVE = "Vivek killed Barkha with a gun near Faridabad on 20-10-2026"
ATTEMPT_NARRATIVE = "Mayank had ill intentions towards Barkha, and he went to her cabin to kill her, but the attempt failed"
OUTPUT = ROOT / ".runtime" / "feature-audit"
TITLES = {
    "Overview": "From narratives to insight",
    "Case explorer": "Explore the case library",
    "Narrative lab": "Read between the lines",
    "Model evaluation": "Measure what the models learn",
    "Topics": "Discover recurring language",
    "Pipeline & syllabus": "A traceable research pipeline",
    "Learning guide": "Understand the entire project",
}


def settled(page):
    page.wait_for_timeout(300)
    page.wait_for_function("""() => {
        const app = document.querySelector('[data-testid="stApp"]');
        return app && app.getAttribute('data-test-script-state') === 'notRunning';
    }""", timeout=90000)
    expect(page.get_by_test_id("stException")).to_have_count(0)


def navigate(page, name):
    page.get_by_test_id("stSidebar").get_by_text(name, exact=True).click()
    expect(page.get_by_role("heading", name=TITLES[name], exact=True)).to_be_visible(timeout=90000)
    settled(page)


def select(page, label, name):
    control = page.get_by_test_id("stSelectbox").filter(has=page.get_by_text(label, exact=True))
    if control.get_by_role("combobox").input_value() == name:
        settled(page)
        return
    control.get_by_role("combobox").press("ArrowDown")
    page.get_by_role("option", name=name, exact=True).click()
    settled(page)


def analyze(page, text=NARRATIVE):
    page.get_by_label("Case narrative", exact=True).fill(text)
    page.get_by_role("button", name="Analyze narrative", exact=True).click()
    expect(page.get_by_text("Annotated narrative", exact=True)).to_be_visible(timeout=90000)
    settled(page)


def download(page, label):
    with page.expect_download() as event:
        page.get_by_role("button", name=label, exact=True).click()
    item = event.value
    path = OUTPUT / "downloads" / item.suggested_filename
    path.parent.mkdir(parents=True, exist_ok=True)
    item.save_as(path)
    assert item.failure() is None
    return path.read_bytes()


def upload(page, label, content, name="case.txt", mime="text/plain"):
    control = page.get_by_test_id("stFileUploader").filter(has=page.get_by_text(label, exact=True))
    control.locator('input[type="file"]').set_input_files({"name": name, "mimeType": mime, "buffer": content})
    settled(page)


def check_navigation(page):
    for name in TITLES:
        navigate(page, name)
    assert page.get_by_role("heading").count() > 2


def check_overview(page):
    navigate(page, "Overview")
    manifest = json.loads((ROOT / "artifacts" / "dataset_manifest.json").read_text(encoding="utf-8"))
    assault_count = manifest["category_counts"]["Assault"]
    expect(page.get_by_test_id("stPlotlyChart")).to_have_count(4, timeout=15000)
    expect(page.get_by_test_id("stMetricValue").first).to_have_text(f"{manifest['valid_unique_reports']:,}")
    category = page.get_by_test_id("stMultiSelect").filter(has=page.get_by_text("Crime categories", exact=True))
    category.get_by_role("combobox").click()
    page.get_by_role("option", name="Assault", exact=True).click()
    page.keyboard.press("Escape")
    settled(page)
    expect(page.get_by_test_id("stMetricValue").first).to_have_text(f"{assault_count:,}")
    expect(page.get_by_test_id("stMetricValue").nth(1)).to_have_text("1")
    district = page.get_by_test_id("stMultiSelect").filter(has=page.get_by_text("Districts", exact=True))
    district.get_by_role("combobox").fill("Delhi")
    page.get_by_role("option", name="Delhi", exact=True).click()
    page.keyboard.press("Escape")
    settled(page)
    expect(page.get_by_test_id("stMetricValue").nth(2)).to_have_text("1", timeout=15000)
    count = int(page.get_by_test_id("stMetricValue").first.inner_text().replace(",", ""))
    assert 0 < count < assault_count, f"Unexpected filtered report count: {count}"


def check_case_search(page):
    navigate(page, "Case explorer")
    query = page.get_by_label("Search narratives", exact=True)
    query.fill("stolen mobile phone")
    query.press("Enter")
    settled(page)
    for mode in ["Keyword", "Semantic", "Hybrid"]:
        page.get_by_text(mode, exact=True).click()
        settled(page)
        assert page.get_by_test_id("stDataFrame").count() > 0
        assert "No matching reports" not in page.inner_text("body")
    payload = download(page, "Export this page CSV")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
    assert len(rows) == 20 and {"report_id", "narrative", "summary"} <= set(rows[0])
    report_id = rows[0]["report_id"]
    query.fill(report_id)
    query.press("Enter")
    settled(page)
    assert report_id in page.inner_text("body")
    query.fill("")
    query.press("Enter")
    settled(page)
    select(page, "Entity type", "WEAPON")
    entity = page.get_by_label("Entity contains", exact=True)
    entity.fill("petrol")
    entity.press("Enter")
    settled(page)
    assert page.get_by_test_id("stDataFrame").count() > 0
    query.fill("zzzzzzunmatchabletoken")
    query.press("Enter")
    page.get_by_text("Keyword", exact=True).click()
    settled(page)
    expect(page.get_by_text("No matching reports. Try different words or broaden the filters.", exact=True)).to_be_visible()


def check_narrative_models(page):
    navigate(page, "Narrative lab")
    analyze(page)
    page.get_by_role("tab", name="Analysis details", exact=True).click()
    expect(page.get_by_text("Raw BERT eight-category suggestion:", exact=False)).to_be_visible()
    page.get_by_role("tab", name="Entities & summary", exact=True).click()
    for model in ["Automatic", "CRF", "BERT"]:
        select(page, "Entity extraction model", model)
        analyze(page)
        assert page.locator("mark").count() >= 4
        assert any("gun" in mention for mention in page.locator("mark").all_text_contents())
    page.get_by_text("Include BERT sequence classification", exact=True).click()
    analyze(page)
    expect(page.get_by_text("Raw BERT eight-category suggestion:", exact=False)).to_have_count(0)
    # Inspect every tab, source citations and download payloads.
    page.get_by_role("tab", name="Syntax & morphology", exact=True).click()
    expect(page.get_by_role("tabpanel", name="Syntax & morphology", exact=True).get_by_test_id("stDataFrame")).to_be_visible()
    assert page.locator("iframe").count() > 0
    expect(page.frame_locator("iframe").locator("svg.displacy")).to_be_visible(timeout=15000)
    page.get_by_role("tab", name="Analysis details", exact=True).click()
    payload = json.loads(download(page, "Download analysis JSON"))
    assert payload["text"] == NARRATIVE
    assert payload["entities"] and payload["tokens"] and payload["explanation"]
    assert abs(sum(payload["topic_distribution"]) - 1) < 1e-6
    assert "bert_prediction" not in payload
    page.get_by_role("tab", name="Entities & summary", exact=True).click()
    redacted = download(page, "Download redacted narrative").decode("utf-8")
    assert "Vivek" not in redacted and "Barkha" not in redacted
    assert "gun" in redacted


def check_evidence(page):
    navigate(page, "Narrative lab")
    analyze(page)
    page.get_by_role("tab", name="Evidence search", exact=True).click()
    field = page.get_by_label("Find supporting evidence in this case", exact=True)
    for query in ["gun", "GUN!", "Faridabad", "20-10-2026"]:
        field.fill(query)
        field.press("Enter")
        settled(page)
        expect(page.get_by_text("Source: custom narrative, sentence 1", exact=True)).to_be_visible()
        expect(page.get_by_text(NARRATIVE, exact=True).last).to_be_visible()
    field.fill("zzzzzzunmatchabletoken")
    field.press("Enter")
    settled(page)
    expect(page.get_by_text("Insufficient evidence in this narrative.", exact=True)).to_be_visible()


def check_attempted_killing_sentiment(page):
    navigate(page, "Narrative lab")
    analyze(page, ATTEMPT_NARRATIVE)
    expect(page.get_by_test_id("stMetricValue").nth(0)).to_have_text("Attempted killing")
    expect(page.get_by_test_id("stMetricValue").nth(1)).to_have_text("Negative")
    expect(page.get_by_text("VADER compound: -0.8481", exact=True)).to_be_visible()
    assert page.get_by_test_id("stMetricValue").nth(0).evaluate("element => element.scrollWidth <= element.clientWidth + 1")
    expect(page.get_by_text("Outcome: Attempt failed.", exact=False).first).to_be_visible()
    expect(page.get_by_text("BERT sequence classification:", exact=False)).to_have_count(0)
    page.get_by_test_id("stMetric").first.evaluate("element => { element.style.scrollMarginTop = '90px'; element.scrollIntoView({block: 'start'}); }")
    page.screenshot(path=str(ROOT / "docs" / "screenshots" / "attempted-killing-sentiment.png"), full_page=True)
    page.get_by_role("tab", name="Analysis details", exact=True).click()
    expect(page.get_by_text("Raw TF-IDF eight-category suggestion: Robbery", exact=False)).to_be_visible()
    expect(page.get_by_text("Raw BERT eight-category suggestion: Robbery", exact=False)).to_be_visible()
    payload = json.loads(download(page, "Download analysis JSON"))
    assert payload["display_category"] == "Attempted killing"
    assert payload["incident_context"]["outcome"] == "Attempt failed"
    assert payload["sentiment"]["label"] == "Negative" and payload["sentiment"]["compound"] == -.8481
    assert {cue["text"] for cue in payload["sentiment"]["lexicon_cues"]} >= {"ill", "kill", "failed"}


def check_text_upload(page):
    navigate(page, "Narrative lab")
    label = "Optional: open a text case file"
    upload(page, label, ("\ufeff" + NARRATIVE).encode("utf-8"))
    expect(page.get_by_label("Case narrative", exact=True)).to_have_value(NARRATIVE)
    analyze(page)
    changed = "Ravi hit Asha near Pune. Camera footage was collected."
    upload(page, label, changed.encode("utf-8"))  # Deliberately reuse the filename.
    expect(page.get_by_label("Case narrative", exact=True)).to_have_value(changed)
    analyze(page, changed)
    upload(page, label, b"\xff\xfeinvalid", name="invalid.txt")
    expect(page.get_by_text("The uploaded text must use UTF-8 encoding.", exact=True)).to_be_visible()


def check_upload_limits(page):
    navigate(page, "Narrative lab")
    label = "Optional: open a text case file"
    upload(page, label, b"x" * 120001, name="large.txt")
    expect(page.get_by_text("Use a UTF-8 text file smaller than 120 KB.", exact=True)).to_be_visible()
    upload(page, label, b"x" * 30001, name="long.txt")
    expect(page.get_by_text("Use plain text with at most 30,000 characters and no null bytes.", exact=True)).to_be_visible()
    upload(page, label, b"An invalid\x00narrative", name="null.txt")
    expect(page.get_by_text("Use plain text with at most 30,000 characters and no null bytes.", exact=True)).to_be_visible()


def check_invalid_input(page):
    navigate(page, "Narrative lab")
    for text, message in [("", "Enter a narrative first."), ("12345", "Enter a plain-text description of what happened, including some words.")]:
        page.get_by_label("Case narrative", exact=True).fill(text)
        page.get_by_role("button", name="Analyze narrative", exact=True).click()
        settled(page)
        expect(page.get_by_text(message, exact=True)).to_be_visible()
    analyze(page)
    page.get_by_label("Case narrative", exact=True).fill("A new report.")
    page.get_by_label("Case narrative", exact=True).press("Tab")
    settled(page)
    expect(page.get_by_text("The text has changed. Select Analyze narrative to update the results.", exact=True)).to_be_visible()


def check_evaluation(page):
    navigate(page, "Model evaluation")
    for model in ["Bag of Words", "TF-IDF", "Word2Vec", "BERT"]:
        select(page, "Inspect model", model)
        assert page.get_by_test_id("stPlotlyChart").count() >= 4
        assert page.get_by_test_id("stDataFrame").count() >= 4
    payload = json.loads(download(page, "Download measured metrics"))
    assert set(payload["models"]) == {"Bag of Words", "TF-IDF", "Word2Vec", "BERT"}


def check_quality_upload(page):
    navigate(page, "Model evaluation")
    page.get_by_text("Evaluate summaries, search and topics", exact=True).click()
    label = "Upload reviewed quality judgments"
    fixture = {"task": "summary", "review_status": "reviewed", "reviewer": "feature-audit-fixture",
               "faithful_to_source": True, "preserves_participants": True, "preserves_negation": True, "covers_main_event": True}
    upload(page, label, json.dumps(fixture).encode(), name="review.jsonl", mime="application/jsonl")
    assert '"judgments":1' in page.inner_text("body").replace(" ", "")
    for payload in [b"not json", b"null", b"[]", b"1", b'"text"']:
        upload(page, label, payload, name="invalid-review.jsonl", mime="application/jsonl")
        assert page.get_by_test_id("stAlert").count() > 0


def check_topics(page):
    navigate(page, "Topics")
    control = page.get_by_test_id("stSelectbox").filter(has=page.get_by_text("Explore a topic", exact=True))
    control.get_by_role("combobox").press("ArrowDown")
    names = page.get_by_role("option").all_text_contents()
    page.keyboard.press("Escape")
    for name in names:
        select(page, "Explore a topic", name)
        assert page.get_by_test_id("stPlotlyChart").count() == 1
        assert page.get_by_test_id("stDataFrame").count() == 1
    page.get_by_text("LDA and NMF comparison", exact=True).click()
    expect(page.get_by_test_id("stJson").last).to_be_visible()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8502")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks = [check_navigation, check_overview, check_case_search, check_narrative_models, check_evidence, check_attempted_killing_sentiment,
              check_text_upload, check_upload_limits, check_invalid_input, check_evaluation, check_quality_upload, check_topics]
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        for check in checks:
            if args.only and check.__name__ not in args.only:
                continue
            context = browser.new_context(viewport={"width": 1440, "height": 1080}, accept_downloads=True)
            page = context.new_page()
            page.set_default_timeout(15000)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            started = time.perf_counter()
            result = {"feature": check.__name__, "status": "passed"}
            try:
                page.goto(args.url, wait_until="domcontentloaded")
                expect(page.get_by_role("heading", name=TITLES["Overview"], exact=True)).to_be_visible(timeout=90000)
                settled(page)
                check(page)
                settled(page)
                assert not errors, errors
            except Exception as error:
                result.update(status="failed", error=str(error), traceback=traceback.format_exc(limit=3), javascript_errors=errors)
                page.screenshot(path=str(OUTPUT / f"{check.__name__}.png"), full_page=True)
            result["seconds"] = round(time.perf_counter() - started, 2)
            results.append(result)
            print(json.dumps(result), flush=True)
            context.close()
        browser.close()
    receipt = {"verified_at": datetime.now(timezone.utc).isoformat(), "url": args.url,
               "checks": results, "status": "failed" if any(row["status"] == "failed" for row in results) else "passed"}
    target = ROOT / "artifacts" / ("feature_audit_browser.json" if not args.only else "feature_audit_browser_targeted.json")
    target.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    raise SystemExit(receipt["status"] != "passed")


if __name__ == "__main__":
    main()
