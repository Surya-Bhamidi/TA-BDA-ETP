"""Create an auditable results sheet and standalone printable project report."""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import ROOT, ARTIFACTS, SETTINGS, write_json


def read(name):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def main():
    from markdown_it import MarkdownIt
    spark, evaluation, ner = read("spark_metrics.json"), read("evaluation.json"), read("ner_evaluation.json")
    topics, enriched = read("topics.json"), read("enrichment.json")
    lines = ["# Executed project results", "", "Generated from saved artifacts; all scores below are measured, not illustrative.", "",
             "## Data and execution", "", f"* Raw reports: **{spark['raw_rows']:,}**", f"* Clean unique reports: **{spark['clean_rows']:,}**",
             f"* Invalid rows removed: **{spark['invalid_rows_removed']:,}**", f"* Duplicates removed: **{spark['duplicates_removed']:,}**",
             f"* Spark: **{spark['spark_version']}**, master **{spark['master']}**, **{spark['input_partitions']}** input partitions",
             f"* Enriched narratives: **{enriched['rows']:,}**", "", "## Same-split model comparison", "",
             f"Train / validation / development / final test: {evaluation['protocol']['train_rows']:,} / {evaluation['protocol']['validation_rows']:,} / {evaluation['protocol']['development_rows']:,} / {evaluation['protocol']['test_rows']:,}.", "",
             "| Model | Accuracy | Macro F1 | Weighted F1 |", "|---|---:|---:|---:|"]
    for name, result in evaluation["models"].items():
        lines.append(f"| {name} | {result['accuracy']:.4f} | {result['macro_f1']:.4f} | {result['weighted_f1']:.4f} |")
    if evaluation["protocol"].get("evaluation_scope"):
        lines += ["", evaluation["protocol"]["evaluation_scope"]]
    lines += ["", "## Entity extraction", "", f"Validation-selected {ner['model']} strict entity F1: **{ner['strict_entity_f1']:.4f}** on {ner['test_rows']} evaluation narratives.", "", ner.get("split", ""), "",
              "| Entity | Precision | Recall | F1 |", "|---|---:|---:|---:|"]
    for label in ner["labels"]:
        result = ner["report"][label]
        lines.append(f"| {label} | {result['precision']:.4f} | {result['recall']:.4f} | {result['f1-score']:.4f} |")
    if "ablation" in ner:
        a = ner["ablation"]
        lines += ["", f"Original CRF test F1: {a['initial_test_f1']:.4f}. Validation F1 changed from {a['initial_validation_f1']:.4f} to {a['context_validation_f1']:.4f} after adding sentence-level observations.", "", a["disclosure"]]
    weak = [label for label in ner["labels"] if ner["report"][label]["recall"] < .8]
    if weak:
        lines += ["", "**Material limitation:** held-out recall is below 80% for " + ", ".join(weak) + ". The overall entity F1 is dominated by more frequent types; consult each row above. Review predicted roles against the original text."]
    lines += ["", "## Distributed baseline and topics", "",
              f"MLlib accuracy: **{spark['mllib']['accuracy']:.4f}**; weighted F1: **{spark['mllib']['weighted_f1']:.4f}**. It uses {spark['mllib']['train_rows']:,} training and {spark['mllib']['test_rows']:,} test rows, so sample sizes differ from the four-model comparison.", "",
              f"LDA held-out perplexity: **{topics['held_out_perplexity']:.2f}**; top-term diversity: **{topics['topic_diversity']:.3f}**.", "", "## Verification", ""]
    test_log = ROOT / "logs" / "tests.log"
    if test_log.exists():
        content = test_log.read_text(encoding="utf-16" if test_log.read_bytes().startswith(b"\xff\xfe") else "utf-8", errors="replace")
        summary = [s.strip() for s in content.splitlines() if re.search(r"\d+ passed", s)]
        lines.append("Pytest: **" + (summary[-1].strip("= ") if summary else "See logs/tests.log for the run outcome") + "**.")
    scala_log = ROOT / "logs" / "scala.log"
    if scala_log.exists():
        content = scala_log.read_text(encoding="utf-16" if scala_log.read_bytes().startswith(b"\xff\xfe") else "utf-8", errors="replace")
        lines.append("\nScala report count verified: **60,000**." if "SCALA_VALID_REPORTS=60000" in content else "\nScala execution: inspect logs/scala.log.")
        if "Exception while deleting Spark temp dir" in content:
            lines.append("\nThe Windows Scala run completed its computations and exited successfully, with a non-fatal Spark temporary-JAR cleanup warning at shutdown.")
    browser_audit = ARTIFACTS / "feature_audit_browser.json"
    browser = ARTIFACTS / "browser_check.json"
    if browser_audit.exists():
        result = read("feature_audit_browser.json")
        passed = sum(check["status"] == "passed" for check in result["checks"])
        lines.append(f"\nCurrent browser feature audit: **{result['status']}**, with **{passed}/{len(result['checks'])}** workflows passed. See docs/FEATURE_AUDIT.md for the full feature matrix and remaining limitations.")
    elif browser.exists():
        result = read("browser_check.json")
        lines.append(f"\nBrowser smoke check: **{result['status']}**. Pages visited: {', '.join(result['pages'])}.")
    lines += ["", "## Classification uncertainty and calibration", "", "| Model | 95% accuracy interval |", "|---|---|"]
    for name, result in evaluation["models"].items():
        low, high = result["accuracy_ci95"]
        lines.append(f"| {name} | {low:.4f} to {high:.4f} |")
    lines += ["", "Intervals use 400 report-level bootstrap resamples and do not cover new-source uncertainty.", "",
              f"Training-only grouped TF-IDF CV mean macro-F1: **{evaluation['grouped_cv']['mean']:.4f}**.", "",
              "| TF-IDF final-test calibration | ECE | Log loss | Coverage at 0.60 | Accepted accuracy |", "|---|---:|---:|---:|---:|"]
    for name in ("before", "after"):
        item = evaluation["calibration_test"][name]
        accepted = f"{item['accuracy_when_accepted']:.4f}" if item["accuracy_when_accepted"] is not None else "not applicable"
        lines.append(f"| {name} | {item['ece']:.4f} | {item['log_loss']:.4f} | {item['coverage']:.4f} | {accepted} |")
    lines += ["", "Temperature is fitted on validation only. Calibration is not guaranteed to improve under a shifted test distribution.", "",
              "## Both entity models", "", "| Model | Validation strict F1 | Diagnostic evaluation strict F1 |", "|---|---:|---:|"]
    for name, result in ner["models"].items():
        lines.append(f"| {name} | {ner['validation'][name]:.4f} | {result['strict_entity_f1']:.4f} |")
    robustness = ARTIFACTS / "robustness_development.json"
    if robustness.exists():
        comparisons = read("robustness_development.json")
        lines += ["", "## Informal English development challenges", "", comparisons["scope"], "",
                  "| Release | Category cases correct / 32 | Exact people + location cases / 8 |", "|---|---:|---:|"]
        for version in ("v2", "v3", "v3.1"):
            if version in comparisons:
                result = comparisons[version]
                lines.append(f"| {version} | {sum(row['correct'] for row in result['cases'])} | {result['exact_role_cases']} |")
        lines += ["", "These hand-written examples informed development. They are not an untouched final benchmark or evidence of flawless real-world performance. Every case is retained in artifacts/robustness_development.json."]
    verification = ARTIFACTS / "evaluation_verification.json"
    if verification.exists():
        checked = read("evaluation_verification.json")
        current_freeze = read("final_evaluation_receipt.json")["frozen_at"]
        if checked.get("frozen_at") == current_freeze:
            lines += ["", "## Independent score recomputation", "", f"Verification: **{checked['status']}**. Recomputed classification metrics on {checked['document_rows']} saved predictions and entity span F1 on {checked['entity_rows']} narratives. Frozen model/prediction hashes, reference spans and test IDs were checked. No models were fitted during verification."]
    if enriched.get("summary"):
        lines += ["", "## Case summaries", "", f"All {enriched['rows']:,} enriched cases include a one-line TextRank summary in Parquet, SQLite and case-search exports. " + enriched["summary"] + ". Summary quality has not been evaluated against human reference summaries."]
    context_receipt = ARTIFACTS / "evidence_context_validation.json"
    if context_receipt.exists():
        context = read("evidence_context_validation.json")
        lines += ["", "## Initial media evidence correction (historical)", "", "This receipt covers the earlier narrow media-only rule. The current broader context layer is measured separately below.", "", context["scope"], "",
                  "The reported Camera footage false-person label is corrected to EVIDENCE by a disclosed context check. Rule corrections have no model probability.", "",
                  f"CRF validation Span-F1: {context['crf_validation']['raw_span_f1']:.4f} before and {context['crf_validation']['context_checked_span_f1']:.4f} after. CRF diagnostic Span-F1: {context['test']['CRF']['raw_span_f1']:.4f} before and {context['test']['CRF']['context_checked_span_f1']:.4f} after. Original model weights and reference labels are unchanged."]
    if (ARTIFACTS / "narrative_context_validation.json").exists():
        context = read("narrative_context_validation.json")
        audit, crosscheck = read("narrative_audit_after.json"), read("narrative_crosscheck.json")
        lines += ["", "## Legacy eight-label context diagnostics", "", context["scope"], "",
                  "These core/context measurements precede the separately scored ten-label role upgrade below. Witness clauses are separated from locations; context repairs have no model probability. Legacy learned weights remain unchanged.", "",
                  "| Diagnostic dataset | Raw Span-F1 | Context pipeline Span-F1 |", "|---|---:|---:|"]
        for name, result in [("CRF validation", context["crf_validation"]), *context["test"].items()]:
            lines.append(f"| {name} | {result['raw_span_f1']:.4f} | {result['context_span_f1']:.4f} |")
        lines += ["", "These are previously viewed synthetic cases. Validation and diagnostic-test changes can differ; the context layer is not an across-the-board benchmark improvement. Some added mentions and article boundaries differ from generator annotations.", "",
                  f"Hand-annotated development audit: {audit['summary']['all_requirements_passed']}/{audit['summary']['cases']} cases meet their specified requirements. Separate variations: {crosscheck['summary']['all_requirements_passed']}/{crosscheck['summary']['cases']}. These examples informed repairs and are not a real-world accuracy estimate. See docs/NARRATIVE_VALIDATION.md."]
        if (ARTIFACTS / "entity_store_validation.json").exists():
            stored = read("entity_store_validation.json")
            lines += ["", f"Stored entity structural validation: **{stored['status']}** across {stored['rows']:,} reports; {stored['failures']} source-offset/overlap failures. This checks data integrity, not semantic correctness."]
    improvement_path = ARTIFACTS / "accuracy_improvements.json"
    if (ARTIFACTS / "material_context_after.json").exists():
        materials = read("material_context_after.json")["summary"]
        lines += ["", "## Petrol and material roles", "",
                  f"Automatic entity pipeline: {materials['passed']}/{materials['cases']} material-role development checks passed. Petrol used to set a shop on fire is highlighted as WEAPON (accelerant); theft and forensic collection contexts are distinguished. Corrections have no model probability. These are development checks, not an independent accuracy estimate. See docs/MATERIAL_HIGHLIGHTING.md for scope and diagnostic tradeoffs."]
        if (ARTIFACTS / "material_context_bert.json").exists():
            bert_materials = read("material_context_bert.json")["summary"]
            lines += ["", f"BERT with context layer: {bert_materials['passed']}/{bert_materials['cases']} material-role checks passed."]
    if improvement_path.exists():
        improvements = read("accuracy_improvements.json")
        lines += ["", "## Validation-selected accuracy revision", "", improvements["scope"], "",
                  "| Model | Previous accuracy | Current accuracy | Change (percentage points) |",
                  "|---|---:|---:|---:|"]
        for name, item in improvements["classification"].items():
            lines.append(f"| {name} | {item['before_accuracy']:.4f} | {item['after_accuracy']:.4f} | {item['accuracy_change_pp']:+.2f} |")
        lines += ["", "| Entity model | Previous strict Span-F1 | Current strict Span-F1 | Change (percentage points) |", "|---|---:|---:|---:|"]
        for name, item in improvements["entities"].items():
            lines.append(f"| {name} | {item['before_span_f1']:.4f} | {item['after_span_f1']:.4f} | {item['change_pp']:+.2f} |")
        for name, rejected in improvements.get("rejected_candidates", {}).items():
            lines += ["", f"**Rejected {name} experiment:** validation-selected candidate accuracy was {rejected['after_accuracy']:.2%}, versus {rejected['before_accuracy']:.2%} for the baseline. The baseline is retained for deployment. This release rollback used the viewed test results; no new fitting or tuning followed. Candidate evidence is preserved under artifacts/experiments/accuracy_20261107."]
        lines += ["", "Training sizes and evaluation cases are unchanged. Model tuning used validation; any test-informed release rollback is disclosed above. The viewed synthetic evaluation does not establish real-world accuracy. See docs/ACCURACY_IMPROVEMENTS.md for recipes and reproduction."]
    if (ARTIFACTS / "hybrid_entity_evaluation.json").exists():
        hybrid = read("hybrid_entity_evaluation.json")
        lines += ["", "## Schema-4 generalization upgrade", "", hybrid["scope"], "",
                  "| Same ten-label diagnostic | Strict Span-F1 |", "|---|---:|"]
        for name, result in hybrid["splits"]["test"].items():
            if name != "rows":
                lines.append(f"| {name.replace('_', ' ')} | {result['strict_entity_f1']:.4f} |")
        specialist = read("role_specialist_broad_evaluation.json")
        lines += ["", "The selected CRF role specialist adds ORGANIZATION/WITNESS while preserving the legacy core. These task-specific results are not comparable with the legacy eight-label table.", "",
                  f"Role-specialist validation F1: {specialist['validation']['CRF']:.4f}; diagnostic F1: {specialist['models']['CRF']['strict_entity_f1']:.4f}. Its high precision and limited recall are documented in UPGRADE_STATUS.md."]
        classifier = read("windowed_classification_experiment.json")
        window_eval = read("windowed_classification_evaluation.json")
        lines += ["", f"Windowed BERT candidate validation macro-F1: {classifier['candidate_validation_macro_f1']:.4f}, versus retained {classifier['legacy_validation_macro_f1']:.4f}. The candidate was rejected. Full-document inference with the retained weights scores {window_eval['full_document']['accuracy']:.4f} on the viewed diagnostic test; no classification accuracy gain is claimed."]
        delivery = read("upgrade_delivery.json")
        lines += ["", f"Source-window retrieval: {delivery['semantic_windows']:,} windows cover {delivery['corpus_rows']:,} reports. Source text, offsets and ordering were verified. The dashboard includes entity correction, opt-in annotation drafts, qualified action/reference notes and per-entity review.", "",
                  "The annotation contract supports ten labels. Agreement, adjudication and source-separated preparation tools are provided. Thirty-six human quality-review forms remain unreviewed; real-source accuracy and human summary/search/topic quality are not measured. See UPGRADE_STATUS.md, ANNOTATION_GUIDELINES_V4.md and REAL_DATA_WORKFLOW.md."]
    lines += ["", "## Interpretation", "", "The corpus is synthetic. Scenario groups are separated, but vocabulary and generator conventions are shared. Model and prediction hashes record the release state; repeat evaluation, prediction reuse and any test-informed release rollback are disclosed in the evaluation receipts and above. No human-reference accuracy is claimed for syntax, sentiment, threat heuristics, search or summarization. " + spark["deployment_note"] + " Docker and multi-host deployment were not exercised. Version benchmarks use different corpora and protocols.", "",
              "Resources: `artifacts/evaluation.json`, `ner_evaluation.json`, `spark_metrics.json`, `topics.json`, split ID CSVs, and `logs/`."]
    markdown = "\n".join(lines) + "\n"
    (ROOT / "docs" / "RESULTS.md").write_text(markdown, encoding="utf-8")
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    source = (ROOT / "docs" / "REPORT.md").read_text(encoding="utf-8")
    body = parser.render(source) + '<div class="new-page"></div>' + parser.render(markdown)
    style = "body{font:14px/1.65 Arial,sans-serif;color:#203245;max-width:900px;margin:45px auto;padding:0 30px}h1{font-size:30px;color:#0c655f;line-height:1.3}h2{font-size:20px;margin-top:30px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #d7e1e6;padding:8px;text-align:left}th{background:#edf4f3}code{font-size:12px;background:#f0f3f5;padding:2px 4px}a{color:#126b78}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f5;padding:12px}td,code{overflow-wrap:anywhere}p,li{orphans:3;widows:3}h1,h2,h3{break-after:avoid}@media print{body{margin:0;max-width:none;font-size:11px}.new-page{break-before:page}h1{font-size:25px}h2{font-size:17px}a{color:inherit}table{break-inside:avoid}}"
    (ROOT / "docs" / "PROJECT_REPORT.html").write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Decoding Crime Narratives — Project Report</title><style>' + style + '</style></head><body>' + body + '</body></html>', encoding="utf-8")
    write_json(ARTIFACTS / "delivery_manifest.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "version": SETTINGS["project"]["version"], "project": "Decoding Crime Narratives using NLP and Big Data Analytics", "reports": spark["clean_rows"], "spark_master": spark["master"], "models": list(evaluation["models"]), "final_models_frozen": True, "docker_executed": False})
    print("Created docs/RESULTS.md and docs/PROJECT_REPORT.html from executed artifacts.")


if __name__ == "__main__":
    main()
