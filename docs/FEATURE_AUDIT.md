# Application feature audit — 11 November 2026

The local dashboard passed **240 automated tests with no skips**, **11 live Edge browser workflows**, the console demonstration, model-integrity checks, an isolated Spark pipeline, and the Scala corpus check. This verifies the tested functionality; it does not establish perfect predictions for arbitrary narratives.

The server at `http://localhost:8502` was restarted so that it loads the current evidence-search and upload code. Its health endpoint returned `ok` after testing.

| Area | What was verified | Result |
| --- | --- | --- |
| Navigation | All seven pages: Overview, Case explorer, Narrative lab, Model evaluation, Topics, Pipeline & syllabus, Learning guide | Passed in the browser and Streamlit tests |
| Overview | Four charts; category and district filters; counts checked against source data; date filtering; empty results | Passed |
| Case explorer | Keyword, Semantic and Hybrid search; exact report IDs; weapon/entity filters; empty results; page size; pagination reset; opening reports | Passed |
| Case export | Downloaded CSV parsed and checked for page length, report IDs, summaries and original narratives | Passed |
| Narrative entry | Blank and numeric-only input; changed-text detection; Unicode and original-text preservation | Passed |
| Text uploads | UTF-8 with BOM; replacement using the same filename; invalid encoding; size limit; character limit; null-byte rejection | Passed after fixing same-filename replacement |
| Classifiers | Live BoW, TF-IDF, Word2Vec and BERT console inference; dashboard BERT toggle; category confidence and review flags | Passed |
| Entity extraction | Automatic, CRF and BERT selection; participant and ownership roles; weapons, property, places, dates, evidence, organizations and witnesses; negation and assertion qualifiers | Passed on the included regression fixtures |
| Entities and summaries | Highlighted text; escaped HTML; extractive one-line summaries; source citations; structured notes; exact character offsets | Passed |
| Syntax and morphology | Dependency diagram rendered inside its iframe; syntax/POS/morphology table; sentence windows and long-document handling | Passed |
| Evidence search | Original gun example; casing and punctuation; names, dates and locations; semantic paraphrases; late source windows; missing match metadata; source citations and abstention tests | Passed, including the live server after restart |
| Analysis exports | Downloaded analysis JSON parsed and checked against input, entities, tokens, explanations and normalized topic distribution; redacted text checked for removed names | Passed |
| Category corrections | Correction saved with a text hash; original narrative omitted | Passed using a temporary database |
| Entity corrections | Offset/label validation; private saves; explicit opt-in for source text; annotation consent reset after changing input | Passed using temporary databases and schema tests |
| Optional password | Wrong password rejected; correct password opens the application | Passed in Streamlit tests |
| Model evaluation | All four model selections; charts and tables; downloaded measured metrics; recomputed saved classification and entity scores | Passed |
| Quality reviews | Valid submitted fixture calculated; malformed JSON and non-object JSON values rejected without crashing | Passed after fixing shape validation |
| Topics | Every topic selected; example reports and timeline rendered; LDA/NMF comparison expanded | Passed in the browser |
| Corpus and indexes | 60,000 stored reports; valid source offsets and non-overlapping spans; scenario/text split isolation; SQLite/report order; search-index alignment | Passed |
| Calibration and review workflows | Probability normalization; confidence abstention; annotation agreement/adjudication; source-separated splits; invalid-import quarantine and duplicate rejection | Passed |
| Model integrity | Frozen model hashes and prediction hashes; four classifier scores recomputed from 1,200 saved predictions; both core entity scores recomputed from 600 saved narratives; role-model experiments checked | Passed |
| Spark pipeline | Actual JSON ingestion, SQL cleaning, deterministic deduplication, split-partitioned Parquet and SQL aggregates; MLlib fit and held-out scoring on an isolated fixture | Passed: 132 raw → 128 clean, with 3 invalid rows and 1 duplicate removed |
| Saved MLlib model | Loaded the Java-backed pipeline through its JVM reader; checked 32 live predictions and probability normalization | Passed |
| Scala | Compiled and ran the Scala analytics against the saved corpus | Passed: 60,000 reports verified |
| Code checks | Repository Ruff configuration and whitespace/error checks | Passed |

## Problems corrected during this audit

1. Replacing an uploaded text file with a different file of the same name retained the previous narrative. The text widget now distinguishes uploads by their filename and content hash, while retaining manual edits across ordinary reruns.
2. Review files containing JSON values such as `null`, arrays, numbers or strings crashed the evaluation page. Review validation now requires each JSONL row to be an object and presents an ordinary validation error.
3. The already-running dashboard had retained the older evidence-search function in memory. Restarting the verified project server activated the keyword retrieval fix. The original narrative and its citation now appear when searching for `gun` in the live browser.

The earlier optional `match_type` compatibility fix is covered both with and without that field. No production model weights were retrained or replaced. Pipeline fixture writes and test downloads are stored under `.runtime/feature-audit`; correction tests use temporary databases.

## Limits and remaining observations

- **Prediction quality is not perfect.** Existing categories contain no separate Homicide class. The supplied killing example is suggested as Assault and is flagged for review. The recorded synthetic classifier accuracies are approximately 83.5% BoW, 92.4% TF-IDF, 82.2% Word2Vec and 81.7% BERT; they are not real-world accuracy estimates.
- Threat-language scoring remains a limited dictionary heuristic. For example, its current dictionary includes `handgun` but does not score `gun` or `killed`. Sentiment, topic quality and summary quality do not gain an independent accuracy benchmark from this audit. Review-upload tests used explicitly named synthetic fixtures.
- An **additional Python-RDD worker probe failed** on this installed Windows runtime, including when using the exact project interpreter. This affects the ordinary Python metadata path used by `PipelineModel.load`. The current application pipeline uses JVM-backed Spark operations; its complete isolated run passed, and the saved all-Java MLlib model was verified through its JVM reader. General Python-RDD worker execution remains unresolved. Diagnostic output is in `logs/feature-audit-worker.log`.
- Logs include a deprecation notice for `streamlit.components.v1.html`, although the syntax diagram still renders. Scala completed successfully but reported its existing temporary-JAR cleanup warning.
- Full production-corpus retraining, fresh resource downloads, Docker, standalone/multi-host cluster deployment, other browsers/devices, and extended load testing were not rerun. The deployed classifiers were exercised with their saved weights, and the frozen artifacts were checked.

## Evidence and reproduction

- [Final automated test results](../artifacts/feature_audit_pytest.xml): 240 passed, 0 failed, 0 skipped.
- [Full browser results](../artifacts/feature_audit_browser.json): all 11 workflows passed with no uncaught page exceptions or JavaScript page errors.
- [Supplementary browser checks](../artifacts/feature_audit_browser_targeted.json): rendered syntax diagram and manifest-based chart counts.
- [Spark fixture and saved-model results](../artifacts/feature_audit_pipeline.json).
- [Saved-score verification](../artifacts/evaluation_verification.json), [role-model verification](../artifacts/upgrade_model_verification.json), [Scala verification](../artifacts/scala_verification.json).
- Console demonstration: `logs/feature-audit-demo.log`, with its JSON output at `.runtime/feature-audit/demo.json`.
- Earlier failing audit receipts and screenshots are retained under `.runtime/feature-audit` for diagnosis.

From the project directory, with the dashboard running on port 8502:

```powershell
.\.venv312\Scripts\python.exe -m pytest -q --junitxml=artifacts/feature_audit_pytest.xml
.\.venv312\Scripts\python.exe scripts/check_features.py
.\.venv312\Scripts\python.exe scripts/check_pipeline_runtime.py
.\.venv312\Scripts\python.exe scripts/verify_evaluation.py
.\.venv312\Scripts\python.exe scripts/verify_upgrade_models.py
.\.venv312\Scripts\python.exe scripts/run_scala.py
.\.venv312\Scripts\ruff.exe check .
```

The browser audit uses the installed Microsoft Edge browser and Playwright from the project-local `.runtime/feature-audit-tools` directory. The Spark audit redirects pipeline writes to a fresh fixture directory and preserves production reports and models.
