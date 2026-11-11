# BERT, CRF, pattern discovery, evaluation and local deployment

The completed implementation uses Scala 2.12.18 / Spark 3.5.5 for the main data-processing and MLlib stage, with Python for the NLP models and dashboard. The [backend comparison](SPARK_BACKEND_COMPARISON.md) verifies unchanged full-corpus data and predictions, fresh NLP scores and 281 passing regression tests. [SCALA_SPARK.md](SCALA_SPARK.md) documents execution and output contracts.

This implementation retains eight crime categories and the legacy eight-label
core, with a separately trained ORGANIZATION/WITNESS specialist bringing the
supported annotation schema to ten entity types. See [UPGRADE_STATUS.md](UPGRADE_STATUS.md).
The existing 60,000-report synthetic corpus is retained. This rebuild fine-tunes
the compact pretrained BERT classifier, trains CRF and BERT entity models, fits
LDA/NMF topics, and enriches every case for the local Streamlit dashboard.

## Requirement mapping

| Requirement | Implementation | Evidence |
|---|---|---|
| Context-aware BERT classification | All layers of pretrained two-layer BERT fine-tuned on 12,000 reports; six-epoch scheduled candidates versus the four-epoch baseline; checkpoint chosen by validation macro-F1 | `crime_nlp/bert_model.py`, `artifacts/evaluation.json`, `docs/ACCURACY_IMPROVEMENTS.md` |
| CRF entity extraction | L-BFGS CRF with context, lexical shape and BIO labels; 8,000 training narratives including ownership and administrative contexts | `crime_nlp/train.py`, `artifacts/ner_evaluation.json` |
| LDA pattern discovery | Eight topics over training-only content-word lemmas; held-out perplexity, coherence, diversity and NMF comparison | `crime_nlp/topics.py`, Topics page, `artifacts/topics.json` |
| One-line case summaries | TextRank ranks sentences; one sentence is retained and whitespace normalized to one line. Original narratives and summary source citations remain available | `crime_nlp/text.py`, `privacy.py`, `enrich.py`; Case explorer and Narrative lab |
| Accuracy, F1 and strict Span-F1 | Fixed splits; fitting/checkpoint selection restricted to train/validation; frozen weights before evaluation; saved row-level predictions and independent metric recomputation | `scripts/verify_evaluation.py`, `artifacts/evaluation_verification.json` |
| Local presentation layer | Seven-page Streamlit dashboard on `http://localhost:8502`, with keyword/semantic/hybrid search, filters, charts, summaries and exports | `app.py`, `START_DASHBOARD.cmd`, UI tests and browser smoke check |

## Evaluation protocol and its limits

Document train/validation/development/test sizes are 12,000/800/800/1,200.
Entity train/validation/evaluation sizes are 8,000/400/600. Before fitting,
cross-split report ID, scenario-group and normalized-text overlap are rejected.
Split fingerprints record IDs and original text. The model freeze records
weight hashes, source hashes and the evaluation plan.

The document test split has already been viewed in earlier releases. The entity
evaluation includes a previously viewed subset and generated context examples
sharing grammar. Repeated scores are diagnostic synthetic evaluations, not a
new untouched holdout or evidence of real-source generalization. No tuning uses
test scores in this rebuild. A new independent dataset is needed for future
model selection or real-world accuracy claims.

Row-level predictions are stored locally in:

- `data/processed/evaluation_predictions.parquet`: report IDs, reference categories and four model predictions.
- `data/processed/entity_evaluation_predictions.jsonl`: original narratives, reference spans, reference BIO tags and both entity models' predictions.

`scripts/verify_evaluation.py` verifies frozen model hashes, prediction hashes,
freeze/evaluation ordering, test IDs, source categories and entity span alignment,
then recomputes Accuracy, macro/weighted F1 and strict entity Span-F1 without
fitting a model. The verification receipt and regenerated `docs/RESULTS.md`
provide the executed results.

Summaries select source sentences and normalize whitespace; they do not invent
or correct facts. Long source sentences can still produce long summaries.
Human-reference summary quality and topic interpretation remain unmeasured.

The schema-4 upgrade adds an entity editor, opt-in annotation drafts, two-reviewer
agreement and adjudicated source splitting, full-document search/BERT windows,
and qualified action/reference notes. Thirty-six unreviewed quality forms are
available; no human quality scores are claimed. The legacy benchmark remains
separate from the new synthetic diagnostic task.

## Reproduce the completed local workflow

Stop the dashboard before replacing models or rebuilding its data. Existing
models and evaluation artifacts were backed up under `.runtime/rebuild-backups/`
before this rebuild; that directory is excluded from Git.

```powershell
.\.venv312\Scripts\python.exe run_pipeline.py --stage train
.\.venv312\Scripts\python.exe run_pipeline.py --stage enrich
.\.venv312\Scripts\python.exe scripts\verify_evaluation.py
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe scripts\export_results.py
.\START_DASHBOARD.cmd
```

The dashboard binds to loopback and uses port 8502. It is a local deployment.
Models and resources run locally after initial downloads; no cloud hosting or
external LLM service is required. Optional browser verification uses
`scripts/browser_check.py` with the installed Playwright/Edge environment.
