# Executed project results

Generated from saved artifacts; all scores below are measured, not illustrative.

## Data and execution

* Raw reports: **62,700**
* Clean unique reports: **60,000**
* Invalid rows removed: **1,200**
* Duplicates removed: **1,500**
* Spark: **3.5.5**, master **spark://127.0.0.1:17077**, **8** input partitions
* Enriched narratives: **60,000**

## Same-split model comparison

Train / validation / development / final test: 12,000 / 800 / 800 / 1,200.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Bag of Words | 0.8350 | 0.8361 | 0.8361 |
| TF-IDF | 0.9242 | 0.9225 | 0.9225 |
| Word2Vec | 0.8217 | 0.8188 | 0.8188 |
| BERT | 0.8167 | 0.8139 | 0.8139 |

Repeat evaluation on previously viewed synthetic cases. BoW, TF-IDF, BERT and CRF candidates were selected on validation. Word2Vec deployment retained its baseline after observing a diagnostic test regression. This release choice used test results; it is not an untouched final benchmark. No further training or tuning was performed.

## Entity extraction

Validation-selected CRF strict entity F1: **0.9223** on 600 evaluation narratives.

Previously viewed synthetic test subset plus generated role/context cases sharing grammar; diagnostic evaluation, not untouched real-source validation.

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUSPECT | 0.9241 | 0.8134 | 0.8652 |
| VICTIM | 0.8891 | 0.8191 | 0.8527 |
| LOCATION | 0.9597 | 0.8389 | 0.8953 |
| WEAPON | 1.0000 | 0.9861 | 0.9930 |
| PROPERTY | 0.9778 | 0.9075 | 0.9413 |
| DATE | 1.0000 | 1.0000 | 1.0000 |
| TIME | 1.0000 | 1.0000 | 1.0000 |
| EVIDENCE | 1.0000 | 1.0000 | 1.0000 |

## Distributed baseline and topics

MLlib accuracy: **0.7970**; weighted F1: **0.7895**. It uses 39,528 training and 6,824 test rows, so sample sizes differ from the four-model comparison.

LDA held-out perplexity: **476.67**; top-term diversity: **0.900**.

## Verification

Pytest: **281 passed, 0 failed, 0 errors, 0 skipped**. Source: artifacts/spark_backend_regressions.json.

Scala report count verified: **60,000**.

The Windows Scala run completed its computations and exited successfully, with a non-fatal Spark temporary-JAR cleanup warning at shutdown.

Current browser feature audit: **passed**, with **12/12** workflows passed. See docs/FEATURE_AUDIT.md for the full feature matrix and remaining limitations.

## Classification uncertainty and calibration

| Model | 95% accuracy interval |
|---|---|
| Bag of Words | 0.8142 to 0.8558 |
| TF-IDF | 0.9083 to 0.9392 |
| Word2Vec | 0.8000 to 0.8425 |
| BERT | 0.7958 to 0.8392 |

Intervals use 400 report-level bootstrap resamples and do not cover new-source uncertainty.

Training-only grouped TF-IDF CV mean macro-F1: **0.6831**.

| TF-IDF final-test calibration | ECE | Log loss | Coverage at 0.60 | Accepted accuracy |
|---|---:|---:|---:|---:|
| before | 0.2826 | 0.5516 | 0.5617 | 0.9970 |
| after | 0.0332 | 0.2195 | 0.9683 | 0.9363 |

Temperature is fitted on validation only. Calibration is not guaranteed to improve under a shifted test distribution.

## Both entity models

| Model | Validation strict F1 | Diagnostic evaluation strict F1 |
|---|---:|---:|
| CRF | 0.9439 | 0.9223 |
| BERT | 0.8537 | 0.7912 |

## Informal English development challenges

Fixed hand-written development/regression examples; not an independent real-world benchmark.

| Release | Category cases correct / 32 | Exact people + location cases / 8 |
|---|---:|---:|
| v2 | 23 | 0 |
| v3 | 31 | 8 |
| v3.1 | 32 | 8 |

These hand-written examples informed development. They are not an untouched final benchmark or evidence of flawless real-world performance. Every case is retained in artifacts/robustness_development.json.

## Independent score recomputation

Verification: **passed**. Recomputed classification metrics on 1200 saved predictions and entity span F1 on 600 narratives. Frozen model/prediction hashes, reference spans and test IDs were checked. No models were fitted during verification.

## Case summaries

All 60,000 enriched cases include a one-line TextRank summary in Parquet, SQLite and case-search exports. One-sentence extractive TextRank, whitespace normalized to one line; source text retained. Summary quality has not been evaluated against human reference summaries.

## Initial media evidence correction (historical)

This receipt covers the earlier narrow media-only rule. The current broader context layer is measured separately below.

Narrow disclosed context-rule correction, not CRF retraining. Existing viewed synthetic cases are diagnostic; reported-example tests are development regressions. Raw model weights and benchmark predictions are retained.

The reported Camera footage false-person label is corrected to EVIDENCE by a disclosed context check. Rule corrections have no model probability.

CRF validation Span-F1: 0.9439 before and 0.9439 after. CRF diagnostic Span-F1: 0.9223 before and 0.9223 after. Original model weights and reference labels are unchanged.

## Legacy eight-label context diagnostics

Diagnostic on previously viewed synthetic evaluation; WITNESS is separate from the eight trained labels. Raw models and predictions unchanged.

These core/context measurements precede the separately scored ten-label role upgrade below. Witness clauses are separated from locations; context repairs have no model probability. Legacy learned weights remain unchanged.

| Diagnostic dataset | Raw Span-F1 | Context pipeline Span-F1 |
|---|---:|---:|
| CRF validation | 0.9439 | 0.9393 |
| CRF | 0.9223 | 0.9221 |
| BERT | 0.7912 | 0.8063 |

These are previously viewed synthetic cases. Validation and diagnostic-test changes can differ; the context layer is not an across-the-board benchmark improvement. Some added mentions and article boundaries differ from generator annotations.

Hand-annotated development audit: 38/38 cases meet their specified requirements. Separate variations: 17/17. These examples informed repairs and are not a real-world accuracy estimate. See docs/NARRATIVE_VALIDATION.md.

Stored entity structural validation: **passed** across 60,000 reports; 0 source-offset/overlap failures. This checks data integrity, not semantic correctness.

## Petrol and material roles

Automatic entity pipeline: 37/37 material-role development checks passed. Petrol used to set a shop on fire is highlighted as WEAPON (accelerant); theft and forensic collection contexts are distinguished. Corrections have no model probability. These are development checks, not an independent accuracy estimate. See docs/MATERIAL_HIGHLIGHTING.md for scope and diagnostic tradeoffs.

BERT with context layer: 37/37 material-role checks passed.

## Validation-selected accuracy revision

Repeat evaluation on previously viewed synthetic cases. BoW, TF-IDF, BERT and CRF candidates were selected on validation. Word2Vec deployment retained its baseline after observing a diagnostic test regression. This release choice used test results; it is not an untouched final benchmark. No further training or tuning was performed.

| Model | Previous accuracy | Current accuracy | Change (percentage points) |
|---|---:|---:|---:|
| Bag of Words | 0.8067 | 0.8350 | +2.83 |
| TF-IDF | 0.8825 | 0.9242 | +4.17 |
| Word2Vec | 0.8217 | 0.8217 | +0.00 |
| BERT | 0.7958 | 0.8167 | +2.08 |

| Entity model | Previous strict Span-F1 | Current strict Span-F1 | Change (percentage points) |
|---|---:|---:|---:|
| CRF | 0.9178 | 0.9223 | +0.45 |
| BERT | 0.7912 | 0.7912 | +0.00 |

**Rejected Word2Vec experiment:** validation-selected candidate accuracy was 75.75%, versus 82.17% for the baseline. The baseline is retained for deployment. This release rollback used the viewed test results; no new fitting or tuning followed. Candidate evidence is preserved under artifacts/experiments/accuracy_20261107.

Training sizes and evaluation cases are unchanged. Model tuning used validation; any test-informed release rollback is disclosed above. The viewed synthetic evaluation does not establish real-world accuracy. See docs/ACCURACY_IMPROVEMENTS.md for recipes and reproduction.

## Schema-4 generalization upgrade

Diagnostic comparison on previously viewed synthetic ten-label annotations; same current core/context rules in both arms

| Same ten-label diagnostic | Strict Span-F1 |
|---|---:|
| without trained roles | 0.5921 |
| with trained roles | 0.6230 |

The selected CRF role specialist adds ORGANIZATION/WITNESS while preserving the legacy core. These task-specific results are not comparable with the legacy eight-label table.

Role-specialist validation F1: 0.7729; diagnostic F1: 0.5410. Its high precision and limited recall are documented in UPGRADE_STATUS.md.

Windowed BERT candidate validation macro-F1: 0.9246, versus retained 0.9270. The candidate was rejected. Full-document inference with the retained weights scores 0.8167 on the viewed diagnostic test; no classification accuracy gain is claimed.

Source-window retrieval: 61,300 windows cover 60,000 reports. Source text, offsets and ordering were verified. The dashboard includes entity correction, opt-in annotation drafts, qualified action/reference notes and per-entity review.

The annotation contract supports ten labels. Agreement, adjudication and source-separated preparation tools are provided. Thirty-six human quality-review forms remain unreviewed; real-source accuracy and human summary/search/topic quality are not measured. See UPGRADE_STATUS.md, ANNOTATION_GUIDELINES_V4.md and REAL_DATA_WORKFLOW.md.

## Interpretation

The corpus is synthetic. Scenario groups are separated, but vocabulary and generator conventions are shared. Model and prediction hashes record the release state; repeat evaluation, prediction reuse and any test-informed release rollback are disclosed in the evaluation receipts and above. No human-reference accuracy is claimed for syntax, sentiment, threat heuristics, search or summarization. Standalone Spark master and worker processes; see cluster_execution.json for verified workers and host scope. Docker and multi-host deployment were not exercised. Version benchmarks use different corpora and protocols.

Resources: `artifacts/evaluation.json`, `ner_evaluation.json`, `spark_metrics.json`, `topics.json`, split ID CSVs, and `logs/`.

## Scala Spark implementation verification

The main Spark stage is Scala 2.12.18 / Spark 3.5.5. Comparison status: **passed**, with 3 fresh-process full-corpus runs per backend.

All cleaned fields, SQL trends, quality reports, held-out MLlib labels and probability vectors matched. Fresh NLP inference reproduced the saved classification and entity scores; retained model hashes were unchanged.

Local median end-to-end runtime: PySpark **25.722 seconds**, Scala **29.549 seconds** (+14.88% for Scala). Scala compilation on every invocation is included; no accuracy or end-to-end speed improvement is claimed.

See docs/SPARK_BACKEND_COMPARISON.md and artifacts/spark_backend_comparison.json for complete results. Historical dataset and cluster receipts retain their original run provenance.
