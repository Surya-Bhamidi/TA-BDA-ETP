# Scala and PySpark comparison — 11 November 2026

Replacing the PySpark data-processing implementation with Scala produced no
measured change in accuracy, F1, cleaned data or predictions. The current Scala
launcher was slower end to end. Its measured execution includes compiling the
Scala application on every invocation.

## Method

`scripts/compare_spark_backends.py --repeats 3` ran each backend three times in
fresh processes, sequentially with alternating order. Both used Spark 3.5.5,
`local[2]`, eight partitions, 2 GB configured driver memory, the same raw JSONL,
the same split assignments and the same MLlib settings. Every run independently
fitted the Spark MLlib pipeline; no NLP model was retrained or tuned.

The six rebuilt Spark datasets/models and fresh NLP predictions are isolated
under the directory recorded in `artifacts/spark_backend_comparison.json`.
Production data, saved NLP weights and historical prediction files were retained.

## Spark accuracy and data integrity

| Measurement | PySpark | Scala | Change |
|---|---:|---:|---:|
| Raw reports | 62,700 | 62,700 | 0 |
| Invalid rows removed | 1,200 | 1,200 | 0 |
| Duplicates removed | 1,500 | 1,500 | 0 |
| Clean reports | 60,000 | 60,000 | 0 |
| MLlib training reports | 39,528 | 39,528 | 0 |
| MLlib test reports | 6,824 | 6,824 | 0 |
| MLlib accuracy | 79.703986% | 79.703986% | 0 percentage points |
| MLlib weighted F1 | 78.949978% | 78.949978% | 0 percentage points |

Every cleaned column matched across the two implementations and the deployed
corpus in all three pairs. SQL trends and complete data-quality reports matched.
The comparison loaded each newly fitted model on a common held-out dataset and
compared all 6,824 predictions per pair: **zero prediction disagreements and a
maximum absolute probability difference of 0.0** in every pair.

## Fresh NLP inference on retained weights

Each classifier ran fresh inference on the original 1,200 document-test IDs
from both backend outputs. Accuracy, macro F1 and weighted F1 matched the saved
baseline exactly. All four classifiers also reproduced their saved row-level
predictions with zero disagreements. Live full-document BERT inference was
checked separately against its full-document baseline.

| Model | Baseline accuracy | PySpark-input accuracy | Scala-input accuracy | Change |
|---|---:|---:|---:|---:|
| Bag of Words | 83.50% | 83.50% | 83.50% | 0 percentage points |
| TF-IDF | 92.416667% | 92.416667% | 92.416667% | 0 percentage points |
| Word2Vec | 82.166667% | 82.166667% | 82.166667% | 0 percentage points |
| BERT benchmark | 81.666667% | 81.666667% | 81.666667% | 0 percentage points |
| BERT full document | 81.666667% | 81.666667% | 81.666667% | 0 percentage points |

Both entity taggers also ran fresh inference on their original 600 saved
evaluation cases, including the augmented synthetic entity cases. Their token
sequences and strict entity F1 matched the saved predictions exactly:

| Model | Baseline strict entity F1 | Fresh strict entity F1 | Change |
|---|---:|---:|---:|
| CRF | 92.227800% | 92.227800% | 0 percentage points |
| BERT NER | 79.118743% | 79.118743% | 0 percentage points |

The 24 frozen NLP model/configuration files were checked by SHA-256 before and
after inference and were unchanged. NER was evaluated on its retained test set;
the taggers were not independently retrained for each Spark language.

## Local execution time

| Backend | Round 1 | Round 2 | Round 3 | Median |
|---|---:|---:|---:|---:|
| PySpark | 26.131 s | 25.722 s | 25.434 s | 25.722 s |
| Scala | 29.783 s | 29.529 s | 29.549 s | 29.549 s |

Scala's median was **3.827 seconds, or 14.88%, longer** in this launcher. Timing
covers the fresh process, JVM startup, Scala compilation for Scala runs, ETL,
MLlib fitting/evaluation, output writes and process exit. The separately recorded
internal stage timings have different language startup boundaries and should
not be presented as an isolated Spark-engine speedup.

These are three local runs per backend on one computer with uncontrolled OS
caches. They do not establish performance on larger datasets or multiple hosts.
The result supports using Scala for the project's Scala/Spark requirement;
this migration did not improve predictive accuracy or end-to-end runtime.

## Reproduce and inspect

The complete regression suite passed: **281 tests, zero failures, zero errors,
zero skips**, in 45.15 seconds. Independent saved-evaluation verification also
passed, including model/prediction hashes, split IDs, source labels, reference
spans and recomputed scores. Repository lint checks passed.

```powershell
.\.venv312\Scripts\python.exe scripts\compare_spark_backends.py --repeats 3
.\.venv312\Scripts\python.exe scripts\verify_evaluation.py
.\.venv312\Scripts\python.exe -m pytest -q
```

- `artifacts/spark_backend_comparison.json`: all runs, exact prediction checks,
  fresh model scores, baseline deltas and timing scope.
- `artifacts/spark_backend_regressions.json`: saved-evaluation integrity and full
  regression-suite results from this comparison session.
- `logs/spark-backend-comparison.log` and `logs/spark-regression-tests.log`: local
  execution details.

These are diagnostic results on previously viewed synthetic test sets, not a
new independent real-world accuracy estimate.
