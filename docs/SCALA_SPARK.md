# Scala Spark execution

Scala has a main processing role in this project. `scala/CrimePipeline.scala`
implements the complete Spark stage; `scala/CrimeAnalytics.scala` remains an
additional typed-Dataset and functional-programming demonstration.

The [full backend comparison](SPARK_BACKEND_COMPARISON.md) records three runs per
backend, fresh NLP inference and regression checks. Accuracy was unchanged;
the current compile-on-launch Scala path took 14.88% longer end to end locally.

```text
Python fictional-data generator
  -> Scala Spark: JSONL ingestion, validation, normalization, deduplication
  -> Scala Spark: SQL trends, partitioned Parquet, MLlib classification
  -> Python: BERT, CRF, other NLP models, enrichment and retrieval
  -> Python Streamlit dashboard
```

## Run the Scala pipeline

From the project folder, using the existing Python 3.12 environment and Java 17:

```powershell
# Reads the existing raw JSONL and rebuilds the Spark outputs using Scala.
.\.venv312\Scripts\python.exe run_pipeline.py --stage spark --spark-engine scala
```

Scala is also the default in `settings.toml`, so `--stage spark` and an ordinary
full pipeline run use Scala without an extra flag. `--master "local[2]"` uses two
local execution threads. `scripts/run_cluster.py` runs the same Scala stage on
its one master and two workers on this computer. The compose cluster profile
uses the same default.

`crime_nlp/scala_runner.py` compiles both Scala source files using the Scala
2.12.18 compiler shipped with the installed Spark 3.5.5 distribution. It creates
`artifacts/crime-analytics-scala.jar` and submits the `CrimePipeline` class using
`spark-submit`. Python is the launcher; the Spark data transformations and
MLlib fitting execute in Scala on the JVM. No separate sbt installation or
Scala download is necessary. The `pyspark` Python package is retained to supply
the Spark distribution/compiler and support the optional comparison backend.

The stage retains the existing output contract:

| Output | Purpose |
|---|---|
| `data/processed/reports/` | Parquet reports partitioned by split |
| `data/processed/trends.parquet` | Single-file SQL aggregates |
| `artifacts/spark_mllib_model/` | Tokenization, stopwords, TF/IDF and logistic regression pipeline |
| `artifacts/spark_metrics.json` | Language, Scala/Spark versions, master, counts and MLlib metrics |
| `artifacts/data_quality.json` | Invalid rows, duplicates, schema and split/category counts |
| `artifacts/spark_execution_plan.txt` | Actual Spark physical execution plan |
| `artifacts/scala_pipeline_verification.json` | Successful completion, source and compiled-JAR hashes |
| `artifacts/scala_pipeline.log` | JVM execution log |

Original narratives are preserved, including spelling and Unicode characters,
because entity character offsets refer to the original text. Deduplication
retains the lowest report ID for each normalized-text hash, matching PySpark.

## Verify without rebuilding the dashboard dataset

```powershell
.\.venv312\Scripts\python.exe scripts\verify_scala_pipeline.py --full
```

The verifier runs both languages on a small fixture containing valid rows,
invalid dates/categories/splits, null/blank/punctuation-only narratives,
missing entity JSON, Unicode text and a duplicate with a lower report ID.
It compares every cleaned field, SQL trends, quality reports and held-out MLlib
metrics. It also loads the Scala-saved MLlib model through PySpark and scores it.

With `--full`, Scala additionally processes the full raw corpus and compares
every cleaned report and SQL trend against the deployed dataset. All outputs
stay under `.runtime/scala-verification/`. The receipt is
`artifacts/scala_migration_verification.json`. Saved dataset, cluster and NLP
benchmark receipts retain their original provenance; this check does not relabel
older runs as Scala runs or retrain the NLP models.

Stop the dashboard before rebuilding production Spark outputs. For the current
unchanged corpus, full verification establishes identical logical reports, so
the existing NLP models and dashboard dataset can be retained. The enrichment
code sorts reports by `report_id` before building its position-based indexes.

```powershell
.\.venv312\Scripts\python.exe run_pipeline.py --stage spark --spark-engine scala
```

If reports or preprocessing change later, run `run_pipeline.py --stage enrich`
to rebuild dashboard data and search indexes together. A changed corpus or
training recipe also requires the normal training and evaluation workflow.
The migration verifier avoids any production rebuild.

## Additional Scala demonstration and PySpark comparison

```powershell
# Typed Dataset operations, immutable collections, traits and pattern matching.
.\.venv312\Scripts\python.exe scripts\run_scala.py

# Original Spark implementation, if explicitly requested.
.\.venv312\Scripts\python.exe run_pipeline.py --stage spark --spark-engine pyspark
```

Scala source edits participate in stage signatures, so `--resume` cannot reuse
an old Spark stage after a Scala implementation change.

## Hadoop distinction

Scala is a programming language, Spark is the processing engine, HDFS is storage,
and YARN is a cluster manager. This project uses local Parquet and Spark local or
standalone execution. It does not deploy HDFS or YARN. Spark's bundled Hadoop
client libraries remain present. Switching to Scala does not remove Windows
native-library requirements; the existing trusted Hadoop native binaries support
Windows execution. The included Linux container avoids that Windows-specific setup.

For a presentation: "Our big-data stage is implemented in Scala using Apache
Spark SQL and MLlib. Python handles the NLP models and dashboard. Storage is
local Parquet, with no HDFS or YARN deployment."
