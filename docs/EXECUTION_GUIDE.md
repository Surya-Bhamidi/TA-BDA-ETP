# Execution methods, demonstration and benefits

This guide distinguishes live inference, saved execution evidence, rebuilding,
and experiments. The current corpus contains 60,000 fictional reports. Keep the
current weights for a presentation; training and rebuild commands replace files.

## 1. Recommended presentation setup

Open PowerShell in the project directory:

```powershell
cd "C:\Users\bhara\OneDrive\Desktop\TA AND BDA - Copy\Final Project"
$projectPython = ".\.venv312\Scripts\python.exe"
```

First try `http://localhost:8502`. If the dashboard is already running, use that
instance. Otherwise double-click `START_DASHBOARD.cmd` or run:

```powershell
& $projectPython -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

The console stays open while the server runs; Ctrl+C stops that foreground
instance. Models/data are already present in this completed workspace.

## 2. One-command console demonstration

Double-click `RUN_DEMO.cmd`, or run:

```powershell
& $projectPython scripts\demo_project.py
```

It performs live classification with BoW, TF-IDF, Word2Vec and BERT; CRF and BERT
entity extraction; TextRank summarization with source citations; spaCy syntax;
LDA topic assignment; VADER polarity and separate threat cues; keyword,
semantic and hybrid search; and cited evidence retrieval. It also displays
saved Spark cleaning counts and saved benchmark metrics, clearly labeled as
recorded evidence. It does not retrain or rerun Spark.

Other console modes:

```powershell
# Faster run: skip BERT comparisons, neural retrieval and semantic evidence.
& $projectPython scripts\demo_project.py --quick

# Your own narrative; original spelling is retained.
& $projectPython scripts\demo_project.py --text "sir my moblie stoln from pocket in bus ystrday" --question "phone"

# UTF-8 text file, a case-library query, and a within-case evidence query.
& $projectPython scripts\demo_project.py --file examples\demo_case.txt --query "stolen phone" --question "camera footage"

# Explicitly save the results, including the submitted text, locally.
& $projectPython scripts\demo_project.py --output artifacts\demos\demo.json
```

The default fictional narrative is:

> Rajiv hit Sneha with a stick near Pune on 29-06-2026. A witness called the police. Camera footage was collected.

In the checked run all four document models predicted Assault. CRF found Rajiv,
Sneha, stick, Pune and the date. The earlier “Camera footage” VICTIM error is
corrected to EVIDENCE by a disclosed media-evidence context check. Such
corrections have no model probability; the original text and offsets remain
intact. BERT labels that evidence correctly but can miss Sneha. A high category
score does not validate every entity. Evidence search can abstain even when a
human sees an answer; these are heuristic similarity thresholds, not verified
question answering. See EVIDENCE_CONTEXT_FIX.md for the correction and checks.

## 3. A 15-minute dashboard demonstration

The participant/witness fixes are documented in [NARRATIVE_VALIDATION.md](NARRATIVE_VALIDATION.md).
The [schema-4 upgrade](UPGRADE_STATUS.md) adds trained ORGANIZATION/WITNESS
recognition alongside the retained core. Context repairs and learned scores
remain distinguished. Run **RUN_UPGRADE_DEMO.cmd** for organizations, witnesses,
accelerants, action/reference notes and full-document search. Live ten-label
diagnostics and raw legacy eight-label F1 are separate evaluations.

| Time | Action | Explain |
|---|---|---|
| 0–1 min | State the problem and open Overview | Narrative text is difficult to search and aggregate. All displayed reports are fictional. |
| 1–3 min | Open Pipeline & syllabus | 62,700 raw rows became 60,000 clean reports after removing 1,200 invalid rows and 1,500 duplicates. Show the saved physical plan and worker record. |
| 3–4 min | Filter categories, districts and dates on Overview | Charts summarize the generated collection; they do not describe real crime rates. |
| 4–6 min | Search “stolen phone” in Case explorer | Compare Keyword, Semantic and Hybrid, apply an entity filter, change pages and export CSV with one-line summaries. Scores are retrieval similarities. |
| 6–10 min | Paste the default narrative into Narrative lab | Show four-way classification in the console, TF-IDF/BERT comparison in the UI, CRF/BERT entities, summary, original text, source citations and feature explanations. Review any mistakes. |
| 10–11 min | Open Syntax & morphology and Evidence search | Explain a noun, verb, lemma, morphology and dependency arrow. Search “Camera footage” to retrieve a source sentence. |
| 11–12 min | Open Topics | Show eight LDA topics, top terms, monthly distribution and representative reports. Topic IDs are learned themes, not predefined crime categories. |
| 12–14 min | Open Model evaluation | Explain Accuracy, macro F1, confusion matrices, exact Span-F1, confidence calibration and the before/after comparison. Disclose the rejected Word2Vec trial. |
| 14–15 min | Open Learning guide and conclude | Summarize benefits, local execution and limitations. Show the report and saved verification evidence. |

Useful extra examples:

- Informal spelling: `sir my moblie stoln from pocket in bus ystrday`.
- Roles: `Mayank stole the phone of Surya while he was sleeping on 29-06-2026`.
- Passive voice: `mayank was killed by surya on 29.06.2026`. Explain that homicide
  is outside the eight-category classifier and triggers review.
- Negation: `No weapon was used. Nobody was injured.` Compare threat-language
  cues with an affirmative description; polarity is a separate measurement.
- Insufficient information: `please help`. Show the request for review.

The report roles are allegations/predictions. Entity confidence is uncalibrated.
Show both successful outputs and mistakes rather than promising perfection.

## 4. Dashboard launch methods

| Method | Command or action | Use |
|---|---|---|
| Double-click | `START_DASHBOARD.cmd` | Easiest presentation launch. |
| Foreground Python | `& $projectPython -m streamlit run app.py` | See server logs and stop with Ctrl+C. Configuration selects port 8502. |
| PowerShell launcher | `.\scripts\launch.ps1` | Same local foreground launch with a fixed address/port. |
| Background launcher | `.\scripts\start_background.ps1` | Starts hidden, checks health, records PID and writes dashboard logs. |
| Alternate port | `.\scripts\start_background.ps1 -Port 8503` | Use when 8502 is occupied by another application; visit the selected port. |
| Console inference | `.\RUN_DEMO.cmd` | Demonstrate algorithms without needing the dashboard server. |

The background launcher refuses an occupied port; it does not stop unrelated
applications. Its logs are `logs/dashboard.log` and `logs/dashboard-error.log`.
Browser-check scripts currently target port 8502.

## 5. First-time setup on a new machine

Use Python 3.12 and Java 17. Windows Spark also needs compatible trusted Hadoop
native libraries; this workspace uses an existing local setup. Language models
are downloaded initially and then used locally. Large weights/data are excluded
from Git, so a source clone requires generation and training.

```powershell
py -3.12 -m venv .venv312
& $projectPython -m pip install -r requirements.txt
& $projectPython scripts\download_resources.py
```

Alternatively, `scripts/setup.ps1 -SkipPipeline` installs resources without
starting a rebuild. Its default invocation also runs the baseline pipeline.
Do not run setup/retraining as part of the current presentation.

## 6. Baseline pipeline: full or stage by stage

Stop the dashboard and preserve the current model/results before rebuilding.
The baseline pipeline does not reproduce the latest tuned models automatically;
use the accuracy workflow below after establishing the baseline.

```powershell
# All four baseline stages, in order.
& $projectPython run_pipeline.py

# Or execute stages individually, in this order.
& $projectPython run_pipeline.py --stage generate --records 60000
& $projectPython run_pipeline.py --stage spark
& $projectPython run_pipeline.py --stage train
& $projectPython run_pipeline.py --stage enrich
```

| Stage | What executes | Output and benefit |
|---|---|---|
| Generate | Deterministic fictional narratives, categories, exact spans and controlled faults | `data/raw/reports.jsonl`, dataset manifest; reproducible training and cleaning demonstrations. |
| Spark | Schema validation, normalization, deterministic deduplication, SQL aggregates and distributed MLlib | Partitioned Parquet, trend data, data-quality report and physical plan. |
| Train | Four document models, CRF/BERT entities, validation selection/calibration, freeze and evaluation, LDA/NMF | Saved weights, metrics, predictions and split IDs; measured model comparison. |
| Enrich | Apply TF-IDF, selected NER, LDA, VADER, threat cues and TextRank to the corpus; build indexes | `dashboard.parquet`, `cases.sqlite`, keyword matrix and MiniLM embeddings; interactive search/analysis. |

Training parameters can be overridden, for example:

```powershell
& $projectPython run_pipeline.py --stage train --train-limit 12000 --validation-limit 800 --test-limit 1200 --bert-epochs 4
```

Changing sample sizes changes the experiment; do not compare such scores as if
the protocol were unchanged. `settings.toml` holds other documented defaults.

## 7. Resume and selective refresh

```powershell
& $projectPython run_pipeline.py --resume
& $projectPython run_pipeline.py --stage enrich --entities-only
```

Resume skips a stage only when recorded source, argument, input and output
hashes match. Changed source/models can cause a full rebuild, so it is not a
presentation shortcut. Entity-only refresh applies a changed entity model to
an otherwise unchanged library; it checks corpus identity and retains the
classifier/retrieval data. For category, topic or corpus changes use full enrichment.

## 8. Spark and Scala execution

```powershell
# Spark default: local[2], two execution threads on this computer.
& $projectPython run_pipeline.py --stage spark --master "local[2]"

# Alternative to that Spark stage: standalone master and two worker processes.
& $projectPython scripts\run_cluster.py

# Executable Scala language and Spark Dataset demonstration.
& $projectPython scripts\run_scala.py
```

`run_cluster.py` starts one master and two workers on this single physical host,
runs Spark cleaning/MLlib, saves worker registration evidence and stops its own
process trees. During execution its web UIs use 18080, 18081 and 18082; after
completion use the saved `cluster_execution.json`. This is not a multi-host
scalability benchmark. The Scala example compiles using PySpark's bundled Scala
compiler, demonstrates case classes, traits, pattern matching, immutable
collections and map/filter/flatMap, and checks the processed report count.

## 9. Accuracy experiments and model activation

These commands are for development rather than a live presentation. Candidate
scripts keep active weights unchanged. Use a fresh output directory and backup
for each new experiment; the paths below describe the completed run.

| Command after `& $projectPython` | Purpose |
|---|---|
| `scripts\tune_classical.py --output .runtime\accuracy-candidates\classical` | Compare BoW, TF-IDF and Word2Vec feature/regularization recipes on validation. |
| `scripts\tune_crf.py --output .runtime\accuracy-candidates\classical` | Compare CRF regularization on validation Span-F1. |
| `scripts\tune_transformers.py --task classification --output .runtime\accuracy-candidates\transformers` | Scheduled BERT classifier candidate. |
| `scripts\tune_transformers.py --task classification --learning-rate 0.00005 --mention-dropout 0.35 --output .runtime\accuracy-candidates\bert-low-lr` | Lower-rate classifier with training-only name/place dropout. |
| `scripts\tune_transformers.py --task entities --output .runtime\accuracy-candidates\transformers` | BERT entity candidate; meaningful validation gain required for replacement. |
| `scripts\check_accuracy_regressions.py --classical .runtime\accuracy-candidates\classical` | Check candidate TF-IDF/CRF against existing development regressions before activation. |

The complete activation command, backup layout and order are in
`docs/ACCURACY_IMPROVEMENTS.md`. `promote_accuracy.py` rechecks validation scores,
activates candidates while the dashboard is stopped, freezes weights and runs
the repeat evaluation. `retain_word2vec_baseline.py` is the disclosed release
rollback for this run's failed Word2Vec trial. It does not perform further
tuning. Do not invoke activation or baseline-retention commands with an old
backup against unrelated current models.

Historical `refine_crf.py` and `refine_entities.py` are maintenance experiments,
not required for the current demonstration. Updating an already active entity
model through the latter requires `--crf-only --update-active`; it replaces
weights and requires subsequent enrichment and re-verification. They are not
alternate ways to launch the app.

## 10. Tests, metric verification and report generation

```powershell
# Recompute saved scores and check weights, prediction hashes, IDs and spans.
& $projectPython scripts\verify_evaluation.py

# Full unit, model, corpus and Streamlit interaction tests.
& $projectPython -m pytest -q

# Hand-written development checks; these are not independent final benchmarks.
& $projectPython scripts\check_robustness.py
& $projectPython scripts\check_entity_context.py

# Regenerate measured results and the standalone HTML report.
& $projectPython scripts\export_results.py
```

Current saved results: BoW 83.50%, TF-IDF 92.42%, Word2Vec 82.17%, BERT 81.67%
classification accuracy; CRF 92.23% and BERT 79.12% strict entity Span-F1. The
latest full suite recorded 64 passing tests. Check fresh command output rather
than promising that count after future changes.

With the dashboard running at port 8502, the optional installed Playwright/Edge
environment can run:

```powershell
python scripts\browser_check.py
python scripts\check_user_ui.py
python scripts\print_report.py
```

On this machine `python` is the separate interpreter with Playwright installed;
the project's `.venv312` is used for NLP. Browser checks visit seven pages and
capture screenshots; `check_user_ui.py` checks exact entity/source-text display;
`print_report.py` prints an existing HTML report without revisiting the dashboard.
These require Playwright and Edge, beyond the basic NLP requirements.

## 11. External JSONL staging

```powershell
& $projectPython scripts\import_jsonl.py .\examples.jsonl --source-url "https://source.example/dataset" --license "Actual permission or license reference"
```

Replace example metadata with real provenance. Each line needs `report_id`,
`narrative`, `reported_at` and `district`; category and exact entity spans are
optional. The importer validates rows, quarantines failures and records a source
hash. Output is a separate `data/external/` staging directory. It does not add
the file to the dashboard or retrain models automatically.

For a quick one-file inference demonstration, use Narrative lab's `.txt` upload
or the console `--file` flag instead of corpus ingestion.

A runnable fictional ingestion example is included. It deliberately has one
valid row and one invalid blank narrative:

```powershell
& $projectPython scripts\import_jsonl.py examples\demo_reports.jsonl --source-url "local-demo://examples/demo_reports.jsonl" --license "Project-generated fictional demonstration text"
```

Expected receipt: one accepted, one rejected. Inspect `validated.jsonl`,
`rejections.jsonl` and `provenance.json` inside the returned staging directory.
This demonstrates validation/quarantine without changing the benchmark.

## 12. Optional containers (not validated on this machine)

Native Windows execution is the verified presentation route. Docker definitions
are supplied for a separate Linux-container experiment. Compose publishes 8501,
whereas the project Streamlit configuration uses 8502, so explicitly set the
container port if trying this method:

```powershell
docker compose build dashboard
docker compose run --rm dashboard python scripts/download_resources.py
docker compose run --rm dashboard python run_pipeline.py
docker compose run --rm --service-ports dashboard python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then visit `http://localhost:8501`. These containers bind-mount the project, so
pipeline commands write into that checkout. Use a separate copy for experiments.
Optional cluster services are started with
`docker compose --profile cluster up -d spark-master spark-worker`, and the
pipeline service with `docker compose --profile cluster run --rm pipeline`.
Multi-host storage/networking and container behavior have not been verified here.

## 13. Benefits and precise claims

| Benefit | What this project provides |
|---|---|
| Easier case discovery | Category/entity filters and keyword, semantic and hybrid retrieval. |
| Faster scanning | One-line extractive summaries in case tables and exports; no measured human-time savings are claimed. |
| Structured information | Suggested participants, locations, weapons, property, dates, times and evidence with original offsets. |
| Traceability | Original narrative, source-sentence citations, feature contributions and review flags. |
| Collection exploration | Category/district/time charts and recurring LDA word themes over the generated corpus. |
| Local operation | Trained resources run locally after downloads, without a paid inference API; optional password and redaction support controlled sharing. |
| Reproducible teaching | Spark/Scala execution, four classifier families, two entity models, preserved predictions, metric recomputation and tests. |

The credible claim is an integrated educational research system with measured
internal improvements. The corpus is fictional and the evaluation has been
viewed; the Word2Vec baseline was retained after a disclosed diagnostic test
regression. The project does not establish real-world crime accuracy, predict
future crime, determine guilt or estimate actual danger. Independent labeled
real narratives, human-reviewed retrieval/summaries and deployment studies are
needed for those broader uses.

## 14. Short presentation script

“This project turns fictional crime narratives into searchable structured
information. Spark cleans 62,700 raw rows into 60,000 valid unique reports. We
compare four category classifiers and CRF/BERT entity extraction, discover LDA
themes, generate one-line TextRank summaries and expose the results in a local
Streamlit dashboard. TF-IDF is our strongest current classifier at 92.42%
accuracy, and CRF reaches 92.23% strict Span-F1 on the diagnostic synthetic
evaluation. Original text and citations remain visible for review. We preserve
failed experiments and do not claim verified accuracy on real police files.”
