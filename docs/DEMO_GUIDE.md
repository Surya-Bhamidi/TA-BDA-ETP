# Beginner demonstration guide — Version 3.1

## Before presenting

Run START_DASHBOARD.cmd in the completed workspace and open http://localhost:8502. A fresh clone requires the setup and full pipeline commands in README.md first. Keep docs/PROJECT_REPORT.pdf and docs/RESULTS.md available.

For the complete command catalog, a 15-minute presentation, benefits and rebuild
precautions, read [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md). Double-click
**RUN_DEMO.cmd** to run all saved classifiers, both entity models, topic
assignment, summaries, syntax, polarity and search on one fictional narrative.
It prints actual inference alongside clearly labeled saved Spark/benchmark
evidence. `--quick` skips neural comparisons. It does not retrain models.

## A ten-minute walkthrough

1. **Problem (one minute):** “Written descriptions are hard to aggregate. This project turns them into searchable structured information.” State that all 60,000 reports are fictional.
2. **Pipeline (one minute):** Open Pipeline & syllabus. Explain 62,700 raw rows, invalid rows, duplicates, partitions and 60,000 clean rows. Show the actual two-worker single-host execution record.
3. **Trends (one minute):** Open Overview and choose one category. Explain that the chart reflects the generator, not real crime statistics.
4. **Retrieval (two minutes):** Search “stolen phone” in Case explorer. Read the one-line TextRank summaries, compare Semantic or Hybrid with a paraphrase, filter an entity and change result pages. Open a case and compare its summary with the cited source sentence. CSV exports include the summary.
5. **Language (two minutes):** Show entity highlights, structured notes and source sentences. Open Syntax & morphology and explain one noun, verb and dependency arrow. Demonstrate that a negated weapon mention differs from a weapon-use claim.
6. **Models (two minutes):** Show the document-model comparison and one confusion matrix. Explain validation versus final test, macro-F1 and exact-span entity scoring. Show that CRF won validation selection even though BERT is also implemented.
7. **Conclusion (one minute):** Open Learning guide and point to the full report. Name the real-data validation and multi-machine benchmark still needed.

## A pasteable fictional example

For an accelerant example, paste `Abebe Bekele poured petrol and set Olga Ivanova's shop on fire near Dubai. kindly take my complaint`. Petrol now highlights as WEAPON, with an explanation that it is used as an accelerant. See [MATERIAL_HIGHLIGHTING.md](MATERIAL_HIGHLIGHTING.md) for other material roles and checks.

For the corrected witness examples, expected labels and reproducible checks, see
[NARRATIVE_VALIDATION.md](NARRATIVE_VALIDATION.md). WITNESS appears as a
separate role, now supported by a trained specialist with syntax fallback.
Learned probabilities and context-only repairs are distinguished. Rahul and
Suresh in those examples should appear as WITNESS. Run RUN_UPGRADE_DEMO.cmd to
also show ORGANIZATION, reference/action notes and the annotation workflow;
see UPGRADE_STATUS.md for measured limits and retained legacy benchmarks.

For informal English, first try `sir my moblie stoln from pocket in bus ystrday`. For unfamiliar names and active/passive roles, compare `rajiv hit sneha with a stick near pune` with `sneha was beaten by rajiv near pune`. Review the highlights against the unchanged source. The fixed development comparisons, including failures, are in `artifacts/robustness_development.json`.

On 2025-03-18 at 21:15, Alex Lee reported a stolen phone near Cedar Market. A witness identified Morgan Reed as a suspect. No knife was reported. Officers reviewed camera footage. The account remains an allegation pending review.

Choose Automatic NER, analyze, then compare BERT if time permits. This example is illustrative; do not promise exact outputs in advance. Inspect errors openly.

## Questions you should be able to answer

- Why do word counts provide a useful baseline?
- What does BERT add, and why might it still perform worse?
- What is the difference between a document category and an entity role?
- Why preserve character offsets?
- Why can sentiment not directly measure danger?
- What did Spark actually distribute?
- Why does a high score on synthetic data require careful interpretation?
- What is excluded from GitHub and how is it rebuilt?

Answers and definitions are in Chapters 4–10 and 17 of the report.
