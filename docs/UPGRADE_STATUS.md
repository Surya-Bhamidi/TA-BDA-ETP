# Generalization upgrade work record

User-approved scope: synthetic experiments and preparation of the real-data
workflow. Real data collection and human judgments remain external inputs.

| Work | Status |
|---|---|
| Versioned ten-label annotation contract | Implemented; legacy eight-label benchmark preserved |
| Source-separated real-data preparation and two-annotator agreement | Implemented; awaiting user-supplied reviewed data |
| Ten-label CRF/BERT experiment | Trained and frozen; results in schema_v4_evaluation.json; not automatically deployed |
| Dedicated trained organization/witness specialists | Trained, frozen and verified; broader CRF selected on validation; deployed as non-overlapping role additions |
| Full-document BERT classifier | Windowed inference enabled with retained weights; newly trained candidate rejected on validation |
| Full-document semantic search and evidence windows | Implemented and indexed: 61,300 windows cover all 60,000 reports |
| Entity correction and opt-in annotation queue | Implemented and tested, including private defaults and opt-in reset on changed input |
| Assertion/attribution, relations and entity review | Implemented and tested; ambiguous references remain unresolved; core and role reliability fitted on synthetic validation |
| Human judgments for summaries, retrieval and topics | Tools and 36 blank review forms prepared; actual judgments remain pending |

The initial broad ten-label experiment generalizes poorly to new witness
sentence patterns. Its disappointing results are retained rather than hidden.
Dedicated role specialists preserve the existing core tagger and are a
validation-selected follow-up; any reuse of viewed synthetic evaluation is
diagnostic. No real-world accuracy claim follows from generated cases or tests.

## Measured results

| Evaluation | Result | Interpretation |
|---|---|---|
| Same ten-label synthetic diagnostic, current core/context without learned roles | 59.21% Span-F1 | Baseline for this specific new task |
| Same examples with selected learned role additions | 62.30% Span-F1 | +3.10 percentage points; not comparable with legacy eight-label F1 |
| Selected two-label specialist | 77.29% validation F1; 54.10% diagnostic F1 | High precision (99.04%) but limited recall (37.22%) on the diagnostic cases |
| New windowed BERT candidate | 92.46% validation macro-F1 versus 92.70% retained | Candidate not activated |
| Retained BERT with full-document windows | 81.67% viewed diagnostic accuracy | Same measured accuracy; removes the first-window coverage limit |
| Legacy model weights/reference predictions | Preserved and verified | Previous benchmark remains separately labeled |

The rejected broad ten-label models and narrower specialist experiment remain
under `artifacts/models/v4_entities` and `v4_roles`; the selected specialist is
in `v4_roles_broad`. Each has frozen file hashes, saved predictions, input
snapshots and verification. Training templates evolved; use the saved input
snapshots when reproducing a historical experiment.

## Try the delivered workflow

1. Run START_DASHBOARD.cmd and open http://localhost:8502.
2. In Narrative lab, paste the example from examples/upgrade_case.txt. Indian Oil
   should be ORGANIZATION; petrol WEAPON/accelerant; Neha WITNESS. Source-linked
   actions, qualifiers and unresolved references appear below the highlights.
3. Open Analysis details → Correct extracted entities. Save a hash/offset draft,
   or explicitly opt into including source text for later human annotation.
4. Open Model evaluation for separate legacy and specialist results and the
   downloadable unreviewed quality forms.
5. Run RUN_UPGRADE_DEMO.cmd for all classifiers, raw/core comparisons, enriched
   entities, reference/action notes, topics, summaries and retrieval.

Useful commands:

```powershell
$projectPython = '.\.venv312\Scripts\python.exe'
& $projectPython scripts\verify_upgrade_models.py
& $projectPython scripts\verify_evaluation.py
& $projectPython scripts\verify_entity_store.py
& $projectPython -m pytest -q
& $projectPython scripts\quality_review.py evaluate --input data\quality_review\reviewed.jsonl --output artifacts\human_quality_review.json
```

See REAL_DATA_WORKFLOW.md for permission/source metadata, independent annotation,
agreement and adjudication. Real-source accuracy, human summary/search/topic
quality, unrestricted coreference and unrestricted multilingual understanding
remain unproven. Core and role reliability are fitted only on synthetic
validation spans and do not estimate omissions or prove real-source calibration.
