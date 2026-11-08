# Narrative classification and entity repairs

The two reported narratives are preserved exactly, including accents and spelling.

| Narrative | Category | Correct mentions |
|---|---|---|
| José García hit Wei Zhang with iron rod near Nairobi, Suresh had seen it | Assault | José García: SUSPECT; Wei Zhang: VICTIM; iron rod: WEAPON; Nairobi: LOCATION; Suresh: WITNESS |
| sir ystrday two fellows showed knife and took my moblie near bus stand, rahul has seen it | Robbery | two fellows: SUSPECT; knife: WEAPON; moblie: PROPERTY; bus stand: LOCATION; rahul: WITNESS |

The original model joined the following witness clause to the location. The pipeline now checks witness clauses and syntax, separates those boundaries, and checks active/passive roles, ownership, spatial complements and acquired evidence. Camera footage remains EVIDENCE in collection contexts; a stolen camera remains PROPERTY. Administrative parent/officer fields and noisy parses are guarded against role overrides. Requests such as “kindly take my complaint” do not create stolen-property mentions.

WITNESS was initially a contextual role. The schema-4 upgrade now adds a separately trained organization/witness specialist, retaining these context checks as fallback. Witnesses remain in `contextual_roles` / `contextual_roles_json` for compatibility. The legacy eight-label checkpoints are unchanged; specialist checkpoints have their own label contract. Learned role mentions keep model scores and provenance; context-only repairs have null confidence. See [UPGRADE_STATUS.md](UPGRADE_STATUS.md). Raw core and specialist predictions remain available separately in live analysis JSON.

## Validation evidence

The original handwritten audit contains 38 cases: 31 category checks and 33 exact entity-set checks. Before repairs, 16/33 entity cases matched exactly; after repairs, 33/33 match. All 31 category checks pass. A further 17 annotated variations include new names, casing, passive voice, witness position, punctuation, multiword names and witness titles; all 17 pass. These cases informed development and are not independent accuracy estimates. Annotation boundaries exclude articles and titles.

The same viewed synthetic evaluation is also scored separately for the broader context pipeline. Raw checkpoint metrics are unchanged. The current measured values are in `artifacts/narrative_context_validation.json` and the generated RESULTS.md. Changes across models and splits are mixed. Several new property/evidence/material mentions are meaningful but absent from generator gold labels, and location article boundaries differ. Other errors and omissions remain possible. This is not a claim of universal correctness or a new untouched test result. See [MATERIAL_HIGHLIGHTING.md](MATERIAL_HIGHLIGHTING.md) for the subsequent petrol/accelerant repair and its separate diagnostic changes.

## Reproduce locally

```powershell
$projectPython = '.\.venv312\Scripts\python.exe'
& $projectPython scripts\audit_narratives.py --output artifacts\narrative_audit_after.json
& $projectPython scripts\audit_narratives.py --crosscheck --output artifacts\narrative_crosscheck.json
& $projectPython scripts\verify_narrative_context.py
& $projectPython scripts\verify_evaluation.py
& $projectPython scripts\verify_entity_store.py
& $projectPython -m pytest -q
```

For an interactive demonstration, run START_DASHBOARD.cmd, open http://localhost:8502, choose Narrative lab and paste either exact example above. Automatic NER selects CRF. RUN_DEMO.cmd also shows contextual roles. Entity-only refresh updates Parquet and SQLite without retraining classifiers or rebuilding retrieval embeddings:

```powershell
& $projectPython run_pipeline.py --stage enrich --entities-only
```

Close the local dashboard before replacing the SQLite dataset, then restart it.

The all-report structural check also found duplicate overlapping witness spans
from clause and syntax extraction in 58 reports. Witness detections are now
consolidated to one complete name, including initials, and leading titles such
as “sir” are excluded. Dedicated regressions cover those cases. The stored-span
check verifies all 60,000 reports; it is not a claim that every role is correct.
