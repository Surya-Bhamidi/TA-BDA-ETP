# Media evidence correction

The reported sentence “Camera footage was collected” should identify Camera
footage as EVIDENCE. The original CRF mislabeled it as VICTIM. Its training
covered phrases such as “Officers collected camera footage” better than
sentence-start passive evidence mentions.

Three isolated CRF training candidates improved evidence-specific diagnostics
but missed the broader validation non-regression gate. Those candidate weights
were not activated; their files remain under `.runtime/evidence-candidate*`.

The deployed fix is a narrow, disclosed context-validation layer after entity
prediction. It recognizes supported media phrases (camera/CCTV/surveillance/
dashcam/video/audio footage or recordings) with collection, review, submission,
preservation or availability context. It does not relabel bare cameras and
preserves larger person/property spans. It preserves the original narrative,
character offsets and negation annotations. Corrected mentions record
`extraction_method`, previous labels and `confidence: null`; a rule correction
is not a learned model probability.

The same layer is used for Narrative lab, case analysis and corpus enrichment.
Analysis caches include its source revision, so old cached results are invalidated.
The reported example, casing variants, other evidence phrases, a negated
mention, stolen-camera property labels and existing person-role/FIR examples
are checked in tests/test_evidence_context.py and tests/test_dashboard.py.
scripts/browser_check.py also verifies the actual EVIDENCE highlight.

`scripts/verify_evidence_context.py` compares outputs before/after the layer on
the existing 400 CRF validation examples and 600 CRF/BERT diagnostic examples.
The checked scores and predictions on those examples are unchanged. Raw model
weights, training data, classification scores and reference labels are retained.
These viewed synthetic cases and the known-error development regressions do not
establish independent real-world accuracy.

```powershell
.\.venv312\Scripts\python.exe -m pytest tests\test_evidence_context.py -q
.\.venv312\Scripts\python.exe scripts\verify_evidence_context.py
.\.venv312\Scripts\python.exe scripts\demo_project.py
```

Stop the dashboard before refreshing the existing library with
`run_pipeline.py --stage enrich --entities-only`. A full retraining run is not
required for this correction.
