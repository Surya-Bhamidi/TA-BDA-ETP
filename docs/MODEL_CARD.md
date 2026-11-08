# Model card — Version 3.1

Entity training mixes 4,000 existing-corpus reports with 4,000 role/ownership/administrative-context reports. The CRF received additional destination and property-list refinement; both retained entity candidates are compared on the same current validation set. The NER evaluation includes a previously viewed V3 subset and generated context cases, so it is a diagnostic evaluation, not untouched real-source validation. See ENTITY_CONTEXT_V3_1.md. Document models and their V3 benchmark are unchanged.

**Task:** English synthetic narrative classification and entity extraction. **Intended use:** university demonstration and experimentation.

**Document models:** normalized binary BoW/logistic regression, word + character TF-IDF/logistic regression, retained train-only Gensim Word2Vec mean pooling/logistic regression, and fine-tuned compact BERT. TF-IDF is the dashboard's default document classifier for speed and explanation. Character 3–5-grams help tolerate unknown spellings without rewriting names or introducing a new algorithm. Bounded candidates are tuned on validation; the IDF-weighted mean/max Word2Vec candidate failed the repeat-test comparison and its baseline is retained through a disclosed test-informed deployment rollback. See ACCURACY_IMPROVEMENTS.md. BERT training uses warmup/decay and train-only participant/place masking; inference never needs gold entities.

**Entity models:** CRF and fine-tuned BERT token classifier. Validation strict span F1 selects the automatic model; see `artifacts/models/v3_1/ner_selection.json`. Eight entity types are supported. CRF uses lexical context, shape and cue distances; no name/country whitelist is used at inference.

**Search:** local pretrained all-MiniLM-L6-v2 embeddings, 384 dimensions, masked mean pooling, L2 normalization, first 128 subword tokens per case. Semantic retrieval is not trained on case relevance judgments.

**Evaluation:** document sample sizes 12,000/800/800/1,200 across train/validation/development/test. NER uses 8,000/400/600 train/validation/test. Vocabulary fitting and Word2Vec use training only. Validation selects checkpoints and fits temperature. Frozen-model hashes precede final scoring. Report-level bootstrap intervals do not include domain shift. The separate hand-written robustness suite is a development/regression set, not an independent final benchmark.

**Evidence:** artifacts/evaluation.json, ner_evaluation.json, evaluation_plan.json, frozen_models.json, final_evaluation_receipt.json, robustness_development.json and docs/RESULTS.md. Prior version benchmarks are historical because corpus and protocol changed.

**Known limitations:** shared synthetic patterns, validation-selection optimism, possible role confusion, bounded negation/reference handling, reliability estimates limited to synthetic validation, summaries that omit context, and no real-source evaluation. The archived BERT benchmark truncates at 192 subwords; live BERT uses validation-selected windowed inference. Search now ranks all source windows rather than only the first 128 subwords. TF-IDF reads the full accepted input. Broader names/places do not imply unrestricted multilingual understanding. VADER polarity and cue levels are not validated danger measures. Review flags cannot detect every error.

**Privacy:** inference is local after resource downloads. Category feedback and default entity feedback store hashes and corrections without submitted text or entity names. An explicit unchecked-by-default annotation-queue option can save the narrative locally for human review. Drafts are not automatically trained or treated as adjudicated gold. Redaction can miss identifiers. Optional shared-password access is not production identity management.

**Current context validation:** syntax and clause-boundary checks repair clear participant, ownership, location, evidence and material-role errors. WITNESS is an additional contextual role, stored separately from the eight trained labels. WEAPON includes accelerants used to start a fire; stolen fuels and forensic material samples receive different roles. Corrections have no model probability and preserve source offsets. Raw model weights and benchmark predictions are retained. The broader layer has separate measured results, including regressions, in `artifacts/narrative_context_validation.json`; see [NARRATIVE_VALIDATION.md](NARRATIVE_VALIDATION.md) and [MATERIAL_HIGHLIGHTING.md](MATERIAL_HIGHLIGHTING.md). The earlier media-only correction and rejected training candidates are historical experiments documented in EVIDENCE_CONTEXT_FIX.md.

**Updates:** use a new independent evaluation protocol for changes motivated by the viewed final-test results. Never overwrite benchmark labels to hide mistakes.

**Schema 4 additions:** the legacy eight-label tagger is retained as the core. A validation-selected CRF specialist adds ORGANIZATION and WITNESS, preserving non-overlapping core mentions. Witness syntax remains a fallback. The specialist's validation Span-F1 is 0.7729; its repeat synthetic diagnostic Span-F1 is 0.5410, with 0.9904 precision and 0.3722 recall. This conservative specialist misses many valid mentions. On the separate ten-label diagnostic task, the same current core/context pipeline scores 0.5921 without the specialist and 0.6230 with it. These tasks differ from the legacy 0.9223 raw CRF benchmark and must not be presented as the same evaluation. See UPGRADE_STATUS.md and the preserved experiment artifacts.

**Confidence:** new role scores have validation-only reliability estimates. Witness cross-fitted Brier loss changes from 0.0603 raw to 0.0529 calibrated on synthetic validation. Organization validation predictions contain no observed errors, so their calibration uses a smoothed empirical constant and has no cross-fitted estimate. Neither calibrator covers missed entities or real-source shift. Core and rule-based scores remain clearly distinguished.

The retained CRF and BERT core taggers also have per-label exact-span reliability
fits on synthetic schema-4 validation, in core_entity_calibration.json. Raw
scores and labels are preserved. These fits are applied on demand only when
their recorded model hash matches; context repairs remain unscored. Cross-fitted
Brier estimates are recorded where there are enough correct and incorrect spans.
One-class labels use a smoothed constant. Validation selection and domain shift
still limit interpretation; calibration is not guaranteed to improve every label.

**Rejected candidates:** the broad ten-label CRF/BERT experiment and the dedicated BERT role candidates are preserved and reported. The new full-document BERT classification candidate reached 0.9246 validation macro-F1 versus 0.9270 for the retained model. It was not activated; full-document inference with the retained weights tied validation and retains 81.67% on the viewed diagnostic test. No accuracy increase is claimed for that classifier.
