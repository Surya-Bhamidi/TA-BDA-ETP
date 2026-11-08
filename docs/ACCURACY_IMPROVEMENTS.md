# Validation-selected accuracy improvements

The existing model families, 60,000-report corpus and sample IDs are retained.
Document training/validation/evaluation sizes remain 12,000/800/1,200; entity
training/validation/evaluation sizes remain 8,000/400/600. No test prediction is
used to choose a candidate. Executed before/after scores are generated into
`docs/RESULTS.md` from `artifacts/accuracy_improvements.json`.

## Changes and candidate selection

- **Bag of Words:** compare binary/count features, optional L2 normalization,
  and logistic-regression regularization. The selected recipe is binary word
  unigrams/bigrams with L2 normalization and C=10.
- **TF-IDF:** compare word/character feature weights and regularization.
  The selected recipe uses character weight 1.5 and C=0.25.
- **Word2Vec:** compare three embedding recipes and mean, IDF-weighted mean,
  and mean-plus-max pooling. IDF fits on training only. Selected: 100-dimensional
  skip-gram embeddings, window 3, 20 epochs, IDF-weighted mean concatenated with
  max pooling, L2 normalization and logistic regression with C=3. This produces
  200-dimensional document features from 100-dimensional word embeddings. This
  candidate failed the diagnostic test comparison (75.75% versus 82.17%), so
  the deployed Word2Vec model retains its original unweighted mean pooling.
- **BERT classification:** compare the retained model with two six-epoch
  candidates, using learning rates 0.0002 and 0.00005, 10% warmup and linear
  decay, dropout 0.2, label smoothing 0.03, and a 192-subword limit. Dynamic
  padding avoids unnecessary computation. The lower-rate candidate also masks
  people/place mentions in 35% of training examples to reduce memorization of
  names and places. Original validation, evaluation and user text is unchanged;
  no gold entities are used at inference. Validation macro-F1 chooses the model
  and checkpoint. The selected recipe is recorded in `accuracy_selection.json`.
- **CRF:** compare three regularization settings with 150 iterations. Selected:
  c1=0.01, c2=0.1, based on strict validation Span-F1. Existing informal,
  ownership and FIR development regressions are checked before activation.
- **BERT entities:** compare a candidate with matching 48-word/32-stride train
  and inference windows, dynamic padding, scheduled learning rate 0.00015,
  label smoothing and reduced outside-label weight. Deployment requires at
  least 0.2 percentage points validation Span-F1 improvement; marginal gains
  retain the baseline. All candidate scores remain in the selection receipt.

## What the results establish

These are validation-selected changes followed by one repeat evaluation on the
same previously viewed synthetic cases. Scores are comparable across this
before/after run because training sizes and evaluation IDs are unchanged.
Shared generator grammar and validation-set tuning can still give optimistic
results. Independent real-source validation remains outstanding. The Word2Vec
candidate is a concrete example: its validation improvement did not transfer.
Its failed scores and predictions remain archived, and deployment retains the
baseline. This rollback used the viewed diagnostic test results, so the final
release is explicitly not validation-only selection or an untouched final
benchmark. No additional fitting or parameter search followed this result.

`artifacts/models/v3_1/accuracy_selection.json` preserves candidates, recipes and
validation scores. `artifacts/frozen_models.json` records selected weights and
source hashes before evaluation. Row-level predictions and
`scripts/verify_evaluation.py` allow metric recomputation. Previous weights,
metrics and predictions remain under `.runtime/accuracy-backups/`.

## Reproduce this revision

Use the project Python environment and existing generated corpus. First make
a new backup of the current model directory, JSON/CSV artifacts and row-level
predictions, as in `.runtime/accuracy-backups/20261107-153000`. These scripts
write candidate outputs separately. Avoid reusing a candidate directory from
an incomplete run.

```powershell
.\.venv312\Scripts\python.exe scripts\tune_classical.py --output .runtime\accuracy-candidates\classical
.\.venv312\Scripts\python.exe scripts\tune_crf.py --output .runtime\accuracy-candidates\classical
.\.venv312\Scripts\python.exe scripts\tune_transformers.py --task classification --output .runtime\accuracy-candidates\transformers
.\.venv312\Scripts\python.exe scripts\tune_transformers.py --task entities --output .runtime\accuracy-candidates\transformers
.\.venv312\Scripts\python.exe scripts\tune_transformers.py --task classification --learning-rate 0.00005 --mention-dropout 0.35 --output .runtime\accuracy-candidates\bert-low-lr
.\.venv312\Scripts\python.exe scripts\check_accuracy_regressions.py --classical .runtime\accuracy-candidates\classical
```

Stop the dashboard before activation. Replace the baseline argument below with
the backup made before that experiment. The promotion script rechecks selected
validation scores and baseline hashes, fits confidence temperature on validation
only, freezes the selected weights, and then evaluates the existing test cases.

```powershell
.\.venv312\Scripts\python.exe scripts\promote_accuracy.py --baseline .runtime\accuracy-backups\20261107-153000 --classical .runtime\accuracy-candidates\classical --transformers .runtime\accuracy-candidates\transformers --alternate-classification .runtime\accuracy-candidates\bert-low-lr
.\.venv312\Scripts\python.exe scripts\retain_word2vec_baseline.py --baseline .runtime\accuracy-backups\20261107-153000
.\.venv312\Scripts\python.exe scripts\verify_evaluation.py
.\.venv312\Scripts\python.exe run_pipeline.py --stage enrich
.\.venv312\Scripts\python.exe scripts\check_robustness.py
.\.venv312\Scripts\python.exe -m pytest -q
.\.venv312\Scripts\python.exe scripts\export_results.py
.\START_DASHBOARD.cmd
```

The original `run_pipeline.py --stage train` remains a baseline training route;
use this workflow to reproduce the validation-selected models.
