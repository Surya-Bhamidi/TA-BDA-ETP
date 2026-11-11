# Incident descriptions and sentiment — 11 November 2026

The reported text was:

> Mayank had ill intentions towards Barkha, and he went to her cabin to kill her, but the attempt failed

Both trained document classifiers returned **Robbery**. Their eight-class training task has no killing or attempted-killing class; features such as “kill” and “to kill” were associated with Robbery in the synthetic training data. A high calibrated score within that restricted label set does not make Robbery a supported interpretation of this text.

The primary dashboard result now shows:

| Field | Result |
| --- | --- |
| Described incident | Attempted killing |
| Outcome stated by the source | Attempt failed |
| Text sentiment | Negative |
| Original VADER compound | -0.8481 |
| Threat-language cue | `kill`, included in the cue dictionary |

## Incident-description contract

`crime_nlp/incident_context.py` uses the existing spaCy parse and bounded source context. It distinguishes planned intent, threats, attempts, prevented actions and direct completed wording. Negation, denial and uncertainty qualify the description. Generic adjacent “The attempt failed” references can be linked only when the document has one killing reference; the outcome retains its own source offsets. Unrelated failures and explicitly named escape attempts do not qualify an otherwise completed killing. Conflicting completion/failure wording is marked for review.

Literal human targets distinguish killing-related verbs from phrases such as “kill time”, killing a computer process, or killing weeds. This remains a conservative heuristic: unusual grammar, long-range references, implicit human targets and multiple events can remain unresolved. It does not assign participants to a legal offence or infer pronoun gender.

The primary display uses `display_category` when a source-based description is available without another incident in the same sentence. For a genuine robbery that also contains a threat to kill, the trained category remains primary and the additional context is displayed separately. Multiple killing references are retained for review rather than merged into one assumed outcome.

The JSON contract retains:

- `crime_type`, `model_score` and `bert_prediction`: the original eight-class outputs, without changing their meaning or inventing a probability for the contextual description.
- `display_category`: the primary description shown in the dashboard.
- `incident_context`: the source-based stage, outcome, assertion, exact source offsets and method. Its confidence is `null`.
- Matching action notes carry the source-based stage and outcome, so an unsuccessful attempt is not presented as a completed killing action.

When a contextual description is primary, raw classifier suggestions appear in **Analysis details**, rather than as the headline or the BERT prediction banner. Review is still required. The weights, class list and archived benchmark predictions were not retrained or relabeled.

## Sentiment interpretation

VADER's original numeric output is preserved. The text above remains negative because it contains negative wording, including “ill”, “kill” and “failed”. “The attempt failed” describes the incident outcome; it does not make the surrounding harmful-intent wording positive. VADER also treats “failed” as negative regardless of whether failure prevented harm.

The UI now provides:

- A Negative / Neutral / Positive label alongside the signed VADER score.
- The scale and label boundaries: Negative ≤ -0.05, Neutral strictly between -0.05 and +0.05, Positive ≥ +0.05.
- Negative, neutral and positive component shares, explicitly distinguished from confidence probabilities.
- Scores for each unchanged source sentence.
- Base lexicon entries with source offsets, clearly labeled as values before negation, intensifier and contrast adjustments.
- An outcome/negation note. For example, “did not try to kill” can receive positive lexical polarity from VADER; it is still shown as negated killing language rather than an affirmed attempt or a claim of happiness.

Sentiment does not measure criminal intent, guilt, dangerousness, a person's emotions, or success/failure. No human-labeled sentiment benchmark was added, so this update makes no new sentiment-accuracy claim.

The threat cue dictionary now includes common firearm and killing words. The live pipeline filters explicit non-human objects for killing verbs; local negation suppression remains in place. It is still a limited word-cue heuristic, and its score is separate from sentiment. Stored corpus cue snapshots retain the values from their earlier enrichment run until the pipeline is refreshed.

## Validation

- [Full regression results](../artifacts/incident_sentiment_pytest.xml): **280 passed**, no failures or skips.
- [Live browser audit](../artifacts/feature_audit_browser.json): includes the exact failed-attempt example, original model suggestions in details, sentiment components and exported JSON.
- [Browser screenshot](screenshots/attempted-killing-sentiment.png).
- Source-based regressions cover other names and wording, intent without an attempt, direct completed wording, denial, negation, prevention, adjacent outcome references, unrelated failures, genuine robbery, non-human uses and contradictory outcomes.
- Tests confirm that the underlying classifier predictions and VADER scores remain unchanged and that polarity-label boundaries are correct. Both new source files are included in the analysis revision fingerprint to invalidate stale cached results.

Reproduce with the project environment:

```powershell
.\.venv312\Scripts\python.exe -m pytest -q --junitxml=artifacts/incident_sentiment_pytest.xml
.\.venv312\Scripts\python.exe scripts/check_features.py
```

Restart an already-running Streamlit server after changing imported inference modules, then select **Analyze narrative** again. The local server was restarted during this verification.
