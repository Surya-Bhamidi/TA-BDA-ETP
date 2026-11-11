"""Transparent VADER text polarity, separate from incident type and outcome."""


def polarity_label(compound):
    return "Positive" if compound >= .05 else "Negative" if compound <= -.05 else "Neutral"


def analyze_sentiment(text, analyzer, doc, incident=None):
    scores = dict(analyzer.polarity_scores(text))
    source_scores = []
    for sentence in doc.sents:
        values = analyzer.polarity_scores(sentence.text)
        source_scores.append({"text": sentence.text, **values, "label": polarity_label(values["compound"])})
    scores.update(label=polarity_label(scores["compound"]), method="NLTK VADER; original lexicon and scoring rules",
                  interpretation="Text polarity on -1 to +1; not crime severity, outcome, or a person's emotional state.",
                  sentences=source_scores,
                  lexicon_cues=[{"text": token.text, "start": token.idx, "end": token.idx + len(token),
                                 "lexicon_valence": float(analyzer.lexicon[token.lower_])}
                                for token in doc if token.lower_ in analyzer.lexicon and analyzer.lexicon[token.lower_] != 0])
    if incident:
        if incident["assertion"] in {"negated", "disputed", "uncertain"}:
            scores["context_note"] = "Harmful language is negated, disputed or uncertain. VADER polarity does not establish that the event happened; negating a negative word can produce a positive score."
        elif incident["outcome"] == "Attempt failed":
            scores["context_note"] = "The source states that the attempt failed. Negative wording such as intent to kill can remain negative; VADER also treats 'failed' as a negative word regardless of what failed."
        else:
            scores["context_note"] = "Read the described incident and outcome separately from the polarity of its wording."
    else:
        scores["context_note"] = "Positive and negative percentages describe VADER's text components, not probabilities that an event is good or bad."
    return scores
