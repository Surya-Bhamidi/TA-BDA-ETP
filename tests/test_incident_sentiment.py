"""Intent, attempt/outcome and text polarity are separate source observations."""
import pytest

from crime_nlp.config import MODELS
from crime_nlp.incident_context import describe_incident
from crime_nlp.sentiment import analyze_sentiment, polarity_label
from crime_nlp.text import threat_language

EXAMPLE = "Mayank had ill intentions towards Barkha, and he went to her cabin to kill her, but the attempt failed"


@pytest.fixture(scope="module")
def nlp():
    import spacy
    if not spacy.util.is_package("en_core_web_sm"):
        pytest.skip("Requires the local English parser")
    return spacy.load("en_core_web_sm")


@pytest.mark.parametrize("text,label,stage,outcome", [
    (EXAMPLE, "Attempted killing", "attempted", "Attempt failed"),
    ("Anaya tried to kill Priya, but the attempt failed.", "Attempted killing", "attempted", "Attempt failed"),
    ("Mayank attempted to murder Barkha.", "Attempted killing", "attempted", "Outcome not stated"),
    ("Mayank attempted murder of Barkha, but the attempt failed.", "Attempted killing", "attempted", "Attempt failed"),
    ("Mayank tried to kill Barkha. The attempt failed.", "Attempted killing", "attempted", "Attempt failed"),
    ("Mayank failed to kill Barkha.", "Attempted killing", "attempted", "Attempt failed"),
    ("Mayank did not succeed in killing Barkha.", "Attempted killing", "attempted", "Attempt failed"),
    ("Mayank planned to kill Barkha.", "Intent to kill", "intent", "Outcome not stated"),
    ("Mayank went to the cabin to kill Barkha.", "Intent to kill", "intent", "Outcome not stated"),
    ("Mayank threatened to kill Barkha.", "Threat to kill", "threatened", "Outcome not stated"),
    ("Mayank killed Barkha.", "Killing described", "completed_wording", "Death described in source"),
    ("Barkha was killed by Mayank.", "Killing described", "completed_wording", "Death described in source"),
    ("Mayank stopped another person from killing Barkha.", "Killing prevented", "prevented", "Prevention described"),
])
def test_source_based_stage_and_outcome(nlp, text, label, stage, outcome):
    context = describe_incident(text, nlp(text))
    assert (context["label"], context["stage"], context["outcome"]) == (label, stage, outcome)
    assert context["confidence"] is None
    source = context["source"]
    assert source["text"] == text[source["start"]:source["end"]]


@pytest.mark.parametrize("text,assertion", [
    ("Mayank did not try to kill Barkha.", "negated"),
    ("Mayank had no intention of killing Barkha.", "negated"),
    ("Mayank denied trying to kill Barkha.", "disputed"),
    ("Mayank might have tried to kill Barkha.", "uncertain"),
    ("Mayank said he tried to kill Barkha.", "alleged"),
])
def test_qualified_claims_not_displayed_as_affirmative_killings(nlp, text, assertion):
    context = describe_incident(text, nlp(text))
    assert context["assertion"] == assertion
    assert context["label"] not in {"Attempted killing", "Killing described"}
    assert context["outcome"] != "Death described in source"


@pytest.mark.parametrize("text", [
    "Mayank went to the cabin to return a book.",
    "Mayank went to kill time in the cabin.",
    "Mayank tried to kill the computer process.",
    "Mayank killed the weeds in the garden.",
    "Mayank murdered the song during karaoke.",
])
def test_nonhuman_and_nonviolent_cases_do_not_get_killing_description(nlp, text):
    doc = nlp(text)
    assert describe_incident(text, doc) is None
    assert threat_language(text, doc)["score"] == 0


def test_unrelated_failed_event_does_not_turn_completed_killing_into_attempt(nlp):
    text = "Mayank killed Barkha, but he failed his examination."
    context = describe_incident(text, nlp(text))
    assert context["stage"] == "completed_wording"
    assert context["outcome"] == "Death described in source"


def test_separate_escape_attempt_failure_not_applied_to_killing(nlp):
    text = "Mayank killed Barkha, but his escape attempt failed."
    context = describe_incident(text, nlp(text))
    assert context["stage"] == "completed_wording"


def test_failed_escape_not_presented_as_successful_escape(nlp):
    text = "Mayank tried to kill Barkha but her attempt to escape failed."
    assert describe_incident(text, nlp(text))["outcome"] == "Outcome not stated"


def test_contradictory_completion_and_failure_requires_review(nlp):
    text = "Mayank killed Barkha, but the attempt failed."
    context = describe_incident(text, nlp(text))
    assert context["label"] == "Conflicting killing outcome"
    assert context["assertion"] == "uncertain"


def test_real_robbery_is_kept_separate_from_a_threat(nlp):
    text = "Mayank threatened to kill Barkha and stole her phone."
    context = describe_incident(text, nlp(text))
    assert context["label"] == "Threat to kill"
    assert context["cooccurring_incident"] is True


@pytest.mark.parametrize("score,label", [(-1., "Negative"), (-.05, "Negative"), (-.0499, "Neutral"),
                                          (0., "Neutral"), (.0499, "Neutral"), (.05, "Positive"), (1., "Positive")])
def test_vader_label_boundaries(score, label):
    assert polarity_label(score) == label


def test_user_example_preserves_raw_models_and_vader_and_adds_correct_description():
    if not (MODELS / "tfidf.joblib").exists():
        pytest.skip("Requires local trained artifacts")
    from crime_nlp.inference import analyze, models
    result = analyze(EXAMPLE, use_bert=True)
    assert result["display_category"] == "Attempted killing"
    assert result["incident_context"]["outcome"] == "Attempt failed"
    assert result["crime_type"] == "Robbery"  # Raw model retained, not relabeled as a trained class.
    assert result["review_required"]
    expected = models()["sentiment"].polarity_scores(EXAMPLE)
    assert {key: result["sentiment"][key] for key in expected} == expected
    assert result["sentiment"]["label"] == "Negative"
    assert result["sentiment"]["compound"] == pytest.approx(-.8481)
    assert "attempt failed" in result["sentiment"]["context_note"].lower()
    assert "kill" in result["threat"]["cues"]
    action = next(event for event in result["relations"]["events"] if event["predicate"] == "kill")
    assert action["stage"] == "attempted" and action["outcome"] == "Attempt failed"


def test_sentiment_preserves_sentence_text_and_lexicon_offsets(nlp):
    class Analyzer:
        lexicon = {"happy": 2., "sad": -2.}
        def polarity_scores(self, text):
            positive = "happy" in text
            return {"compound": .5 if positive else -.5, "pos": .5 if positive else 0., "neg": 0. if positive else .5, "neu": .5}
    text = "I am happy. I am sad."
    result = analyze_sentiment(text, Analyzer(), nlp(text))
    assert [row["label"] for row in result["sentences"]] == ["Positive", "Negative"]
    assert [row["text"] for row in result["sentences"]] == ["I am happy.", "I am sad."]
    assert all(text[row["start"]:row["end"]] == row["text"] for row in result["lexicon_cues"])


@pytest.mark.parametrize("text", ["No gun was reported.", "Mayank did not try to kill Barkha."])
def test_new_threat_cues_respect_local_negation(text):
    assert threat_language(text)["score"] == 0
