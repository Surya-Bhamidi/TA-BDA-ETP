"""Source-based descriptions for killing language outside the eight trained classes.

These descriptions have no model probability and do not assign a legal offence
or resolve who a pronoun refers to. Keep the trained model output separately.
"""
import re

HUMAN_NOUNS = {"person", "man", "woman", "child", "boy", "girl", "victim", "wife", "husband", "people", "someone", "somebody"}
INTENT = {"intend", "plan", "want", "wish", "hope", "mean", "aim", "intention", "intent", "plot"}
ATTEMPT = {"try", "attempt"}
OTHER_INCIDENTS = {"steal", "rob", "snatch", "hack", "defraud", "vandalize", "burn"}


def _scope(token):
    chain = [token]
    for parent in token.ancestors:
        if parent.sent != token.sent:
            break
        chain.append(parent)
        # A purpose clause can inherit intent from its own governing verb,
        # without inheriting negation from a separate coordinated statement.
        if parent.dep_ == "conj":
            break
    return chain


def _negated(token):
    return any(child.dep_ == "neg" or child.lower_ in {"no", "neither", "never", "without"} for child in token.children)


def _human_target(token):
    targets = [child for child in token.children if child.dep_ in {"dobj", "obj", "nsubjpass"}]
    return any(target.pos_ == "PROPN" or target.pos_ == "PRON" and target.lower_ in {"him", "her", "them", "he", "she", "they", "me", "you", "us"}
               or target.lemma_.lower() in HUMAN_NOUNS for target in targets)


def describe_incident(text, doc):
    descriptions = []
    sentences = list(doc.sents)
    for token in doc:
        noun_reference = token.pos_ == "NOUN" and token.lemma_.lower() in {"murder", "homicide", "killing"}
        if not noun_reference and (token.lemma_.lower() not in {"kill", "murder"} or token.pos_ != "VERB"):
            continue
        if noun_reference and token.dep_ == "compound" and token.head.lemma_.lower() != "attempt":
            continue
        if not noun_reference and not _human_target(token):
            continue  # "kill time", a process, weeds, etc. are not homicide.
        chain = _scope(token)
        lemmas = {item.lemma_.lower() for item in chain}
        sentence = list(token.sent)
        # Generic "the attempt failed" may qualify a single killing reference.
        # A named escape/theft attempt or an unrelated failure must not do so.
        same_actions = [item for item in sentence if item.lemma_.lower() in {"kill", "murder"} and item.pos_ == "VERB"]
        failed_attempt = len(same_actions) == 1 and any(
            item.lemma_.lower() == "fail" and not _negated(item) and any(
                child.lemma_.lower() == "attempt" and child.dep_ in {"nsubj", "nsubjpass"}
                and not any(mod.dep_ in {"compound", "amod", "poss", "acl", "relcl", "prep"} for mod in child.children)
                for child in item.children)
            for item in sentence)
        # Link only a simple adjacent reference, with one killing reference in
        # the document. Preserve its separate source offsets for inspection.
        next_sentence = next((item for item in sentences if item.start == token.sent.end), None)
        outcome_source = None
        if next_sentence is not None and sum(item.lemma_.lower() in {"kill", "murder"} and item.pos_ == "VERB" for item in doc) == 1:
            if re.fullmatch(r"(?:but\s+)?(?:the|this|that)\s+attempt\s+(?:had\s+)?failed[.!]?", next_sentence.text.strip(), re.I):
                failed_attempt = True
                outcome_source = {"start": next_sentence.start_char, "end": next_sentence.end_char, "text": next_sentence.text}
        unsuccessful_governor = any(item.lemma_.lower() == "fail" and not _negated(item)
                                      or item.lemma_.lower() in {"succeed", "manage"} and _negated(item) for item in chain)
        negated = any(_negated(item) for item in chain if item.lemma_.lower() not in {"fail", "succeed", "manage"})
        denied = bool(lemmas & {"deny", "dispute", "refute"})
        uncertain = any(child.lower_ in {"might", "may", "could", "possibly", "perhaps", "if"}
                        for item in chain for child in item.children)
        attributed = bool(lemmas & {"say", "report", "claim", "allege", "state"})
        prevented = bool(lemmas & {"stop", "prevent", "avert", "block"})
        attempting = bool(lemmas & ATTEMPT) or any(child.lemma_.lower() in ATTEMPT for child in token.children) or failed_attempt or unsuccessful_governor
        planning = bool(lemmas & INTENT) or any(item.lemma_.lower() == "go" for item in chain)
        threatened = bool(lemmas & {"threaten", "threat"})
        outcome = "Outcome not stated"
        if prevented:
            stage, label, outcome = "prevented", "Killing prevented", "Prevention described"
        elif threatened:
            stage, label = "threatened", "Threat to kill"
        elif attempting:
            stage, label = "attempted", "Attempted killing"
            if failed_attempt or unsuccessful_governor:
                outcome = "Attempt failed"
            elif any(item.lemma_.lower() == "escape" and not any(_negated(scope) or scope.lemma_.lower() in {"fail", "try", "attempt"}
                                                                  for scope in _scope(item)) for item in sentence):
                outcome = "Escape described; death not stated"
        elif planning or any(child.lower_ in {"will", "would", "shall"} for item in chain for child in item.children):
            stage, label = "intent", "Intent to kill"
        elif not noun_reference and token.tag_ in {"VBD", "VBN", "VBZ", "VBP"}:
            stage, label, outcome = "completed_wording", "Killing described", "Death described in source"
        else:
            stage, label = "unspecified", "Killing mentioned"
        assertion = "disputed" if denied else "negated" if negated else "uncertain" if uncertain else "alleged" if attributed else "mentioned"
        if failed_attempt and not noun_reference and token.tag_ in {"VBD", "VBN"} and not (lemmas & (ATTEMPT | INTENT)):
            assertion, label, outcome = "uncertain", "Conflicting killing outcome", "Source has conflicting completion and failure wording"
        if assertion == "disputed":
            label, outcome = "Killing claim disputed", "Outcome not established"
        elif assertion == "negated":
            label, outcome = "Killing language negated", "No affirmative killing or attempt established"
        elif assertion == "uncertain" and label != "Conflicting killing outcome":
            label, outcome = "Possible " + label.lower(), "Outcome not established"
        elif assertion == "alleged":
            label = "Reported " + label.lower()
        descriptions.append({"label": label, "stage": stage, "outcome": outcome, "assertion": assertion,
                             "trigger_start": token.idx, "trigger_end": token.idx + len(token),
                             "source": {"start": token.sent.start_char, "end": token.sent.end_char,
                                        "text": text[token.sent.start_char:token.sent.end_char]},
                             "cooccurring_incident": any(item.lemma_.lower() in OTHER_INCIDENTS and item.pos_ == "VERB" for item in sentence),
                             "confidence": None, "method": "Source-based syntax heuristic; outside the eight trained categories"})
        if outcome_source:
            descriptions[-1]["outcome_source"] = outcome_source
    # Multiple killing clauses can describe different events. Preserve all of
    # them for review instead of choosing an outcome from another clause.
    if not descriptions:
        return None
    if len(descriptions) == 1:
        return descriptions[0]
    return {"label": "Multiple killing references", "stage": "multiple", "outcome": "Review each source clause",
            "assertion": "uncertain", "descriptions": descriptions, "cooccurring_incident": True,
            "confidence": None, "method": "Source-based syntax heuristic; multiple events require review"}
