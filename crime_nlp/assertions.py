"""Bounded grammatical assertion scope, distinct from entity detection."""
REPORTING = {"say", "claim", "allege", "accuse", "report", "state", "tell"}
DENIAL = {"deny", "dispute", "refute"}


def qualify_entities(text, entities, doc):
    for entity in entities:
        span = doc.char_span(entity["start"], entity["end"], alignment_mode="expand")
        if span is None:
            continue
        root = span.root
        chain = [root, *root.ancestors]
        verbs = [token for token in chain if token.pos_ in {"VERB", "AUX"}]
        assertion = entity.get("assertion", "mentioned")
        without_positions = [index for index, token in enumerate(chain) if token.lower_ == "without"]
        without_action = any(any(token.pos_ == "VERB" for token in chain[:index]) for index in without_positions)
        if without_positions and any(token.lemma_.lower() in {"permission", "consent", "warning", "delay"} for token in chain[:without_positions[0]]):
            assertion = "mentioned"
        attribution = None
        if any(token.lemma_.lower() in DENIAL for token in verbs):
            assertion = "disputed"
        elif any(token.dep_ == "neg" for verb in verbs[:1] for token in verb.children) or any(child.lower_ in {"no", "neither"} for child in root.children) or without_action:
            assertion = "negated"
        elif any(token.lower_ in {"may", "might", "could", "possibly", "perhaps"} for verb in verbs[:1] for token in verb.children):
            assertion = "uncertain"
        elif any(token.lemma_.lower() in REPORTING for token in verbs):
            assertion = "alleged"
            reporter = next(token for token in verbs if token.lemma_.lower() in REPORTING)
            subject = next((child for child in reporter.children if child.dep_ == "nsubj"), None)
            if subject is not None:
                from .narrative_context import _noun_bounds
                bounds = _noun_bounds(doc, subject)
                if bounds:
                    attribution = {"text": text[bounds[0]:bounds[1]], "start": bounds[0], "end": bounds[1]}
        entity["assertion"] = assertion
        entity["assertion_method"] = "Dependency scope with lexical fallback; not full logical inference"
        if attribution:
            entity["attribution"] = attribution
    return entities


def entity_review(entities):
    for entity in entities:
        reasons = []
        if entity.get("confidence") is not None and entity["confidence"] < .65:
            reasons.append("Low uncalibrated model score")
        if entity.get("calibrated_confidence") is not None and entity["calibrated_confidence"] < .8:
            reasons.append("Limited reliability on synthetic validation spans")
        if entity.get("previous_labels") and entity["label"] not in entity["previous_labels"]:
            reasons.append("Context check changed the learned label")
        if entity.get("assertion") in {"alleged", "disputed", "uncertain"}:
            reasons.append("Qualified or attributed claim")
        entity["review_required"] = bool(reasons)
        entity["review_reasons"] = reasons
    return [entity for entity in entities if entity["review_required"]]
