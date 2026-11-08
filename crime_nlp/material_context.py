"""Disclosed material-role repairs using grammatical links, not bare keywords.

WEAPON includes an accelerant used to start a fire in this project's ontology.
No checkpoint labels or learned probabilities are changed.
"""
import re

MATERIAL = re.compile(r"\b(?:petrol|gasoline|kerosene|diesel|accelerant|paint\s+thinner|lighter\s+fluid)(?:\s+(?:samples?|residues?|traces?|cans?|bottles?))?\b", re.I)
APPLICATION = {"pour", "splash", "douse", "sprinkle", "use"}
COLLECTION = {"collect", "recover", "preserve", "seize", "examine", "submit"}
THEFT = {"steal", "take", "snatch"}
APPLICATION_PREFIX = re.compile(r"\b(?:poured|splashed|doused|sprinkled)\s+(?:(?:some|the)\s+)?$", re.I)
CLAUSE = r"(?:\b[^\W\d_]\.|[^.!?;])"
FIRE_SEQUENCE = re.compile(r"^\s+(?:(?:over|on)\s+" + CLAUSE + r"{1,70}?\s+)?(?:and\s+(?:then\s+)?|then\s+)(?:(?:deliberately|intentionally)\s+)?(?:set\b" + CLAUSE + r"{1,100}?\bon\s+fire\b|(?:started|lit|ignited)\b" + CLAUSE + r"{1,60}?\b(?:fire|blaze|match)\b)", re.I)
DENIAL = re.compile(r"\b(?:not|never|without|didn't|didn’t)\b", re.I)
CONTROLLED_FIRE = re.compile(r"\b(?:stove|campfire|barbecue|bonfire|rubbish|waste)\b", re.I)


def _explicit_accelerant(text, first, last):
    # All-capital text and unfamiliar names can make the parser tag SET as a
    # proper noun. Require an explicit, contiguous same-actor action sequence.
    prefix = re.split(r"[.!?;]", text[max(0, first - 80):first])[-1]
    sequence = FIRE_SEQUENCE.match(text[last:])
    return bool(APPLICATION_PREFIX.search(prefix) and sequence
                and not DENIAL.search(prefix + sequence[0]) and not CONTROLLED_FIRE.search(sequence[0]))


def _verb(token):
    return next((ancestor for ancestor in token.ancestors if ancestor.pos_ == "VERB"), None)


def _fire(verb):
    if any(child.dep_ == "neg" for child in verb.children):
        return False
    words = {token.lower_ for token in verb.subtree if _verb(token) == verb}
    if words & {"stove", "campfire", "barbecue", "bonfire", "rubbish", "waste"}:
        return False
    lemma = verb.lemma_.lower()
    return (lemma in {"burn", "torch", "ignite"}
            or lemma in {"set", "start"} and bool(words & {"fire", "flame", "flames", "blaze"})
            or lemma == "light" and bool(words & {"fire", "match", "flames"}))


def _fire_actions(verb):
    candidates = [verb]
    parent = _verb(verb)
    if parent is not None and verb.dep_ in {"advcl", "xcomp", "conj"}:
        candidates.append(parent)
    subjects = {child.lower_ for child in verb.children if child.dep_ in {"nsubj", "nsubjpass"}}
    for child in verb.children:
        if child.pos_ != "VERB" or child.dep_ not in {"conj", "advcl", "xcomp"}:
            continue
        own_subjects = {item.lower_ for item in child.children if item.dep_ in {"nsubj", "nsubjpass"}}
        if not own_subjects or own_subjects <= subjects:
            candidates.append(child)
            # "used petrol ... and used it to burn ...": follow only a
            # same-actor, unnegated reuse of "it", without intervening objects.
            if child.lemma_.lower() == "use" and child.dep_ == "conj" and not any(item.dep_ == "neg" for item in child.children):
                pronoun = any(item.dep_ in {"dobj", "obj"} and item.lower_ == "it" for item in child.children)
                competing = any(item.dep_ in {"dobj", "obj"} and item.head != verb for item in verb.doc[verb.i + 1:child.i])
                if pronoun and not competing:
                    candidates.extend(item for item in child.children if item.pos_ == "VERB" and item.dep_ == "xcomp")
    return [candidate for candidate in candidates if _fire(candidate)]


def _linked_fire(verb):
    return bool(_fire_actions(verb))


def _reused_material(verb):
    return any(action.head != verb and action.head.dep_ == "conj" and action.head.lemma_.lower() == "use"
               for action in _fire_actions(verb))


def _arson_arguments(text, entities, doc, verb):
    """Repair clear arguments only after a material is linked to damaging fire."""
    from .narrative_context import _noun_bounds, _is_human
    result = entities

    def put(token, label):
        nonlocal result
        bounds = _noun_bounds(doc, token)
        if bounds is None:
            return
        first, last = bounds
        if label == "PROPERTY":
            first = min([token.idx] + [child.idx for child in token.children if child.dep_ == "compound"])
        overlapping = [entity for entity in result if entity["start"] < last and entity["end"] > first]
        if any(entity["label"] == label and entity["start"] <= token.idx < entity["end"] for entity in overlapping):
            return
        if any(entity["start"] < first or entity["end"] > last for entity in overlapping):
            return
        result = [entity for entity in result if entity not in overlapping]
        result.append({"text": text[first:last], "start": first, "end": last, "label": label,
                       "confidence": None, "extraction_method": "material-linked arson argument context",
                       "previous_labels": sorted({entity["label"] for entity in overlapping})})

    for subject in verb.children:
        if subject.dep_ == "nsubj" and _is_human(subject):
            put(subject, "SUSPECT")
    for action in _fire_actions(verb):
        # A verb mis-tagged as a participant is not a person's name.
        result = [entity for entity in result if not (entity["label"] in {"SUSPECT", "VICTIM"}
                  and entity["start"] == action.idx and entity["end"] == action.idx + len(action))]
        for item in action.children:
            if item.dep_ in {"dobj", "obj"} and item.lower_ not in {"fire", "flame", "flames", "blaze", "match", "it"} and not _is_human(item):
                put(item, "PROPERTY")
                for owner in item.children:
                    if owner.dep_ == "poss" and _is_human(owner):
                        put(owner, "VICTIM")
        # Physical scene complements, including an unfamiliar locality.
        for prep in action.children:
            if prep.lower_ == "in" and prep.dep_ == "prep":
                for place in prep.children:
                    if place.dep_ == "pobj" and place.pos_ == "PROPN" and place.ent_type_ != "PERSON":
                        put(place, "LOCATION")
    # The locality of a named fuel supplier: "from Supplier in Locality".
    for prep in verb.children:
        if prep.lower_ != "from" or prep.dep_ != "prep":
            continue
        for supplier in prep.children:
            for area in supplier.children:
                if area.lower_ == "in" and area.dep_ == "prep":
                    for place in area.children:
                        if place.dep_ == "pobj" and place.pos_ == "PROPN" and place.ent_type_ != "PERSON":
                            put(place, "LOCATION")
    return result


def validate_material_context(text, entities, doc):
    result = [dict(entity) for entity in entities]
    for match in MATERIAL.finditer(text):
        span = doc.char_span(*match.span(), alignment_mode="expand")
        if span is None:
            continue
        verb = _verb(span.root)
        explicit = _explicit_accelerant(text, *match.span())
        if verb is None and not explicit:
            continue
        lemma = verb.lemma_.lower() if verb is not None else ""
        clause = " ".join(token.text for token in verb.subtree if _verb(token) == verb) if verb is not None else ""
        label, subtype = None, None
        if lemma in THEFT:
            label = "PROPERTY"
        elif lemma in COLLECTION and (re.search(r"\b(?:samples?|residues?|traces?)\b", match[0], re.I)
                or re.search(r"\bas evidence\b|\bfor (?:forensic|testing|analysis)\b", clause, re.I)):
            label = "EVIDENCE"
        elif explicit or verb is not None and (lemma in APPLICATION or _fire(verb) and span.root.dep_ == "pobj"
              and span.root.head.lower_ in {"with", "using"}) and not any(child.dep_ == "neg" for child in verb.children) and _linked_fire(verb):
            label, subtype = "WEAPON", "accelerant"
        if label is None:
            continue
        first, last = match.span()
        overlapping = [entity for entity in result if entity["start"] < last and entity["end"] > first]
        # Keep a correctly learned containing span, such as "a petrol can".
        correct = next((entity for entity in overlapping if entity["label"] == label
                        and entity["start"] <= first and entity["end"] >= last), None)
        if correct is not None:
            if subtype:
                correct["instrument_type"] = subtype
                if verb is not None and _reused_material(verb):
                    result = _arson_arguments(text, result, doc, verb)
            continue
        # Do not cut apart larger learned entities or unrelated neighbouring fields.
        if any(entity["start"] < first or entity["end"] > last for entity in overlapping):
            continue
        result = [entity for entity in result if entity not in overlapping]
        correction = {"text": text[first:last], "start": first, "end": last, "label": label,
                      "confidence": None, "extraction_method": "material role context",
                      "previous_labels": sorted({entity["label"] for entity in overlapping})}
        if subtype:
            correction["instrument_type"] = subtype
        result.append(correction)
        if subtype and verb is not None and _reused_material(verb):
            result = _arson_arguments(text, result, doc, verb)
    return sorted(result, key=lambda entity: (entity["start"], entity["end"]))
