"""Disclosed syntax/context repairs, including separately displayed witnesses.

The eight trained entity labels are unchanged. WITNESS is a contextual role,
not a newly trained CRF/BERT label. Corrections never receive model confidence.
"""
import re
from functools import lru_cache

HUMAN_HEADS = set("man men woman women person persons people fellow fellows guy guys youth seller hacker stranger thief robber assailant attacker boy girl operator".split())
NON_PEOPLE = set("camera footage recording recordings statement report evidence photographs screenshots video audio logs records it this that website websie login software app system kindly please incident event attack fire accident theft robbery arson damage scene police".split())
NON_PROPERTY = set("anything nothing something everything it them complaint place someone somone somebody person people".split())
TITLES = set("mr mrs ms dr shri smt sir suspect victim complainant witness".split())
VIOLENCE = set("hit punch beat attack assault slap stab shoot kill murder kick strike".split())
THEFT = set("steal take snatch rob".split())
OTHER_PARTY_ACTIONS = set("cheat trick defraud damage smash vandalize hack".split())
LOCATION_HEADS = set("stand station stop market road street park bridge office school hospital shop mall terminal court plaza square lane avenue airport warehouse".split())
NAME = r"[^\W\d_][\w’'\-]*(?:\s+[^\W\d_][\w’'\-]*){0,3}?"
WITNESS_CLAUSE = re.compile(r"(?:^|[.!?;,\n])\s*(?:and\s+)?(?P<name>" + NAME + r")\s+(?:(?:has|had|have)\s+)?(?:seen|saw|witnessed|observed)\b", re.I)
EXPLICIT_WITNESS = re.compile(r"\bwitness\s+(?P<name>" + NAME + r")\s+(?:said|told|reported|stated|saw|witnessed|observed)\b", re.I)


@lru_cache(maxsize=512)
def _lowercase_place(value):
    from .inference import language_model
    return any(entity.label_ in {"GPE", "LOC", "FAC"} for entity in language_model()("near " + value.title()).ents)


def _noun_bounds(doc, token):
    chunk = next((chunk for chunk in doc.noun_chunks if chunk.root.i == token.i), None)
    if chunk is not None:
        left, right = chunk.start, chunk.end
    else:
        compounds = [child.i for child in token.children if child.dep_ in {"compound", "amod", "nummod"}]
        left, right = min([token.i] + compounds), token.i + 1
    while left < right and (doc[left].is_space or doc[left].dep_ in {"det", "poss"} or doc[left].text in {"'s", "’s"} or doc[left].lower_.rstrip(".") in TITLES or doc[left].is_punct):
        left += 1
    while right > left and (doc[right - 1].is_space or doc[right - 1].is_punct or doc[right - 1].text in {"'s", "’s"}):
        right -= 1
    return (doc[left].idx, doc[right - 1].idx + len(doc[right - 1])) if left < right else None


def _is_human(token):
    return token.pos_ == "PROPN" and token.lower_ not in NON_PEOPLE or token.lower_ in HUMAN_HEADS


def _subject(verb):
    subjects = [child for child in verb.children if child.dep_ in {"nsubj", "nsubjpass"}]
    if not subjects and verb.dep_ == "conj" and verb.head != verb:
        subjects = [child for child in verb.head.children if child.dep_ in {"nsubj", "nsubjpass"}]
    return subjects


def _correct(text, entity, first, last, label, method):
    return {"text": text[first:last], "start": first, "end": last, "label": label,
            "confidence": None, "extraction_method": method,
            "previous_labels": sorted({e["label"] for e in entity})}


def resolve_narrative_context(text, entities, doc):
    result = [dict(entity) for entity in entities]
    witnesses = []
    for pattern in [WITNESS_CLAUSE, EXPLICIT_WITNESS]:
        for match in pattern.finditer(text):
            first, last = match.span("name")
            title = re.match(r"(?:witness|sir|mr\.?|mrs\.?|ms\.?|dr\.?)\s+", text[first:last], re.I)
            if title:
                first += title.end()
            description = re.match(r"(?:(?:a|an|the)\s+)?(?:bystander|observer|eyewitness|witness)\s+(?:(?:named|called)\s+)?", text[first:last], re.I)
            if description:
                first += description.end()
            name = text[first:last]
            if name.lower() in {"i", "we", "he", "she", "they", "someone", "nobody", "police"} or any(word.lower() in NON_PEOPLE for word in name.split()):
                continue
            if not any(role["start"] == first and role["end"] == last for role in witnesses):
                witnesses.append(_correct(text, [], first, last, "WITNESS", "witness clause context"))
    # Syntax also handles "Rahul saw Ravi hit Asha" and names after conjunctions.
    for verb in doc:
        if verb.lemma_.lower() not in {"see", "witness", "observe"} or any(child.dep_ == "neg" for child in verb.children):
            continue
        for subject in _subject(verb):
            if subject.dep_ == "nsubjpass" or subject.pos_ == "PRON" or subject.lower_ in NON_PEOPLE:
                continue
            if subject.lower_ in {"bystander", "witness", "observer", "eyewitness"}:
                names = [token for token in subject.subtree if token.pos_ == "PROPN"]
                if not names:
                    continue
                subject = names[-1]
            bounds = _noun_bounds(doc, subject)
            if bounds and not any(role["start"] == bounds[0] and role["end"] == bounds[1] for role in witnesses):
                witnesses.append(_correct(text, [], *bounds, "WITNESS", "witness syntax context"))
        if any(child.dep_ == "auxpass" for child in verb.children):
            for agent in verb.children:
                if agent.dep_ != "agent" or agent.lower_ != "by":
                    continue
                for observer in agent.children:
                    if observer.dep_ == "pobj" and observer.pos_ == "PROPN":
                        bounds = _noun_bounds(doc, observer)
                        if bounds and not any(role["start"] == bounds[0] and role["end"] == bounds[1] for role in witnesses):
                            witnesses.append(_correct(text, [], *bounds, "WITNESS", "passive observation context"))
    # Regex and syntax can identify different parts of one multiword name.
    # Keep the complete mention once, including initials when syntax found them.
    distinct = []
    for witness in sorted(witnesses, key=lambda role: (-(role["end"] - role["start"]), role["start"])):
        if not any(role["start"] < witness["end"] and role["end"] > witness["start"] for role in distinct):
            distinct.append(witness)
    witnesses = distinct
    # A witness clause is not a continuation of a preceding place/person span.
    for witness in witnesses:
        repaired = []
        for entity in result:
            if entity["start"] < witness["end"] and entity["end"] > witness["start"]:
                if entity["start"] < witness["start"]:
                    prefix = text[entity["start"]:witness["start"]].rstrip(" ,;\t\r\n")
                    prefix = re.sub(r"\s+(?:and|but)$", "", prefix, flags=re.I)
                    if prefix and any(character.isalpha() for character in prefix):
                        repaired.append(_correct(text, [entity], entity["start"], entity["start"] + len(prefix), entity["label"], "witness clause boundary repair"))
                continue
            repaired.append(entity)
        result = repaired

    def assign(token, label, method="syntactic role context"):
        nonlocal result
        bounds = _noun_bounds(doc, token)
        if not bounds:
            return
        first, last = bounds
        if label == "PROPERTY":
            first = min([token.idx] + [child.idx for child in token.children if child.dep_ == "compound"])
        if label in {"SUSPECT", "VICTIM"} and re.search(
                r"\b(?:s\s*/\s*o|d\s*/\s*o|father(?:'s)?(?:\s+name)?|mother(?:'s)?(?:\s+name)?|parent)\s*[:\-]?\s*$",
                text[max(0, first - 30):first], re.I):
            return
        if any(role["start"] < last and role["end"] > first for role in witnesses):
            return
        overlapping = [entity for entity in result if entity["start"] < last and entity["end"] > first]
        # A noisy parse must not turn a learned place into an actor, or merge
        # multiple independently extracted fields into one noun phrase.
        if label in {"SUSPECT", "VICTIM", "PROPERTY"} and any(entity["label"] in {"LOCATION", "DATE", "TIME", "EVIDENCE"} for entity in overlapping):
            return
        if label == "LOCATION" and any(entity["label"] != "LOCATION" for entity in overlapping):
            return
        if any(entity["start"] == first and entity["end"] == last and entity["label"] == label for entity in overlapping):
            return
        if label in {"SUSPECT", "VICTIM"} and any(entity["label"] == label and entity["start"] <= first and entity["end"] >= last
                and not any(t.pos_ in {"VERB", "AUX"} for t in doc if entity["start"] <= t.idx < entity["end"]) for entity in overlapping):
            return
        # Keep an already-correct property/weapon span that omits an adjective.
        if label in {"PROPERTY", "WEAPON", "LOCATION"} and any(entity["label"] == label and entity["start"] <= token.idx < entity["end"] for entity in overlapping):
            return
        result = [entity for entity in result if entity not in overlapping]
        result.append(_correct(text, overlapping, first, last, label, method))

    for verb in doc:
        lemma = verb.lemma_.lower()
        arson = lemma in {"burn", "torch"} or lemma == "set" and any(child.lower_ == "fire" for child in verb.children)
        if not arson and lemma not in VIOLENCE | THEFT | OTHER_PARTY_ACTIONS | {"show", "threaten"}:
            continue
        if any(child.dep_ == "neg" for child in verb.children) or verb.head.lower_ == "without":
            continue
        if lemma == "take" and re.search(r"\b(?:kindly|please)\s*$", text[max(0, verb.idx - 20):verb.idx], re.I):
            continue
        subjects = _subject(verb)
        passive = any(child.dep_ == "auxpass" for child in verb.children) or any(subject.dep_ == "nsubjpass" for subject in subjects)
        if not passive and not any(child.dep_ == "nsubj" for child in verb.children) and any(
                entity["label"] == "SUSPECT" and verb.sent.start_char <= entity["start"] < verb.sent.end_char for entity in result):
            subjects = []
        for subject in subjects:
            if passive:
                if lemma in VIOLENCE and _is_human(subject):
                    assign(subject, "VICTIM")
                elif lemma in THEFT and not _is_human(subject):
                    assign(subject, "PROPERTY")
                    for owner in subject.children:
                        if owner.dep_ == "poss" and _is_human(owner):
                            assign(owner, "VICTIM")
            elif _is_human(subject):
                assign(subject, "SUSPECT")
        for child in verb.children:
            if child.dep_ in {"dobj", "obj"}:
                if lemma in VIOLENCE | {"rob", "cheat", "trick", "defraud"} and _is_human(child):
                    assign(child, "VICTIM")
                elif lemma in THEFT | {"damage", "smash", "vandalize"} and not _is_human(child) and child.lower_ not in NON_PROPERTY:
                    if lemma in {"damage", "smash", "vandalize"} and not any(_is_human(subject) for subject in subjects):
                        continue
                    assign(child, "PROPERTY")
                    for owner in child.children:
                        if owner.dep_ == "poss" and _is_human(owner):
                            assign(owner, "VICTIM")
                        if owner.dep_ == "prep" and owner.lower_ == "of":
                            for person in owner.children:
                                if person.dep_ == "pobj" and _is_human(person):
                                    assign(person, "VICTIM")
                elif lemma == "show" and child.lower_ in {"knife", "gun", "pistol", "weapon"}:
                    assign(child, "WEAPON")
            if child.dep_ in {"agent", "prep"} and child.lower_ == "by" and passive:
                for person in child.children:
                    if person.dep_ == "pobj" and _is_human(person):
                        assign(person, "SUSPECT")
            if child.dep_ == "prep" and child.lower_ == "with" and lemma in VIOLENCE:
                for instrument in child.children:
                    if instrument.dep_ == "pobj" and not _is_human(instrument):
                        assign(instrument, "WEAPON")
            if arson and child.dep_ == "prep" and child.lower_ == "to":
                for item in child.children:
                    if item.dep_ == "pobj" and not _is_human(item):
                        assign(item, "PROPERTY")
                        for owner in item.children:
                            if owner.dep_ == "poss" and _is_human(owner):
                                assign(owner, "VICTIM")
    # Spatial complements can be missed or joined to a following witness clause.
    for prep in doc:
        if prep.lower_ not in {"near", "outside", "inside"}:
            continue
        for place in prep.children:
            if place.dep_ != "pobj":
                continue
            bounds = _noun_bounds(doc, place)
            if not bounds:
                continue
            first, last = bounds
            if re.search(r"\b(?:No|number)\.?$", text[first:last], re.I):
                continue
            if re.match(r"gate\s+(?:no\.?|number)\b", text[first:first + 30], re.I):
                continue
            named = any(entity.start_char <= place.idx < entity.end_char and entity.label_ in {"GPE", "LOC", "FAC"} for entity in doc.ents)
            recognized = named or place.lower_ in LOCATION_HEADS and not any(child.lower_ in {"vacant", "empty", "unoccupied"} for child in place.children)
            recognized = recognized or place.pos_ == "PROPN" and not any(entity.start_char <= place.idx < entity.end_char and entity.label_ == "PERSON" for entity in doc.ents)
            # Existing LOCATION evidence is retained for unfamiliar place names.
            recognized = recognized or any(entity["label"] == "LOCATION" and entity["start"] <= place.idx < entity["end"] for entity in result)
            if not recognized and place.pos_ == "NOUN" and place.text.islower() and place.is_alpha and place.lower_ not in HUMAN_HEADS and not any(child.dep_ in {"det", "amod"} for child in place.children):
                recognized = _lowercase_place(place.text)
            if recognized:
                assign(place, "LOCATION", "spatial complement context")
    from .material_context import validate_material_context
    result = validate_material_context(text, result, doc)
    return sorted(result, key=lambda entity: (entity["start"], entity["end"])), sorted(witnesses, key=lambda role: role["start"])
