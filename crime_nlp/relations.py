"""Source-linked action and ownership notes; no learned relation-model claim."""
from .material_context import _fire_actions

ACTIONS = {"hit", "beat", "attack", "assault", "stab", "shoot", "kill", "murder", "punch", "steal", "snatch", "take", "rob", "burn", "torch", "ignite", "hack", "cheat", "trick", "defraud", "damage", "smash", "vandalize", "threaten"}


def action_notes(text, entities, doc, references=None):
    def mentions(token, labels):
        direct = [entity for entity in entities if entity["label"] in labels and entity["start"] <= token.idx < entity["end"]]
        if direct:
            return direct
        return [reference["antecedent"] for reference in (references or []) if reference["pronoun"]["start"] == token.idx
                and reference["status"] == "resolved" and reference["antecedent"]["label"] in labels]
    compact = lambda entity: {key: entity[key] for key in ("text", "start", "end", "label")}
    events, ownership = [], []
    for verb in doc:
        if verb.lemma_.lower() not in ACTIONS:
            continue
        subjects = [child for child in verb.children if child.dep_ in {"nsubj", "nsubjpass"}]
        parent = verb
        for _ in range(3):
            if subjects or parent.head == parent or parent.dep_ not in {"conj", "xcomp", "advcl"}:
                break
            parent = parent.head
            subjects = [child for child in parent.children if child.dep_ in {"nsubj", "nsubjpass"}]
        actors = [entity for subject in subjects for entity in mentions(subject, {"SUSPECT"})]
        for agent in verb.children:
            if agent.dep_ == "agent" and agent.lower_ == "by":
                actors.extend(entity for token in agent.children for entity in mentions(token, {"SUSPECT"}))
        objects = [child for child in verb.children if child.dep_ in {"dobj", "obj", "nsubjpass"}]
        affected = [entity for token in objects for entity in mentions(token, {"VICTIM", "PROPERTY"})]
        for token in objects:
            for owner in token.children:
                if owner.dep_ == "poss":
                    for property_entity in mentions(token, {"PROPERTY"}):
                        for person in mentions(owner, {"VICTIM"}):
                            ownership.append({"predicate": "owned_by", "property": compact(property_entity), "owner": compact(person)})
        instruments = []
        for entity in entities:
            if entity["label"] != "WEAPON" or not verb.sent.start_char <= entity["start"] < verb.sent.end_char:
                continue
            span = doc.char_span(entity["start"], entity["end"], alignment_mode="expand")
            if span is None:
                continue
            ancestors = list(span.root.ancestors)
            linked = verb in ancestors
            if entity.get("instrument_type") == "accelerant":
                linked = linked or any(verb in _fire_actions(ancestor) for ancestor in ancestors if ancestor.pos_ == "VERB")
            if linked:
                instruments.append(entity)
        participating = actors + affected + instruments
        if not participating:
            continue
        qualifiers = {entity.get("assertion", "mentioned") for entity in participating}
        from .assertions import qualify_entities
        action_qualifier = qualify_entities(text, [{"text": verb.text, "start": verb.idx, "end": verb.idx + len(verb), "label": "EVENT"}], doc)[0]["assertion"]
        qualifiers.add(action_qualifier)
        assertion = next((value for value in ["disputed", "negated", "uncertain", "alleged"] if value in qualifiers), "mentioned")
        events.append({"predicate": verb.lemma_.lower(), "action_start": verb.idx, "action_end": verb.idx + len(verb),
                       "actors": [compact(entity) for entity in actors], "affected": [compact(entity) for entity in affected],
                       "instruments": [compact(entity) for entity in instruments], "assertion": assertion,
                       "source": {"start": verb.sent.start_char, "end": verb.sent.end_char, "text": text[verb.sent.start_char:verb.sent.end_char]},
                       "method": "Bounded syntax and entity-role links; review against source"})
    return {"events": events, "ownership": ownership, "scope": "Heuristic source-linked notes; ambiguous pronouns and complex attribution can remain unresolved"}
