"""Conservative reference links; ambiguous pronouns remain unresolved."""
from .material_context import _reused_material, MATERIAL

PEOPLE = {"SUSPECT", "VICTIM", "WITNESS"}
PERSON_PRONOUNS = {"he", "she", "they", "him", "her", "them", "his", "their"}


def reference_notes(text, entities, doc):
    result = []
    for token in doc:
        if token.lower_ not in PERSON_PRONOUNS | {"it", "its"}:
            continue
        previous = [entity for entity in entities if entity["end"] <= token.idx and token.idx - entity["end"] <= 350]
        target, basis = None, None
        if token.lower_ == "it" and token.head.lemma_.lower() == "use":
            parent = token.head.head
            if parent.pos_ == "VERB" and _reused_material(parent):
                candidates = [entity for entity in previous if entity.get("instrument_type") == "accelerant" and MATERIAL.fullmatch(entity["text"])]
                if len(candidates) == 1:
                    target, basis = candidates[0], "Same-actor material reuse linked to fire; no competing intervening object"
        if token.lower_ in PERSON_PRONOUNS:
            candidates = {entity["text"].casefold(): entity for entity in previous if entity["label"] in PEOPLE}
            reporting_scope = any(ancestor.lemma_.lower() in {"say", "claim", "report", "tell", "allege"} for ancestor in token.ancestors)
            if len(candidates) == 1 and not reporting_scope:
                target, basis = next(iter(candidates.values())), "Single explicit nearby person; gender is not inferred from a name"
        else:
            candidates = {entity["text"].casefold(): entity for entity in previous if entity["label"] in {"PROPERTY", "WEAPON", "ORGANIZATION"}}
        result.append({"pronoun": {"text": token.text, "start": token.idx, "end": token.idx + len(token)},
                       "antecedent": {key: target[key] for key in ("text", "start", "end", "label")} if target else None,
                       "status": "resolved" if target else "ambiguous" if len(candidates) > 1 else "unresolved",
                       "basis": basis or "Insufficient unambiguous evidence; review the source"})
    return result
