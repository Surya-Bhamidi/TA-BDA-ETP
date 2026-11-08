"""Conservative, disclosed context validation for media evidence mentions.

This is a rule layer after entity prediction, not a learned CRF probability.
It does not classify bare cameras, or relabel every recording in every context.
"""
import re

MEDIA = re.compile(r"\b(?:(?:security\s+camera|camera|cctv|surveillance|dashcam|video|audio)\s+(?:footage|recordings?)|photographs|screenshots|(?:witness|written)\s+statements?|fingerprint\s+samples|access\s+logs)\b", re.I)
CONTEXT = re.compile(r"\b(?:collected|reviewed|obtained|preserved|submitted|recovered|attached|examined|supplied|available)\b", re.I)


def validate_evidence_context(text, entities):
    from .text import sentences
    result = [dict(entity) for entity in entities]
    offset = 0
    for sentence in sentences(text):
        start = text.find(sentence, offset)
        if start < 0:
            continue
        offset = start + len(sentence)
        if not CONTEXT.search(sentence):
            continue
        for match in MEDIA.finditer(sentence):
            first, last = start + match.start(), start + match.end()
            overlapping = [entity for entity in result if entity["start"] < last and entity["end"] > first]
            if any(entity["start"] == first and entity["end"] == last and entity["label"] == "EVIDENCE" for entity in overlapping):
                continue
            # Preserve larger annotations rather than cutting up a person or
            # property span when the noun phrase is not independently tagged.
            if any(entity["end"] > last or (entity["start"] < first and not re.fullmatch(
                    r"\s*(?:no|the|some|any|a|an)\s+", text[entity["start"]:first], re.I)) for entity in overlapping):
                continue
            result = [entity for entity in result if entity not in overlapping]
            result.append({"start": first, "end": last, "text": text[first:last], "label": "EVIDENCE",
                           "confidence": None, "extraction_method": "media evidence context check",
                           "previous_labels": sorted({entity["label"] for entity in overlapping})})
    return sorted(result, key=lambda entity: (entity["start"], entity["end"]))
