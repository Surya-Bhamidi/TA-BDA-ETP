"""Versioned annotation contract; legacy benchmark labels remain unchanged."""
import json
from .config import ENTITY_LABELS

SCHEMA_VERSION = "4.0"
LABELS = [*ENTITY_LABELS, "ORGANIZATION", "WITNESS"]


def validate_record(row):
    text = row["narrative"]
    if not isinstance(text, str) or not text.strip() or len(text) > 30000 or "\x00" in text:
        raise ValueError("Narrative must be nonempty source text of at most 30,000 characters")
    entities = row.get("entities", row.get("entities_json", []))
    if isinstance(entities, str):
        entities = json.loads(entities)
    ordered = sorted(entities, key=lambda entity: (entity["start"], entity["end"]))
    previous = 0
    for entity in ordered:
        first, last = entity["start"], entity["end"]
        if type(first) is not int or type(last) is not int or entity["label"] not in LABELS or not 0 <= first < last <= len(text) or first < previous:
            raise ValueError("Unsupported label, invalid boundary or overlapping annotation")
        if text[first:last] != entity["text"] or entity["text"] != entity["text"].strip():
            raise ValueError("Annotation must match the exact source text without outer whitespace")
        if entity.get("assertion", "mentioned") not in {"mentioned", "negated", "alleged", "disputed", "uncertain"}:
            raise ValueError("Unsupported assertion qualifier")
        previous = last
    return ordered
