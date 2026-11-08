"""Conservative redaction and mention-level negation for optional exports."""
import re
from .text import word_tokens


def annotate_assertions(text, entities):
    for entity in entities:
        before = re.split(r"[.!?;,]", text[:entity["start"]])[-1]
        cues = set(word_tokens(before)[-4:])
        entity["assertion"] = "negated" if cues & {"no", "not", "without", "denied", "never"} else "mentioned"
        entity["normalized"] = " ".join(entity["text"].lower().split())
    return entities


def redact(text, entities):
    result, end = [], 0
    for entity in sorted(entities, key=lambda e: e["start"]):
        if entity["label"] not in {"SUSPECT", "VICTIM", "PERSON", "WITNESS"} or entity["start"] < end:
            continue
        result.extend([text[end:entity["start"]], f"[{entity['label']}]"])
        end = entity["end"]
    result.append(text[end:])
    value = "".join(result)
    value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", value)
    value = re.sub(r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\d[ -]?){9,14}\d(?!\w)", "[PHONE]", value)
    return value


def structured_summary(text, entities, summary):
    from .text import sentences
    source = sentences(text)
    facts = {label: sorted({e["text"] for e in entities if e["label"] == label and e.get("assertion", "mentioned") == "mentioned"})
             for label in ["SUSPECT", "VICTIM", "LOCATION", "WEAPON", "PROPERTY", "DATE", "TIME", "EVIDENCE", "ORGANIZATION"]}
    summary_sentences = {" ".join(sentence.split()) for sentence in sentences(summary)}
    citations = [{"sentence": i + 1, "text": sentence} for i, sentence in enumerate(source)
                 if " ".join(sentence.split()) in summary_sentences]
    return {"summary": summary, "mentioned_facts": facts, "citations": citations,
            "qualified_mentions": [entity for entity in entities if entity.get("assertion", "mentioned") != "mentioned"],
            "note": "Extracted mentions may be allegations; model errors and omissions are possible."}
