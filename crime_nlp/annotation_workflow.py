"""Human-review preparation and source-separated evaluation contracts."""
import hashlib
import json
import random
import re
from .annotation_schema import validate_record, SCHEMA_VERSION
from .config import SEED


def source_split(records):
    groups = sorted({row["source_group"] for row in records})
    if len(groups) < 3:
        raise ValueError("At least three independent source groups are required for train/validation/test")
    random.Random(SEED).shuffle(groups)
    train_end = min(len(groups) - 2, max(1, round(len(groups) * .6)))
    val_end = min(len(groups) - 1, max(train_end + 1, round(len(groups) * .8)))
    assignment = {group: "train" if i < train_end else "validation" if i < val_end else "test" for i, group in enumerate(groups)}
    seen_ids, seen_text = set(), {}
    result = []
    for record in records:
        row = dict(record)
        validate_record(row)
        if row["report_id"] in seen_ids:
            raise ValueError("Duplicate report ID")
        fingerprint = hashlib.sha256(re.sub(r"[\W_]+", " ", row["narrative"].lower()).strip().encode()).hexdigest()
        split = assignment[row["source_group"]]
        if fingerprint in seen_text:
            raise ValueError("Duplicate normalized narrative; deduplicate before splitting")
        seen_ids.add(row["report_id"])
        seen_text[fingerprint] = split
        row.update(split=split, annotation_schema=SCHEMA_VERSION, text_sha256=hashlib.sha256(row["narrative"].encode()).hexdigest())
        result.append(row)
    return result


def agreement(first, second):
    left = {row["report_id"]: row for row in first}
    right = {row["report_id"]: row for row in second}
    if len(left) != len(first) or len(right) != len(second):
        raise ValueError("Each annotator's report IDs must be unique")
    if set(left) != set(right):
        raise ValueError("Both annotators must annotate the same report IDs")
    matches, total_a, total_b, disagreements = 0, 0, 0, []
    for report_id, row in left.items():
        other = right[report_id]
        if row["narrative"] != other["narrative"]:
            raise ValueError("Annotators must use identical source text")
        a = {(e["start"], e["end"], e["label"], e.get("assertion", "mentioned")) for e in validate_record(row)}
        b = {(e["start"], e["end"], e["label"], e.get("assertion", "mentioned")) for e in validate_record(other)}
        matches += len(a & b)
        total_a += len(a)
        total_b += len(b)
        if a != b:
            disagreements.append({"report_id": report_id, "only_first": sorted(a - b), "only_second": sorted(b - a)})
    return {"reports": len(left), "exact_entity_and_assertion_f1": 2 * matches / max(1, total_a + total_b),
            "reports_requiring_adjudication": len(disagreements), "disagreements": disagreements,
            "scope": "Agreement between submitted annotations; not model accuracy or proof of annotator independence"}


def validate_adjudicated(row):
    validate_record(row)
    if row.get("review_status") != "adjudicated" or not row.get("reviewer"):
        raise ValueError("Gold export requires adjudication status and a reviewer identifier")
    if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ["source_group", "permission", "reviewer"]):
        raise ValueError("Gold export requires source grouping and recorded permission")
    return row


def load_jsonl(path):
    with open(path, encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]
