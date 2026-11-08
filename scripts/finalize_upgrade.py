"""Verify corpus/index lineage and record delivered capabilities and limits."""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from crime_nlp.config import ARTIFACTS, PROCESSED, ROOT, write_json
from crime_nlp.audit import corpus_fingerprint

frame = pd.read_parquet(PROCESSED / "dashboard.parquet", columns=["report_id", "narrative"])
source = pd.read_parquet(PROCESSED / "reports", columns=["report_id", "narrative"]).sort_values("report_id").reset_index(drop=True)
if not frame.equals(source):
    raise RuntimeError("Stored corpus differs from its preserved source")
rows = np.load(ARTIFACTS / "semantic_chunk_rows.npy", mmap_mode="r")
offsets = np.load(ARTIFACTS / "semantic_chunk_offsets.npy", mmap_mode="r")
vectors = np.load(ARTIFACTS / "semantic_chunk_embeddings.npy", mmap_mode="r")
if len(rows) != len(offsets) or len(rows) != len(vectors) or set(rows.tolist()) != set(range(len(frame))):
    raise RuntimeError("Source-window index coverage is incomplete")
previous = {}
for row, (first, last) in zip(rows.tolist(), offsets.tolist()):
    text = frame.narrative.iloc[row]
    if not 0 <= first < last <= len(text) or text[previous.get(row, 0):first].strip():
        raise RuntimeError("Source-window boundary or coverage mismatch")
    previous[row] = max(previous.get(row, 0), last)
if any(frame.narrative.iloc[row][last:].strip() for row, last in previous.items()):
    raise RuntimeError("A report tail is missing from its source windows")
digest = corpus_fingerprint(frame.report_id, frame.narrative)
for filename, key in [("index_manifest.json", "report_text_sha256"), ("semantic_chunk_manifest.json", "source_narratives_sha256")]:
    path = ARTIFACTS / filename
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get(key) and manifest[key] != digest:
        raise RuntimeError("An existing corpus fingerprint disagrees; rebuild the index instead of changing its provenance")
    manifest[key] = digest
    write_json(path, manifest)
read = lambda name: json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))
test_log = (ROOT / "logs" / "tests.log").read_bytes()
test_text = test_log.decode("utf-16" if test_log.startswith(b"\xff\xfe") else "utf-8", errors="replace")
test_counts = re.findall(r"(\d+) passed", test_text)
write_json(ARTIFACTS / "upgrade_delivery.json", {
    "verified_at": datetime.now(timezone.utc).isoformat(), "corpus_rows": len(frame), "semantic_windows": len(rows),
    "index_source_coverage": "passed: all source characters outside whitespace covered; report order and source text preserved",
    "role_model": read("active_entity_roles.json"), "classifier_mode": read("active_document_model.json"),
    "verification": {"automated_tests_passed": int(test_counts[-1]) if test_counts else None,
                     "legacy_benchmark": read("evaluation_verification.json")["status"],
                     "new_model_experiments": read("upgrade_model_verification.json")["status"],
                     "stored_span_integrity": read("entity_store_validation.json")["status"],
                     "browser": read("browser_check.json")["status"]},
    "core_reliability": {"fit_split": read("core_entity_calibration.json")["fit_split"],
                         "model_hashes": read("core_entity_calibration.json")["legacy_core_sha256"],
                         "scope": "Fitted on synthetic validation; rules remain unscored and omissions are not calibrated"},
    "hybrid_diagnostic": {name: result["strict_entity_f1"] for name, result in read("hybrid_entity_evaluation.json")["splits"]["test"].items() if name != "rows"},
    "implemented": ["Versioned ten-label annotation contract and preserved data snapshots", "Validation-selected trained organization/witness additions", "Validation-only core/role reliability and per-entity review", "Negation, denial, attribution and bounded reference/action notes", "Full-document semantic search and windowed BERT inference", "Entity correction with text-hash default and explicit annotation opt-in", "Source-separated human annotation, agreement and adjudication workflow", "Unreviewed human quality forms and metric evaluator"],
    "still_requires_external_input": ["Permissioned real narratives", "Independent human annotation/adjudication", "Human relevance, summary and topic judgments"],
    "limitations": "Synthetic diagnostic improvements do not establish real-world accuracy. Role recall remains limited. General coreference and full logical inference are not claimed. Rejected candidates are preserved. Legacy benchmark files are unchanged."})
print({"reports": len(frame), "semantic_windows": len(rows), "source_coverage": "passed"})
