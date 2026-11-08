"""Select trained role additions on validation; retain the core model."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import ARTIFACTS, write_json
from crime_nlp.train import file_hash

parser = argparse.ArgumentParser()
parser.add_argument("--reports", nargs="+", default=["role_specialist_evaluation.json", "role_specialist_broad_evaluation.json"])
args = parser.parse_args()
candidates = []
for name in args.reports:
    report = json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))
    for model, score in report["validation"].items():
        candidates.append((score, model, Path(report["candidate_directory"]).resolve(), name))
score, model, directory, report_name = max(candidates, key=lambda row: (row[0], row[1] == "CRF"))
freeze = json.loads((directory / "frozen.json").read_text(encoding="utf-8"))
for name, digest in freeze["models"].items():
    if file_hash(directory / name) != digest:
        raise RuntimeError(f"Frozen candidate file changed: {name}")
write_json(ARTIFACTS / "active_entity_roles.json", {"directory": directory.relative_to(ARTIFACTS).as_posix(), "selected": model,
           "labels": ["ORGANIZATION", "WITNESS"], "validation_span_f1": score, "selection_artifact": report_name,
           "strategy": "Retain legacy eight-label core and context safeguards; add non-overlapping learned roles. No diagnostic test score is used for selection."})
print(f"Activated {model} specialist from {directory.name}; validation F1={score:.4f}. Core weights retained.")
