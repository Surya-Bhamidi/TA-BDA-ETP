"""Recheck changed 'without' scope without rerunning taggers or changing spans."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from crime_nlp.config import ARTIFACTS, PROCESSED, write_json
from crime_nlp.inference import language_model
from crime_nlp.privacy import annotate_assertions
from crime_nlp.assertions import qualify_entities

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()
path = PROCESSED / "dashboard.parquet"
frame = pd.read_parquet(path)
selected = frame.loc[frame.narrative.str.contains(r"\bwithout\b", case=False, regex=True)]
changed = 0
for (position, row), doc in zip(selected.iterrows(), language_model().pipe(selected.narrative.tolist(), batch_size=64)):
    updated = {}
    for column in ["predicted_entities_json", "contextual_roles_json"]:
        before = json.loads(row[column])
        after = qualify_entities(row.narrative, annotate_assertions(row.narrative, [dict(entity) for entity in before]), doc)
        if before != after:
            updated[column] = json.dumps(after, ensure_ascii=False)
    if updated:
        changed += 1
        for column, value in updated.items():
            frame.at[position, column] = value
if changed and args.apply:
    pending = path.with_suffix(".building.parquet")
    frame.to_parquet(pending, index=False)
    pending.replace(path)
    from crime_nlp.store import build_store
    build_store(frame)
write_json(ARTIFACTS / "assertion_subset_refresh.json", {"checked_rows": len(selected), "changed_rows": changed,
           "applied": args.apply, "scope": "Qualifier metadata only; report text, span boundaries, types and learned scores unchanged"})
print({"checked_rows": len(selected), "changed_rows": changed, "applied": args.apply})
