"""Check every stored source span and report conflicting mentions."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from crime_nlp.config import PROCESSED, ARTIFACTS, write_json

frame = pd.read_parquet(PROCESSED / "dashboard.parquet")
failures = []
for row in frame.itertuples():
    mentions = sorted(json.loads(row.predicted_entities_json) + json.loads(row.contextual_roles_json),
                      key=lambda entity: (entity["start"], entity["end"]))
    conflicts = [(a, b) for a, b in zip(mentions, mentions[1:]) if a["end"] > b["start"]]
    mismatches = [entity for entity in mentions if row.narrative[entity["start"]:entity["end"]] != entity["text"]]
    if conflicts or mismatches:
        failures.append({"report_id": row.report_id, "text": row.narrative, "conflicts": conflicts, "mismatches": mismatches})
result = {"status": "failed" if failures else "passed", "verified_at": datetime.now(timezone.utc).isoformat(),
          "scope": "All stored source offsets and span overlaps; this does not measure semantic accuracy",
          "rows": len(frame), "failures": len(failures), "examples": failures[:12]}
write_json(ARTIFACTS / "entity_store_validation.json", result)
print(json.dumps(result, ensure_ascii=True, indent=2))
sys.exit(bool(failures))
