"""Refresh fuel/material narratives, verifying all other entity rows are retained."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from crime_nlp.config import ARTIFACTS, PROCESSED, write_json
from crime_nlp.enrich import refresh_entity_predictions
from crime_nlp.material_context import MATERIAL

path = PROCESSED / "dashboard.parquet"
columns = ["report_id", "narrative", "predicted_entities_json", "contextual_roles_json"]
before = pd.read_parquet(path, columns=columns)
affected = before.narrative.str.contains(MATERIAL)
refresh_entity_predictions(materials_only=True)
after = pd.read_parquet(path, columns=columns)
if not before[["report_id", "narrative"]].equals(after[["report_id", "narrative"]]):
    raise RuntimeError("Report text or ordering changed unexpectedly.")
if not before.loc[~affected].equals(after.loc[~affected]):
    raise RuntimeError("An unrelated entity row changed unexpectedly.")
receipt = {"status": "passed", "rows": len(after), "material_rows_refreshed": int(affected.sum()),
           "unrelated_rows_preserved": int((~affected).sum()), "scope": "Rerun raw selected NER and current context checks for material narratives; preserve other rows, models, classifier outputs and retrieval indexes"}
write_json(ARTIFACTS / "material_refresh.json", receipt)
print(receipt)
