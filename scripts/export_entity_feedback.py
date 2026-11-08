"""Export explicitly opted-in local annotation drafts for human review."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import PROCESSED

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
database = PROCESSED / "feedback.sqlite"
rows = []
if database.exists():
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        if connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='entity_feedback'").fetchone():
            rows = connection.execute("SELECT created_at,narrative_sha256,payload_json FROM entity_feedback WHERE includes_text=1 ORDER BY created_at DESC").fetchall()
seen = set()
count = 0
with args.output.open("x", encoding="utf-8") as target:
    for created_at, fingerprint, payload in rows:
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        row = json.loads(payload)
        row.update(report_id="feedback-" + fingerprint[:16], captured_at=created_at, source_group="", permission="", reviewer="")
        target.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
print(f"Exported {count} latest unreviewed opted-in drafts; adjudication and source metadata are required before gold use.")
