"""Disk-backed case retrieval and explicit opt-in feedback; no full text in overview memory."""
import json
import sqlite3
import hashlib
from contextlib import closing
from datetime import datetime, timezone
import pandas as pd
from .config import PROCESSED

DB = PROCESSED / "cases.sqlite"


def build_store(frame):
    target = DB.with_suffix(".building.sqlite")
    rows = frame.copy()
    rows.insert(0, "row_number", range(len(rows)))
    rows["event_time"] = rows.event_time.astype(str)
    with closing(sqlite3.connect(target)) as connection:
        rows.to_sql("reports", connection, if_exists="replace", index=False, chunksize=2000)
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_row ON reports(row_number)")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_report ON reports(report_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_filters ON reports(crime_type,district,reported_at)")
        connection.commit()
    target.replace(DB)


def fetch_reports(row_numbers):
    positions = [int(i) for i in row_numbers]
    if not positions:
        return pd.DataFrame()
    with closing(sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)) as connection:
        frame = pd.read_sql_query(f"SELECT * FROM reports WHERE row_number IN ({','.join('?' for _ in positions)})", connection, params=positions)
    return frame.set_index("row_number").reindex(positions)


def save_feedback(text, predicted, correction, comment=""):
    """Store a hash and correction only. Submitted narrative text is not persisted."""
    path = PROCESSED / "feedback.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS feedback (created_at TEXT, narrative_sha256 TEXT, predicted TEXT, corrected TEXT, comment TEXT)")
        connection.execute("INSERT INTO feedback VALUES (?,?,?,?,?)", (datetime.now(timezone.utc).isoformat(), hashlib.sha256(text.encode()).hexdigest(), predicted, correction, comment[:1000]))
        connection.commit()


def save_entity_feedback(text, predicted, corrected, revision, include_text=False):
    """Store hash + offset/label corrections; source text requires explicit opt-in."""
    from .annotation_schema import validate_record, SCHEMA_VERSION
    corrected = validate_record({"narrative": text, "entities": corrected})
    def minimal(entities):
        return [{key: entity[key] for key in ("start", "end", "label")} for entity in entities]
    payload = {"annotation_schema": SCHEMA_VERSION, "model_revision": revision,
               "predicted": minimal(predicted), "corrected": minimal(corrected), "review_status": "unreviewed"}
    if include_text:
        payload.update(narrative=text, entities=corrected)
    with closing(sqlite3.connect(PROCESSED / "feedback.sqlite")) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS entity_feedback (created_at TEXT, narrative_sha256 TEXT, includes_text INTEGER, payload_json TEXT)")
        connection.execute("INSERT INTO entity_feedback VALUES (?,?,?,?)", (datetime.now(timezone.utc).isoformat(),
                           hashlib.sha256(text.encode()).hexdigest(), int(include_text), json.dumps(payload, ensure_ascii=False)))
        connection.commit()


def entity_matches(label="Any", value=""):
    clauses, params = [], []
    if label != "Any":
        clauses.append("json_extract(e.value,'$.label') = ?")
        params.append(label)
    if value.strip():
        clauses.append("instr(lower(json_extract(e.value,'$.text')),lower(?)) > 0")
        params.append(value.strip())
    if not clauses:
        return None
    with closing(sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(reports)")}
        sources = ["r.predicted_entities_json"]
        if "contextual_roles_json" in columns:
            sources.append("r.contextual_roles_json")
        tests = ["EXISTS (SELECT 1 FROM json_each(" + source + ") e WHERE " + " AND ".join(clauses) + ")" for source in sources]
        sql = "SELECT row_number FROM reports r WHERE " + " OR ".join(tests)
        return [row[0] for row in connection.execute(sql, params * len(sources))]


def record_audit(event, report_id):
    """Opt-in local access events contain only case IDs and event names."""
    with closing(sqlite3.connect(PROCESSED / "audit.sqlite")) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS events (at TEXT, event TEXT, report_id TEXT)")
        connection.execute("INSERT INTO events VALUES (?,?,?)", (datetime.now(timezone.utc).isoformat(), event, report_id))
        connection.commit()
