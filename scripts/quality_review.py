"""Prepare blank human-review forms or evaluate completed judgments."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from crime_nlp.config import PROCESSED, ARTIFACTS, write_json
from crime_nlp.quality_review import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("action", choices=["prepare", "evaluate"])
parser.add_argument("--input", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.action == "evaluate":
    if args.input is None:
        parser.error("evaluate requires --input")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    result = evaluate(rows)
    write_json(args.output, result)
    print(result)
else:
    frame = pd.read_parquet(PROCESSED / "dashboard.parquet")
    sample = frame[frame.split == "development"].sample(24, random_state=42)
    rows = [{"task": "summary", "report_id": row.report_id, "narrative": row.narrative, "summary": row.summary,
             "review_status": "unreviewed", "reviewer": "", "faithful_to_source": None, "preserves_participants": None,
             "preserves_negation": None, "covers_main_event": None} for row in sample.itertuples()]
    topics = json.loads((ARTIFACTS / "topics.json").read_text(encoding="utf-8"))
    rows.extend({"task": "topic", "topic_id": topic["id"], "terms": topic["terms"], "coherent": None,
                 "human_label": "", "review_status": "unreviewed", "reviewer": ""} for topic in topics["topics"])
    from crime_nlp.semantic import semantic_scores
    queries = ["stolen mobile phone", "a fire started using petrol", "account password changed by a hacker", "money paid for goods never delivered"]
    for query in queries:
        scores = semantic_scores(query)
        ranked = scores.argsort()[::-1][:10]
        rows.append({"task": "retrieval", "query": query, "ranked_ids": frame.report_id.iloc[ranked].tolist(),
                     "candidate_texts": frame.narrative.iloc[ranked].tolist(), "relevant_ids": [], "k": 10,
                     "review_status": "unreviewed", "reviewer": "", "instructions": "Judge relevance against the library; include relevant IDs beyond this candidate list when assessing recall."})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Prepared {len(rows)} unreviewed forms; no quality scores have been measured.")
