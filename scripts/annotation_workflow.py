"""Prepare reviewed real-data annotations without mixing them into benchmarks."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.annotation_workflow import load_jsonl, source_split, agreement, validate_adjudicated
from crime_nlp.config import write_json

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="action", required=True)
compare = sub.add_parser("agreement")
compare.add_argument("first", type=Path)
compare.add_argument("second", type=Path)
compare.add_argument("--output", type=Path, required=True)
split = sub.add_parser("split")
split.add_argument("input", type=Path)
split.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.action == "agreement":
    result = agreement(load_jsonl(args.first), load_jsonl(args.second))
    write_json(args.output, result)
    print({key: value for key, value in result.items() if key != "disagreements"})
else:
    rows = source_split([validate_adjudicated(row) for row in load_jsonl(args.input)])
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("train", "validation", "test"):
        with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as target:
            for row in rows:
                if row["split"] == name:
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.output / "manifest.json", {"rows": len(rows), "scope": "User-supplied adjudicated annotations, source-separated; reviewer metadata is recorded, not independently verified", "counts": {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")}})
    print(f"Prepared source-separated annotations in {args.output}")
