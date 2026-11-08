"""Record actual predictions on explicit material-role development examples."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import write_json
from crime_nlp.inference import analyze
from crime_nlp.material_cases import CASES

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--ner", choices=["Automatic", "CRF", "BERT"], default="Automatic")
args = parser.parse_args()
rows = []
for name, text, mention, expected in CASES:
    result = analyze(text, ner_choice=args.ner)
    first = text.index(mention)
    matches = [entity for entity in result["entities"] if entity["start"] < first + len(mention) and entity["end"] > first]
    passed = (any(entity["text"] == mention and entity["label"] == expected for entity in matches) if expected
              else not any(entity["label"] in {"WEAPON", "EVIDENCE"} for entity in matches))
    rows.append({"id": name, "text": text, "mention": mention, "expected": expected,
                 "actual": matches, "category": result["crime_type"], "passed": passed})
summary = {"cases": len(rows), "passed": sum(row["passed"] for row in rows), "ner": args.ner}
write_json(args.output, {"scope": "Hand-written development checks; neutral cases prohibit false WEAPON/EVIDENCE claims, not all possible entity labels", "summary": summary, "cases": rows})
print(summary)
for row in rows:
    if not row["passed"]:
        print(row["id"], repr(row["actual"]))
