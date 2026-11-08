"""Audit explicitly annotated development narratives without fitting models."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.challenges import cases
from crime_nlp.inference import analyze, language_model
from crime_nlp.config import write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--syntax", action="store_true")
    parser.add_argument("--crosscheck", action="store_true")
    args = parser.parse_args()
    if args.crosscheck:
        from crime_nlp.crosscheck_cases import cases
    results = []
    for case in cases():
        result = analyze(case["text"], ner_choice="Automatic")
        actual = result["entities"] + result.get("contextual_roles", [])
        expected_spans = {(e["start"], e["end"], e["label"]) for e in case["entities"]}
        actual_spans = {(e["start"], e["end"], e["label"]) for e in actual}
        missing = [e for e in case["entities"] if (e["start"], e["end"], e["label"]) not in actual_spans]
        extra = [e for e in actual if (e["start"], e["end"], e["label"]) not in expected_spans]
        category_ok = case["category"] is None or result["crime_type"] == case["category"]
        entity_ok = not case["score_entities"] or not missing and not extra
        row = {**case, "prediction": result["crime_type"], "review_required": result["review_required"],
               "actual": actual, "missing": missing if case["score_entities"] else [],
               "extra": extra if case["score_entities"] else [], "category_ok": category_ok, "entities_ok": bool(entity_ok)}
        results.append(row)
        if not category_ok or not entity_ok:
            print(json.dumps({"case": case["id"], "category": [case["category"], result["crime_type"]],
                              "missing": row["missing"], "extra": [{k: e[k] for k in ["text", "label", "start", "end"]} for e in row["extra"]]}, ensure_ascii=False), flush=True)
        if args.syntax and case["id"].startswith("user_"):
            print([(t.text, t.pos_, t.dep_, t.head.text) for t in language_model()(case["text"])], flush=True)
    summary = {"cases": len(results), "categories_scored": sum(r["category"] is not None for r in results),
               "category_correct": sum(r["category_ok"] for r in results if r["category"] is not None),
               "entities_scored": sum(r["score_entities"] for r in results),
               "entity_exact_cases": sum(r["entities_ok"] for r in results if r["score_entities"]),
               "all_requirements_passed": sum(r["category_ok"] and r["entities_ok"] for r in results)}
    write_json(args.output, {"scope": "Explicit hand-written development audit; not an independent final benchmark", "summary": summary, "cases": results})
    print(json.dumps(summary, indent=2))
