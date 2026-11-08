"""Train two-label role specialists, retaining the stronger legacy core tagger.

The v4 diagnostic test has already been viewed. This follow-up is selected on
validation and reports repeat diagnostic scores; it is not an untouched test.
"""
import gc
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
import sklearn_crfsuite
from crime_nlp.config import ARTIFACTS, PROCESSED, write_json
from crime_nlp.entity_corpus_v4 import generate
from crime_nlp.train import sequence_data, file_hash
from crime_nlp.evaluation import entity_metrics, split_integrity
from crime_nlp.ner_model import train_ner, load_ner, sequences, predict_tags

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=ARTIFACTS / "models" / "v4_roles")
parser.add_argument("--report", default="role_specialist_evaluation.json")
args = parser.parse_args()
labels = ["ORGANIZATION", "WITNESS"]
target = args.output
if target.exists():
    raise RuntimeError("Use a fresh specialist directory; previous experiments must be preserved")
target.mkdir(parents=True)
data = PROCESSED / "annotation_v4"
legacy = pd.read_json(data / "train.jsonl", lines=True)
legacy = legacy[legacy.source != "synthetic-explicit-schema-v4"]
positive = legacy[legacy.entities_json.str.contains("WITNESS")]
negative = legacy[~legacy.index.isin(positive.index)].sample(800, random_state=42)
train = pd.concat([positive, negative, generate("train", 6000)], ignore_index=True)
validation = pd.read_json(data / "validation.jsonl", lines=True)

def roles_only(frame):
    result = frame.copy()
    result["entities_json"] = [json.dumps([entity for entity in json.loads(value) if entity["label"] in labels], ensure_ascii=False) for value in result.entities_json]
    return result

train, validation = roles_only(train), roles_only(validation)
split_integrity({"train": train, "validation": validation})
train.to_json(target / "training_annotations.jsonl", orient="records", lines=True, force_ascii=False)
plan = {"labels": labels, "train_rows": len(train), "validation_rows": len(validation),
        "selection": "Validation-only specialist selection; preserve core model; no user examples in training",
        "scope": "Synthetic role-specialist follow-up after viewing v4 diagnostic test scores. Repeat diagnostic evaluation, not independent real data."}
write_json(target / "plan.json", plan)
x, y = sequence_data(train)
vx, vy = sequence_data(validation)
crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=.01, c2=.1, max_iterations=180, all_possible_transitions=True).fit(x, y)
joblib.dump(crf, target / "crf.joblib")
crf_validation = entity_metrics(vy, crf.predict(vx))
print(f"Role CRF validation F1: {crf_validation['strict_entity_f1']:.4f}", flush=True)
write_json(target / "crf_validation.json", crf_validation)
del x, y
gc.collect()
details = train_ner(train, validation, epochs=6, scheduled=True, target_dir=target / "bert_ner", entity_labels=labels)
chosen = "BERT" if details["validation_f1"] > crf_validation["strict_entity_f1"] else "CRF"
write_json(target / "ner_selection.json", {"selected": chosen, "labels": labels,
           "crf_validation_f1": crf_validation["strict_entity_f1"], "bert_validation_f1": details["validation_f1"]})
freeze = {"frozen_at": datetime.now(timezone.utc).isoformat(), "models": {
    path.relative_to(target).as_posix(): file_hash(path) for path in sorted(target.rglob("*")) if path.is_file()}, "plan": plan}
write_json(target / "frozen.json", freeze)
test = roles_only(pd.read_json(data / "test.jsonl", lines=True))
tx, ty = sequence_data(test)
crf_tags = list(crf.predict(tx))
bert, tokenizer = load_ner(target / "bert_ner")
tokens, _ = sequences(test)
bert_tags, _ = predict_tags(bert, tokenizer, tokens)
scores = {"CRF": entity_metrics(ty, crf_tags), "BERT": entity_metrics(ty, bert_tags)}
test["gold_bio"], test["crf_bio"], test["bert_bio"] = ty, crf_tags, bert_tags
test.to_json(target / "diagnostic_predictions.jsonl", orient="records", lines=True, force_ascii=False)
write_json(ARTIFACTS / args.report, {"plan": plan, "selected": chosen,
           "validation": {"CRF": crf_validation["strict_entity_f1"], "BERT": details["validation_f1"]}, "models": scores,
           "frozen_at": freeze["frozen_at"], "evaluated_at": datetime.now(timezone.utc).isoformat(), "candidate_directory": str(target)})
print({name: score["strict_entity_f1"] for name, score in scores.items()}, flush=True)
