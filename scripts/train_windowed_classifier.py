"""Compare a full-document BERT candidate using validation, preserving legacy weights."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from crime_nlp.config import ARTIFACTS, MODELS, PROCESSED, write_json
from crime_nlp.train import sample_split, file_hash
from crime_nlp.bert_model import fit_bert, predict_bert
from crime_nlp.tuning import mask_training_mentions
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=ARTIFACTS / "models" / "v4_classifier")
parser.add_argument("--epochs", type=int, default=8)
args = parser.parse_args()
if args.output.exists():
    raise RuntimeError("Use a fresh candidate directory")
args.output.mkdir(parents=True)
frame = pd.read_parquet(PROCESSED / "reports")
train, validation = sample_split(frame, "train", 12000), sample_split(frame, "validation", 800)
baseline = AutoModelForSequenceClassification.from_pretrained(MODELS / "bert_classifier", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODELS / "bert_classifier", local_files_only=True)
legacy_prediction = predict_bert(baseline, tokenizer, validation.narrative.tolist())
windowed_prediction = predict_bert(baseline, tokenizer, validation.narrative.tolist(), full_text=True)
baseline_score = float(f1_score(validation.crime_type, legacy_prediction, average="macro"))
legacy_windowed_score = float(f1_score(validation.crime_type, windowed_prediction, average="macro"))
write_json(args.output / "baseline_validation.json", {"legacy_macro_f1": baseline_score, "same_weights_full_document_macro_f1": legacy_windowed_score,
           "scope": "Existing previously used validation split; no test-based selection"})
del baseline
import gc
gc.collect()
texts = mask_training_mentions(train, .35)
_, details = fit_bert(texts, train.crime_type.tolist(), validation.narrative.tolist(), validation.crime_type.tolist(), [],
                      epochs=args.epochs, learning_rate=5e-5, scheduled=True, max_tokens=256, target_dir=args.output / "bert_classifier", windowed=True)
candidate_score = details["best_validation_macro_f1"]
choice = "candidate" if candidate_score >= max(baseline_score, legacy_windowed_score) + .002 else "legacy_windowed" if legacy_windowed_score >= baseline_score else "legacy"
write_json(ARTIFACTS / "windowed_classification_experiment.json", {"legacy_validation_macro_f1": baseline_score,
           "legacy_windowed_validation_macro_f1": legacy_windowed_score, "candidate_validation_macro_f1": candidate_score,
           "validation_choice": choice, "candidate_directory": str(args.output), "details": details,
           "candidate_weights_sha256": file_hash(args.output / "bert_classifier" / "model.safetensors"),
           "scope": "Validation-selected experiment on existing synthetic data; legacy test remains unchanged and is not used for selection"})
print({"validation_choice": choice, "baseline": baseline_score, "windowed": legacy_windowed_score, "candidate": candidate_score})
