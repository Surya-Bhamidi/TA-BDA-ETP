"""Train bounded BERT candidates using train/validation only, with no test scoring."""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from threadpoolctl import threadpool_limits
from crime_nlp.config import ARTIFACTS, MODELS, PROCESSED, write_json
from crime_nlp.train import sample_split, entity_split
from crime_nlp.tuning import validation_score, choose_candidate, mask_training_mentions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "entities"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--mention-dropout", type=float, default=0.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    threadpool_limits(4)
    import torch
    torch.set_num_threads(4)
    frame = pd.read_parquet(PROCESSED / "reports")
    if args.task == "classification":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from crime_nlp.bert_model import fit_bert, predict_bert
        train, val = sample_split(frame, "train", 12000), sample_split(frame, "validation", 800)
        old = AutoModelForSequenceClassification.from_pretrained(MODELS / "bert_classifier", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(MODELS / "bert_classifier", local_files_only=True)
        baseline = validation_score(val.crime_type, predict_bert(old, tokenizer, val.narrative.tolist()))
        del old
        gc.collect()
        texts = mask_training_mentions(train, args.mention_dropout) if args.mention_dropout else train.narrative.tolist()
        _, details = fit_bert(texts, train.crime_type.tolist(), val.narrative.tolist(),
                              val.crime_type.tolist(), [], epochs=6, learning_rate=args.learning_rate, scheduled=True,
                              max_tokens=192, target_dir=args.output / "bert_classifier")
        details["recipe"]["train_only_mention_dropout"] = args.mention_dropout
        candidate = {"macro_f1": details["best_validation_macro_f1"]}
        write_json(args.output / "classification_selection.json", {"baseline": baseline, "candidate": candidate,
                   "use_candidate": choose_candidate(baseline, candidate), "details": details, "selection_split": "validation"})
    else:
        from crime_nlp.ner_model import train_ner
        baseline = json.loads((ARTIFACTS / "ner_evaluation.json").read_text())["validation"]["BERT"]
        train, val = entity_split(frame, "train", 8000), entity_split(frame, "validation", 400)
        details = train_ner(train, val, epochs=6, scheduled=True, target_dir=args.output / "bert_ner")
        write_json(args.output / "entity_selection.json", {"baseline_f1": baseline, "candidate_f1": details["validation_f1"],
                   "use_candidate": details["validation_f1"] >= baseline + .002, "minimum_gain": .002,
                   "details": details, "selection_split": "validation"})


if __name__ == "__main__":
    main()
