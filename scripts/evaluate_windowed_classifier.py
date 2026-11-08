"""Score the fixed live windowed mode on the viewed legacy diagnostic split."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from crime_nlp.config import ARTIFACTS, MODELS, PROCESSED, write_json
from crime_nlp.inference import bert_model
from crime_nlp.bert_model import predict_bert
from crime_nlp.evaluation import classification_metrics
from crime_nlp.train import file_hash

reference = pd.read_parquet(PROCESSED / "evaluation_predictions.parquet")
texts = pd.read_parquet(PROCESSED / "reports", columns=["report_id", "narrative"]).set_index("report_id").loc[reference.report_id].narrative.tolist()
model, tokenizer = bert_model()
prediction = predict_bert(model, tokenizer, texts, full_text=True)
reference["BERT_full_document"] = prediction
reference.to_parquet(PROCESSED / "windowed_evaluation_predictions.parquet", index=False)
write_json(ARTIFACTS / "windowed_classification_evaluation.json", {"rows": len(reference),
           "legacy": classification_metrics(reference.crime_type, reference.BERT), "full_document": classification_metrics(reference.crime_type, prediction),
           "legacy_weights_sha256": file_hash(MODELS / "bert_classifier" / "model.safetensors"),
           "scope": "Mode selected on validation before this repeat diagnostic test. Existing legacy predictions and benchmarks are unchanged."})
print({"legacy_accuracy": float((reference.BERT == reference.crime_type).mean()), "full_document_accuracy": float((reference.BERT_full_document == reference.crime_type).mean())})
