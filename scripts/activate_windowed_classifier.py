"""Activate a validation-selected classifier mode without replacing legacy files."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import ARTIFACTS, write_json

experiment = json.loads((ARTIFACTS / "windowed_classification_experiment.json").read_text(encoding="utf-8"))
choice = experiment["validation_choice"]
directory = "models/v4_classifier/bert_classifier" if choice == "candidate" else "models/v3_1/bert_classifier"
write_json(ARTIFACTS / "active_document_model.json", {"directory": directory, "full_text": choice != "legacy",
           "validation_choice": choice, "selection_artifact": "windowed_classification_experiment.json",
           "legacy_benchmark": "Unchanged archived truncating inference; live full-document mode is reported separately"})
print(f"Activated {choice}; legacy classifier files preserved.")
