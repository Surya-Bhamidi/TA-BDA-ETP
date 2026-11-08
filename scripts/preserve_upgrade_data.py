"""Keep experiment inputs alongside each archived model bundle."""
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import ARTIFACTS, PROCESSED, write_json
from crime_nlp.train import file_hash

source = PROCESSED / "annotation_v4"
for name in ["v4_entities", "v4_roles", "v4_roles_broad"]:
    directory = ARTIFACTS / "models" / name
    target = directory / "data_snapshot"
    target.mkdir(exist_ok=True)
    files = ["train.jsonl", "validation.jsonl", "development.jsonl", "test.jsonl"] if name == "v4_entities" else ["validation.jsonl", "test.jsonl"]
    for filename in files:
        destination = target / filename
        if destination.exists() and file_hash(destination) != file_hash(source / filename):
            raise RuntimeError("Existing experiment data snapshot differs; refusing to overwrite it")
        if not destination.exists():
            shutil.copy2(source / filename, destination)
    manifest = {filename: file_hash(target / filename) for filename in files}
    if name == "v4_entities":
        freeze = json.loads((directory / "frozen.json").read_text(encoding="utf-8"))
        if manifest["train.jsonl"] != freeze["training_data_sha256"] or manifest["validation.jsonl"] != freeze["validation_data_sha256"]:
            raise RuntimeError("Initial ten-label inputs no longer match their frozen hashes")
    write_json(directory / "data_snapshot.json", {"files": manifest, "scope": "Frozen experiment inputs; later generator changes do not redefine these archived annotations"})
print("Preserved annotation inputs for all three entity experiments.")
