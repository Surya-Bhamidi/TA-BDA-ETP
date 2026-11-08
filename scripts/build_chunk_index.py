"""Build source-window semantic retrieval while retaining the legacy index."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from crime_nlp.config import ARTIFACTS, PROCESSED, write_json
from crime_nlp.semantic import encoder, encode
from crime_nlp.windowing import source_windows
from crime_nlp.train import file_hash
from crime_nlp.audit import corpus_fingerprint

frame = pd.read_parquet(PROCESSED / "dashboard.parquet", columns=["report_id", "narrative"])
_, tokenizer = encoder()
embeddings, row_ids, offsets = [], [], []
for first in range(0, len(frame), 512):
    texts = frame.narrative.iloc[first:first + 512].tolist()
    chunks = source_windows(texts, tokenizer)
    embeddings.append(encode([chunk["text"] for chunk in chunks]).astype(np.float16))
    row_ids.extend(first + chunk["source"] for chunk in chunks)
    offsets.extend((chunk["start"], chunk["end"]) for chunk in chunks)
    if first % 5120 == 0:
        print(f"Chunked retrieval {min(first + 512, len(frame)):,}/{len(frame):,}", flush=True)
arrays = {"semantic_chunk_embeddings.npy": np.concatenate(embeddings),
          "semantic_chunk_rows.npy": np.asarray(row_ids, dtype=np.int32),
          "semantic_chunk_offsets.npy": np.asarray(offsets, dtype=np.int32)}
for name, array in arrays.items():
    target = ARTIFACTS / name
    pending = target.with_name(target.stem + ".building.npy")
    np.save(pending, array)
    pending.replace(target)
write_json(ARTIFACTS / "semantic_chunk_manifest.json", {"reports": len(frame), "chunks": len(row_ids),
           "window_subwords": 128, "overlap_subwords": 32, "ranking": "Maximum chunk cosine per report, then existing hybrid rank fusion",
           "source_ids_sha256": __import__("hashlib").sha256("\n".join(frame.report_id).encode()).hexdigest(),
           "source_narratives_sha256": corpus_fingerprint(frame.report_id, frame.narrative),
           "files": {name: file_hash(ARTIFACTS / name) for name in arrays}, "scope": "Full-document coverage; quality requires human relevance judgments"})
print(f"Saved {len(row_ids):,} source windows for {len(frame):,} reports.", flush=True)
