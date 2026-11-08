"""Demonstrate saved models, NLP, retrieval and measured results without training."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy import sparse
from threadpoolctl import threadpool_limits

from crime_nlp.config import ARTIFACTS, MODELS, write_json
from crime_nlp.features import keyword_vectorizer
from crime_nlp.inference import analyze, entity_transformer
from crime_nlp.ner_model import extract_bert
from crime_nlp.semantic import semantic_scores, rank_scores, semantic_evidence
from crime_nlp.store import fetch_reports
from crime_nlp.text import threat_language
from crime_nlp.train import document_vectors

EXAMPLE = ("Rajiv hit Sneha with a stick near Pune on 29-06-2026. "
           "A witness called the police. Camera footage was collected.")


def read(name):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def show(title, value):
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)
    if isinstance(value, pd.DataFrame):
        print(value.to_string(index=False), flush=True)
    elif isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)
    else:
        print(value, flush=True)


def retrieve(query, classifier, quick):
    vectorizer = keyword_vectorizer(classifier)
    matrix = sparse.load_npz(ARTIFACTS / "search_matrix.npz")
    lexical = np.asarray((matrix @ vectorizer.transform([query]).T).toarray()).ravel()
    semantic = None if quick else semantic_scores(query)
    results = {}
    for mode in (["Keyword"] if quick else ["Keyword", "Semantic", "Hybrid"]):
        scores = lexical if mode == "Keyword" else rank_scores(lexical, semantic, mode)
        threshold = {"Keyword": .02, "Semantic": .25, "Hybrid": 0.}[mode]
        positions = [int(i) for i in np.argsort(-scores) if scores[i] > threshold][:3]
        records = fetch_reports(positions)
        rows = [{"report_id": str(row.report_id), "summary": str(row.summary), "score": float(scores[i])}
                for i, row in records.iterrows()]
        results[mode] = rows
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Analyze this narrative instead of the fictional example")
    source.add_argument("--file", type=Path, help="Read a UTF-8 narrative file")
    parser.add_argument("--query", default="stolen mobile phone", help="Search query for the case library")
    parser.add_argument("--question", default="Camera footage", help="Retrieve evidence within the input narrative")
    parser.add_argument("--quick", action="store_true", help="Skip BERT comparisons, neural search and semantic evidence")
    parser.add_argument("--output", type=Path, help="Explicitly save this demonstration as JSON, including the input text")
    args = parser.parse_args()
    if not (MODELS / "tfidf.joblib").exists() or not (ARTIFACTS / "search_matrix.npz").exists():
        parser.error("Saved models/data are missing. Follow README.md setup and pipeline instructions first.")
    text = args.file.read_text(encoding="utf-8-sig") if args.file else (args.text if args.text is not None else EXAMPLE)
    threadpool_limits(4)
    if not args.quick:
        import torch
        torch.set_num_threads(4)
    started = time.perf_counter()
    show("Execution mode", "LIVE inference and search with saved models. Spark/training scores below are saved execution evidence; no training or ETL is run.")
    show("Input narrative (fictional default)", text)
    result = analyze(text, use_bert=not args.quick, ner_choice="Automatic")
    spark = read("spark_metrics.json")
    show("Recorded Spark cleaning", {key: spark[key] for key in ["raw_rows", "invalid_rows_removed", "duplicates_removed", "clean_rows", "master"]})
    classifier = joblib.load(MODELS / "tfidf.joblib")
    embedding = Word2Vec.load(str(MODELS / "word2vec.model"))
    embedding_classifier = joblib.load(MODELS / "word2vec_classifier.joblib")
    recipe_path = MODELS / "word2vec_features.json"
    recipe = json.loads(recipe_path.read_text()) if recipe_path.exists() else {"pooling": "mean"}
    predictions = {"Bag of Words": str(joblib.load(MODELS / "bow.joblib").predict([text])[0]),
                   "TF-IDF": result["crime_type"],
                   "Word2Vec": str(embedding_classifier.predict(document_vectors(embedding, [text], recipe))[0])}
    if not args.quick:
        predictions["BERT"] = result["bert_prediction"]
    show("Live category predictions", predictions)
    show("Category confidence and review", {key: result[key] for key in ["model_score", "review_required", "review_reasons"]})
    show("TF-IDF feature explanation", pd.DataFrame(result["explanation"]))
    show(f"Live entities and contextual roles: {result['ner_model']}", pd.DataFrame(sorted(result["entities"] + result.get("contextual_roles", []), key=lambda entity: entity["start"])))
    if result.get("entity_context_corrections"):
        show("Disclosed context corrections/roles (no model probability)", result["entity_context_corrections"])
    entity_comparison = {}
    if not args.quick:
        model, tokenizer = entity_transformer()
        entity_comparison["BERT"] = extract_bert(text, model, tokenizer)
        show("Legacy BERT core entities (before role additions and context checks)", pd.DataFrame(entity_comparison["BERT"]))
    show("Source-linked action and ownership notes", result.get("relations", {}))
    show("Reference links and unresolved ambiguity", result.get("references", []))
    show("Mentions flagged for review", result.get("entity_review", []))
    show("One-line TextRank summary", result["summary"])
    show("Original source sentence citations", result["structured_summary"]["citations"])
    show("spaCy syntax and morphology (first 12 tokens)", pd.DataFrame(result["tokens"][:12]))
    topics = read("topics.json")
    topic_rows = [{"topic": topic["id"] + 1, "probability": result["topic_distribution"][topic["id"]],
                   "top_terms": ", ".join(topic["terms"][:5])} for topic in topics["topics"]]
    show("Live LDA topic distribution", pd.DataFrame(topic_rows))
    show("Recorded topic quality", {"held_out_perplexity": topics["held_out_perplexity"], "top_term_diversity": topics["topic_diversity"]})
    show("VADER polarity and separate threat cues", {"polarity": result["sentiment"], "threat_language": result["threat"],
         "negation_example": threat_language("No weapon was used. Nobody was injured.")})
    search = retrieve(args.query, classifier, args.quick)
    for method, rows in search.items():
        show(f"Live {method} search: {args.query}", pd.DataFrame(rows))
    evidence = None if args.quick else semantic_evidence(args.question, text, "demo narrative")
    if evidence is not None:
        show(f"Live evidence search: {args.question}", evidence)
    evaluation, ner = read("evaluation.json"), read("ner_evaluation.json")
    show("Saved document evaluation (not accuracy of this one example)", pd.DataFrame([
        {"model": name, "accuracy_percent": round(item["accuracy"] * 100, 2), "macro_F1_percent": round(item["macro_f1"] * 100, 2)}
        for name, item in evaluation["models"].items()]))
    show("Saved legacy eight-label entity evaluation: strict Span-F1", pd.DataFrame([
        {"model": name, "span_F1_percent": round(item["strict_entity_f1"] * 100, 2)} for name, item in ner["models"].items()]))
    if (ARTIFACTS / "hybrid_entity_evaluation.json").exists():
        hybrid = read("hybrid_entity_evaluation.json")
        show("Separate ten-label synthetic diagnostic comparison", {"scope": hybrid["scope"],
             "test_span_f1": {name: item["strict_entity_f1"] for name, item in hybrid["splits"]["test"].items() if name != "rows"}})
    show("Evaluation scope", evaluation["protocol"]["evaluation_scope"])
    show("Interpretation", "Predictions and role mentions need review. Polarity/threat cues are not real-world danger estimates. Search scores are similarities. Topic and summary quality have no human-reference accuracy benchmark here.")
    if args.output:
        write_json(args.output, {"input": text, "category_predictions": predictions, "analysis": result,
                   "entity_comparison": entity_comparison, "search": search, "evidence": evidence,
                   "topic_distribution": topic_rows, "evaluation_scope": evaluation["protocol"]["evaluation_scope"]})
        print(f"\nExplicitly saved demonstration JSON, including input text: {args.output.resolve()}", flush=True)
    print(f"\nDemonstration completed in {time.perf_counter() - started:.1f}s. Dashboard: http://localhost:8502", flush=True)


if __name__ == "__main__":
    main()
