"""Bounded candidate comparisons using training and validation data only."""
import json
import time
from collections import Counter

import joblib
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from .config import CATEGORIES, SEED, write_json
from .features import robust_tfidf
from .text import word_tokens


def validation_score(truth, guesses):
    return {"macro_f1": float(f1_score(truth, guesses, labels=CATEGORIES, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(truth, guesses))}


def choose_candidate(incumbent, candidate):
    """Strict validation improvement; ties retain the incumbent."""
    return candidate["macro_f1"] > incumbent["macro_f1"] + 1e-12


def mask_training_mentions(frame, probability=.35, seed=SEED):
    """Train-only name/place dropout; original narratives are never modified."""
    random = np.random.default_rng(seed)
    texts = []
    for row in frame.itertuples():
        text = row.narrative
        if random.random() < probability:
            for span in sorted(json.loads(row.entities_json), key=lambda e: e["start"], reverse=True):
                if span["label"] in {"SUSPECT", "VICTIM", "LOCATION"}:
                    replacement = "somewhere" if span["label"] == "LOCATION" else "someone"
                    text = text[:span["start"]] + replacement + text[span["end"]:]
        texts.append(text)
    return texts


def tune_documents(train, validation, target, baseline):
    from .train import document_vectors, stable_hash
    x, y = train.narrative.tolist(), train.crime_type.tolist()
    vx, vy = validation.narrative.tolist(), validation.crime_type.tolist()
    results = {"selection_split": "validation", "objective": "macro F1; ties retain incumbent",
               "train_rows": len(train), "validation_rows": len(validation), "models": {}}
    fitted, timing = {}, {}
    for name, filename in [("Bag of Words", "bow.joblib"), ("TF-IDF", "tfidf.joblib")]:
        started = time.perf_counter()
        best_model = joblib.load(baseline / filename)
        best = {"name": "retained baseline", **validation_score(vy, best_model.predict(vx))}
        baseline_result = dict(best)
        trials = [dict(best)]
        if name == "Bag of Words":
            specs = [(binary, norm, c) for binary, norm in [(False, False), (True, False), (False, True), (True, True)]
                     for c in [.1, 1., 10.]]
        else:
            specs = [(weight, c) for weight in [.3, .75, 1.5] for c in [.25, 4., 16.]]
        for spec in specs:
            if name == "Bag of Words":
                binary, norm, c = spec
                steps = [("vectorizer", CountVectorizer(max_features=18000, ngram_range=(1, 2), min_df=2, binary=binary))]
                if norm:
                    steps.append(("normalizer", Normalizer()))
                description = f"binary={binary}, L2={norm}, C={c}"
            else:
                weight, c = spec
                steps = [("vectorizer", robust_tfidf(char_weight=weight))]
                description = f"character weight={weight}, C={c}"
            steps.append(("classifier", LogisticRegression(C=c, max_iter=700, random_state=SEED)))
            candidate = Pipeline(steps).fit(x, y)
            score = {"name": description, **validation_score(vy, candidate.predict(vx))}
            trials.append(score)
            if choose_candidate(best, score):
                best_model, best = candidate, score
        fitted[name] = best_model
        joblib.dump(best_model, target / filename)
        timing[name] = time.perf_counter() - started
        results["models"][name] = {"baseline": baseline_result, "selected": best, "trials": trials, "search_seconds": timing[name]}
        print(f"Selected {name}: {best['name']}; validation F1={best['macro_f1']:.4f}", flush=True)
        write_json(target / "validation_selection.json", results)

    started = time.perf_counter()
    old_w2v = Word2Vec.load(str(baseline / "word2vec.model"))
    old_classifier = joblib.load(baseline / "word2vec_classifier.joblib")
    recipe_path = baseline / "word2vec_features.json"
    old_recipe = json.loads(recipe_path.read_text()) if recipe_path.exists() else {"pooling": "mean"}
    best = {"name": "retained baseline", **validation_score(vy, old_classifier.predict(document_vectors(old_w2v, vx, old_recipe)))}
    baseline_result, trials = dict(best), [dict(best)]
    best_w2v, best_classifier, best_recipe = old_w2v, old_classifier, old_recipe
    tokenized = [word_tokens(text) for text in x]
    frequency = Counter(word for words in tokenized for word in set(words))
    idf = {word: float(np.log((1 + len(x)) / (1 + count)) + 1) for word, count in frequency.items()}
    for size, window, sg in [(100, 3, 1), (200, 5, 1), (100, 5, 0)]:
        embedding = Word2Vec(tokenized, vector_size=size, window=window, min_count=2, workers=1, sg=sg,
                             seed=SEED, epochs=20, hashfxn=stable_hash)
        for pooling in ["mean", "idf", "idf_max"]:
            recipe = {"pooling": pooling, "idf": idf if pooling != "mean" else {}}
            features, val_features = document_vectors(embedding, x, recipe), document_vectors(embedding, vx, recipe)
            for c in [.3, 3., 30.]:
                classifier = Pipeline([("normalizer", Normalizer()),
                    ("classifier", LogisticRegression(C=c, max_iter=700, random_state=SEED))]).fit(features, y)
                score = {"name": f"size={size}, window={window}, sg={sg}, pooling={pooling}, C={c}",
                         **validation_score(vy, classifier.predict(val_features))}
                trials.append(score)
                if choose_candidate(best, score):
                    best_w2v, best_classifier, best_recipe, best = embedding, classifier, recipe, score
        print(f"Word2Vec candidate size={size}, sg={sg}; best validation F1={best['macro_f1']:.4f}", flush=True)
    best_w2v.save(str(target / "word2vec.model"))
    joblib.dump(best_classifier, target / "word2vec_classifier.joblib")
    write_json(target / "word2vec_features.json", best_recipe)
    timing["Word2Vec"] = time.perf_counter() - started
    results["models"]["Word2Vec"] = {"baseline": baseline_result, "selected": best, "trials": trials, "search_seconds": timing["Word2Vec"]}
    write_json(target / "validation_selection.json", results)
    return fitted, best_w2v, best_classifier, best_recipe, timing, results
