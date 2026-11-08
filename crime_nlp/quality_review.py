"""Compute quality metrics only from explicitly submitted reviewed judgments."""
import math


def evaluate(judgments):
    if not judgments or any(row.get("review_status") != "reviewed" or not row.get("reviewer") for row in judgments):
        raise ValueError("Quality evaluation requires reviewed judgments and reviewer identifiers")
    retrieval, summaries, topics = [], [], []
    for row in judgments:
        if row["task"] == "retrieval":
            ranked = row["ranked_ids"]
            if len(set(ranked)) != len(ranked):
                raise ValueError("Ranked report IDs must be unique")
            relevant = set(row["relevant_ids"])
            k = row.get("k", min(10, len(ranked)))
            if k < 1 or not relevant:
                raise ValueError("Retrieval judgments need positive k and at least one relevant report")
            hits = [int(identifier in relevant) for identifier in ranked[:k]]
            dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
            ideal = sum(1 / math.log2(index + 2) for index in range(min(len(relevant), k)))
            retrieval.append({"precision_at_k": sum(hits) / k, "recall_within_judged_set_at_k": sum(hits) / len(relevant),
                              "ndcg_at_k": dcg / ideal, "reciprocal_rank": next((1 / (index + 1) for index, hit in enumerate(hits) if hit), 0.)})
        elif row["task"] == "summary":
            keys = ["faithful_to_source", "preserves_participants", "preserves_negation", "covers_main_event"]
            if any(type(row.get(key)) is not bool for key in keys):
                raise ValueError("Summary quality fields must be explicit boolean judgments")
            summaries.append({key: int(row[key]) for key in keys})
        elif row["task"] == "topic":
            if type(row.get("coherent")) is not bool:
                raise ValueError("Topic coherence requires a boolean judgment")
            topics.append({"coherent": int(row["coherent"])})
        else:
            raise ValueError("Unsupported quality-review task")
    mean = lambda rows: {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]} if rows else None
    return {"retrieval": {"judgments": len(retrieval), "metrics": mean(retrieval)},
            "summaries": {"judgments": len(summaries), "metrics": mean(summaries)},
            "topics": {"judgments": len(topics), "metrics": mean(topics)},
            "scope": "Metrics from submitted reviewed judgments; tool does not certify reviewer independence or real-world representativeness"}
