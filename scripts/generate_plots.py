"""Generate standalone model plots from evaluation artifacts into plots/ folder."""
import json
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import ARTIFACTS, ROOT

PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

# ── Shared style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#fafbfc",
    "axes.facecolor": "#fafbfc",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
})

PALETTE = ["#087F75", "#3E6DA8", "#DA9C42", "#9167AD", "#DC7B65", "#60AEB0", "#768BC1", "#9DAD71"]
MODEL_COLORS = {"Bag of Words": PALETTE[0], "TF-IDF": PALETTE[1], "Word2Vec": PALETTE[2], "BERT": PALETTE[3]}

evaluation = json.loads((ARTIFACTS / "evaluation.json").read_text(encoding="utf-8"))
ner_evaluation = json.loads((ARTIFACTS / "ner_evaluation.json").read_text(encoding="utf-8"))
topics = json.loads((ARTIFACTS / "topics.json").read_text(encoding="utf-8"))

LABELS = evaluation["models"]["TF-IDF"]["labels"]
count = 0


def save(fig, name):
    global count
    fig.savefig(PLOTS / name)
    plt.close(fig)
    count += 1
    print(f"  [SAVED] {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MODEL COMPARISON – grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating model comparison plots...")

fig, ax = plt.subplots(figsize=(10, 5))
models = list(evaluation["models"].keys())
metrics = ["accuracy", "macro_f1", "weighted_f1"]
metric_labels = ["Accuracy", "Macro F1", "Weighted F1"]
x = np.arange(len(models))
width = 0.22
for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    values = [evaluation["models"][m][metric] for m in models]
    bars = ax.bar(x + i * width, values, width, label=label, color=PALETTE[i], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="600")
ax.set_xticks(x + width)
ax.set_xticklabels(models, fontweight="600")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.08)
ax.set_title("Document Classification — Model Comparison")
ax.legend(frameon=False)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
save(fig, "01_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONFUSION MATRICES – one per model
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating confusion matrices...")

for model_name in models:
    m = evaluation["models"][model_name]
    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color=color, fontweight="600")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    safe = model_name.lower().replace(" ", "_")
    save(fig, f"02_confusion_matrix_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PER-CLASS F1 – grouped bars for all models
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating per-class F1 comparison...")

fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(LABELS))
width = 0.18
for i, model_name in enumerate(models):
    report = evaluation["models"][model_name]["classification_report"]
    f1_scores = [report[label]["f1-score"] for label in LABELS]
    bars = ax.bar(x + i * width, f1_scores, width, label=model_name,
                  color=MODEL_COLORS[model_name], edgecolor="white", linewidth=0.5)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(LABELS, rotation=30, ha="right", fontweight="600")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.08)
ax.set_title("Per-Class F1 Score — All Models")
ax.legend(frameon=False, loc="lower right")
save(fig, "03_per_class_f1.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PRECISION vs RECALL – per model scatter
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating precision vs recall plots...")

for model_name in models:
    report = evaluation["models"][model_name]["classification_report"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for j, label in enumerate(LABELS):
        p = report[label]["precision"]
        r = report[label]["recall"]
        ax.scatter(r, p, s=report[label]["support"] * 0.7 + 40, color=PALETTE[j % len(PALETTE)],
                   edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(label, (r, p), textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.plot([0, 1], [0, 1], "--", color="#bbb", linewidth=1, zorder=1)
    ax.set_xlim(0.55, 1.02)
    ax.set_ylim(0.55, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision vs Recall — {model_name}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    safe = model_name.lower().replace(" ", "_")
    save(fig, f"04_precision_recall_{safe}.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ACCURACY CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating accuracy confidence interval plot...")

fig, ax = plt.subplots(figsize=(8, 4.5))
for i, model_name in enumerate(models):
    m = evaluation["models"][model_name]
    lo, hi = m["accuracy_ci95"]
    acc = m["accuracy"]
    ax.errorbar(acc, i, xerr=[[acc - lo], [hi - acc]], fmt="o", color=MODEL_COLORS[model_name],
                capsize=6, capthick=2, markersize=9, linewidth=2, zorder=3)
    ax.text(hi + 0.005, i, f"{acc:.3f} [{lo:.3f}, {hi:.3f}]", va="center", fontsize=9)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontweight="600")
ax.set_xlabel("Accuracy")
ax.set_title("95% Bootstrap Confidence Intervals — Accuracy")
ax.grid(True, axis="x", alpha=0.2)
ax.set_xlim(0.7, 0.95)
save(fig, "05_accuracy_confidence_intervals.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CALIBRATION PLOT (before / after temperature scaling)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating calibration plot...")

fig, ax = plt.subplots(figsize=(6.5, 6))
for phase, color, marker in [("before", PALETTE[4], "s"), ("after", PALETTE[0], "o")]:
    bins = evaluation["calibration_test"][phase]["bins"]
    confs = [b["mean_confidence"] for b in bins]
    accs = [b["accuracy"] for b in bins]
    ax.plot(confs, accs, f"-{marker}", color=color, label=f"{phase.capitalize()} (ECE={evaluation['calibration_test'][phase]['ece']:.3f})",
            markersize=7, linewidth=2)
ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1, label="Perfect calibration")
ax.set_xlabel("Mean Predicted Confidence")
ax.set_ylabel("Observed Accuracy")
ax.set_title("TF-IDF Calibration — Before & After Temperature Scaling")
ax.legend(frameon=False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.grid(True, alpha=0.15)
save(fig, "06_calibration_curve.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BERT TRAINING HISTORY (classification)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating BERT training history...")

history = evaluation["models"]["BERT"]["history"]
epochs = [h["epoch"] for h in history]

fig, ax1 = plt.subplots(figsize=(8, 4.5))
color_loss = PALETTE[4]
color_f1 = PALETTE[0]
ax1.plot(epochs, [h["loss"] for h in history], "-o", color=color_loss, linewidth=2, markersize=7, label="Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color=color_loss)
ax1.tick_params(axis="y", labelcolor=color_loss)
ax1.set_xticks(epochs)

ax2 = ax1.twinx()
ax2.plot(epochs, [h["validation_macro_f1"] for h in history], "-s", color=color_f1, linewidth=2, markersize=7, label="Validation Macro F1")
ax2.set_ylabel("Validation Macro F1", color=color_f1)
ax2.tick_params(axis="y", labelcolor=color_f1)
ax2.spines["top"].set_visible(False)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right")
ax1.set_title("BERT Classifier — Training History")
save(fig, "07_bert_training_history.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. NER ENTITY F1 – per entity type (CRF vs BERT)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating NER entity F1 comparison...")

entity_labels = ner_evaluation["labels"]
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(entity_labels))
width = 0.32
for i, (ner_name, color) in enumerate([("CRF", PALETTE[0]), ("BERT", PALETTE[3])]):
    report = ner_evaluation["models"][ner_name]["report"]
    f1s = [report[label]["f1-score"] for label in entity_labels]
    bars = ax.bar(x + i * width, f1s, width, label=f"{ner_name} (overall F1={ner_evaluation['models'][ner_name]['strict_entity_f1']:.3f})",
                  color=color, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, fontweight="600")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(entity_labels, rotation=30, ha="right", fontweight="600")
ax.set_ylabel("F1 Score (Strict Span)")
ax.set_ylim(0, 1.12)
ax.set_title("Named Entity Recognition — Per-Entity F1 (CRF vs BERT)")
ax.legend(frameon=False)
save(fig, "08_ner_entity_f1.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. NER PRECISION vs RECALL scatter (CRF)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating NER precision-recall scatter...")

fig, ax = plt.subplots(figsize=(7, 6))
crf_report = ner_evaluation["models"]["CRF"]["report"]
for j, label in enumerate(entity_labels):
    p = crf_report[label]["precision"]
    r = crf_report[label]["recall"]
    ax.scatter(r, p, s=crf_report[label]["support"] * 0.15 + 50, color=PALETTE[j % len(PALETTE)],
               edgecolor="white", linewidth=1, zorder=3)
    ax.annotate(label, (r, p), textcoords="offset points", xytext=(8, 5), fontsize=9)
ax.plot([0, 1], [0, 1], "--", color="#bbb", linewidth=1, zorder=1)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("CRF Entity Extraction — Precision vs Recall")
ax.set_xlim(0.75, 1.02)
ax.set_ylim(0.82, 1.02)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)
save(fig, "09_ner_precision_recall_crf.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. NER BERT TRAINING HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
if ner_evaluation.get("bert_history"):
    print("Generating NER BERT training history...")
    ner_history = ner_evaluation["bert_history"]
    epochs_ner = [h["epoch"] for h in ner_history]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs_ner, [h["loss"] for h in ner_history], "-o", color=PALETTE[4], linewidth=2, markersize=7, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=PALETTE[4])
    ax1.tick_params(axis="y", labelcolor=PALETTE[4])
    ax1.set_xticks(epochs_ner)

    ax2 = ax1.twinx()
    ax2.plot(epochs_ner, [h["validation_entity_f1"] for h in ner_history], "-s", color=PALETTE[0], linewidth=2, markersize=7, label="Validation Entity F1")
    ax2.set_ylabel("Validation Entity F1", color=PALETTE[0])
    ax2.tick_params(axis="y", labelcolor=PALETTE[0])
    ax2.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right")
    ax1.set_title("BERT NER Tagger — Training History")
    save(fig, "10_ner_bert_training_history.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. TRAINING TIME COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating training time comparison...")

fig, ax = plt.subplots(figsize=(8, 4))
times = [evaluation["models"][m]["train_seconds"] for m in models]
bars = ax.barh(models, times, color=[MODEL_COLORS[m] for m in models], edgecolor="white", height=0.55)
for bar, t in zip(bars, times):
    ax.text(bar.get_width() + max(times) * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{t:.1f}s", va="center", fontsize=10, fontweight="600")
ax.set_xlabel("Training Time (seconds)")
ax.set_title("Model Training Time Comparison")
ax.invert_yaxis()
save(fig, "11_training_time.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. TF-IDF PER-CLASS PRECISION / RECALL / F1 grouped bar
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating TF-IDF detailed per-class metrics...")

fig, ax = plt.subplots(figsize=(11, 5))
report = evaluation["models"]["TF-IDF"]["classification_report"]
x = np.arange(len(LABELS))
width = 0.25
for i, (metric, label, color) in enumerate([("precision", "Precision", PALETTE[1]),
                                              ("recall", "Recall", PALETTE[2]),
                                              ("f1-score", "F1 Score", PALETTE[0])]):
    values = [report[lbl][metric] for lbl in LABELS]
    bars = ax.bar(x + i * width, values, width, label=label, color=color, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, fontweight="600")
ax.set_xticks(x + width)
ax.set_xticklabels(LABELS, rotation=30, ha="right", fontweight="600")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.12)
ax.set_title("TF-IDF + Logistic Regression — Per-Class Precision, Recall & F1")
ax.legend(frameon=False)
save(fig, "12_tfidf_per_class_detail.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. BAG OF WORDS PER-CLASS DETAIL
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Bag of Words detailed per-class metrics...")

fig, ax = plt.subplots(figsize=(11, 5))
report = evaluation["models"]["Bag of Words"]["classification_report"]
x = np.arange(len(LABELS))
for i, (metric, label, color) in enumerate([("precision", "Precision", PALETTE[1]),
                                              ("recall", "Recall", PALETTE[2]),
                                              ("f1-score", "F1 Score", PALETTE[0])]):
    values = [report[lbl][metric] for lbl in LABELS]
    bars = ax.bar(x + i * width, values, width, label=label, color=color, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, fontweight="600")
ax.set_xticks(x + width)
ax.set_xticklabels(LABELS, rotation=30, ha="right", fontweight="600")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.12)
ax.set_title("Bag of Words + Logistic Regression — Per-Class Precision, Recall & F1")
ax.legend(frameon=False)
save(fig, "13_bow_per_class_detail.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 14. WORD2VEC PER-CLASS DETAIL
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Word2Vec detailed per-class metrics...")

fig, ax = plt.subplots(figsize=(11, 5))
report = evaluation["models"]["Word2Vec"]["classification_report"]
x = np.arange(len(LABELS))
for i, (metric, label, color) in enumerate([("precision", "Precision", PALETTE[1]),
                                              ("recall", "Recall", PALETTE[2]),
                                              ("f1-score", "F1 Score", PALETTE[0])]):
    values = [report[lbl][metric] for lbl in LABELS]
    bars = ax.bar(x + i * width, values, width, label=label, color=color, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, fontweight="600")
ax.set_xticks(x + width)
ax.set_xticklabels(LABELS, rotation=30, ha="right", fontweight="600")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.12)
ax.set_title("Word2Vec + Logistic Regression — Per-Class Precision, Recall & F1")
ax.legend(frameon=False)
save(fig, "14_word2vec_per_class_detail.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 15. BERT PER-CLASS DETAIL
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating BERT detailed per-class metrics...")

fig, ax = plt.subplots(figsize=(11, 5))
report = evaluation["models"]["BERT"]["classification_report"]
x = np.arange(len(LABELS))
for i, (metric, label, color) in enumerate([("precision", "Precision", PALETTE[1]),
                                              ("recall", "Recall", PALETTE[2]),
                                              ("f1-score", "F1 Score", PALETTE[0])]):
    values = [report[lbl][metric] for lbl in LABELS]
    bars = ax.bar(x + i * width, values, width, label=label, color=color, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, fontweight="600")
ax.set_xticks(x + width)
ax.set_xticklabels(LABELS, rotation=30, ha="right", fontweight="600")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.12)
ax.set_title("BERT — Per-Class Precision, Recall & F1")
ax.legend(frameon=False)
save(fig, "15_bert_per_class_detail.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 16. LDA TOPIC COHERENCE
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating LDA topic coherence plot...")

fig, ax = plt.subplots(figsize=(8, 4.5))
topic_ids = list(range(1, len(topics["umass_coherence"]) + 1))
bars = ax.bar(topic_ids, topics["umass_coherence"], color=PALETTE[:len(topic_ids)], edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, topics["umass_coherence"]):
    ax.text(bar.get_x() + bar.get_width() / 2, min(val, 0) - 0.15, f"{val:.2f}",
            ha="center", va="top", fontsize=8, fontweight="600")
ax.set_xlabel("Topic")
ax.set_ylabel("UMass Coherence")
ax.set_title(f"LDA Topic Coherence (Perplexity: {topics['held_out_perplexity']:.1f}, Diversity: {topics['topic_diversity']:.0%})")
ax.set_xticks(topic_ids)
ax.axhline(0, color="#ccc", linewidth=0.8)
save(fig, "16_lda_topic_coherence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 17. LDA TOP TERMS HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating LDA top terms heatmap...")

fig, ax = plt.subplots(figsize=(12, 6))
term_matrix = []
ylabels = []
for topic in topics["topics"]:
    term_matrix.append(topic["weights"][:8])
    ylabels.append(f"Topic {topic['id'] + 1}")
term_matrix = np.array(term_matrix)
norm = term_matrix / term_matrix.max(axis=1, keepdims=True)

im = ax.imshow(norm, cmap="YlGnBu", aspect="auto")
ax.set_yticks(range(len(ylabels)))
ax.set_yticklabels(ylabels, fontsize=9)
# Use terms from first topic for x-axis as position labels
ax.set_xticks(range(8))
ax.set_xticklabels([f"Term {i+1}" for i in range(8)], fontsize=8)
# Overlay actual term text
for i, topic in enumerate(topics["topics"]):
    for j, term in enumerate(topic["terms"][:8]):
        ax.text(j, i, term, ha="center", va="center", fontsize=7, fontweight="bold",
                color="white" if norm[i, j] > 0.6 else "black")
ax.set_title("LDA — Top Terms per Topic (normalized weight)")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04, label="Relative Weight")
save(fig, "17_lda_top_terms_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 18. GROUPED CV (TF-IDF cross-validation folds)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating cross-validation fold plot...")

cv = evaluation["grouped_cv"]
fig, ax = plt.subplots(figsize=(6, 4))
fold_scores = cv["fold_macro_f1"]
bars = ax.bar(range(1, len(fold_scores) + 1), fold_scores, color=PALETTE[1], edgecolor="white", width=0.5)
ax.axhline(cv["mean"], color=PALETTE[4], linewidth=2, linestyle="--", label=f"Mean: {cv['mean']:.4f}")
for bar, val in zip(bars, fold_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="600")
ax.set_xlabel("Fold")
ax.set_ylabel("Macro F1")
ax.set_ylim(0.8, max(fold_scores) + 0.03)
ax.set_title("TF-IDF — Grouped 3-Fold Cross-Validation")
ax.set_xticks(range(1, len(fold_scores) + 1))
ax.legend(frameon=False)
save(fig, "18_tfidf_cross_validation.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 19. CLASS DISTRIBUTION (support per class from test set)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating class distribution plot...")

fig, ax = plt.subplots(figsize=(8, 4.5))
report = evaluation["models"]["TF-IDF"]["classification_report"]
supports = [report[label]["support"] for label in LABELS]
bars = ax.bar(LABELS, supports, color=PALETTE[:len(LABELS)], edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, supports):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(int(val)),
            ha="center", va="bottom", fontsize=9, fontweight="600")
ax.set_ylabel("Number of Test Samples")
ax.set_title("Test Set — Class Distribution")
plt.xticks(rotation=30, ha="right")
save(fig, "19_class_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 20. SUMMARY RADAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating model radar chart...")

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
categories_radar = ["Accuracy", "Macro F1", "Weighted F1"]
N = len(categories_radar)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for model_name in models:
    m = evaluation["models"][model_name]
    values = [m["accuracy"], m["macro_f1"], m["weighted_f1"]]
    values += values[:1]
    ax.plot(angles, values, "-o", color=MODEL_COLORS[model_name], linewidth=2, markersize=6, label=model_name)
    ax.fill(angles, values, color=MODEL_COLORS[model_name], alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=11, fontweight="600")
ax.set_ylim(0.7, 0.95)
ax.set_title("Model Comparison Radar", fontsize=14, fontweight="bold", pad=25)
ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0), frameon=False)
save(fig, "20_model_radar.png")


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Done! {count} plots saved to: {PLOTS}")
print(f"{'='*60}")
