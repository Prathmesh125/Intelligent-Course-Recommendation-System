"""
evaluation.py
-------------
Phase 7: Research Evaluation Module
Computes Precision@K, Recall@K, F1@K for the NLP model and the
keyword-search baseline, then generates comparison plots.

This is the core research contribution required for publication:
  → "Our NLP model improved Precision by X% over keyword baseline."
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from recommender import recommend, keyword_search

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR   = os.path.join(BASE_DIR, "assets")
RESULTS_PATH = os.path.join(BASE_DIR, "dataset", "eval_results.json")
os.makedirs(ASSETS_DIR, exist_ok=True)


# ── Ground-truth test set ─────────────────────────────────────────────────────
# Format: { "user_query": ["relevant course title 1", ...], ... }
# Manually curated — this is how papers validate recommenders.
TEST_SET = {
    "learn python programming basics for beginners": [
        "Python for Everybody",
        "Introduction to Computer Science with Python",
        "Introduction to Programming with Java",
    ],
    "I want to enter AI but I am weak at math beginner": [
        "Introduction to Artificial Intelligence",
        "Python for Everybody",
        "Ethics in Artificial Intelligence",
        "Prompt Engineering for AI Applications",
    ],
    "data science with python statistics machine learning": [
        "Data Science with Python",
        "Machine Learning by Andrew Ng",
        "Statistics for Data Science",
        "Applied Machine Learning",
        "Feature Engineering for Machine Learning",
    ],
    "advanced deep learning neural networks convolutional": [
        "Deep Learning Specialization",
        "Introduction to Neural Networks",
        "Computer Vision with OpenCV",
    ],
    "web development HTML CSS JavaScript front end": [
        "Web Development Bootcamp",
        "React.js Front-End Development",
    ],
    "database SQL queries data analysis": [
        "SQL for Data Analysis",
        "Data Visualization with Tableau",
        "Excel for Data Analysis",
    ],
    "natural language processing text classification sentiment": [
        "Natural Language Processing with Python",
        "Generative AI with Large Language Models",
        "Speech Recognition and Audio Processing",
    ],
    "cloud computing aws deployment devops docker": [
        "Cloud Computing on AWS",
        "Docker and Kubernetes",
        "FastAPI and Python Backend Development",
    ],
    "mathematics for machine learning linear algebra calculus": [
        "Linear Algebra for Machine Learning",
        "Calculus for Machine Learning",
        "Mathematics for Computer Science",
        "Probability and Statistics",
    ],
    "recommendation systems collaborative filtering": [
        "Recommender Systems",
        "Applied Machine Learning",
        "Natural Language Processing with Python",
    ],
}


# ── Metric functions ──────────────────────────────────────────────────────────
def precision_at_k(predicted: list, relevant: list, k: int) -> float:
    """Fraction of top-K predicted that are relevant."""
    if k == 0:
        return 0.0
    top_k = predicted[:k]
    hits  = sum(1 for p in top_k if p in relevant)
    return hits / k


def recall_at_k(predicted: list, relevant: list, k: int) -> float:
    """Fraction of relevant items found in top-K predicted."""
    if not relevant:
        return 0.0
    top_k = predicted[:k]
    hits  = sum(1 for p in top_k if p in relevant)
    return hits / len(relevant)


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Evaluate a single model over all test queries ─────────────────────────────
def evaluate_model(recommend_fn, k: int = 5, label: str = "Model") -> dict:
    """
    Runs recommend_fn on every test query and aggregates metrics.
    Returns dict of mean Precision@K, Recall@K, F1@K.
    """
    precisions, recalls, f1s = [], [], []

    per_query = []
    for query, relevant in TEST_SET.items():
        recs = recommend_fn(query, top_n=k)
        predicted = recs["course_title"].tolist() if not recs.empty else []

        p = precision_at_k(predicted, relevant, k)
        r = recall_at_k(predicted, relevant, k)
        f = f1_at_k(p, r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        per_query.append({
            "query":     query,
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f, 4),
            "predicted": predicted,
            "relevant":  relevant,
        })

    return {
        "label":         label,
        "k":             k,
        "precision_mean": round(np.mean(precisions), 4),
        "recall_mean":    round(np.mean(recalls),    4),
        "f1_mean":        round(np.mean(f1s),        4),
        "per_query":     per_query,
    }


# ── Run full comparison ────────────────────────────────────────────────────────
def run_evaluation(k: int = 5, save: bool = True) -> tuple:
    """
    Compares NLP semantic model vs keyword baseline.
    Returns (nlp_results, baseline_results).
    """
    print(f"\n{'='*60}")
    print(f"NLPRec — Evaluation  (K={k})")
    print(f"{'='*60}")

    print("\n[1/2] Evaluating NLP Semantic Model …")
    nlp_res  = evaluate_model(
        lambda q, top_n: recommend(q, top_n=top_n),
        k=k,
        label="NLP Semantic (TF-IDF + Cosine)"
    )

    print("[2/2] Evaluating Keyword Baseline …")
    base_res = evaluate_model(
        lambda q, top_n: keyword_search(q, top_n=top_n),
        k=k,
        label="Baseline (Keyword Matching)"
    )

    # Print summary table
    print(f"\n{'Model':<35} {'Precision@'+str(k):<16} {'Recall@'+str(k):<14} {'F1@'+str(k)}")
    print("-" * 75)
    for res in [nlp_res, base_res]:
        print(f"{res['label']:<35} {res['precision_mean']:<16.4f} "
              f"{res['recall_mean']:<14.4f} {res['f1_mean']:.4f}")

    # Improvement percentages
    for metric in ["precision_mean", "recall_mean", "f1_mean"]:
        nlp_val  = nlp_res[metric]
        base_val = base_res[metric]
        if base_val > 0:
            delta = (nlp_val - base_val) / base_val * 100
            name  = metric.replace("_mean", "").capitalize()
            print(f"\n  ✔ NLP model improved {name} by {delta:+.1f}% over baseline")
        else:
            print(f"\n  ✔ NLP model {metric}: {nlp_val:.4f} (baseline=0)")

    if save:
        payload = {"nlp": nlp_res, "baseline": base_res}
        with open(RESULTS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[Saved] {RESULTS_PATH}")

    return nlp_res, base_res


# ── Plot functions ────────────────────────────────────────────────────────────
def plot_comparison(nlp_res: dict, base_res: dict, save: bool = True):
    """Side-by-side bar chart: NLP model vs Keyword baseline."""
    sns.set_theme(style="whitegrid", palette="muted")

    metrics = ["Precision", "Recall", "F1 Score"]
    nlp_vals  = [nlp_res["precision_mean"],  nlp_res["recall_mean"],  nlp_res["f1_mean"]]
    base_vals = [base_res["precision_mean"], base_res["recall_mean"], base_res["f1_mean"]]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width/2, nlp_vals,  width, label="NLP Semantic Model",   color="#4C72B0", zorder=3)
    bars2 = ax.bar(x + width/2, base_vals, width, label="Keyword Baseline",      color="#DD8452", zorder=3)

    ax.set_xlabel("Metric", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title(f"NLPRec: Model Comparison @ K={nlp_res['k']}", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, zorder=0)

    # Value labels on bars
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, "comparison_chart.png")
    if save:
        plt.savefig(path, dpi=150)
        print(f"[Saved] {path}")
    return fig


def plot_per_query_heatmap(nlp_res: dict, save: bool = True):
    """Heatmap of per-query Precision / Recall / F1 for the NLP model."""
    pq = nlp_res["per_query"]
    short_queries = [q["query"][:45] + "…" if len(q["query"]) > 45 else q["query"]
                     for q in pq]
    data = pd.DataFrame({
        "Precision": [q["precision"] for q in pq],
        "Recall":    [q["recall"]    for q in pq],
        "F1":        [q["f1"]        for q in pq],
    }, index=short_queries)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title("NLP Model — Per-Query Metrics Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Query")
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=8, rotation=0)
    plt.tight_layout()

    path = os.path.join(ASSETS_DIR, "per_query_heatmap.png")
    if save:
        plt.savefig(path, dpi=150)
        print(f"[Saved] {path}")
    return fig


def plot_metric_radar(nlp_res: dict, base_res: dict, save: bool = True):
    """Radar / spider chart comparing the two models across 3 metrics."""
    from matplotlib.patches import FancyArrowPatch

    labels   = np.array(["Precision", "Recall", "F1 Score"])
    stats_nlp  = np.array([nlp_res["precision_mean"],  nlp_res["recall_mean"],  nlp_res["f1_mean"]])
    stats_base = np.array([base_res["precision_mean"], base_res["recall_mean"], base_res["f1_mean"]])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    stats_nlp_plot  = np.concatenate([stats_nlp,  stats_nlp[:1]])
    stats_base_plot = np.concatenate([stats_base, stats_base[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats_nlp_plot,  "o-", linewidth=2, color="#4C72B0",  label="NLP Model")
    ax.fill(angles, stats_nlp_plot,  alpha=0.25, color="#4C72B0")
    ax.plot(angles, stats_base_plot, "s--", linewidth=2, color="#DD8452", label="Keyword Baseline")
    ax.fill(angles, stats_base_plot, alpha=0.15, color="#DD8452")

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Radar: NLP vs Baseline", y=1.08, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.2), fontsize=10)
    plt.tight_layout()

    path = os.path.join(ASSETS_DIR, "radar_chart.png")
    if save:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {path}")
    return fig


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    nlp_res, base_res = run_evaluation(k=5)
    plot_comparison(nlp_res, base_res)
    plot_per_query_heatmap(nlp_res)
    plot_metric_radar(nlp_res, base_res)
    print("\nAll evaluation plots saved to assets/ folder.")
