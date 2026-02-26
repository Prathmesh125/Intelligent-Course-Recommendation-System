"""
recommender.py
--------------
Phase 5: Recommendation Engine  (v2 — Engagement-Boosted)
Computes cosine similarity between user query and course corpus,
then blends in a small engagement-boost signal learned from
collective user behaviour (clicks + saves via behavior_tracker).
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from vectorizer import load_tfidf_model, transform_query
from behavior_tracker import get_engagement_boost


# ── Load model once at module level (cached) ──────────────────────────────────
_vectorizer   = None
_tfidf_matrix = None
_courses_df   = None


def _ensure_model():
    global _vectorizer, _tfidf_matrix, _courses_df
    if _vectorizer is None:
        _vectorizer, _tfidf_matrix, _courses_df = load_tfidf_model()


# ── Core recommender ──────────────────────────────────────────────────────────
def recommend(
    user_query: str,
    top_n: int = 5,
    difficulty_filter: str = "All",
    min_rating: float = 0.0,
    source_filter: str = "All",
) -> pd.DataFrame:
    """
    Main recommendation function.

    Parameters
    ----------
    user_query       : Natural-language string from user
    top_n            : Number of results to return
    difficulty_filter: 'All' | 'Beginner' | 'Intermediate' | 'Advanced'
    min_rating       : Minimum course rating (0.0 – 5.0)
    source_filter    : 'All' | 'Coursera' | 'edX' | 'freeCodeCamp' | 'Khan Academy'
    """
    _ensure_model()

    if not user_query.strip():
        return pd.DataFrame()

    # 1. Vectorize user query
    query_vec = transform_query(user_query, _vectorizer)

    # 2. Cosine similarity against all courses
    scores = cosine_similarity(query_vec, _tfidf_matrix).flatten()

    # 3. Attach scores to dataframe
    results = _courses_df.copy()
    results["similarity_score"] = scores

    # 3b. Blend engagement boost (learned from clicks + saves)
    results["similarity_score"] = results.apply(
        lambda row: row["similarity_score"] + get_engagement_boost(row["course_title"]),
        axis=1,
    )

    # 4. Apply difficulty filter
    if difficulty_filter != "All":
        results = results[results["difficulty"].str.lower() == difficulty_filter.lower()]

    # 5. Apply source filter
    if source_filter != "All" and "source" in results.columns:
        results = results[results["source"] == source_filter]

    # 6. Apply rating filter
    if min_rating > 0:
        results = results[results["rating"] >= min_rating]

    # 7. Sort by similarity (desc), then rating (desc) as tie-breaker
    results = results.sort_values(
        by=["similarity_score", "rating"],
        ascending=[False, False],
    )

    # 8. Take top-N
    results = results.head(top_n).reset_index(drop=True)
    results["rank"] = results.index + 1

    cols = ["rank", "course_title", "difficulty", "rating",
            "similarity_score", "description", "skills", "url"]
    if "source" in results.columns:
        cols.append("source")
    return results[cols]


# ── Baseline: keyword search (for comparison in evaluation) ───────────────────
def keyword_search(
    user_query: str,
    top_n: int = 5,
    difficulty_filter: str = "All",
) -> pd.DataFrame:
    """
    Naive keyword matching baseline.
    Counts exact word overlaps between query tokens and course text.
    Used to demonstrate improvement of NLP semantic model over baseline.
    """
    _ensure_model()

    query_tokens = set(user_query.lower().split())
    df = _courses_df.copy()

    def _count_matches(row):
        text = " ".join([
            str(row.get("course_title", "")),
            str(row.get("description",  "")),
            str(row.get("skills",       "")),
        ]).lower()
        return sum(1 for t in query_tokens if t in text)

    df["similarity_score"] = df.apply(_count_matches, axis=1)

    if difficulty_filter != "All":
        df = df[df["difficulty"].str.lower() == difficulty_filter.lower()]

    df = df.sort_values("similarity_score", ascending=False)
    df = df.head(top_n).reset_index(drop=True)
    df["rank"] = df.index + 1

    cols = ["rank", "course_title", "difficulty", "rating",
            "similarity_score", "description", "skills", "url"]
    if "source" in df.columns:
        cols.append("source")
    return df[cols]


# ── Invalidate in-memory cache (call after scraping new data) ─────────────────
def invalidate_cache():
    global _vectorizer, _tfidf_matrix, _courses_df
    _vectorizer   = None
    _tfidf_matrix = None
    _courses_df   = None


# ── Get available difficulties ─────────────────────────────────────────────────
def get_difficulties():
    _ensure_model()
    return ["All"] + sorted(_courses_df["difficulty"].unique().tolist())


# ── Get available sources ──────────────────────────────────────────────────────
def get_sources():
    _ensure_model()
    if "source" in _courses_df.columns:
        return ["All"] + sorted(_courses_df["source"].unique().tolist())
    return ["All"]


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        "I want to learn AI but I am weak at math and I am a beginner",
        "data science with python for beginners",
        "advanced deep learning for computer vision",
        "web development from scratch no coding experience",
        "machine learning with statistics",
    ]

    print("=" * 70)
    print("NLPRec — Recommender Engine Test")
    print("=" * 70)

    for q in queries:
        print(f"\nQuery   : {q}")
        print("-" * 70)
        recs = recommend(q, top_n=3)
        for _, row in recs.iterrows():
            score = f"{row['similarity_score']:.4f}"
            print(f"  {row['rank']}. [{row['difficulty']:12s}] "
                  f"{row['course_title']}  (sim={score}, ★{row['rating']})")
