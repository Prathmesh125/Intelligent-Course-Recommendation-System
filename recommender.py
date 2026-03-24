"""
recommender.py
--------------
Phase 5: Recommendation Engine  (v2 — Engagement-Boosted)
Computes cosine similarity between user query and course corpus,
then blends in a small engagement-boost signal learned from
collective user behaviour (clicks + saves via behavior_tracker).
"""

from typing import Optional
import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from vectorizer import load_tfidf_model, transform_query
from behavior_tracker import get_engagement_boost

log = logging.getLogger("NLPRec-Recommender")


# ── Load model once at module level (cached) ──────────────────────────────────
_vectorizer: Optional[object] = None
_tfidf_matrix: Optional[np.ndarray] = None
_courses_df: Optional[pd.DataFrame] = None


def _ensure_model() -> None:
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

    # Input validation
    if not user_query or not user_query.strip():
        log.warning("Empty query provided to recommend()")
        return pd.DataFrame()
    
    if len(user_query) > 1000:
        log.warning(f"Query too long ({len(user_query)} chars), truncating to 1000")
        user_query = user_query[:1000]
    
    # Validate top_n range
    if not isinstance(top_n, int) or top_n < 1:
        log.warning(f"Invalid top_n={top_n}, defaulting to 5")
        top_n = 5
    elif top_n > 100:
        log.warning(f"top_n={top_n} exceeds limit, capping to 100")
        top_n = 100
    
    # Validate rating range
    if not isinstance(min_rating, (int, float)) or not (0.0 <= min_rating <= 5.0):
        log.warning(f"Invalid min_rating={min_rating}, defaulting to 0.0")
        min_rating = 0.0
    
    # Validate difficulty filter
    valid_difficulties = ["All", "Beginner", "Intermediate", "Advanced"]
    if difficulty_filter not in valid_difficulties:
        log.warning(f"Invalid difficulty_filter={difficulty_filter}, defaulting to 'All'")
        difficulty_filter = "All"

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


# ── Cold-start helper: top-rated courses with no query required ───────────────
def get_top_rated_courses(
    top_n: int = 5,
    difficulty_filter: str = "All",
    source_filter: str = "All",
    min_rating: float = 4.0,
) -> pd.DataFrame:
    """Return highest-rated courses; useful for landing pages and cold-start."""
    _ensure_model()
    results = _courses_df.copy()

    if difficulty_filter != "All":
        results = results[results["difficulty"].str.lower() == difficulty_filter.lower()]

    if source_filter != "All" and "source" in results.columns:
        results = results[results["source"] == source_filter]

    if min_rating > 0:
        results = results[results["rating"] >= min_rating]

    results = results.sort_values("rating", ascending=False)
    results = results.head(top_n).reset_index(drop=True)
    results["rank"] = results.index + 1
    results["similarity_score"] = 0.0

    cols = ["rank", "course_title", "difficulty", "rating",
            "similarity_score", "description", "skills", "url"]
    if "source" in results.columns:
        cols.append("source")
    return results[cols]


# ── Content-based "more like this" lookup ────────────────────────────────────
def get_similar_courses(course_title: str, top_n: int = 5) -> pd.DataFrame:
    """Return courses most similar to the given course title using TF-IDF cosine similarity."""
    _ensure_model()
    if _courses_df is None or _tfidf_matrix is None:
        return pd.DataFrame()

    titles_lower = _courses_df["course_title"].str.lower()
    match = titles_lower[titles_lower == course_title.lower()]
    if match.empty:
        match = titles_lower[titles_lower.str.contains(course_title.lower(), regex=False)]
    if match.empty:
        log.warning(f"get_similar_courses: no match for '{course_title}'")
        return pd.DataFrame()

    idx = match.index[0]
    scores = cosine_similarity(_tfidf_matrix[idx], _tfidf_matrix).flatten()
    scores[idx] = -1  # exclude the course itself

    results = _courses_df.copy()
    results["similarity_score"] = scores
    results = results.sort_values("similarity_score", ascending=False).head(top_n).reset_index(drop=True)
    results["rank"] = results.index + 1

    cols = ["rank", "course_title", "difficulty", "rating", "similarity_score", "description", "skills", "url"]
    if "source" in results.columns:
        cols.append("source")
    return results[cols]


# ── Get available difficulties ─────────────────────────────────────────────────
def get_difficulties():
    try:
        _ensure_model()
        if _courses_df is not None and "difficulty" in _courses_df.columns:
            return ["All"] + sorted(_courses_df["difficulty"].unique().tolist())
    except Exception as e:
        print(f"[Recommender] Error getting difficulties: {e}")
    return ["All", "Beginner", "Intermediate", "Advanced"]


# ── Get available sources ──────────────────────────────────────────────────────
def get_sources():
    try:
        _ensure_model()
        if _courses_df is not None and "source" in _courses_df.columns:
            return ["All"] + sorted(_courses_df["source"].unique().tolist())
    except Exception as e:
        print(f"[Recommender] Error getting sources: {e}")
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
