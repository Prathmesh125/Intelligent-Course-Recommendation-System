"""
tests/test_pipeline.py
Unit tests for NLPRec modules.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd

from text_preprocessing import preprocess_text, build_corpus, tokenize
from vectorizer import load_tfidf_model, transform_query
from recommender import recommend, keyword_search
from user_profile import (load_profile, log_search, save_course,
                           remove_course, enrich_query, get_stats, clear_history)
from evaluation import precision_at_k, recall_at_k, f1_at_k


# ── text_preprocessing ────────────────────────────────────────────────────────
class TestPreprocessing:
    def test_lowercase(self):
        result = preprocess_text("Python FOR Data SCIENCE")
        assert result == result.lower()

    def test_removes_punctuation(self):
        result = preprocess_text("hello, world! python.")
        assert "," not in result and "!" not in result and "." not in result

    def test_removes_stopwords(self):
        result = preprocess_text("I want to learn the basics of programming")
        tokens = result.split()
        # Common stopwords like 'the', 'to', 'of' should be removed
        assert "the" not in tokens
        assert "of"  not in tokens

    def test_lemmatization(self):
        result = preprocess_text("running learning building")
        # lemmatizer should reduce to base forms
        assert "run" in result or "learn" in result

    def test_empty_input(self):
        assert preprocess_text("") == ""
        assert preprocess_text("   ") == ""

    def test_non_string(self):
        assert preprocess_text(None) == ""

    def test_tokenize(self):
        tokens = tokenize("data science python")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_build_corpus(self):
        df = pd.DataFrame([{
            "course_title": "Python Basics",
            "description":  "Learn python programming",
            "skills":       "python, programming",
        }])
        corpus = build_corpus(df)
        assert len(corpus) == 1
        assert isinstance(corpus[0], str)
        assert len(corpus[0]) > 0


# ── vectorizer ────────────────────────────────────────────────────────────────
class TestVectorizer:
    @pytest.fixture(scope="class")
    def model(self):
        return load_tfidf_model()

    def test_model_loads(self, model):
        vectorizer, matrix, df = model
        assert vectorizer is not None
        assert matrix is not None
        assert len(df) > 0

    def test_matrix_shape(self, model):
        vectorizer, matrix, df = model
        assert matrix.shape[0] == len(df)

    def test_transform_query(self, model):
        vectorizer, _, _ = model
        q_vec = transform_query("python data science", vectorizer)
        assert q_vec.shape[1] == vectorizer.get_feature_names_out().shape[0]

    def test_transform_empty_query(self, model):
        vectorizer, _, _ = model
        q_vec = transform_query("", vectorizer)
        assert q_vec.shape[0] == 1  # still returns 1-row sparse matrix


# ── recommender ───────────────────────────────────────────────────────────────
class TestRecommender:
    def test_returns_dataframe(self):
        result = recommend("learn python", top_n=5)
        assert isinstance(result, pd.DataFrame)

    def test_returns_n_results(self):
        result = recommend("data science machine learning", top_n=5)
        assert len(result) <= 5

    def test_has_required_columns(self):
        result = recommend("web development", top_n=3)
        required = {"rank", "course_title", "difficulty", "rating",
                    "similarity_score", "description", "skills"}
        assert required.issubset(set(result.columns))

    def test_similarity_in_valid_range(self):
        result = recommend("python beginners", top_n=5)
        assert (result["similarity_score"] >= 0).all()
        assert (result["similarity_score"] <= 1).all()

    def test_difficulty_filter(self):
        result = recommend("machine learning", top_n=10, difficulty_filter="Beginner")
        if not result.empty:
            assert all(result["difficulty"] == "Beginner")

    def test_empty_query(self):
        result = recommend("")
        assert result.empty

    def test_keyword_search_returns_dataframe(self):
        result = keyword_search("python data", top_n=5)
        assert isinstance(result, pd.DataFrame)

    def test_results_ranked_descending(self):
        result = recommend("deep learning neural networks", top_n=5)
        scores = result["similarity_score"].tolist()
        assert scores == sorted(scores, reverse=True)


# ── user_profile ──────────────────────────────────────────────────────────────
class TestUserProfile:
    def test_default_profile(self):
        p = load_profile("test_unit_user_xyz")
        assert p["username"] == "test_unit_user_xyz"
        assert p["search_history"] == []

    def test_log_search(self):
        p = load_profile("test_unit_user_xyz")
        p = log_search(p, "learn python", "Beginner")
        assert len(p["search_history"]) == 1
        assert p["search_history"][0]["query"] == "learn python"
        assert p["preferred_difficulty"] == "Beginner"

    def test_save_course(self):
        p = load_profile("test_unit_user_xyz")
        p = save_course(p, "Python for Everybody")
        assert "Python for Everybody" in p["saved_courses"]

    def test_remove_course(self):
        p = load_profile("test_unit_user_xyz")
        p = save_course(p, "Python for Everybody")
        p = remove_course(p, "Python for Everybody")
        assert "Python for Everybody" not in p["saved_courses"]

    def test_enrich_query(self):
        p = load_profile("test_unit_user_xyz")
        p["interests"] = ["python", "data", "science"]
        enriched = enrich_query(p, "learn more")
        assert "python" in enriched or "data" in enriched

    def test_enrich_empty_interests(self):
        p = load_profile("test_unit_user_xyz")
        p["interests"] = []
        enriched = enrich_query(p, "learn more")
        assert enriched == "learn more"

    def test_clear_history(self):
        p = load_profile("test_unit_user_xyz")
        p = log_search(p, "some query")
        p = clear_history(p)
        assert p["search_history"] == []
        assert p["interests"] == []

    def test_get_stats(self):
        p = load_profile("test_unit_user_xyz")
        stats = get_stats(p)
        assert "total_searches" in stats
        assert "saved_courses"  in stats


# ── evaluation metrics ────────────────────────────────────────────────────────
class TestMetrics:
    def test_precision_perfect(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_precision_zero(self):
        assert precision_at_k(["x", "y", "z"], ["a", "b", "c"], k=3) == 0.0

    def test_precision_partial(self):
        p = precision_at_k(["a", "x", "b"], ["a", "b", "c"], k=3)
        assert abs(p - 2/3) < 1e-9

    def test_recall_perfect(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0

    def test_recall_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], k=2) == 0.0

    def test_f1_balanced(self):
        f = f1_at_k(0.5, 0.5)
        assert abs(f - 0.5) < 1e-9

    def test_f1_zero_when_both_zero(self):
        assert f1_at_k(0.0, 0.0) == 0.0

    def test_f1_harmonic_mean(self):
        p, r = 0.8, 0.4
        expected = 2 * p * r / (p + r)
        assert abs(f1_at_k(p, r) - expected) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
