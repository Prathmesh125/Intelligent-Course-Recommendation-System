"""
vectorizer.py
-------------
Phase 4: Text Representation (TF-IDF Embedding)
Builds TF-IDF vectors for all courses and persists them so the
recommender can load them instantly without re-computing.
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from text_preprocessing import build_corpus, preprocess_text

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "courses.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
TFIDF_PATH   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH  = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
COURSES_PATH = os.path.join(MODELS_DIR, "courses_df.pkl")


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"course_title", "description", "skills", "difficulty", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df.fillna("")
    return df


# ── Build & save TF-IDF model ─────────────────────────────────────────────────
def build_and_save_tfidf(df: pd.DataFrame = None):
    """
    1. Loads dataset (if df not supplied)
    2. Builds cleaned corpus
    3. Fits TF-IDF vectorizer
    4. Saves vectorizer + matrix + dataframe with pickle
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    if df is None:
        df = load_dataset()

    print(f"[Vectorizer] Building TF-IDF on {len(df)} courses …")

    corpus = build_corpus(df)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=1,
        sublinear_tf=True,    # apply log normalization to term frequency
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Persist
    with open(TFIDF_PATH,   "wb") as f:
        pickle.dump(vectorizer,   f)
    with open(MATRIX_PATH,  "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open(COURSES_PATH, "wb") as f:
        pickle.dump(df, f)

    print(f"[Vectorizer] TF-IDF matrix shape : {tfidf_matrix.shape}")
    print(f"[Vectorizer] Saved → {MODELS_DIR}")
    return vectorizer, tfidf_matrix, df


# ── Load pre-built model (used by recommender) ────────────────────────────────
def load_tfidf_model():
    """
    Returns (vectorizer, tfidf_matrix, courses_df).
    Builds the model if pickle files are not found.
    """
    if not all(os.path.exists(p) for p in [TFIDF_PATH, MATRIX_PATH, COURSES_PATH]):
        print("[Vectorizer] Pre-built model not found — building now …")
        return build_and_save_tfidf()

    try:
        with open(TFIDF_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        with open(MATRIX_PATH, "rb") as f:
            tfidf_matrix = pickle.load(f)
        with open(COURSES_PATH, "rb") as f:
            df = pickle.load(f)
        return vectorizer, tfidf_matrix, df
    except Exception as e:
        # Pickles can be brittle across library versions (notably NumPy / SciPy).
        # If loading fails, rebuild deterministically from the CSV dataset.
        print(f"[Vectorizer] Failed to load pre-built model ({type(e).__name__}: {e}) — rebuilding now …")
        for path in (TFIDF_PATH, MATRIX_PATH, COURSES_PATH):
            try:
                os.remove(path)
            except OSError:
                pass
        return build_and_save_tfidf()


# ── Transform a single user query ─────────────────────────────────────────────
def transform_query(query: str, vectorizer: TfidfVectorizer):
    """Preprocesses query and converts to TF-IDF vector."""
    cleaned = preprocess_text(query)
    return vectorizer.transform([cleaned])


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    vec, mat, courses = build_and_save_tfidf()
    sample_query = "I want to learn python for data science"
    q_vec = transform_query(sample_query, vec)
    print(f"\nQuery     : {sample_query}")
    print(f"Query vec shape : {q_vec.shape}")
    print(f"Non-zero features : {q_vec.nnz}")
