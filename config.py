"""
config.py
---------
Centralized Configuration & Constants for NLPRec System
Provides a single source of truth for system-wide settings,
paths, hyperparameters, and feature flags.
"""

import os

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEHAVIOR_DIR = os.path.join(DATASET_DIR, "behavior")
PROFILES_DIR = os.path.join(DATASET_DIR, "profiles")

# ── Dataset Paths ─────────────────────────────────────────────────────────────
COURSES_CSV = os.path.join(DATASET_DIR, "courses.csv")
BEHAVIOR_STORE = os.path.join(BEHAVIOR_DIR, "behavior_store.json")

# ── Model Paths ───────────────────────────────────────────────────────────────
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
COURSES_DF_PATH = os.path.join(MODELS_DIR, "courses_df.pkl")

# ── TF-IDF Hyperparameters ────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)  # unigrams + bigrams
TFIDF_MIN_DF = 1
TFIDF_SUBLINEAR_TF = True

# ── Recommendation Engine Settings ────────────────────────────────────────────
DEFAULT_TOP_N = 5
DEFAULT_MIN_RATING = 0.0

# ── Behavior Tracker Settings ─────────────────────────────────────────────────
CLICK_WEIGHT = 0.015   # per normalized click → added to cosine score
SAVE_WEIGHT = 0.025    # per normalized save
MAX_ENGAGEMENT_BOOST = 0.12  # cap so popular courses don't dominate completely

# ── Query Engine Settings ─────────────────────────────────────────────────────
MIN_SPELL_CHECK_LENGTH = 3  # don't spell-check words shorter than this
MAX_QUERY_LENGTH = 500      # truncate queries longer than this

# ── Feature Flags ─────────────────────────────────────────────────────────────
ENABLE_ENGAGEMENT_BOOST = True
ENABLE_SPELL_CORRECTION = True
ENABLE_QUERY_SUGGESTIONS = True

# ── Logging Configuration ─────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ── UI Settings ───────────────────────────────────────────────────────────────
PAGE_TITLE = "NLPRec — Course Intelligence"
PAGE_ICON = "N"
LAYOUT = "wide"
