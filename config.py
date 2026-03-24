"""
config.py
---------
Centralized Configuration & Constants for NLPRec System
Provides a single source of truth for system-wide settings,
paths, hyperparameters, and feature flags.
"""

import os
import logging

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEHAVIOR_DIR = os.path.join(DATASET_DIR, "behavior")
PROFILES_DIR = os.path.join(DATASET_DIR, "profiles")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

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
DEFAULT_PAGE_SIZE = 10        # results per page in paginated views
MAX_CACHE_SIZE = 256          # max entries held by in-process LRU caches
SEARCH_HISTORY_LIMIT = 50     # hard cap on stored search-history entries per user

# ── Behavior Tracker Settings ─────────────────────────────────────────────────
CLICK_WEIGHT = 0.015   # per normalized click → added to cosine score
SAVE_WEIGHT = 0.025    # per normalized save
MAX_ENGAGEMENT_BOOST = 0.12  # cap so popular courses don't dominate completely

# ── Query Engine Settings ─────────────────────────────────────────────────────
MIN_SPELL_CHECK_LENGTH = 3  # don't spell-check words shorter than this
MAX_QUERY_LENGTH = 500      # truncate queries longer than this
MAX_RECENT_QUERIES = 20     # max queries kept in user search history

# ── Recommendation Thresholds ────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.05   # minimum cosine score to surface a result
CACHE_TTL_SECONDS = 3600      # how long in-memory model cache stays valid (1 h)

# ── Feature Flags ─────────────────────────────────────────────────────────────
ENABLE_ENGAGEMENT_BOOST = True
ENABLE_SPELL_CORRECTION = True
ENABLE_QUERY_SUGGESTIONS = True

# ── Logging Configuration ─────────────────────────────────────────────────────
# Environment variable override: export NLPREC_LOG_LEVEL=DEBUG
LOG_LEVEL = os.getenv("NLPREC_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log file settings
LOG_TO_FILE = True
LOG_FILE = os.path.join(LOGS_DIR, "nlprec.log")
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5  # Keep 5 backup log files

# Module-specific log levels (override default LOG_LEVEL)
MODULE_LOG_LEVELS = {
    "NLPRec-App": "INFO",
    "NLPRec-Recommender": "INFO",
    "NLPRec-Vectorizer": "INFO",
    "NLPRec-Scraper": "WARNING",
    "NLPRec-Utils": "WARNING",
}


def setup_logging():
    """
    Configure logging for the entire NLPRec system.
    Call this once at application startup (e.g., in app.py).
    """
    # Create logs directory if needed
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Configure root logger
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]  # Console output
    
    if LOG_TO_FILE:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
        )
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )
    
    # Apply module-specific log levels
    for module_name, level_str in MODULE_LOG_LEVELS.items():
        logger = logging.getLogger(module_name)
        logger.setLevel(getattr(logging, level_str.upper(), logging.INFO))
    
    logging.info(f"Logging initialized: level={LOG_LEVEL}, file={LOG_FILE if LOG_TO_FILE else 'disabled'}")


# ── UI Settings ───────────────────────────────────────────────────────────────
PAGE_TITLE = "NLPRec — Course Intelligence"
PAGE_ICON = "N"
LAYOUT = "wide"
