"""
text_preprocessing.py
---------------------
Phase 3: NLP Preprocessing Module
Handles text cleaning, tokenization, stopword removal, and lemmatization.
"""

import re
import ssl
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Download required NLTK data (safe to call multiple times) ──────────────────
def download_nltk_resources():
    """Download NLTK resources with SSL workaround for macOS."""
    # macOS SSL certificate fix
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = {
        "tokenizers/punkt":     "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords":    "stopwords",
        "corpora/wordnet":      "wordnet",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

download_nltk_resources()

# ── Singleton helpers ──────────────────────────────────────────────────────────
_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))

# Keep domain-relevant words that NLTK marks as stopwords
_KEEP_WORDS = {
    "not", "no", "nor", "never",          # negations matter for intent
    "when", "where", "what", "how",       # query words
    "me", "my", "i",                      # first-person user context
}
_STOP_WORDS = _stop_words - _KEEP_WORDS


# ── Core preprocessing function ───────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Full NLP pipeline:
      1. Lowercase
      2. Remove URLs, special characters, numbers
      3. Tokenize
      4. Remove stopwords (with domain exceptions)
      5. Lemmatize
    Returns a single cleaned string (space-joined tokens).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove punctuation & digits
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\d+", " ", text)

    # 4. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Tokenize
    tokens = word_tokenize(text)

    # 6. Remove stopwords
    tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    # 7. Lemmatize
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ── Corpus builder (used by vectorizer) ───────────────────────────────────────
def build_corpus(df):
    """
    Combines course_title + description + skills into one text per course,
    preprocesses it, and returns the cleaned corpus list.

    Parameters
    ----------
    df : pd.DataFrame  (must have columns: course_title, description, skills)

    Returns
    -------
    List[str]  – one cleaned string per row
    """
    def _combine(row):
        parts = [
            str(row.get("course_title", "")),
            str(row.get("description",  "")),
            str(row.get("skills",       "")),
        ]
        return " ".join(parts)

    corpus = df.apply(_combine, axis=1).apply(preprocess_text).tolist()
    return corpus


# ── Convenience: tokenize only (for analysis) ─────────────────────────────────
def tokenize(text: str):
    """Returns list of cleaned tokens without joining."""
    cleaned = preprocess_text(text)
    return cleaned.split() if cleaned else []


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "I want to learn Data Science but I don't know coding",
        "I am a beginner and weak in math, want to enter AI",
        "Show me advanced deep learning courses for professionals",
        "Need SQL courses for database management",
    ]
    print("=" * 60)
    print("NLPRec — Text Preprocessing Test")
    print("=" * 60)
    for s in samples:
        cleaned = preprocess_text(s)
        print(f"\nOriginal : {s}")
        print(f"Cleaned  : {cleaned}")
