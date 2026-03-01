# NLPRec — Intelligent Course Recommendation System
---

## Problem Statement

Traditional recommendation systems rely on keyword matching, categories, and ratings.  
They **do not understand user intent** written in plain English.

| User writes | Typical system | NLPRec |
|---|---|---|
| *"I want to enter AI but I am weak at math and am a beginner"* | Shows: Advanced Deep Learning ❌ | Shows: Python Basics, Intro to AI ✅ |

---

## System Architecture

```
User Query (Natural Language)
        ↓
  Text Preprocessing
  (lowercase → tokenize → stopwords → lemmatize)
        ↓
  TF-IDF Vectorization
  (query vector + course corpus vectors)
        ↓
  Cosine Similarity Engine
        ↓
  Personalization Filter
  (difficulty, rating, user history)
        ↓
  Top-N Recommendations
        ↓
  Evaluation (Precision / Recall / F1)
```

---

## Project Structure

```
NLPRec/
├── app.py                   # Streamlit UI (Phase 8)
├── text_preprocessing.py    # NLP pipeline (Phase 3)
├── vectorizer.py            # TF-IDF embedding (Phase 4)
├── recommender.py           # Cosine similarity engine (Phase 5)
├── user_profile.py          # Personalization (Phase 6)
├── evaluation.py            # Research metrics (Phase 7)
├── requirements.txt
├── setup.sh                 # One-time setup
├── run.sh                   # Start the app
├── dataset/
│   └── courses.csv          # 50 course dataset
├── models/                  # Saved TF-IDF model (auto-generated)
├── assets/                  # Evaluation charts (auto-generated)
└── tests/
    └── test_pipeline.py     # Unit tests
```

---

## Quick Start

### 1. Setup (first time only)

```bash
cd /path/to/NLPRec
bash setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Download NLTK data
- Build and save the TF-IDF model

### 2. Run the App

```bash
bash run.sh
```

Or manually:

```bash
source venv/bin/activate
streamlit run app.py
```

Open → **http://localhost:8501**

---

## Manual Setup (step by step)

```bash
# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python3 -c "import nltk; [nltk.download(r) for r in ['punkt','punkt_tab','stopwords','wordnet']]"

# Build TF-IDF model
python3 vectorizer.py

# (Optional) Test individual modules
python3 text_preprocessing.py
python3 recommender.py
python3 evaluation.py

# Run the app
streamlit run app.py
```

---

## Run Unit Tests

```bash
source venv/bin/activate
pip install pytest
python3 -m pytest tests/ -v
```

---

## Features

| Tab | Feature |
|-----|---------|
| 🔍 Recommend | Natural language search + difficulty/rating filter + save courses |
| ⚖️ NLP vs Keyword | Side-by-side model comparison |
| 📊 Evaluation | Precision@K, Recall@K, F1@K + charts for paper |
| 🔖 Saved Courses | Bookmarked courses per user |
| ℹ️ About | Architecture + tech stack |

---

## Research Contribution (for Publication)

The `evaluation.py` module produces the **exact comparison required by research journals**:

```
              Precision@5   Recall@5   F1@5
NLP Semantic  0.xxxx        0.xxxx     0.xxxx
Keyword Base  0.xxxx        0.xxxx     0.xxxx
Improvement   +X.X%         +X.X%      +X.X%
```

> *"Our NLP semantic model improved Precision by X% over the keyword-matching baseline."*

Charts saved to `assets/`:
- `comparison_chart.png`  — bar chart for paper
- `radar_chart.png`       — radar plot
- `per_query_heatmap.png` — heatmap per query

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| scikit-learn | TF-IDF vectorizer, cosine similarity |
| NLTK | Tokenization, stopwords, lemmatization |
| pandas / numpy | Data processing |
| Streamlit | Web interface |
| matplotlib / seaborn | Evaluation plots |

---

*NLPRec — Intelligent Course Recommendation System*
