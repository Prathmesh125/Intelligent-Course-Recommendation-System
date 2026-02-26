#!/usr/bin/env bash
# setup.sh — One-time project setup for NLPRec
# Usage: bash setup.sh

set -e
cd "$(dirname "$0")"

echo "======================================================"
echo " NLPRec — Project Setup"
echo "======================================================"

# 1. Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment …"
    python3 -m venv venv
else
    echo "[1/4] Virtual environment already exists — skipping."
fi

# 2. Activate venv
source venv/bin/activate
echo "[2/4] Virtual environment activated."

# 3. Install dependencies
echo "[3/4] Installing Python packages …"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "      Packages installed."

# 4. Build TF-IDF model & download NLTK data
echo "[4/4] Building TF-IDF model and downloading NLTK resources …"
python3 - <<'PYEOF'
import nltk
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(resource, quiet=True)
print("  NLTK resources downloaded.")

from vectorizer import build_and_save_tfidf
build_and_save_tfidf()
print("  TF-IDF model built and saved.")
PYEOF

echo ""
echo "======================================================"
echo " Setup complete!"
echo " Run the app with:  bash run.sh"
echo " Or directly:       source venv/bin/activate && streamlit run app.py"
echo "======================================================"
