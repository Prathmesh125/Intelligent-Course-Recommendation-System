#!/usr/bin/env bash
# run.sh — Start the NLPRec Streamlit app
# Usage: bash run.sh

set -e
cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Run 'bash setup.sh' first."
    exit 1
fi

# Ensure TF-IDF model is built
python3 - <<'PYEOF'
import os, sys
if not os.path.exists("models/tfidf_vectorizer.pkl"):
    print("Building TF-IDF model …")
    from vectorizer import build_and_save_tfidf
    build_and_save_tfidf()
PYEOF

echo "======================================================"
echo " Starting NLPRec …  http://localhost:8501"
echo " Press Ctrl+C to stop."
echo "======================================================"
streamlit run app.py --server.port 8501
