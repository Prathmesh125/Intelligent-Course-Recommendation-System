"""
build_model.py
--------------
Script to download necessary NLTK resources and build the TF-IDF model.
Run this script to initialize the models required by the recommender system.
"""

import ssl
import nltk
from vectorizer import build_and_save_tfidf


def main() -> None:
    """
    Downloads NLTK resources and builds the TF-IDF model.
    Applies a fix for macOS SSL certificate issues during download.
    """
    # macOS SSL certificate fix
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for r in resources:
        nltk.download(r, quiet=True)
    print('NLTK resources ready.')

    build_and_save_tfidf()
    print('TF-IDF model built successfully.')


if __name__ == "__main__": .
    main()
