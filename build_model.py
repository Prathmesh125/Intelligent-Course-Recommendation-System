"""
build_model.py
--------------
Script to download necessary NLTK resources and build the TF-IDF model.
Run this script to initialize the models required by the recommender system.
"""

import ssl
import nltk
import logging
import argparse
from vectorizer import build_and_save_tfidf


def setup_logging(verbose: bool) -> None:
    """Configures the logging level based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


def download_nltk_resources(quiet: bool = True) -> None:
    """
    Downloads required NLTK resources.
    Applies a fix for macOS SSL certificate issues during download.
    """
    # macOS SSL certificate fix
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for r in resources:
        logging.debug(f'Downloading NLTK resource: {r}')
        nltk.download(r, quiet=quiet)
    logging.info('NLTK resources ready.')


def main() -> None:
    """
    Main execution function.
    Downloads NLTK resources and builds the TF-IDF model.
    """
    parser = argparse.ArgumentParser(description="Build TF-IDF model and download NLTK resources.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    download_nltk_resources(quiet=not args.verbose)

    logging.info('Building TF-IDF model...')
    try:
        build_and_save_tfidf()
        logging.info('TF-IDF model built successfully.')
    except Exception as e:
        logging.error(f"Failed to build TF-IDF model: {e}")
        raise


if __name__ == "__main__":
    main()
