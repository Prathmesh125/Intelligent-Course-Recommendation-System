import ssl, nltk
from vectorizer import build_and_save_tfidf

def main():
    # macOS SSL certificate fix
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    
    for r in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
        nltk.download(r, quiet=True)
    print('NLTK resources ready.')

    build_and_save_tfidf()
    print('TF-IDF model built successfully.')

if __name__ == "__main__":
    main()
