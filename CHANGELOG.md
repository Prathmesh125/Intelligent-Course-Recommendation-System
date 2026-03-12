# Changelog

All notable changes to NLPRec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Centralized configuration module (`config.py`) for system constants
- Common utility functions in `utils/helpers.py` module
- Performance monitoring decorators (timing, retry, memoization)
- Comprehensive `.gitignore` patterns
- `.editorconfig` for consistent code formatting
- `CONTRIBUTING.md` with developer guidelines
- Type hints to recommender module
- Comprehensive logging to vectorizer module
- Named constants for behavior tracker thresholds

### Changed
- Enhanced docstrings with examples in `text_preprocessing.py`
- Improved input validation in recommend function

### Fixed
- None yet

## [1.0.0] - 2026-03-12

### Added
- Initial release of NLPRec — Intelligent Course Recommendation System
- TF-IDF based recommendation engine with engagement boost
- Natural language query understanding engine
- Spell correction and query expansion
- User behavior tracking and analytics
- Personalized user profiles with saved courses
- Multi-platform course scraping (Coursera, edX, freeCodeCamp, Khan Academy)
- Real-time live search capabilities
- Comprehensive evaluation metrics (Precision@K, Recall@K, NDCG@K, MRR)
- Interactive Streamlit web interface
- Session-based user tracking
- Query suggestions and trending topics
- Difficulty and source filtering
- Rating-based filtering
- Dark mode UI with premium design

### Features
- **Smart Query Processing**: Handles typos, abbreviations, casual language
- **Engagement Learning**: Learns from user clicks and saves
- **Personalization**: User-specific recommendations and history
- **Multi-source**: Aggregates courses from multiple platforms
- **Analytics**: Comprehensive behavior tracking and statistics
- **Evaluation**: Built-in comparison against baseline methods

---

## Version History

- **v1.0.0** (2026-03-12): Initial public release
