# Contributing to NLPRec

Thank you for your interest in contributing to NLPRec — Intelligent Course Recommendation System! This document provides guidelines and instructions for contributing.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, virtualenv, or conda)

### Setup Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/Prathmesh125/Intelligent-Course-Recommendation-System.git
   cd Intelligent-Course-Recommendation-System
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Build the TF-IDF model**
   ```bash
   python build_model.py
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📋 Development Guidelines

### Code Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use 4 spaces for indentation
- Maximum line length: 120 characters
- Use meaningful variable and function names
- Add type hints where appropriate

### Documentation

- Add docstrings to all functions, classes, and modules
- Use Google-style or NumPy-style docstrings
- Update README.md if adding new features
- Comment complex logic

### Testing

- Write tests for new features
- Ensure existing tests pass before submitting PR
- Run tests with: `python -m pytest tests/`

### Commit Messages

Follow conventional commit format:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add personalized query suggestions based on user history
fix: resolve issue with special characters in course titles
docs: update installation instructions in README
refactor: improve vectorizer loading performance
```

## 🔧 Project Structure

```
├── app.py                    # Streamlit UI
├── recommender.py            # Core recommendation engine
├── query_engine.py           # Query understanding & expansion
├── vectorizer.py             # TF-IDF model building
├── text_preprocessing.py     # NLP preprocessing
├── behavior_tracker.py       # User behavior analytics
├── user_profile.py           # User profile management
├── config.py                 # Configuration constants
├── dataset/                  # Data files
├── models/                   # Trained models
├── utils/                    # Utility functions
└── tests/                    # Test files
```

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. Description of the issue
2. Steps to reproduce
3. Expected behavior
4. Actual behavior
5. Environment details (OS, Python version, etc.)
6. Screenshots if applicable

## 💡 Feature Requests

We welcome feature requests! Please provide:

1. Clear description of the feature
2. Use case and motivation
3. Potential implementation approach
4. Any relevant examples or references

## 🔀 Pull Request Process

1. **Create a new branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests if applicable
   - Update documentation

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**
   - Provide a clear title and description
   - Reference any related issues
   - Wait for review and address feedback

## 🎯 Areas for Contribution

We're especially interested in contributions for:

- **New Features**: Query suggestions, more filters, course comparison
- **Improvements**: Better NLP preprocessing, enhanced recommendation algorithms
- **Data Sources**: Integration with additional course platforms
- **UI/UX**: Better visualizations, improved user experience
- **Testing**: More comprehensive test coverage
- **Documentation**: Tutorials, examples, API documentation
- **Performance**: Optimization and caching improvements

## 📞 Getting Help

- Open an issue for questions or discussions
- Check existing issues and pull requests
- Read the README.md and DEPLOYMENT.md

## 📜 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards others

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's license.

---

Thank you for contributing to NLPRec! 🎓✨
