"""
query_suggestions.py
---------------------
Dynamic Contextual Query Suggestion Engine for NLPRec
=====================================================
Generates **contextually relevant**, ever-changing query chips shown
below the main search bar.  The chips evolve with every search so they
are always connected to what the user just asked.

Sources (blended together):
 1. Progression    — beginner → intermediate → advanced for the current topic
 2. Adjacent topics — semantic neighbours of the detected topic
 3. Skill combos   — common skill pairings found in the topic's domain
 4. User history   — user's own past queries resurfaced as reminders
 5. Trending        — globally popular queries from behavior_tracker

Public API
----------
    generate_suggestions(query, profile, n)  → list[str]
    get_trending_chips(n)                    → list[str]
"""

import re
import random
from typing import Optional

from behavior_tracker import get_trending_topics, get_popular_queries


# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC KNOWLEDGE GRAPH
# Maps a core topic → related exploration paths
# ═══════════════════════════════════════════════════════════════════════════════
# Format:
#   "canonical_topic": {
#       "progression": ["beginner …", "intermediate …", "advanced …"],
#       "adjacent":    ["related topic A", "related topic B", …],
#       "combos":      ["topic + skill", …],
#   }

_TOPIC_GRAPH: dict[str, dict] = {
    # ── Python ────────────────────────────────────────────────────────────────
    "python": {
        "progression": [
            "Python basics for absolute beginners",
            "Python intermediate — functions, OOP, and modules",
            "Python advanced — decorators, generators, async programming",
        ],
        "adjacent": [
            "data science with Python",
            "Python for automation scripting",
            "Python web development with Django",
            "FastAPI REST API with Python",
            "Python for machine learning",
        ],
        "combos": [
            "Python and pandas for data analysis",
            "Python with NumPy for scientific computing",
            "Python automation with Selenium",
            "Python and Flask build web apps",
        ],
    },
    # ── Machine Learning ─────────────────────────────────────────────────────
    "machine learning": {
        "progression": [
            "machine learning for beginners no math background",
            "supervised learning algorithms from scratch",
            "advanced machine learning — ensemble methods and tuning",
        ],
        "adjacent": [
            "deep learning and neural networks",
            "natural language processing NLP",
            "computer vision with OpenCV",
            "reinforcement learning fundamentals",
            "MLOps and model deployment",
        ],
        "combos": [
            "machine learning with Python scikit-learn",
            "machine learning and statistics essentials",
            "machine learning for time series data",
            "applied machine learning projects portfolio",
        ],
    },
    # ── Deep Learning ─────────────────────────────────────────────────────────
    "deep learning": {
        "progression": [
            "neural networks for beginners",
            "convolutional neural networks for image classification",
            "advanced deep learning — transformers and attention",
        ],
        "adjacent": [
            "machine learning foundations",
            "computer vision and image recognition",
            "natural language processing NLP",
            "generative AI and diffusion models",
            "PyTorch from scratch",
        ],
        "combos": [
            "deep learning with TensorFlow and Keras",
            "deep learning with PyTorch step by step",
            "deep learning for NLP text classification",
        ],
    },
    # ── Data Science ──────────────────────────────────────────────────────────
    "data science": {
        "progression": [
            "data science for absolute beginners",
            "data analysis with pandas and matplotlib",
            "advanced data science — feature engineering and ML pipelines",
        ],
        "adjacent": [
            "machine learning for data scientists",
            "SQL for data analysis",
            "data visualization with Tableau",
            "statistics for data science",
            "big data with Apache Spark",
        ],
        "combos": [
            "data science with Python end to end project",
            "data science and business analytics",
            "data science interview preparation",
        ],
    },
    # ── Web Development ───────────────────────────────────────────────────────
    "web development": {
        "progression": [
            "HTML and CSS for complete beginners",
            "JavaScript intermediate — DOM, fetch API, ES6",
            "full-stack web development with React and Node",
        ],
        "adjacent": [
            "React frontend development",
            "Node.js backend development",
            "Next.js full-stack applications",
            "REST API design with FastAPI",
            "TypeScript for web developers",
        ],
        "combos": [
            "web development with Python Django",
            "web development and databases SQL",
            "web development portfolio project ideas",
        ],
    },
    # ── React ─────────────────────────────────────────────────────────────────
    "react": {
        "progression": [
            "React basics — components, JSX, props",
            "React hooks and state management",
            "React advanced — performance, testing, Next.js",
        ],
        "adjacent": [
            "JavaScript ES6 fundamentals",
            "TypeScript with React",
            "Next.js full-stack apps",
            "Redux for complex state",
            "React Native mobile apps",
        ],
        "combos": [
            "React and Node.js full-stack project",
            "React with GraphQL",
            "React and TailwindCSS UI",
        ],
    },
    # ── JavaScript ────────────────────────────────────────────────────────────
    "javascript": {
        "progression": [
            "JavaScript for beginners from zero",
            "JavaScript intermediate — async, promises, fetch",
            "JavaScript advanced — design patterns, performance",
        ],
        "adjacent": ["TypeScript", "React", "Node.js", "Vue.js"],
        "combos": ["JavaScript and DOM manipulation", "JavaScript and REST API"],
    },
    # ── AI / LLMs ─────────────────────────────────────────────────────────────
    "artificial intelligence": {
        "progression": [
            "artificial intelligence for beginners",
            "machine learning path in AI",
            "large language models and LLM engineering",
        ],
        "adjacent": [
            "prompt engineering for ChatGPT",
            "LangChain and LLM application development",
            "Retrieval Augmented Generation RAG",
            "AI ethics and responsible AI",
        ],
        "combos": [
            "AI with Python end-to-end",
            "AI for healthcare applications",
            "AI automation business use cases",
        ],
    },
    # ── NLP ───────────────────────────────────────────────────────────────────
    "natural language processing": {
        "progression": [
            "NLP basics — text preprocessing and tokenization",
            "NLP with transformers BERT and sentiment analysis",
            "advanced NLP — fine-tuning LLMs and RAG",
        ],
        "adjacent": [
            "machine learning for NLP",
            "deep learning for text",
            "large language models GPT",
            "text classification and named entity recognition",
        ],
        "combos": [
            "NLP with Python spaCy and NLTK",
            "NLP and chatbot development",
            "NLP for search and recommendation systems",
        ],
    },
    # ── Computer Vision ───────────────────────────────────────────────────────
    "computer vision": {
        "progression": [
            "computer vision beginners OpenCV basics",
            "object detection with YOLO",
            "advanced computer vision — GANs and segmentation",
        ],
        "adjacent": ["deep learning", "image processing", "medical imaging AI"],
        "combos": [
            "computer vision with PyTorch",
            "computer vision robotics applications",
        ],
    },
    # ── DevOps / Cloud ────────────────────────────────────────────────────────
    "devops": {
        "progression": [
            "DevOps fundamentals for beginners",
            "Docker and Kubernetes container orchestration",
            "advanced DevOps — GitOps, observability, SRE",
        ],
        "adjacent": ["cloud computing AWS", "Linux administration", "CI/CD pipelines"],
        "combos": ["DevOps with Python", "DevOps and security"],
    },
    "cloud": {
        "progression": [
            "cloud computing basics AWS for beginners",
            "AWS solutions architect associate",
            "advanced cloud architecture multi-region",
        ],
        "adjacent": ["DevOps", "Kubernetes", "serverless computing", "data engineering"],
        "combos": ["cloud and machine learning AWS SageMaker", "cloud and database RDS"],
    },
    # ── Cybersecurity ─────────────────────────────────────────────────────────
    "cybersecurity": {
        "progression": [
            "cybersecurity for beginners ethical hacking",
            "penetration testing and vulnerability assessment",
            "advanced cybersecurity — red team SOC analyst",
        ],
        "adjacent": ["networking and protocols", "Linux for security", "OSCP certification"],
        "combos": ["cybersecurity and Python scripting", "cybersecurity cloud security"],
    },
    # ── Blockchain / Web3 ─────────────────────────────────────────────────────
    "blockchain": {
        "progression": [
            "blockchain basics and cryptocurrency",
            "smart contract development with Solidity",
            "advanced DeFi protocol development",
        ],
        "adjacent": ["Web3 development", "Ethereum", "NFT marketplace", "decentralized apps"],
        "combos": ["blockchain and JavaScript", "blockchain and Python"],
    },
    # ── SQL / Databases ───────────────────────────────────────────────────────
    "sql": {
        "progression": [
            "SQL for beginners — SELECT, WHERE, JOIN",
            "SQL intermediate — window functions, subqueries",
            "advanced SQL — query optimization, indexing",
        ],
        "adjacent": ["data analysis with SQL", "PostgreSQL", "NoSQL MongoDB", "data engineering"],
        "combos": ["SQL and Python for analytics", "SQL and Tableau dashboard"],
    },
    # ── Game Development ─────────────────────────────────────────────────────
    "game development": {
        "progression": [
            "game development for beginners Unity",
            "2D game development Unity C#",
            "advanced 3D game development Unreal Engine",
        ],
        "adjacent": ["C# programming", "game design principles", "Blender 3D modelling"],
        "combos": ["Unity game development mobile", "game development and physics"],
    },
    # ── Mobile / Android / iOS ────────────────────────────────────────────────
    "mobile development": {
        "progression": [
            "mobile app development for beginners Flutter",
            "Android development with Kotlin",
            "React Native cross-platform apps",
        ],
        "adjacent": ["Flutter and Dart", "iOS Swift development", "mobile UX design"],
        "combos": ["mobile and Firebase backend", "mobile and REST API integration"],
    },
    # ── Data Engineering ─────────────────────────────────────────────────────
    "data engineering": {
        "progression": [
            "data engineering fundamentals pipelines",
            "Apache Spark big data processing",
            "advanced data engineering — lakehouse, dbt, orchestration",
        ],
        "adjacent": ["SQL and databases", "cloud data warehouses", "Airflow orchestration"],
        "combos": ["data engineering with Python", "data engineering dbt and Snowflake"],
    },
    # ── Finance / Fintech ─────────────────────────────────────────────────────
    "finance": {
        "progression": [
            "financial literacy and personal finance basics",
            "financial modeling in Excel",
            "algorithmic trading and quantitative finance",
        ],
        "adjacent": ["Python for finance", "blockchain and crypto", "data analysis for finance"],
        "combos": ["finance and Excel advanced", "finance and Python pandas"],
    },
    # ── Music ─────────────────────────────────────────────────────────────────
    "guitar": {
        "progression": [
            "guitar for absolute beginners — chords and strumming",
            "guitar intermediate — scales, solos, music theory",
            "guitar advanced — fingerpicking, improvisation, modes",
        ],
        "adjacent": ["music theory fundamentals", "piano for beginners", "music production"],
        "combos": ["guitar and songwriting", "guitar and music theory"],
    },
    "music": {
        "progression": [
            "music theory for beginners",
            "music production with Ableton",
            "advanced mixing and mastering",
        ],
        "adjacent": ["guitar", "piano", "sound design", "music business"],
        "combos": ["music production and beat making", "music theory and composition"],
    },
    # ── Excel / Analytics ─────────────────────────────────────────────────────
    "excel": {
        "progression": [
            "Excel basics for beginners",
            "Excel intermediate — VLOOKUP pivot tables",
            "Excel advanced — VBA macros Power Query",
        ],
        "adjacent": ["data analysis", "Power BI", "SQL", "Python pandas"],
        "combos": ["Excel and data visualization", "Excel for financial analysis"],
    },
    # ── Design / UX ───────────────────────────────────────────────────────────
    "design": {
        "progression": [
            "UI UX design fundamentals for beginners",
            "Figma for UI designers",
            "advanced design systems and product design",
        ],
        "adjacent": ["graphic design", "web development", "product management"],
        "combos": ["design and front-end development", "UX research and prototyping"],
    },
}

# ── Difficulty progression templates ──────────────────────────────────────────
_DIFF_TIERS = ["for absolute beginners", "intermediate level", "advanced deep dive"]

# ── Domain adjacency fallback (keyword → canonical graph key) ─────────────────
_KEYWORD_MAP: dict[str, str] = {
    "python":              "python",
    "py":                  "python",
    "django":              "python",
    "flask":               "python",
    "fastapi":             "python",
    "ml":                  "machine learning",
    "machine":             "machine learning",
    "learning":            "machine learning",
    "deep":                "deep learning",
    "neural":              "deep learning",
    "tensorflow":          "deep learning",
    "pytorch":             "deep learning",
    "data":                "data science",
    "science":             "data science",
    "pandas":              "data science",
    "numpy":               "data science",
    "web":                 "web development",
    "html":                "web development",
    "css":                 "web development",
    "react":               "react",
    "reactjs":             "react",
    "nextjs":              "react",
    "redux":               "react",
    "javascript":          "javascript",
    "js":                  "javascript",
    "typescript":          "javascript",
    "ts":                  "javascript",
    "ai":                  "artificial intelligence",
    "artificial":          "artificial intelligence",
    "intelligence":        "artificial intelligence",
    "gpt":                 "artificial intelligence",
    "llm":                 "artificial intelligence",
    "langchain":           "artificial intelligence",
    "nlp":                 "natural language processing",
    "natural":             "natural language processing",
    "language":            "natural language processing",
    "bert":                "natural language processing",
    "cv":                  "computer vision",
    "vision":              "computer vision",
    "yolo":                "computer vision",
    "devops":              "devops",
    "docker":              "devops",
    "kubernetes":          "devops",
    "cloud":               "cloud",
    "aws":                 "cloud",
    "azure":               "cloud",
    "gcp":                 "cloud",
    "security":            "cybersecurity",
    "cybersecurity":       "cybersecurity",
    "hacking":             "cybersecurity",
    "blockchain":          "blockchain",
    "web3":                "blockchain",
    "solidity":            "blockchain",
    "crypto":              "blockchain",
    "sql":                 "sql",
    "database":            "sql",
    "postgresql":          "sql",
    "mysql":               "sql",
    "mongodb":             "sql",
    "game":                "game development",
    "unity":               "game development",
    "unreal":              "game development",
    "mobile":              "mobile development",
    "android":             "mobile development",
    "flutter":             "mobile development",
    "kotlin":              "mobile development",
    "swift":               "mobile development",
    "engineering":         "data engineering",
    "spark":               "data engineering",
    "airflow":             "data engineering",
    "finance":             "finance",
    "trading":             "finance",
    "excel":               "excel",
    "guitar":              "guitar",
    "music":               "music",
    "design":              "design",
    "figma":               "design",
    "uxui":                "design",
}


# ══════════════════════════════════════════════════════════════════════════════
# Core suggestion logic
# ══════════════════════════════════════════════════════════════════════════════

def _detect_topic(query: str) -> Optional[str]:
    """
    Map the query to a canonical topic key in _TOPIC_GRAPH.
    Returns None if no known topic is found.
    """
    q = query.lower()
    tokens = re.findall(r"[a-z][a-z0-9+#]*", q)

    # Direct canonical-name match in query text
    for canon in _TOPIC_GRAPH:
        if canon in q:
            return canon

    # Keyword map lookup
    for token in tokens:
        if token in _KEYWORD_MAP:
            return _KEYWORD_MAP[token]

    return None


def _user_history_suggestions(profile: dict, current_query: str, n: int = 3) -> list[str]:
    """
    Pick recent user searches that are different from the current query
    and likely form a learning thread.
    """
    history = profile.get("search_history", [])
    current_lower = current_query.lower()
    seen: set[str] = set()
    result: list[str] = []

    # Walk backwards through history (most recent first)
    for entry in reversed(history[-20:]):
        q = entry.get("query", "").strip()
        ql = q.lower()
        if (
            q
            and ql != current_lower
            and ql not in seen
            and len(q) > 4
        ):
            seen.add(ql)
            result.append(q)

        if len(result) >= n:
            break

    return result


def _trending_suggestions(n: int = 3) -> list[str]:
    """
    Turn trending topic tokens into readable query strings.
    Falls back to popular full queries if topics are sparse.
    """
    topics = get_trending_topics(n + 3, days=7)
    results: list[str] = []

    for t in topics:
        # Capitalise nicely
        canon = _KEYWORD_MAP.get(t)
        if canon and canon in _TOPIC_GRAPH:
            adj = _TOPIC_GRAPH[canon]["adjacent"]
            if adj:
                results.append(random.choice(adj))
        else:
            results.append(f"{t} tutorial for beginners")

        if len(results) >= n:
            break

    # Supplement with popular full queries if needed
    if len(results) < n:
        popular = get_popular_queries(n * 2, days=14)
        for pq in popular:
            if pq not in results:
                results.append(pq)
            if len(results) >= n:
                break

    return results[:n]


def generate_suggestions(
    query: str,
    profile: dict,
    n: int = 6,
) -> list[str]:
    """
    Generate *n* dynamic query suggestion strings that are contextually
    connected to *query* and personalised with *profile*.

    Strategy
    --------
    1. Detect the core topic of the current query.
    2. Pull topic-graph progressions & adjacents (3–4 picks, randomised).
    3. Add 1–2 personalised suggestions from the user's history.
    4. Fill remaining slots with trending / popular queries.
    5. Deduplicate, shuffle slightly, and return exactly n.
    """
    suggestions: list[str] = []
    seen: set[str] = set()

    def _add(s: str):
        sl = s.strip().lower()
        if sl and sl not in seen and s.strip().lower() != query.strip().lower():
            seen.add(sl)
            suggestions.append(s.strip())

    topic = _detect_topic(query)

    if topic and topic in _TOPIC_GRAPH:
        node = _TOPIC_GRAPH[topic]

        # ① Progression (at most 2)
        prog = list(node.get("progression", []))
        random.shuffle(prog)
        for s in prog[:2]:
            _add(s)

        # ② Adjacent topics (at most 2, randomised)
        adj = list(node.get("adjacent", []))
        random.shuffle(adj)
        for s in adj[:2]:
            _add(s)

        # ③ Skill combos (1)
        combos = list(node.get("combos", []))
        if combos:
            _add(random.choice(combos))

    # ④ User history (up to 2, most recent)
    for s in _user_history_suggestions(profile, query, n=2):
        _add(s)

    # ⑤ Trending fill
    if len(suggestions) < n:
        for s in _trending_suggestions(n):
            _add(s)
            if len(suggestions) >= n:
                break

    # ⑥ Generic fallback if still not enough
    fallback = [
        "Python for data science beginners",
        "machine learning with scikit-learn",
        "web development with React",
        "SQL for data analysis",
        "deep learning with PyTorch",
        "cloud computing AWS essentials",
        "data visualization with Python",
        "JavaScript ES6 modern features",
    ]
    for s in fallback:
        if len(suggestions) >= n:
            break
        _add(s)

    # Slight shuffle so the same set doesn't appear every time
    top = suggestions[:max(2, n // 2)]
    rest = suggestions[max(2, n // 2):n]
    random.shuffle(rest)

    return (top + rest)[:n]


def get_trending_chips(n: int = 4) -> list[str]:
    """
    Return n short trending query strings for displaying as 🔥 Trending chips.
    Based on actual multi-user query patterns.
    """
    topics = get_trending_topics(n + 4, days=7)
    chips: list[str] = []

    for t in topics:
        canon = _KEYWORD_MAP.get(t)
        label: str
        if canon and canon in _TOPIC_GRAPH:
            adj = _TOPIC_GRAPH[canon].get("adjacent", [])
            label = random.choice(adj) if adj else f"{canon} for beginners"
        else:
            label = f"{t.title()} crash course"
        if label.lower() not in {c.lower() for c in chips}:
            chips.append(label)
        if len(chips) >= n:
            break

    # Fallback
    defaults = [
        "Python for data science",
        "machine learning for beginners",
        "React web development",
        "cloud computing AWS",
    ]
    for d in defaults:
        if len(chips) >= n:
            break
        if d.lower() not in {c.lower() for c in chips}:
            chips.append(d)

    return chips[:n]
