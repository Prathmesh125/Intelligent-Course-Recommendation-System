"""
live_search.py
--------------
Real-Time Internet Course Search for NLPRec
============================================
When a user enters ANY query, this module searches the live internet —
no predefined sources, no fixed dataset. It scans the entire web for
relevant courses on the topic and returns ranked results.

How it works:
  1. Run multiple DuckDuckGo searches for the query (free, no API key)
  2. Collect results from across the entire internet
  3. Infer platform, difficulty, skills from URL + title + description
  4. Re-rank by NLP cosine similarity to the original query
  5. Return deduplicated, ranked course cards with working live URLs

This covers: Coursera, Udemy, edX, YouTube, LinkedIn Learning, Pluralsight,
Codecademy, Khan Academy, freeCodeCamp, Udacity, DataCamp, DeepLearning.AI,
MIT OCW, Harvard, Stanford, fast.ai, Brilliant … and any other site on the
internet that teaches the topic.
"""

import re
import time
import logging
from urllib.parse import urlparse

log = logging.getLogger("NLPRec-LiveSearch")

# ═══════════════════════════════════════════════════════════════════════════════
# Platform inference from domain
# ═══════════════════════════════════════════════════════════════════════════════
_DOMAIN_TO_PLATFORM = {
    "coursera.org":          "Coursera",
    "udemy.com":             "Udemy",
    "edx.org":               "edX",
    "youtube.com":           "YouTube",
    "youtu.be":              "YouTube",
    "linkedin.com":          "LinkedIn Learning",
    "pluralsight.com":       "Pluralsight",
    "codecademy.com":        "Codecademy",
    "khanacademy.org":       "Khan Academy",
    "freecodecamp.org":      "freeCodeCamp",
    "udacity.com":           "Udacity",
    "datacamp.com":          "DataCamp",
    "deeplearning.ai":       "DeepLearning.AI",
    "fast.ai":               "fast.ai",
    "brilliant.org":         "Brilliant",
    "ocw.mit.edu":           "MIT OCW",
    "openlearninglibrary.mit.edu": "MIT OpenLearn",
    "pll.harvard.edu":       "Harvard Online",
    "online.stanford.edu":   "Stanford Online",
    "w3schools.com":         "W3Schools",
    "tutorialspoint.com":    "TutorialsPoint",
    "geeksforgeeks.org":     "GeeksForGeeks",
    "javatpoint.com":        "JavaTpoint",
    "skillshare.com":        "Skillshare",
    "domestika.org":         "Domestika",
    "futurelearn.com":       "FutureLearn",
    "openlearn.open.ac.uk":  "Open University",
    "alison.com":            "Alison",
    "saylor.org":            "Saylor Academy",
    "theodinproject.com":    "The Odin Project",
    "fullstackopen.com":     "Full Stack Open",
    "scrimba.com":           "Scrimba",
    "egghead.io":            "Egghead",
    "frontendmasters.com":   "Frontend Masters",
    "laracasts.com":         "Laracasts",
    "neetcode.io":           "NeetCode",
    "leetcode.com":          "LeetCode",
    "hackerrank.com":        "HackerRank",
    "kaggle.com":            "Kaggle",
    "roadmap.sh":            "Roadmap.sh",
    "cs50.harvard.edu":      "Harvard CS50",
    "missing.csail.mit.edu": "MIT Missing Semester",
    "developer.mozilla.org": "MDN Web Docs",
    "microsoft.com":         "Microsoft Learn",
    "learn.microsoft.com":   "Microsoft Learn",
    "docs.microsoft.com":    "Microsoft Docs",
    "aws.amazon.com":        "AWS Training",
    "google.qwiklabs.com":   "Google Cloud Skills",
    "cloudskillsboost.google":"Google Cloud Skills",
    "developers.google.com": "Google Developers",
    "academy.google.com":    "Google Career Certificates",
}

# Colours for platform badges in the UI
PLATFORM_COLORS = {
    "Coursera":          ("#0056D2", "#fff"),
    "Udemy":             ("#a435f0", "#fff"),
    "edX":               ("#02262B", "#fff"),
    "YouTube":           ("#FF0000", "#fff"),
    "LinkedIn Learning": ("#0A66C2", "#fff"),
    "Pluralsight":       ("#f15b2a", "#fff"),
    "Codecademy":        ("#105f2a", "#fff"),
    "Khan Academy":      ("#14BF96", "#fff"),
    "freeCodeCamp":      ("#0a0a23", "#99c9ff"),
    "Udacity":           ("#02B3E4", "#fff"),
    "DataCamp":          ("#03EF62", "#111"),
    "DeepLearning.AI":   ("#cc0000", "#fff"),
    "fast.ai":           ("#4b0082", "#fff"),
    "Brilliant":         ("#ff6934", "#fff"),
    "MIT OCW":           ("#8A0000", "#fff"),
    "Harvard Online":    ("#A41034", "#fff"),
    "Stanford Online":   ("#8C1515", "#fff"),
    "Harvard CS50":      ("#A41034", "#fff"),
    "Google Cloud Skills":("#4285F4", "#fff"),
    "Microsoft Learn":   ("#0078D4", "#fff"),
    "AWS Training":      ("#FF9900", "#111"),
    "Skillshare":        ("#00e599", "#111"),
    "W3Schools":         ("#04AA6D", "#fff"),
    "GeeksForGeeks":     ("#2f8d46", "#fff"),
    "Kaggle":            ("#20beff", "#fff"),
    "Scrimba":           ("#2c2c2c", "#fff"),
    "Frontend Masters":  ("#c02d28", "#fff"),
    "The Odin Project":  ("#d35a1f", "#fff"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Difficulty inference from text
# ═══════════════════════════════════════════════════════════════════════════════
_BEGINNER_TOKENS = re.compile(
    r"\b(beginner|introduction|intro|basics|fundamentals|getting started|"
    r"from scratch|101|zero to|for dummies|crash course|overview|primer|"
    r"complete newbie|absolute beginner)\b", re.I
)
_ADVANCED_TOKENS = re.compile(
    r"\b(advanced|expert|mastery|master|deep dive|professional|enterprise|"
    r"architect|senior|production|internals|under the hood|in depth|in-depth|"
    r"hardcore|specialization|research|graduate|phd)\b", re.I
)

def _infer_difficulty(text: str) -> str:
    if _BEGINNER_TOKENS.search(text):
        return "Beginner"
    if _ADVANCED_TOKENS.search(text):
        return "Advanced"
    return "Intermediate"


# ═══════════════════════════════════════════════════════════════════════════════
# Platform inference from URL
# ═══════════════════════════════════════════════════════════════════════════════
def _infer_platform(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower().lstrip("www.")
        # Exact match
        if netloc in _DOMAIN_TO_PLATFORM:
            return _DOMAIN_TO_PLATFORM[netloc]
        # Suffix match (e.g. subdomain.coursera.org)
        for domain, name in _DOMAIN_TO_PLATFORM.items():
            if netloc.endswith(domain):
                return name
    except Exception:
        pass
    return "Web"


# ═══════════════════════════════════════════════════════════════════════════════
# NLP re-ranking: TF-IDF cosine similarity against the user's query
# ═══════════════════════════════════════════════════════════════════════════════
def _rerank(query: str, results: list[dict]) -> list[dict]:
    """Re-rank a list of raw search results by TF-IDF cosine similarity to query."""
    if not results:
        return results
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        docs = [f"{r['course_title']} {r['description']}" for r in results]
        corpus = [query] + docs
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit(corpus)
        mat = vec.transform(corpus)
        sims = cosine_similarity(mat[0:1], mat[1:])[0]
        for i, r in enumerate(results):
            r["similarity_score"] = float(round(sims[i], 4))
            r["rank"] = 0  # will be set after sorting
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
    except Exception as e:
        log.warning(f"Re-ranking failed, returning original order: {e}")
        for i, r in enumerate(results):
            r.setdefault("similarity_score", 0.0)
            r["rank"] = i + 1
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Noise filtering — skip pages that are not actual courses
# ═══════════════════════════════════════════════════════════════════════════════
_SKIP_DOMAINS = re.compile(
    r"\b(reddit|facebook|twitter|instagram|pinterest|quora|medium\.com"
    r"|stackoverflow|wikipedia|amazon\.com|ebay|linkedin\.com/posts"
    r"|linkedin\.com/in|github\.com/[^/]+$|news\.|blog\.)\b",
    re.I,
)
_COURSE_SIGNALS = re.compile(
    r"\b(course|tutorial|learn|lesson|lecture|training|bootcamp|workshop|"
    r"curriculum|syllabus|class|module|certification|certificate|mooc|"
    r"program|degree|academy|school|university|college)\b",
    re.I,
)

# Patterns that indicate a listicle/review article, NOT an actual course page
_LISTICLE_TITLE = re.compile(
    r"(?:^|\b)"
    r"(\d+\s+(?:best|top|free|paid|great|must.?take|must.?have)\s+(?:online\s+)?(?:courses?|tutorials?|resources?|websites?|platforms?)|"
    r"(?:best|top|free)\s+\d+\s+(?:courses?|tutorials?|resources?)|"
    r"(?:best|top|greatest|ultimate|complete|comprehensive)\s+(?:list\s+of\s+)?(?:online\s+)?(?:courses?|tutorials?|resources?|websites?)\s+(?:for|to|in|on)|"
    r"(?:courses?|tutorials?)\s+(?:ranked|reviewed|compared|vs\.?)|"
    r"where\s+to\s+learn|how\s+to\s+learn\s+.*\s+(?:free|online)\s*$)",
    re.I,
)
# Article/blog URL path patterns
_LISTICLE_URL = re.compile(
    r"/(?:blog|article|articles|post|posts|news|guide|guides|resources|roundup|lists?|rankings?)/"
    r"|\.(?:medium|substack|hashnode|dev\.to|towards?data|analyticsvidhya|towardsdatascience)\.com"
    r"|/(?:top|best|\d+)-",
    re.I,
)


def _is_course_like(title: str, body: str, url: str) -> bool:
    """Return True only if the result is an actual course/tutorial page, not an article about courses."""
    if _SKIP_DOMAINS.search(url):
        return False
    # Reject listicle titles: "10 best Python courses", "Top courses for ..."
    if _LISTICLE_TITLE.search(title):
        return False
    # Reject blog/article URLs
    if _LISTICLE_URL.search(url):
        return False
    text = f"{title} {body} {url}"
    return bool(_COURSE_SIGNALS.search(text))


# ═══════════════════════════════════════════════════════════════════════════════
# Core search function
# ═══════════════════════════════════════════════════════════════════════════════
def search_courses_live(
    query: str,
    top_n: int = 15,
    difficulty_filter: str = "All",
    progress_callback=None,
) -> tuple[list[dict], dict]:
    """
    Search the entire internet for courses matching `query`.

    The query is first passed through query_engine.understand_query() which:
      - Fixes spelling errors (pythn → python, mchine → machine)
      - Expands abbreviations  (ML → machine learning, FE → frontend)
      - Removes slang / filler (bro, wanna, literally, just, etc.)
      - Extracts intent & difficulty level signals
      - Returns enriched search queries for better coverage

    Parameters
    ----------
    query : str
        The user's raw free-text query — any spelling, any wording.
    top_n : int
        How many results to return after dedup + NLP ranking.
    difficulty_filter : str
        'All', 'Beginner', 'Intermediate', or 'Advanced'.
        If 'All', auto-inferred difficulty from query_engine is used
        to bias search query suffixes (but not hard-filter results).
    progress_callback : callable(message, pct) | None

    Returns
    -------
    (list[dict], query_info dict)
        list[dict]: ranked course results
        query_info: from understand_query() — includes corrected text, topic, difficulty
    """
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    from query_engine import understand_query

    def _prog(msg, pct):
        log.info(f"[LiveSearch {pct}%] {msg}")
        if progress_callback:
            progress_callback(msg, pct)

    # ── Step 1: Understand the query ───────────────────────────────────────
    _prog("Understanding your query ...", 3)
    query_info = understand_query(query)
    topic      = query_info["topic"]
    inferred_diff = query_info["difficulty"]
    # Respect explicit user filter; fall back to inferred
    diff_to_use = difficulty_filter if difficulty_filter != "All" else inferred_diff
    effective_search_queries = query_info["search_queries"]

    _prog(f'Searching internet for: "{topic}" ...', 5)

    raw: list[dict] = []
    seen_urls: set[str] = set()

    # ── Step 2: Search passes using enriched queries from query_engine ─────
    # Plus 3 fixed site-targeted passes for high-quality platforms
    site_passes = [
        f"{topic} course site:coursera.org OR site:udemy.com OR site:edx.org",
        f"{topic} tutorial site:youtube.com OR site:freecodecamp.org OR site:khanacademy.org",
        f"{topic} course site:datacamp.com OR site:pluralsight.com OR site:linkedin.com/learning",
    ]
    all_search_passes = effective_search_queries + site_passes
    per_query = max(8, (top_n * 3) // len(all_search_passes))

    ddgs = DDGS()
    for i, sq in enumerate(all_search_passes):
        pct = 5 + int((i / len(all_search_passes)) * 55)
        _prog(f"Scanning: {sq[:65]} ...", pct)
        try:
            hits = ddgs.text(sq, max_results=per_query)
            for h in (hits or []):
                url   = (h.get("href") or h.get("url", "")).strip()
                title = (h.get("title", "")).strip()
                body  = (h.get("body",  "") or h.get("description", "")).strip()

                if not url or not title:
                    continue
                if url in seen_urls:
                    continue
                if not _is_course_like(title, body, url):
                    continue

                seen_urls.add(url)
                raw.append({"_title": title, "_body": body, "_url": url})
        except Exception as e:
            log.warning(f"Search pass failed for '{sq[:40]}': {e}")
        time.sleep(0.25)

    _prog(f"Found {len(raw)} raw results — filtering & ranking ...", 62)

    # ── Step 3: Build structured course dicts ─────────────────────────────
    courses = []
    for item in raw:
        title = item["_title"]
        body  = item["_body"]
        url   = item["_url"]
        platform   = _infer_platform(url)
        difficulty = _infer_difficulty(f"{title} {body}")
        skills     = _extract_skills(f"{title} {body}", topic)

        courses.append({
            "course_title":     title,
            "description":      body if body else f"{title} — learn {topic} online.",
            "url":              url,
            "source":           platform,
            "difficulty":       difficulty,
            "rating":           0.0,
            "skills":           skills,
            "similarity_score": 0.0,
            "rank":             0,
        })

    # ── Step 4: Difficulty filter ─────────────────────────────────────────
    if difficulty_filter and difficulty_filter != "All":
        courses = [c for c in courses if c["difficulty"] == difficulty_filter]

    # ── Step 5: NLP re-ranking using the cleaned topic as reference ────────
    _prog("Ranking by relevance ...", 75)
    courses = _rerank(topic, courses)

    final = courses[:top_n]
    _prog(f"Done — {len(final)} courses found.", 100)
    return final, query_info


# ═══════════════════════════════════════════════════════════════════════════════
# Skill keyword extraction (lightweight)
# ═══════════════════════════════════════════════════════════════════════════════
_TECH_TERMS = re.compile(
    r"\b(python|javascript|typescript|java|c\+\+|c#|rust|go|swift|kotlin|php|"
    r"ruby|scala|r|matlab|sql|html|css|react|vue|angular|node|django|flask|"
    r"spring|tensorflow|pytorch|keras|scikit-learn|pandas|numpy|docker|"
    r"kubernetes|aws|gcp|azure|git|linux|bash|machine learning|deep learning|"
    r"nlp|data science|data analysis|statistics|algebra|calculus|physics|"
    r"chemistry|biology|finance|economics|accounting|design|ux|ui|figma|"
    r"photoshop|blender|unity|unreal|blockchain|web3|solidity|cybersecurity|"
    r"networking|devops|mlops|cloud|api|rest|graphql|microservices)\b",
    re.I,
)

def _extract_skills(text: str, query: str) -> str:
    """Extract skill keywords from text + query."""
    found = set(_TECH_TERMS.findall(text.lower()))
    found.update(_TECH_TERMS.findall(query.lower()))
    return ", ".join(sorted(found)) if found else query.title()


# ═══════════════════════════════════════════════════════════════════════════════
# Pandas helper — convert list[dict] to DataFrame matching recommender format
# ═══════════════════════════════════════════════════════════════════════════════
def results_to_df(results: list[dict]):
    """Convert list[dict] from search_courses_live into a DataFrame."""
    import pandas as pd
    if not results:
        return pd.DataFrame(columns=[
            "rank", "course_title", "description", "skills",
            "difficulty", "rating", "url", "source", "similarity_score",
        ])
    return pd.DataFrame(results)[[
        "rank", "course_title", "description", "skills",
        "difficulty", "rating", "url", "source", "similarity_score",
    ]]


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "machine learning for beginners"
    print(f"\n[LiveSearch] Searching internet for: '{q}'\n")
    hits = search_courses_live(q, top_n=10)
    for r in hits:
        print(f"  #{r['rank']:2d} [{r['source']:20s}] [{r['difficulty']:12s}] {r['course_title']}")
        print(f"       {r['url']}")
        print(f"       sim={r['similarity_score']:.4f}  desc: {r['description'][:80]}...")
        print()
