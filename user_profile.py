"""
user_profile.py
---------------
Phase 6: Personalization Module
Persists user search history and preferred difficulty level across sessions.
Builds a cumulative interest profile that enriches future queries.
"""

import os
import json
from datetime import datetime

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(BASE_DIR, "dataset", "profiles")


def _profile_path(username: str) -> str:
    os.makedirs(PROFILES_DIR, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in username)
    return os.path.join(PROFILES_DIR, f"{safe}.json")


# ── Load / create profile ─────────────────────────────────────────────────────
def load_profile(username: str) -> dict:
    """Load user profile from disk or return a fresh default profile."""
    path = _profile_path(username)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return _default_profile(username)


def _default_profile(username: str) -> dict:
    return {
        "username":          username,
        "created_at":        datetime.now().isoformat(),
        "preferred_difficulty": "All",
        "search_history":    [],      # list of {query, difficulty, timestamp}
        "saved_courses":     [],      # list of course_titles user bookmarked
        "interests":         [],      # accumulated keywords from all queries
    }


def save_profile(profile: dict):
    """Persist profile to disk."""
    path = _profile_path(profile["username"])
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)


# ── Log a new search ──────────────────────────────────────────────────────────
def log_search(profile: dict, query: str, difficulty: str = "All") -> dict:
    """
    Appends query to search_history and extracts keywords into interests.
    Call save_profile() afterwards to persist.
    """
    entry = {
        "query":      query,
        "difficulty": difficulty,
        "timestamp":  datetime.now().isoformat(),
    }
    profile["search_history"].append(entry)

    # Keep last 20 searches only
    profile["search_history"] = profile["search_history"][-20:]

    # Accumulate unique interests (simple word-level)
    words = [w.lower() for w in query.split() if len(w) > 3]
    existing = set(profile.get("interests", []))
    profile["interests"] = list(existing | set(words))[:50]  # cap at 50

    # Update preferred difficulty if user keeps picking one
    if difficulty != "All":
        profile["preferred_difficulty"] = difficulty

    return profile


# ── Save a course ─────────────────────────────────────────────────────────────
def save_course(profile: dict, course_title: str) -> dict:
    saved = profile.get("saved_courses", [])
    if course_title not in saved:
        saved.append(course_title)
    profile["saved_courses"] = saved[-50:]  # cap at 50
    return profile


def remove_course(profile: dict, course_title: str) -> dict:
    profile["saved_courses"] = [
        c for c in profile.get("saved_courses", []) if c != course_title
    ]
    return profile


# ── Personalised query enrichment ─────────────────────────────────────────────
def enrich_query(profile: dict, raw_query: str) -> str:
    """
    Appends top interest keywords to the query so the TF-IDF model
    picks up on prior context when making recommendations.
    """
    interests = profile.get("interests", [])
    if not interests:
        return raw_query

    # Add up to 5 most-recent interests (last entries are most recent)
    recent = interests[-5:]
    enriched = raw_query + " " + " ".join(recent)
    return enriched.strip()


# ── Stats helper ──────────────────────────────────────────────────────────────
def get_stats(profile: dict) -> dict:
    history = profile.get("search_history", [])
    return {
        "total_searches":   len(history),
        "saved_courses":    len(profile.get("saved_courses", [])),
        "top_interests":    profile.get("interests", [])[-10:],
        "last_search":      history[-1]["query"] if history else None,
        "preferred_difficulty": profile.get("preferred_difficulty", "All"),
    }


# ── Clear history ─────────────────────────────────────────────────────────────
def clear_history(profile: dict) -> dict:
    profile["search_history"] = []
    profile["interests"]      = []
    return profile


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_user = "demo_user"
    p = load_profile(test_user)

    p = log_search(p, "I want to learn Python for data science", "Beginner")
    p = log_search(p, "machine learning algorithms", "Intermediate")
    p = save_course(p, "Python for Everybody")

    save_profile(p)

    print("Profile stats:")
    for k, v in get_stats(p).items():
        print(f"  {k}: {v}")

    raw   = "recommend me something new"
    enriched = enrich_query(p, raw)
    print(f"\nEnriched query: {enriched}")
