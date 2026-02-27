"""
user_profile.py
---------------
Phase 6: Personalization Module  (v2 — Adaptive Learning)
Persists user search history and preferred difficulty level across sessions.
Builds a cumulative interest profile that enriches future queries.

v2 additions
------------
• session_start / last_active timestamps for retention analytics
• difficulty_counts — tracks how often each level is picked (drift detection)
• topic_frequency   — weighted frequency of topics searched (auto-learning)
• click_history     — which courses the user actually opened (implicit feedback)
"""

import os
import json
import re
from datetime import datetime, timedelta
from collections import Counter

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
        "username":             username,
        "created_at":           datetime.now().isoformat(),
        "preferred_difficulty": "All",
        "search_history":       [],   # list of {query, difficulty, timestamp}
        "saved_courses":        [],   # list of course_titles user bookmarked
        "interests":            [],   # accumulated keywords from all queries
        # ── v2 adaptive fields ──────────────────────────────────────────────
        "difficulty_counts":    {"Beginner": 0, "Intermediate": 0, "Advanced": 0},
        "topic_frequency":      {},   # topic → cumulative weighted count
        "click_history":        [],   # list of {course_title, timestamp}
        "session_count":        0,
        "total_retention_secs": 0,    # total time spent across all sessions
        "last_active":          None,
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
    Also updates v2 adaptive fields (topic_frequency, difficulty_counts).
    Call save_profile() afterwards to persist.
    """
    entry = {
        "query":      query,
        "difficulty": difficulty,
        "timestamp":  datetime.now().isoformat(),
    }
    profile["search_history"].append(entry)
    # Keep last 50 searches (up from 20 — richer history for suggestions)
    profile["search_history"] = profile["search_history"][-50:]

    # Accumulate unique interests (simple word-level)
    words = [w.lower() for w in query.split() if len(w) > 3]
    existing = set(profile.get("interests", []))
    profile["interests"] = list(existing | set(words))[:50]  # cap at 50

    # ── v2: track difficulty selection frequency ──────────────────────────
    dc = profile.setdefault("difficulty_counts", {"Beginner": 0, "Intermediate": 0, "Advanced": 0})
    if difficulty in dc:
        dc[difficulty] += 1
    # Auto-infer preferred difficulty from accumulated picks
    non_all = {k: v for k, v in dc.items() if v > 0}
    if non_all:
        profile["preferred_difficulty"] = max(non_all, key=non_all.get)
    elif difficulty != "All":
        profile["preferred_difficulty"] = difficulty

    # ── v2: track topic frequency with recency weighting ─────────────────
    tf = profile.setdefault("topic_frequency", {})
    _STOP = {
        "want", "learn", "know", "about", "from", "with", "into",
        "help", "that", "this", "what", "have", "will", "some",
        "also", "just", "need", "teach", "show", "make", "more",
        "than", "then", "they", "like", "good", "best", "free",
        "course", "courses", "tutorial", "beginner", "beginners",
        "intermediate", "advanced", "online", "video",
    }
    topics = [t for t in re.findall(r"[a-z][a-z0-9+#]*", query.lower())
              if len(t) > 3 and t not in _STOP]
    for t in topics:
        # Recency-weighted: new topics get +1.0, boost existing by 0.5
        tf[t] = tf.get(t, 0) + (0.5 if t in tf else 1.0)
    # Cap at 100 topics; drop lowest-weight ones
    if len(tf) > 100:
        sorted_tf = sorted(tf.items(), key=lambda x: x[1], reverse=True)
        profile["topic_frequency"] = dict(sorted_tf[:100])

    # Update last active
    profile["last_active"] = datetime.now().isoformat()

    return profile


# ── Log a course click (implicit feedback) ────────────────────────────────────
def log_click(profile: dict, course_title: str) -> dict:
    """Record that the user opened / clicked on *course_title*."""
    clicks = profile.setdefault("click_history", [])
    clicks.append({"course_title": course_title, "timestamp": datetime.now().isoformat()})
    profile["click_history"] = clicks[-100:]  # keep last 100
    profile["last_active"]   = datetime.now().isoformat()
    return profile


# ── Record session retention ──────────────────────────────────────────────────
def record_session(profile: dict, retention_secs: float) -> dict:
    """Accumulate session time for retention analytics."""
    profile["total_retention_secs"] = (
        profile.get("total_retention_secs", 0) + max(0, retention_secs)
    )
    profile["session_count"] = profile.get("session_count", 0) + 1
    profile["last_active"]   = datetime.now().isoformat()
    return profile


# ── Save a course ─────────────────────────────────────────────────────────────
def save_course(profile: dict, course_title: str, metadata: dict = None) -> dict:
    saved = profile.get("saved_courses", [])
    # Normalise: convert any legacy plain strings to dicts
    saved = [{"title": c} if isinstance(c, str) else c for c in saved]
    titles = [c["title"] for c in saved]
    if course_title not in titles:
        entry = {"title": course_title}
        if metadata:
            entry.update({
                "url":         metadata.get("url", ""),
                "description": metadata.get("description", ""),
                "difficulty":  metadata.get("difficulty", ""),
                "source":      metadata.get("source", ""),
                "rating":      metadata.get("rating", 0),
            })
        saved.append(entry)
    profile["saved_courses"] = saved[-50:]
    return profile


def remove_course(profile: dict, course_title: str) -> dict:
    saved = profile.get("saved_courses", [])
    profile["saved_courses"] = [
        c for c in saved
        if (c if isinstance(c, str) else c.get("title", "")) != course_title
    ]
    return profile


# ── Personalised query enrichment ─────────────────────────────────────────────
def enrich_query(profile: dict, raw_query: str) -> str:
    """
    Appends top interest keywords to the query so the TF-IDF model
    picks up on prior context when making recommendations.

    v2: prefers topic_frequency (weighted) over flat interests list.
    """
    # Use frequency-weighted topics if available, else fall back to interests
    tf = profile.get("topic_frequency", {})
    if tf:
        sorted_topics = sorted(tf.items(), key=lambda x: x[1], reverse=True)
        recent = [t for t, _ in sorted_topics[:5]]
    else:
        interests = profile.get("interests", [])
        recent = interests[-5:]

    if not recent:
        return raw_query

    enriched = raw_query + " " + " ".join(recent)
    return enriched.strip()


# ── Stats helper ──────────────────────────────────────────────────────────────
def get_stats(profile: dict) -> dict:
    history = profile.get("search_history", [])

    # Top topics by frequency weight
    tf = profile.get("topic_frequency", {})
    top_topics_weighted = sorted(tf.items(), key=lambda x: x[1], reverse=True)
    top_topics = [t for t, _ in top_topics_weighted[:10]]
    if not top_topics:
        top_topics = profile.get("interests", [])[-10:]

    # Avg retention (minutes per session)
    sc = max(1, profile.get("session_count", 1))
    avg_retention = round(profile.get("total_retention_secs", 0) / sc / 60, 1)

    return {
        "total_searches":       len(history),
        "saved_courses":        len(profile.get("saved_courses", [])),
        "top_interests":        top_topics,
        "last_search":          history[-1]["query"] if history else None,
        "preferred_difficulty": profile.get("preferred_difficulty", "All"),
        "difficulty_counts":    profile.get("difficulty_counts", {}),
        "avg_retention_mins":   avg_retention,
        "session_count":        profile.get("session_count", 0),
        "click_count":          len(profile.get("click_history", [])),
    }


# ── Clear history ─────────────────────────────────────────────────────────────
def clear_history(profile: dict) -> dict:
    profile["search_history"]       = []
    profile["interests"]            = []
    profile["topic_frequency"]      = {}
    profile["click_history"]        = []
    profile["difficulty_counts"]    = {"Beginner": 0, "Intermediate": 0, "Advanced": 0}
    profile["preferred_difficulty"] = "All"
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
