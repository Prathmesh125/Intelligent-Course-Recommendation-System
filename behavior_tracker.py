"""
behavior_tracker.py
-------------------
Adaptive Behavior Learning Engine for NLPRec
============================================
Automatically learns from user activity **without any manual labelling**:

  • Per-user:   query patterns, topic progression, session retention,
                click-through on courses, difficulty preference drift
  • Cross-user: trending topics, popular queries, engagement statistics
  • Model side: engagement-weighted boost fed back to recommender

Data is persisted to  dataset/behavior/behavior_store.json
Every write is atomic (temp-file + rename) so concurrent Streamlit
re-runs do not corrupt the store.

Public API
----------
    log_query(username, query, difficulty)
    log_click(username, course_title)
    log_save(username, course_title)
    log_session_end(username, session_start_ts)
    get_trending_topics(n, days)      → list[str]
    get_popular_queries(n, days)      → list[str]
    get_engagement_boost(course_title) → float
    get_user_behavior_summary(username) → dict
"""

import os
import json
import time
import math
import shutil
import tempfile
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Optional

# ── Storage path ───────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
BEHAVIOR_DIR  = os.path.join(BASE_DIR, "dataset", "behavior")
STORE_PATH    = os.path.join(BEHAVIOR_DIR, "behavior_store.json")

# ── Boost hyperparameters ──────────────────────────────────────────────────────
CLICK_WEIGHT  = 0.015   # per normalized click → added to cosine score
SAVE_WEIGHT   = 0.025   # per normalized save
MAX_BOOST     = 0.12    # cap so popular courses don't dominate completely


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _now_iso() -> str:
    return datetime.now().isoformat()


def _load_store() -> dict:
    """Load (or initialize) the behavior store from disk."""
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return _empty_store()


def _empty_store() -> dict:
    return {
        # LIST of {username, query, difficulty, ts}
        "query_log":      [],
        # dict  course_title → {clicks, saves, last_interacted}
        "course_engagement": {},
        # dict  username → {session_count, total_retention_secs,
        #                    last_seen, query_count, click_count, save_count}
        "user_stats":     {},
        "created_at":     _now_iso(),
        "last_updated":   _now_iso(),
    }


def _save_store(store: dict):
    """Atomically write the store to disk."""
    os.makedirs(BEHAVIOR_DIR, exist_ok=True)
    store["last_updated"] = _now_iso()
    fd, tmp_path = tempfile.mkstemp(dir=BEHAVIOR_DIR, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(store, f, indent=2)
        shutil.move(tmp_path, STORE_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _ensure_user(store: dict, username: str):
    if username not in store["user_stats"]:
        store["user_stats"][username] = {
            "session_count":        0,
            "total_retention_secs": 0,
            "last_seen":            None,
            "query_count":          0,
            "click_count":          0,
            "save_count":           0,
            # Rolling 30-query topic memory
            "recent_topics":        [],
        }


def _extract_topics(query: str) -> list[str]:
    """
    Extract meaningful topic tokens from a query (length > 3, not stopwords).
    Very lightweight — no ML dependency so this module stays fast.
    """
    _STOP = {
        "want", "learn", "know", "about", "from", "with", "into",
        "help", "that", "this", "what", "have", "will", "some",
        "also", "just", "need", "teach", "show", "make", "more",
        "than", "then", "they", "like", "good", "best", "free",
        "course", "courses", "tutorial", "tutorials", "beginner", "beginners",
        "intermediate", "advanced", "online", "video",
    }
    tokens = re.findall(r"[a-z][a-z0-9+#]*", query.lower())
    return [t for t in tokens if len(t) > 3 and t not in _STOP]


def _cutoff_ts(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# Public API — write side
# ══════════════════════════════════════════════════════════════════════════════

def log_query(username: str, query: str, difficulty: str = "All"):
    """
    Record that *username* submitted *query*.
    Updates cross-user query log and user topic memory.
    """
    store = _load_store()
    _ensure_user(store, username)

    entry = {
        "username":   username,
        "query":      query,
        "difficulty": difficulty,
        "ts":         _now_iso(),
        "topics":     _extract_topics(query),
    }
    store["query_log"].append(entry)
    # Keep at most 5 000 entries to avoid unbounded growth
    store["query_log"] = store["query_log"][-5000:]

    # Update per-user stats
    u = store["user_stats"][username]
    u["query_count"] += 1
    u["last_seen"]    = _now_iso()
    # Rolling topic memory (last 30 queries)
    u["recent_topics"] = (u["recent_topics"] + entry["topics"])[-60:]

    _save_store(store)


def log_click(username: str, course_title: str):
    """Record that *username* clicked / opened *course_title*."""
    store = _load_store()
    _ensure_user(store, username)

    eng = store["course_engagement"]
    if course_title not in eng:
        eng[course_title] = {"clicks": 0, "saves": 0, "last_interacted": None}
    eng[course_title]["clicks"]           += 1
    eng[course_title]["last_interacted"]   = _now_iso()

    store["user_stats"][username]["click_count"] += 1
    store["user_stats"][username]["last_seen"]    = _now_iso()
    _save_store(store)


def log_save(username: str, course_title: str):
    """Record that *username* saved *course_title*."""
    store = _load_store()
    _ensure_user(store, username)

    eng = store["course_engagement"]
    if course_title not in eng:
        eng[course_title] = {"clicks": 0, "saves": 0, "last_interacted": None}
    eng[course_title]["saves"]            += 1
    eng[course_title]["last_interacted"]   = _now_iso()

    store["user_stats"][username]["save_count"] += 1
    store["user_stats"][username]["last_seen"]   = _now_iso()
    _save_store(store)


def log_session_end(username: str, session_start_ts: float):
    """
    Call when a user's session ends (or periodically during the session).
    *session_start_ts* is a Unix timestamp (time.time()).
    Accumulates total retention time for the user.
    """
    duration = max(0, time.time() - session_start_ts)
    store    = _load_store()
    _ensure_user(store, username)

    u = store["user_stats"][username]
    u["total_retention_secs"] += duration
    u["session_count"]        += 1
    u["last_seen"]             = _now_iso()
    _save_store(store)


# ══════════════════════════════════════════════════════════════════════════════
# Public API — read side
# ══════════════════════════════════════════════════════════════════════════════

def get_trending_topics(n: int = 8, days: int = 7) -> list[str]:
    """
    Return the top-N topic tokens seen across **all users** in the last
    *days* days.  Each token appears at most once in the result.
    """
    store  = _load_store()
    cutoff = _cutoff_ts(days)
    counter: Counter = Counter()

    for entry in store["query_log"]:
        if entry["ts"] >= cutoff:
            for t in entry.get("topics", []):
                counter[t] += 1

    return [t for t, _ in counter.most_common(n)]


def get_popular_queries(n: int = 6, days: int = 14) -> list[str]:
    """
    Return the N most-submitted full query strings across all users
    in the last *days* days (deduped, most popular first).
    """
    store  = _load_store()
    cutoff = _cutoff_ts(days)
    counter: Counter = Counter()

    for entry in store["query_log"]:
        if entry["ts"] >= cutoff:
            q = entry["query"].strip().lower()
            if 5 < len(q) < 120:
                counter[q] += 1

    return [q for q, _ in counter.most_common(n)]


def get_engagement_boost(course_title: str) -> float:
    """
    Return a small additive boost (0 – MAX_BOOST) for *course_title*
    based on aggregate clicks and saves from all users.
    Uses log-normalization so a single mega-popular course doesn't
    overshadow everything else.
    """
    store = _load_store()
    eng   = store["course_engagement"].get(course_title)
    if not eng:
        return 0.0

    raw = eng["clicks"] * CLICK_WEIGHT + eng["saves"] * SAVE_WEIGHT
    # log-dampen: reward early interactions strongly, later ones less
    boosted = math.log1p(raw) * 0.05
    return min(boosted, MAX_BOOST)


def get_user_behavior_summary(username: str) -> dict:
    """
    Returns a dict with aggregated behavioral stats for *username*:
        session_count, avg_retention_mins, query_count,
        click_count, save_count, top_topics (list)
    """
    store = _load_store()
    u     = store["user_stats"].get(username)
    if not u:
        return {
            "session_count":       0,
            "avg_retention_mins":  0.0,
            "query_count":         0,
            "click_count":         0,
            "save_count":          0,
            "top_topics":          [],
        }

    sc    = max(1, u["session_count"])
    avg_r = round(u["total_retention_secs"] / sc / 60, 1)

    topic_counter: Counter = Counter(u.get("recent_topics", []))
    top_topics = [t for t, _ in topic_counter.most_common(10)]

    return {
        "session_count":      u["session_count"],
        "avg_retention_mins": avg_r,
        "query_count":        u["query_count"],
        "click_count":        u["click_count"],
        "save_count":         u["save_count"],
        "top_topics":         top_topics,
    }


def get_all_users_stats() -> dict:
    """
    Aggregate stats across all tracked users — useful for admin/analytics view.
    Returns:
        total_users, total_queries, total_clicks, total_saves,
        trending_topics (list[str]), top_courses (list[str])
    """
    store = _load_store()
    umap  = store["user_stats"]

    total_users   = len(umap)
    total_queries = sum(u["query_count"]  for u in umap.values())
    total_clicks  = sum(u["click_count"]  for u in umap.values())
    total_saves   = sum(u["save_count"]   for u in umap.values())

    # Top 5 courses by combined engagement
    eng = store["course_engagement"]
    sorted_courses = sorted(
        eng.keys(),
        key=lambda c: eng[c]["clicks"] * 1 + eng[c]["saves"] * 2,
        reverse=True,
    )

    return {
        "total_users":     total_users,
        "total_queries":   total_queries,
        "total_clicks":    total_clicks,
        "total_saves":     total_saves,
        "trending_topics": get_trending_topics(8, days=7),
        "top_courses":     sorted_courses[:5],
    }
