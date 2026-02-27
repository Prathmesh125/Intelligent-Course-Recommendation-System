"""
app.py
------
Phase 8: Streamlit Interface
Full-featured UI for NLPRec — Intelligent Course Recommendation System.
Run with: streamlit run app.py
"""

import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ── Page config (must be FIRST Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="NLPRec — Course Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ───────────────────────────────────────────────────────────
from recommender   import recommend, keyword_search, get_difficulties, get_sources, invalidate_cache
from user_profile  import (load_profile, save_profile, log_search, log_click, record_session,
                            save_course, remove_course, enrich_query,
                            get_stats, clear_history)
from evaluation    import (run_evaluation, plot_comparison,
                            plot_per_query_heatmap, plot_metric_radar,
                            RESULTS_PATH)
from scraper       import scrape_all, get_last_scrape_info
from vectorizer    import build_and_save_tfidf
from live_search   import search_courses_live, results_to_df, PLATFORM_COLORS as _LIVE_PLATFORM_COLORS
import behavior_tracker as bt
from query_suggestions import generate_suggestions, get_trending_chips
import streamlit.components.v1 as components
import time

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* -------------------------------------------------------------------------- */
    /*                                 VARIABLES                                  */
    /* -------------------------------------------------------------------------- */
    :root {
        --primary: #818CF8;
        --primary-light: #A5B4FC;
        --secondary: #F472B6;
        --bg-color: #0F172A;
        --surface-color: #1E293B;
        --text-color: #E2E8F0;
        --text-light: #94A3B8;
        --border-color: #334155;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.3);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.4), 0 2px 4px -2px rgb(0 0 0 / 0.4);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.4), 0 4px 6px -4px rgb(0 0 0 / 0.4);
        --radius: 16px;
    }

    /* -------------------------------------------------------------------------- */
    /*                                GLOBAL RESET                                */
    /* -------------------------------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
        color: var(--text-color);
        background-color: var(--bg-color);
    }
    /* Force main container to respect dark mode */
    .stApp {
        background-color: var(--bg-color);
    }
    
    /* Streamlit structure overrides */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 5rem;
        max-width: 1200px;
    }

    /* -------------------------------------------------------------------------- */
    /*                                COMPONENTS                                  */
    /* -------------------------------------------------------------------------- */
    
    /* HERO SECTION */
    .hero-container {
        text-align: center;
        padding: 4rem 0 3rem 0;
        background: radial-gradient(circle at top center, rgba(79, 70, 229, 0.05) 0%, transparent 70%);
        margin-bottom: 2rem;
        border-radius: var(--radius);
        animation: fadeIn 0.8s ease-out;
    }
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        line-height: 1.1;
    }
    .subtitle {
        color: var(--text-light);
        font-size: 1.25rem;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* CARDS */
    .course-card {
        background: var(--surface-color);
        border: 1px solid var(--border-color);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin-bottom: 0;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .course-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-light);
    }
    /* Accent line on left */
    .course-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 6px;
        height: 100%;
        background: linear-gradient(to bottom, var(--primary), var(--secondary));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .course-card:hover::before {
        opacity: 1;
    }

    .course-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.75rem;
        line-height: 1.3;
        letter-spacing: -0.01em;
    }

    .course-meta {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }

    .course-desc {
        color: var(--text-light);
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1.25rem;
    }
    
    .course-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color);
    }

    /* BADGES */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        transition: background-color 0.2s;
    }
    .badge-beginner     { background: rgba(16, 185, 129, 0.2); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-intermediate { background: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-advanced     { background: rgba(239, 68, 68, 0.2);  color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.3); }
    
    /* PROGRESS BARS */
    .similarity-bar-bg {
        background: #334155;
        border-radius: 9999px;
        height: 8px;
        margin: 0.75rem 0;
        overflow: hidden;
    }
    .similarity-bar-fill {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* BUTTONS & INPUTS */
    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
        border: none;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    /* Primary buttons */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary) 0%, #6366F1 100%);
        color: white;
    }
    /* Save bookmark button row — sits flush under the card */
    .save-row {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1.5rem;
    }
    .save-row div.stButton > button {
        background: var(--surface-color);
        border: 1px solid var(--border-color) !important;
        color: var(--text-light);
        font-size: 0.78rem;
        font-weight: 600;
        padding: 5px 14px;
        border-radius: 0 0 12px 12px !important;
        border-top: none !important;
        margin-top: -1px;
        letter-spacing: 0.03em;
        box-shadow: none;
    }
    .save-row div.stButton > button:hover {
        background: rgba(129, 140, 248, 0.12);
        color: var(--primary-light);
        border-color: var(--primary) !important;
        transform: none;
        box-shadow: none;
    }
    .save-row div.stButton > button[kind="primary"] {
        background: rgba(129, 140, 248, 0.18);
        color: var(--primary-light);
        border: 1px solid var(--primary) !important;
        border-top: none !important;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-color); /* Match global background instead of forcing white */
        border-right: 1px solid var(--border-color);
    }
    .sidebar-header {
        text-align: center;
        padding: 2rem 1rem;
        border-bottom: 1px solid var(--border-color);
        margin: -1rem -1rem 1.5rem -1rem;
        background: transparent;
    }
    /* Ensure text visibility in sidebar regardless of theme */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {
        color: var(--text-color);
    }
    /* Fix specific contrast issues for headers if they are too light */
    section[data-testid="stSidebar"] h4 {
        color: var(--primary);
        font-weight: 700;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid var(--border-color);
    }
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        color: var(--text-light);
        font-weight: 500;
        padding: 0 1rem;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary);
        background-color: #EEF2FF;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary) !important;
        background-color: transparent !important;
        border-bottom: 2px solid var(--primary) !important;
        font-weight: 700;
    }
    
    /* METRICS */
    div[data-testid="stMetricValue"] {
        font-size: 2.25rem;
        font-weight: 800;
        color: var(--primary);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--text-light);
        font-weight: 500;
    }
    
    /* TOASTS */
    div[data-baseweb="toast"] {
        border-radius: 12px;
        box-shadow: var(--shadow-lg);
        background: white;
        border: 1px solid var(--border-color);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
```
    

</style>
""", unsafe_allow_html=True)


# ── Session state initializer ─────────────────────────────────────────────────
def _init_session():
    if "profile" not in st.session_state:
        st.session_state.profile = load_profile("guest")
    if "last_results_nlp" not in st.session_state:
        st.session_state.last_results_nlp  = pd.DataFrame()
    if "last_results_kw" not in st.session_state:
        st.session_state.last_results_kw   = pd.DataFrame()
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Recommend"
    if "scrape_log" not in st.session_state:
        st.session_state.scrape_log = []
    if "live_results" not in st.session_state:
        st.session_state.live_results = pd.DataFrame()
    if "live_query_info" not in st.session_state:
        st.session_state.live_query_info = {}
    if "live_page" not in st.session_state:
        st.session_state.live_page = 0
    if "live_price_filter" not in st.session_state:
        st.session_state.live_price_filter = "All"
    # ── v2: session start + last query for dynamic suggestions ───────────────
    if "session_start_ts" not in st.session_state:
        st.session_state.session_start_ts = time.time()
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "dynamic_suggestions" not in st.session_state:
        st.session_state.dynamic_suggestions = []

_init_session()

# ── Retention tracking (accumulate time on every meaningful page interaction) ─
_now = time.time()
_last_hb = st.session_state.get("last_heartbeat_ts", _now)
_elapsed  = _now - _last_hb
# Count only gaps > 10 s (real reading time) and < 30 min (not idle overnight)
if 10 < _elapsed < 1800:
    _uname = st.session_state.profile.get("username", "guest")
    bt.log_session_end(_uname, _now - _elapsed)               # global tracker
    _prof = record_session(st.session_state.profile, _elapsed) # per-user profile
    save_profile(_prof)
    st.session_state.profile = _prof
st.session_state["last_heartbeat_ts"] = _now


# ── Difficulty badge helper ────────────────────────────────────────────────────
def _difficulty_badge(level: str) -> str:
    level = (level or "Beginner").strip()
    cls = {
        "Beginner":     "badge-beginner",
        "Intermediate": "badge-intermediate",
        "Advanced":     "badge-advanced",
    }.get(level, "badge-beginner")
    return f'<span class="badge {cls}">{level}</span>'


# ── Source badge helper ────────────────────────────────────────────────────────
# Use cleaner colors for the refresh (Dark Mode Adjusted)
_SOURCE_COLORS = {
    "Coursera":     ("rgba(59, 130, 246, 0.2)", "#93C5FD"),  # Blue-500/20% bg, Blue-300 text
    "MIT OCW":      ("rgba(239, 68, 68, 0.2)",  "#FCA5A5"),  # Red-500/20% bg, Red-300 text
    "freeCodeCamp": ("rgba(17, 24, 39, 0.8)",   "#F3F4F6"),  # Gray-900/80% bg, Gray-100 text
    "Khan Academy": ("rgba(16, 185, 129, 0.2)", "#6EE7B7"),  # Emerald-500/20% bg, Emerald-300 text
    "Udemy":        ("rgba(168, 85, 247, 0.2)", "#D8B4FE"),  # Purple-500/20% bg, Purple-300 text
    "YouTube":      ("rgba(220, 38, 38, 0.2)",  "#FCA5A5"),  # Red-600/20% bg, Red-300 text
    "edX":          ("rgba(79, 70, 229, 0.2)",  "#A5B4FC"),  # Indigo-600/20% bg, Indigo-300 text
}
# Fallback to existing logic but preferring above
def _source_badge(source: str) -> str:
    if source in _SOURCE_COLORS:
        bg, fg = _SOURCE_COLORS[source]
    else:
        # Try to find in imported map, else gray
        pair = _LIVE_PLATFORM_COLORS.get(source)
        bg, fg = pair if pair else ("rgba(75, 85, 99, 0.3)", "#D1D5DB") # Default gray

    return (
        f'<span class="badge" style="background:{bg};color:{fg};'
        f'border:1px solid {fg}40;">{source}</span>'
    )


# ── Render a single course card ───────────────────────────────────────────────
def render_course_card(row, index: int, saved_titles: list, show_save: bool = True):
    score_pct  = min(int(row["similarity_score"] * 100 / max(row["similarity_score"], 0.001)), 100)
    diff_badge = _difficulty_badge(row["difficulty"])
    src_badge  = _source_badge(row.get("source", "")) if row.get("source") else ""
    rating_val = float(row.get("rating", 0) or 0)
    stars      = "★" * int(rating_val) + "☆" * (5 - int(rating_val))
    rating_str = f"({rating_val}/5)" if rating_val > 0 else ""

    desc = str(row['description'])
    if len(desc) > 240:
        desc = desc[:240].rsplit(' ', 1)[0] + "…"

    st.markdown(f"""
    <div class="course-card">
        <div class="course-title">
            <span style="color:#A5B4FC; margin-right:6px; font-weight:400;">#{index}</span>
            <a href="{row['url']}" target="_blank" style="color:inherit; text-decoration:none;" onmouseover="this.style.textDecoration='underline'" onmouseout="this.style.textDecoration='none'">{row['course_title']}</a>
        </div>
        
        <div class="course-meta">
            {src_badge}
            {diff_badge}
            <span style="display:inline-flex; align-items:center; gap:4px; margin-left: 8px;">
                <span style="color:#F59E0B; letter-spacing:1px; font-size:1rem;">{stars}</span>
                <span style="color:#94A3B8; font-size:0.85rem;">{rating_str}</span>
            </span>
            <span style="margin-left: auto; color:#C7D2FE; font-size:0.85rem; font-weight:700; background: rgba(79, 70, 229, 0.3); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(79, 70, 229, 0.4);">
                Match: {int(score_pct)}%
            </span>
        </div>
        
        <div class="similarity-bar-bg">
            <div class="similarity-bar-fill" style="width:{score_pct}%"></div>
        </div>
        
        <div class="course-desc">
            {desc}
        </div>
        
        <div style="font-size:0.85rem; color:var(--text-light); background: rgba(255,255,255,0.05); padding:8px 12px; border-radius:8px; border:1px solid rgba(255,255,255,0.1);">
            <b style="color:var(--text-color);">Skills:</b> {row['skills']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if show_save:
        is_saved = row["course_title"] in saved_titles
        _, save_col = st.columns([3, 1])
        with save_col:
            st.markdown('<div class="save-row">', unsafe_allow_html=True)
            if is_saved:
                if st.button("★  Saved", key=f"save_{index}_{row['course_title'][:15]}", type="primary", use_container_width=True):
                    profile = st.session_state.profile
                    profile = remove_course(profile, row["course_title"])
                    save_profile(profile)
                    st.session_state.profile = profile
                    st.rerun()
            else:
                if st.button("☆  Save", key=f"save_{index}_{row['course_title'][:15]}", use_container_width=True):
                    profile = st.session_state.profile
                    profile = save_course(profile, row["course_title"], metadata={
                        "url":         row.get("url", ""),
                        "description": row.get("description", ""),
                        "difficulty":  row.get("difficulty", ""),
                        "source":      row.get("source", ""),
                        "rating":      row.get("rating", 0),
                    })
                    bt.log_save(profile["username"], row["course_title"])
                    save_profile(profile)
                    st.session_state.profile = profile
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2 style="margin:0; color:var(--primary); font-weight:800; font-size:1.8rem;">NLPRec</h2>
        <p style="color:var(--text-light); margin:4px 0 0 0; font-size:0.9rem;">Course Intelligence Engine</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── User Profile ──────────────────────────────────────────────────────────
    st.markdown("#### 👤 User Profile")
    username = st.text_input("Username", value="guest", key="username_input")
    if username != st.session_state.profile.get("username"):
        st.session_state.profile = load_profile(username)
    
    profile = st.session_state.profile
    stats   = get_stats(profile)
        
    st.divider()

    # ── Live Data Status ──────────────────────────────────────────────────────
    st.markdown("#### 🌐 Live Data Index")
    scrape_info = get_last_scrape_info()
    
    if scrape_info["exists"]:
        st.markdown(
            f"""<div style="background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.2); padding:12px; border-radius:12px; margin-bottom:16px;">
                <div style="color:#34D399; font-weight:700; font-size:1.1rem;">{scrape_info['count']} Courses</div>
                <div style="color:#6EE7B7; font-size:0.8rem;">Updated: {scrape_info['last_updated']}</div>
            </div>""", 
            unsafe_allow_html=True
        )
    else:
        st.warning("No course data yet.")

    with st.expander("⚙️ Scraper Settings"):
        coursera_lim  = st.slider("Coursera",   50, 300, 100, 50)
        mit_ocw_lim   = st.slider("MIT OCW",    20, 200,  80, 20)
        include_fcc   = st.checkbox("Include freeCodeCamp", value=True)
        include_khan  = st.checkbox("Include Khan Academy", value=True)
        
        if st.button("🔄 Fetch New Data", use_container_width=True):
            prog_bar    = st.progress(0)
            status_txt  = st.empty()
            log_area    = st.empty()
            scrape_log  = []

            def _progress_cb(msg, pct):
                prog_bar.progress(pct / 100)
                status_txt.caption(f"⏳ {msg}")
                scrape_log.append(f"[{pct:3d}%] {msg}")
                log_area.code("\n".join(scrape_log[-6:]), language=None)

            with st.spinner("Scraping live courses …"):
                try:
                    df_new = scrape_all(
                        coursera_limit=coursera_lim,
                        mit_ocw_limit=mit_ocw_lim,
                        include_fcc=include_fcc,
                        include_khan=include_khan,
                        progress_callback=_progress_cb,
                    )

                    if df_new.empty:
                        st.error("Scraper returned no courses.")
                    else:
                        _progress_cb("Rebuilding NLP Model...", 95)
                        build_and_save_tfidf(df_new)
                        invalidate_cache()
                        st.session_state.scrape_log = scrape_log
                        prog_bar.progress(1.0)
                        status_txt.success("✅ Database Updated!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # ── Search Preferences ────────────────────────────────────────────────────
    st.markdown("#### ⚡ Filters")
    
    difficulties = get_difficulties()
    default_diff = profile.get("preferred_difficulty", "All")
    diff_idx     = difficulties.index(default_diff) if default_diff in difficulties else 0
    difficulty   = st.selectbox("Max Difficulty", difficulties, index=diff_idx)

    sources_list = get_sources()
    source_filter = st.selectbox("Platform Preference", sources_list, index=0)

    with st.expander("Advanced Options"):
        min_rating   = st.slider("Min Rating", 0.0, 5.0, 0.0, 0.5)
        top_n        = st.select_slider("Results Count", options=[10, 20, 30, 40, 50], value=30)
        personalize  = st.toggle("Personalize Results", value=True)

    st.divider()
    
    # ── Stats & Trends ────────────────────────────────────────────────────────
    st.markdown("#### 📊 Insights")
    c1, c2 = st.columns(2)
    c1.metric("Searches", stats["total_searches"])
    c2.metric("Saved",   stats["saved_courses"])

    behavior = bt.get_user_behavior_summary(username)
    if behavior["session_count"] > 0:
        with st.expander("My Activity Analysis", expanded=False):
            st.caption(f"Avg Session: {behavior['avg_retention_mins']}m")
            st.caption(f"Engagement Score: {behavior['click_count'] + behavior['save_count']*2}")
            if behavior["top_topics"]:
                st.markdown("**Top Topics:**")
                for t in behavior["top_topics"][:3]:
                    st.markdown(f"- {t}")

    st.markdown("---")
    if st.button("🗑 Clear All History", use_container_width=True):
        profile = clear_history(profile)
        save_profile(profile)
        st.session_state.profile = profile
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — Tabs
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('''
<div class="hero-container">
    <h1 class="main-title">NLPRec</h1>
    <p class="subtitle">Intelligent Course Intelligence & Recommendation Engine</p>
</div>
''', unsafe_allow_html=True)

tab_rec, tab_compare, tab_eval, tab_saved = st.tabs([
    "🔍 Discover",
    "⚖️ Model Compare",
    "📊 Performance",
    "🔖 Saved",
])

# ── TAB 1: Recommendations ────────────────────────────────────────────────────
with tab_rec:

    st.markdown("### 🌐 What do you want to learn today?")
    st.caption(
        "Search across **Coursera, Udemy, YouTube, MIT OCW, Harvard, Stanford** and 30+ other platforms concurrently. "
        "Our AI understands natural language, slangs, and context."
    )
    
    st.markdown('<div style="margin-bottom: 8px;"></div>', unsafe_allow_html=True)

    query = st.text_area(
        "Search Query",
        placeholder=(
            "e.g.  'machine learning for absolute beginners'  or  "
            "'how to build a react native app'  or  "
            "'python data science bootcamp'"
        ),
        height=68,
        label_visibility="collapsed",
        key="main_query",
    )

    # Intercept Enter → submit, Shift+Enter → new line
    components.html(
        """
        <script>
        (function() {
            function attachHandler() {
                var doc = window.parent.document;
                var textareas = doc.querySelectorAll('textarea[data-testid="stTextArea"], textarea');
                textareas.forEach(function(ta) {
                    if (ta._enterSubmitAttached) return;
                    ta._enterSubmitAttached = true;
                    ta.addEventListener('keydown', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            // Click the first primary button (Find Courses)
                            var btn = doc.querySelector('button[data-testid="baseButton-primary"]');
                            if (btn) btn.click();
                        }
                    });
                });
            }
            // Run once immediately and again after a short delay to catch late renders
            attachHandler();
            setTimeout(attachHandler, 500);
            setTimeout(attachHandler, 1500);
        })();
        </script>
        """,
        height=0,
    )

    col_btn, col_hint = st.columns([1.5, 4])
    with col_btn:
        search_clicked = st.button("🔎 Find Courses", type="primary", use_container_width=True)
    with col_hint:
        st.markdown(
            "<div style='padding-top:10px; color:#6B7280; font-size:0.9rem;'> "
            "<i>💡 Type naturally. We handle typos & context.</i></div>", 
            unsafe_allow_html=True
        )

    st.divider()

    # ── Dynamic query suggestions (context-aware, change per query) ────────
    # Re-generate whenever the query changes so chips are always connected
    current_q = st.session_state.get("main_query", "") or ""
    if current_q != st.session_state.get("last_query", ""):
        st.session_state.last_query = current_q
        st.session_state.dynamic_suggestions = generate_suggestions(
            current_q or "machine learning",
            st.session_state.profile,
            n=4,
        )

    suggestions = st.session_state.dynamic_suggestions
    if not suggestions:
        suggestions = generate_suggestions("machine learning", st.session_state.profile, n=4)

    st.markdown("**Try these:**")
    q_cols = st.columns(4)
    triggered_preset = None
    for i, preset in enumerate(suggestions[:4]):
        with q_cols[i]:
            if st.button(preset[:48] + ("…" if len(preset) > 48 else ""),
                         key=f"preset_{i}", use_container_width=True,
                         help=preset):
                triggered_preset = preset

    # ── Trending row (based on real multi-user data) ──────────────────────
    trending_chips = get_trending_chips(4)
    if any(c.lower() not in {s.lower() for s in suggestions} for c in trending_chips):
        st.markdown("🔥 **Trending:**")
        t_cols = st.columns(4)
        for i, chip in enumerate(trending_chips[:4]):
            with t_cols[i]:
                if st.button(chip[:48] + ("…" if len(chip) > 48 else ""),
                             key=f"trend_{i}", use_container_width=True,
                             help=chip):
                    triggered_preset = chip

    effective_query = triggered_preset if triggered_preset else query

    # ── Execute search ─────────────────────────────────────────────────────
    if search_clicked or triggered_preset:
        if not effective_query.strip():
            st.warning("Please enter a topic before searching.")
        else:
            profile    = st.session_state.profile
            prog_bar   = st.progress(0)
            status_txt = st.empty()

            def _live_prog(msg, pct):
                prog_bar.progress(pct / 100)
                status_txt.caption(f"⏳ {msg}")

            try:
                raw_results, query_info = search_courses_live(
                    effective_query,
                    top_n=top_n,
                    difficulty_filter=difficulty,
                    progress_callback=_live_prog,
                )
                df_live = results_to_df(raw_results)
                prog_bar.progress(1.0)
                status_txt.empty()

                # ── Log to behavior tracker + user profile ─────────────────
                bt.log_query(profile["username"], effective_query, difficulty)
                profile = log_search(profile, effective_query, difficulty)
                save_profile(profile)
                st.session_state.profile         = profile

                # ── Evolve suggestions immediately for the new query ────────
                st.session_state.dynamic_suggestions = generate_suggestions(
                    effective_query, profile, n=4
                )
                st.session_state.last_query = effective_query

                st.session_state.live_results    = df_live
                st.session_state.live_query_info = query_info
                st.session_state.live_page       = 0  # reset to page 1 on new search

            except Exception as e:
                prog_bar.empty()
                status_txt.empty()
                st.error(f"Live search failed: {e}")
                st.session_state.live_results    = pd.DataFrame()
                st.session_state.live_query_info = {}

    # ── Display results ────────────────────────────────────────────────────
    df_live    = st.session_state.live_results
    query_info = st.session_state.live_query_info

    # Back-compat: old cached results may not have a 'price' column
    if not df_live.empty and "price" not in df_live.columns:
        df_live["price"] = "Free*"
        st.session_state.live_results = df_live

    # "Did you mean / Interpreted as" banner
    correction = query_info.get("display_correction")
    if correction:
        st.info(f"🔍 **Interpreted as:** {correction}", icon="💡")

    if not df_live.empty:
        PAGE_SIZE = 10

        # ── Count banner ───────────────────────────────────────────────
        st.success(f"Found **{len(df_live)}** courses from across the internet.")

        # ── Top page-number pills ─────────────────────────────────────────
        # (computed against unfiltered df so pills don't jump while toggling)
        raw_total_pages = max(1, (len(df_live) + PAGE_SIZE - 1) // PAGE_SIZE)
        current_pg      = st.session_state.live_page

        pg_cols = st.columns([1, 6, 1])
        with pg_cols[0]:
            if st.button("◀ Prev", disabled=(current_pg == 0), use_container_width=True):
                st.session_state.live_page -= 1
                st.rerun()
        with pg_cols[1]:
            pill_cols = st.columns(min(raw_total_pages, 8))
            for pg_i in range(min(raw_total_pages, 8)):
                with pill_cols[pg_i]:
                    label = f"**{pg_i+1}**" if pg_i == current_pg else str(pg_i + 1)
                    if st.button(label, key=f"pgbtn_{pg_i}", use_container_width=True):
                        st.session_state.live_page = pg_i
                        st.rerun()
        with pg_cols[2]:
            if st.button("Next ▶", disabled=(current_pg >= raw_total_pages - 1), use_container_width=True):
                st.session_state.live_page += 1
                st.rerun()

        # ── Price filter (inline, below page pills) ───────────────────────
        st.markdown("")
        pf_left, pf_mid, pf_right = st.columns([1, 3, 4])
        with pf_left:
            st.markdown(
                '<span style="font-weight:700;font-size:0.95rem;">&#128176; Price:</span>',
                unsafe_allow_html=True,
            )
        with pf_mid:
            price_options = ["All", "Free", "Paid"]
            price_sel = st.radio(
                "price_sel",
                price_options,
                index=price_options.index(st.session_state.live_price_filter),
                horizontal=True,
                label_visibility="collapsed",
                key="price_radio_inline",
            )
        with pf_right:
            st.markdown(
                '<span style="color:#6c757d;font-size:0.8rem">'  
                '✓ Free — fully free &nbsp;| '  
                '◑ Free to Audit — free course, paid certificate &nbsp;| '  
                '$ Paid — requires payment'  
                '</span>',
                unsafe_allow_html=True,
            )

        # Reset to page 0 when price filter changes
        if price_sel != st.session_state.live_price_filter:
            st.session_state.live_price_filter = price_sel
            st.session_state.live_page = 0
            st.rerun()

        # ── Apply price filter client-side ───────────────────────────────
        price_sel = st.session_state.live_price_filter
        if price_sel == "Free":
            display_df = df_live[df_live["price"].isin(["Free", "Free*"])].reset_index(drop=True)
        elif price_sel == "Paid":
            display_df = df_live[df_live["price"] == "Paid"].reset_index(drop=True)
        else:
            display_df = df_live.reset_index(drop=True)

        total_filtered = len(display_df)
        total_pages    = max(1, (total_filtered + PAGE_SIZE - 1) // PAGE_SIZE)
        current_pg     = min(st.session_state.live_page, total_pages - 1)

        st.divider()

        if total_filtered == 0:
            st.warning(
                f"No **{'Free' if price_sel == 'Free' else 'Paid'}** courses found in these results. "
                "Try switching to **All** or searching again."
            )
        else:
            st.caption(
                f"Showing **{total_filtered}** {'free' if price_sel == 'Free' else 'paid' if price_sel == 'Paid' else ''} "
                f"courses — Page **{current_pg + 1}** of **{total_pages}**"
            )

            # ── Courses for this page ──────────────────────────────────────
            start_idx    = current_pg * PAGE_SIZE
            end_idx      = start_idx + PAGE_SIZE
            page_df      = display_df.iloc[start_idx:end_idx]
            saved_titles = [
                c if isinstance(c, str) else c.get("title", "")
                for c in st.session_state.profile.get("saved_courses", [])
            ]

            for pos, (_, row) in enumerate(page_df.iterrows(), start=start_idx + 1):
                diff_badge  = _difficulty_badge(row["difficulty"])
                src_badge   = _source_badge(row.get("source", "")) if row.get("source") else ""
                price_val   = row.get("price", "Free*")
                price_color = {
                    "Free":  ("#DCFCE7", "#166534"),
                    "Free*": ("#E0F2FE", "#075985"),
                    "Paid":  ("#FEF3C7", "#92400E"),
                }.get(price_val, ("#F3F4F6", "#4B5563"))
                price_label = {
                    "Free":  "✓ Free",
                    "Free*": "◑ Free to Audit",
                    "Paid":  "$ Paid",
                }.get(price_val, price_val)
                price_badge = (
                    f'<span class="badge" style="background:{price_color[0]};color:{price_color[1]};'
                    f'border:1px solid {price_color[1]}40">{price_label}</span>'
                )
                score      = float(row.get("similarity_score", 0))
                title_disp = str(row["course_title"])[:90] + ("…" if len(str(row["course_title"])) > 90 else "")
                desc_disp  = str(row["description"])[:200] + ("…" if len(str(row["description"])) > 200 else "")

                st.markdown(
                    f"""
                    <div class="course-card">
                        <div class="course-title">
                            <span style="color:#A5B4FC; margin-right:6px; font-weight:400;">#{pos}</span>
                            <a href="{row['url']}" target="_blank" style="color:inherit; text-decoration:none;" onmouseover="this.style.textDecoration='underline'" onmouseout="this.style.textDecoration='none'">{title_disp}</a>
                        </div>
                        <div class="course-meta">
                            {price_badge}{src_badge}{diff_badge}
                            <span style="margin-left: auto; color:#C7D2FE; font-size:0.85rem; font-weight:700; background: rgba(79, 70, 229, 0.3); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(79, 70, 229, 0.4);">
                                Match: {int(score * 100)}%
                            </span>
                        </div>
                        <div class="course-desc">
                            {desc_disp}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                is_saved  = row["course_title"] in saved_titles
                btn_label = "★  Saved" if is_saved else "☆  Save"
                btn_type  = "primary" if is_saved else "secondary"
                _, save_col = st.columns([3, 1])
                with save_col:
                    st.markdown('<div class="save-row">', unsafe_allow_html=True)
                    if st.button(btn_label, key=f"save_live_{pos}_{str(row['course_title'])[:10]}",
                                 type=btn_type, use_container_width=True):
                        profile = st.session_state.profile
                        if is_saved:
                            profile = remove_course(profile, row["course_title"])
                            st.toast(f"Removed: {row['course_title'][:40]}")
                        else:
                            profile = save_course(profile, row["course_title"], metadata={
                                "url":         row.get("url", ""),
                                "description": row.get("description", ""),
                                "difficulty":  row.get("difficulty", ""),
                                "source":      row.get("source", ""),
                                "rating":      row.get("rating", 0),
                            })
                            bt.log_click(profile["username"], row["course_title"])
                            bt.log_save(profile["username"], row["course_title"])
                            st.toast(f"Saved: {row['course_title'][:40]}")
                        save_profile(profile)
                        st.session_state.profile = profile
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            # ── Pagination controls (bottom) ────────────────────────────────
            st.divider()
            bt_cols = st.columns([1, 6, 1])
            with bt_cols[0]:
                if st.button("◀ Prev ", disabled=(current_pg == 0), use_container_width=True):
                    st.session_state.live_page -= 1
                    st.rerun()
            with bt_cols[2]:
                if st.button(" Next ▶", disabled=(current_pg >= total_pages - 1), use_container_width=True):
                    st.session_state.live_page += 1
                    st.rerun()

    elif search_clicked or triggered_preset:
        st.info("No results found. Try rephrasing your query.")


# ── TAB 2: NLP vs Keyword Comparison ─────────────────────────────────────────
with tab_compare:
    st.markdown("### Side-by-side: NLP Semantic vs Keyword Search")
    st.caption("Enter the same query to see how differently the two models respond.")

    cmp_query = st.text_input(
        "Enter query for comparison",
        placeholder="machine learning for beginners with weak math",
        key="cmp_query",
    )
    cmp_btn = st.button("⚖️ Compare Models", type="primary")

    if cmp_btn and cmp_query.strip():
        with st.spinner("Running both models …"):
            nlp_res = recommend(cmp_query, top_n=top_n, difficulty_filter=difficulty, source_filter=source_filter)
            kw_res  = keyword_search(cmp_query, top_n=top_n, difficulty_filter=difficulty)

        st.session_state.last_results_nlp = nlp_res
        st.session_state.last_results_kw  = kw_res

    nlp_res = st.session_state.last_results_nlp
    kw_res  = st.session_state.last_results_kw

    if not nlp_res.empty or not kw_res.empty:
        col_nlp, col_kw = st.columns(2)

        with col_nlp:
            st.markdown("#### 🤖 NLP Semantic Model")
            st.caption("*TF-IDF + Cosine Similarity*")
            if nlp_res.empty:
                st.info("No results.")
            else:
                for _, row in nlp_res.iterrows():
                    diff_badge = _difficulty_badge(row["difficulty"])
                    st.markdown(
                        f"""<div class="course-card" style="padding: 1.25rem;">
                            <div class="course-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                                <span style="color:#A5B4FC; margin-right:4px;">#{int(row["rank"])}</span> {row["course_title"]}
                            </div>
                            <div class="course-meta" style="margin-bottom:0;">
                                {diff_badge} 
                                <span style="color:var(--primary); font-weight:700; font-size:0.85rem; background:#EEF2FF; padding:2px 8px; border-radius:4px;">
                                    Sim: {row["similarity_score"]:.4f}
                                </span>
                                <span style="color:#F59E0B; font-size:0.9rem;">★{row["rating"]}</span>
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )

        with col_kw:
            st.markdown("#### 🔑 Keyword Baseline")
            st.caption("*Word-count matching*")
            if kw_res.empty:
                st.info("No results.")
            else:
                for _, row in kw_res.iterrows():
                    diff_badge = _difficulty_badge(row["difficulty"])
                    st.markdown(
                        f"""<div class="course-card" style="padding: 1.25rem; border-left: 3px solid #F59E0B;">
                            <div class="course-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">
                                <span style="color:#FCD34D; margin-right:4px;">#{int(row["rank"])}</span> {row["course_title"]}
                            </div>
                            <div class="course-meta" style="margin-bottom:0;">
                                {diff_badge}
                                <span style="color:#B45309; font-weight:700; font-size:0.85rem; background:#FEF3C7; padding:2px 8px; border-radius:4px;">
                                    Matches: {int(row["similarity_score"])}
                                </span>
                                <span style="color:#F59E0B; font-size:0.9rem;">★{row["rating"]}</span>
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )


# ── TAB 3: Evaluation ─────────────────────────────────────────────────────────
with tab_eval:
    st.markdown("### Research Evaluation: Precision, Recall & F1")
    st.caption(
        "Runs the NLP model and keyword baseline on a curated test set of "
        "10 queries with known relevant courses. Produces metrics required for publication."
    )

    k_val = st.slider("K (top-N for evaluation)", 3, 10, 5, key="eval_k")

    run_eval_btn = st.button("▶ Run Evaluation", type="primary")

    # Load cached results if available
    if not run_eval_btn and st.session_state.eval_results is None:
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH) as f:
                cached = json.load(f)
            st.session_state.eval_results = cached
            st.info("Showing cached results. Click **Run Evaluation** to refresh.")

    if run_eval_btn:
        with st.spinner(f"Evaluating both models at K={k_val} …"):
            nlp_eval, base_eval = run_evaluation(k=k_val, save=True)
            st.session_state.eval_results = {"nlp": nlp_eval, "baseline": base_eval}
        st.success("Evaluation complete!")

    ev = st.session_state.eval_results
    if ev:
        nlp_ev  = ev["nlp"]
        base_ev = ev["baseline"]
        k_used  = nlp_ev["k"]

        # Metric cards
        c1, c2, c3 = st.columns(3)
        for col, metric, label in [
            (c1, "precision_mean", f"Precision@{k_used}"),
            (c2, "recall_mean",    f"Recall@{k_used}"),
            (c3, "f1_mean",        f"F1@{k_used}"),
        ]:
            nlp_val  = nlp_ev[metric]
            base_val = base_ev[metric]
            delta    = nlp_val - base_val
            col.metric(f"NLP {label}",  f"{nlp_val:.4f}",  delta=f"{delta:+.4f} vs baseline")

        st.divider()

        # Plots
        pcol, rcol = st.columns(2)
        with pcol:
            st.markdown("**Model Comparison (Bar Chart)**")
            fig = plot_comparison(nlp_ev, base_ev, save=True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with rcol:
            st.markdown("**Radar Chart**")
            fig2 = plot_metric_radar(nlp_ev, base_ev, save=True)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        st.divider()
        st.markdown("**Per-Query Heatmap**")
        fig3 = plot_per_query_heatmap(nlp_ev, save=True)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        st.divider()
        st.markdown("**Per-Query Results (NLP Model)**")
        pq_df = pd.DataFrame(nlp_ev["per_query"])[["query","precision","recall","f1"]]
        pq_df.columns = ["Query", "Precision", "Recall", "F1"]
        st.dataframe(pq_df, use_container_width=True)


# ── TAB 4: Saved Courses ──────────────────────────────────────────────────────
with tab_saved:
    profile = st.session_state.profile
    raw_saved = profile.get("saved_courses", [])
    # Normalise: support both legacy str entries and new dict entries
    saved = [{"title": c} if isinstance(c, str) else c for c in raw_saved]

    st.markdown(f"### 🔖 Your Saved Courses  ({len(saved)})")

    if not saved:
        st.info("No saved courses yet. Hit ☆ Save on any recommendation to bookmark it here.")
    else:
        for i, entry in enumerate(saved):
            title   = entry.get("title", "Untitled")
            url     = entry.get("url", "")
            desc    = entry.get("description", "")
            diff    = entry.get("difficulty", "")
            source  = entry.get("source", "")
            rating  = float(entry.get("rating", 0) or 0)
            stars   = "★" * int(rating) + "☆" * (5 - int(rating)) if rating else ""

            diff_badge = _difficulty_badge(diff) if diff else ""
            src_badge  = _source_badge(source) if source else ""
            desc_html  = f'<div style="color:var(--text-light); font-size:0.9rem; line-height:1.5; margin-top:8px;">{desc[:200] + "…" if len(desc) > 200 else desc}</div>' if desc else ""
            title_html = (
                f'<a href="{url}" target="_blank" style="color:inherit; text-decoration:none;" '
                f'onmouseover="this.style.textDecoration=\'underline\'" onmouseout="this.style.textDecoration=\'none\'">{title}</a>'
                if url else title
            )

            st.markdown(
                f'<div class="course-card">'
                f'<div class="course-title" style="font-size:1.15rem;">{title_html}</div>'
                f'<div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:8px 0;">'
                f'{src_badge}{diff_badge}'
                f'{"<span style=\"color:#F59E0B; font-size:0.9rem;\">" + stars + "</span>" if stars else ""}'
                f'</div>'
                f'{desc_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
            _, rm_col = st.columns([3, 1])
            with rm_col:
                st.markdown('<div class="save-row">', unsafe_allow_html=True)
                if st.button("✕  Remove", key=f"rm_{i}_{title[:20]}", use_container_width=True):
                    profile = remove_course(profile, title)
                    save_profile(profile)
                    st.session_state.profile = profile
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)


