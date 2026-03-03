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
    page_title="NLPRec — Course Intelligence",
    page_icon="N",
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
import time

# ── Premium SaaS UI (dark) ───────────────────────────────────────────────────
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;650;700&display=swap');

  :root{
    --background:#0B0D12;
    --surface:#111522;
    --surface-elevated:#151A2B;
    --border:rgba(255,255,255,0.08);
    --text-primary:rgba(255,255,255,0.92);
    --text-secondary:rgba(255,255,255,0.64);
    --accent:#7C5CFF;

    --shadow-soft: 0 10px 26px rgba(0,0,0,0.28);
    --shadow-elevated: 0 16px 40px rgba(0,0,0,0.42);
    --radius:14px;
    --radius-lg:18px;
  }

  html, body, [data-testid="stAppViewContainer"]{
    background: var(--background) !important;
    color: var(--text-primary) !important;
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif;
  }

  .block-container{
    padding-top: 26px !important;
    padding-bottom: 44px !important;
    max-width: 1240px !important;
  }

  /* Hide Streamlit chrome */
  header[data-testid="stHeader"], section[data-testid="stToolbar"], div[data-testid="stToolbar"], .stDeployButton{ display:none !important; }
  #MainMenu{ visibility:hidden !important; }
  footer{ visibility:hidden !important; }

  /* Sidebar */
  section[data-testid="stSidebar"]{
    background: rgba(17,21,34,0.72) !important;
    border-right: 1px solid var(--border) !important;
  }
  section[data-testid="stSidebar"] > div{ padding-top: 18px !important; }

  /* Typography */
  h1, h2, h3{ letter-spacing: -0.01em; }
  h1{ font-size: 22px !important; font-weight: 700 !important; color: var(--text-primary) !important; }
  h2{ font-size: 16px !important; font-weight: 650 !important; color: var(--text-primary) !important; }
  p, span, label{ color: var(--text-primary); }

  /* Containers (used as cards/panels) */
  div[data-testid="stContainer"]{
    background: linear-gradient(135deg, rgba(21,26,43,0.95), rgba(17,21,34,0.98));
    border: 1.5px solid rgba(255,255,255,0.12);
    border-radius: var(--radius);
    box-shadow: 0 4px 16px rgba(0,0,0,0.35), 0 1px 3px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(8px);
  }
  div[data-testid="stContainer"] > div{ padding: 20px 20px !important; }

  /* Course card hover lift (only when the container contains .nlprec-course) */
  div[data-testid="stContainer"]:has(.nlprec-course){
    transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease, background 180ms ease;
    will-change: transform;
  }
  div[data-testid="stContainer"]:has(.nlprec-course):hover{
    transform: translateY(-3px);
    background: linear-gradient(135deg, rgba(21,26,43,1), rgba(17,21,34,1));
    box-shadow: 0 12px 32px rgba(0,0,0,0.45), 0 2px 8px rgba(124,92,255,0.15), inset 0 1px 0 rgba(255,255,255,0.08);
    border-color: rgba(124,92,255,0.25);
  }

  /* Buttons */
  .stButton > button{
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: rgba(17,21,34,0.66) !important;
    color: var(--text-primary) !important;
    padding: 10px 14px !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    word-break: keep-all !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 100% !important;
    display: block !important;
    transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease, border-color 180ms ease;
  }
  .stButton > button:hover{
    transform: translateY(-1px);
    background: var(--surface-elevated) !important;
    border-color: rgba(255,255,255,0.14) !important;
    box-shadow: var(--shadow-soft) !important;
  }
  .stButton > button:active{ transform: translateY(0px); box-shadow: none !important; }

  .stButton > button[kind="primary"]{
    background: var(--accent) !important;
    border-color: rgba(124,92,255,0.55) !important;
    color: rgba(255,255,255,0.96) !important;
  }
  .stButton > button[kind="primary"]:hover{ box-shadow: var(--shadow-elevated) !important; }

  /* Inputs */
    /* Streamlit/BaseWeb: background is usually applied to nested wrapper divs.
         Force dark theme on wrapper + inner wrapper + actual input/textarea. */
    div[data-testid="stTextInput"] [data-baseweb="input"],
    div[data-testid="stTextArea"]  [data-baseweb="textarea"]{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Inner wrapper (this is the one that often stays white on Cloud) */
    div[data-testid="stTextInput"] [data-baseweb="input"] > div,
    div[data-testid="stTextArea"]  [data-baseweb="textarea"] > div{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05) !important;
        transition: background 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }
    div[data-testid="stTextInput"] [data-baseweb="input"] > div:hover,
    div[data-testid="stTextArea"]  [data-baseweb="textarea"] > div:hover{
        background: var(--surface-elevated) !important;
        border-color: rgba(255,255,255,0.14) !important;
    }
    div[data-testid="stTextInput"] [data-baseweb="input"] > div:focus-within,
    div[data-testid="stTextArea"]  [data-baseweb="textarea"] > div:focus-within{
        border-color: rgba(124,92,255,0.38) !important;
        box-shadow: 0 0 0 2px rgba(124,92,255,0.35) !important;
    }

    /* Actual editable elements */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea{
        background: transparent !important;
        color: var(--text-primary) !important;
        caret-color: var(--text-primary) !important;
    }
    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stTextArea"] textarea::placeholder{
        color: rgba(255,255,255,0.40) !important;
    }

    /* Remove the default outline that can look harsh on dark */
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus{
        outline: none !important;
    }

  /* Select */
  [data-baseweb="select"] > div{
    background: rgba(17,21,34,0.62) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
  }
  [data-baseweb="select"] *{ color: var(--text-primary) !important; }

  /* Radio as segmented control (works well for small sets) */
  div[role="radiogroup"]{
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 6px 8px;
  }

  /* Chips / badges */
  .nlprec-chip{
    display:inline-flex;
    align-items:center;
    gap:6px;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.03);
    color: var(--text-secondary);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
    white-space: nowrap;
  }
  .nlprec-chip.accent{
    background: rgba(124,92,255,0.12);
    border-color: rgba(124,92,255,0.25);
    color: rgba(255,255,255,0.88);
  }

  /* Links styled as buttons */
  a.nlprec-linkbtn{
    display:inline-flex;
    align-items:center;
    padding: 8px 12px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: rgba(17,21,34,0.55);
    color: var(--text-primary);
    text-decoration: none;
    font-weight: 600;
    font-size: 13px;
    transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease;
  }
  a.nlprec-linkbtn:hover{
    transform: translateY(-1px);
    background: var(--surface-elevated);
    box-shadow: var(--shadow-soft);
  }

  /* Metrics */
  [data-testid="stMetric"]{
    background: linear-gradient(180deg, rgba(255,255,255,0.030), rgba(255,255,255,0.012));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-soft);
    padding: 12px 12px;
  }
  div[data-testid="stMetricValue"]{ color: rgba(255,255,255,0.94) !important; }
  div[data-testid="stMetricLabel"]{ color: var(--text-secondary) !important; }

  /* Dataframe */
  .stDataFrame{
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
  }

    /* Alerts (st.info / st.warning / st.error) */
    div[data-testid="stAlert"]{
        border-radius: var(--radius) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        background: rgba(124,92,255,0.08) !important;
        box-shadow: var(--shadow-soft) !important;
    }
    div[data-testid="stAlert"] *{ color: var(--text-primary) !important; }

  /* Skeleton */
  @keyframes nlprecShimmer{ 0%{ background-position: 0% 0%; } 100%{ background-position: 200% 0%; } }
  .nlprec-skeleton{
    border-radius: var(--radius);
    border: 1.5px solid rgba(255,255,255,0.10);
    background: linear-gradient(90deg,
      rgba(21,26,43,0.7) 0%,
      rgba(21,26,43,0.85) 30%,
      rgba(21,26,43,0.7) 60%);
    background-size: 200% 100%;
    animation: nlprecShimmer 1.2s ease-in-out infinite;
    height: 145px;
    margin-bottom: 12px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
  }

  /* Small utility text */
  .nlprec-muted{ color: var(--text-secondary); font-size: 12.5px; }
  .nlprec-title{ font-size: 18px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.01em; }
  .nlprec-subtitle{ color: var(--text-secondary); font-size: 13.5px; line-height: 1.45; }
</style>
""",
    unsafe_allow_html=True,
)



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
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Discover"
    if "pending_search_query" not in st.session_state:
        st.session_state.pending_search_query = None
    if "pending_search_original" not in st.session_state:
        st.session_state.pending_search_original = None
    if "pending_compare_query" not in st.session_state:
        st.session_state.pending_compare_query = None
    if "last_search_error" not in st.session_state:
        st.session_state.last_search_error = None
    # ── v2: session start + last query for dynamic suggestions ───────────────
    if "session_start_ts" not in st.session_state:
        st.session_state.session_start_ts = time.time()
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "dynamic_suggestions" not in st.session_state:
        st.session_state.dynamic_suggestions = []

_init_session()

# ── Check if models exist, show warning if building for first time ────────────
import os
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

if not os.path.exists(TFIDF_PATH):
    st.info("First-time setup: building NLP models from the course dataset. This may take a minute.")

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
    level = (level or "Intermediate").strip() or "Intermediate"
    return f'<span class="nlprec-chip">{level}</span>'


# ── Source badge helper ────────────────────────────────────────────────────────
# Use cleaner colors for the refresh
_SOURCE_COLORS = {
    "Coursera":     ("#DBEAFE", "#1E40AF"),  # Blue-100/800
    "MIT OCW":      ("#FEE2E2", "#991B1B"),  # Red-100/800
    "freeCodeCamp": ("#1F2937", "#F9FAFB"),  # Gray-800/50
    "Khan Academy": ("#D1FAE5", "#065F46"),  # Emerald-100/800
    "Udemy":        ("#F3E8FF", "#6B21A8"),  # Purple-100/800
    "YouTube":      ("#FEE2E2", "#DC2626"),  # Red-100/600
    "edX":          ("#E0E7FF", "#3730A3"),  # Indigo-100/800
}
# Fallback to existing logic but preferring above
def _source_badge(source: str) -> str:
    if not source:
        return ""

    # Prefer map from live_search for brand colors
    pair = _LIVE_PLATFORM_COLORS.get(source)
    if pair:
        bg, fg = pair
        # Convert to subtle border tint (avoid loud pills on dark UI)
        return (
            '<span class="nlprec-chip" '
            f'style="border-color: {bg}55; background: {bg}14; color: var(--text-primary);">'
            f'{source}</span>'
        )

    return f'<span class="nlprec-chip">{source}</span>'


def _price_badge(price_val: str) -> str:
    price_val = (price_val or "Free*").strip() or "Free*"
    label = {"Free": "Free", "Free*": "Free to audit", "Paid": "Paid"}.get(price_val, price_val)
    style = {
        "Free":  "border-color: rgba(46,204,113,0.35); background: rgba(46,204,113,0.10);",
        "Free*": "border-color: rgba(56,189,248,0.35); background: rgba(56,189,248,0.10);",
        "Paid":  "border-color: rgba(245,165,36,0.35); background: rgba(245,165,36,0.10);",
    }.get(price_val, "")
    return f'<span class="nlprec-chip" style="{style}">{label}</span>'


# ── Render a single course card ───────────────────────────────────────────────
def _render_skeleton_grid(total_cards: int = 6, cols: int = 1):
    """Render skeleton placeholders for loading state (full-width horizontal cards)."""
    for i in range(total_cards):
        st.markdown('<div class="nlprec-skeleton"></div>', unsafe_allow_html=True)


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def render_course_card(row, index: int, saved_titles: list, show_save: bool = True, key_prefix: str = "course"):
    """Premium course card rendered inside a Streamlit container (so buttons are truly inside the card)."""
    title = str(row.get("course_title", ""))
    url = str(row.get("url", "#"))
    source = str(row.get("source", ""))
    difficulty = str(row.get("difficulty", "Intermediate"))
    duration = str(row.get("duration", "") or "—")
    price = str(row.get("price", "Free*"))
    rating = float(row.get("rating", 0) or 0)
    sim = float(row.get("similarity_score", 0) or 0)
    match_pct = int(max(0, min(100, round(sim * 100))))

    # Clean description: remove "Missing:" artifacts and truncate
    raw_desc = str(row.get("description", ""))
    clean_desc = raw_desc.split("Missing:")[0].strip()
    desc = _truncate(clean_desc, 140) if clean_desc else ""

    src_badge = _source_badge(source) if source else ""
    diff_badge = _difficulty_badge(difficulty)
    dur_badge = f'<span class="nlprec-chip">{duration}</span>'
    price_badge = _price_badge(price)
    rating_badge = (
        f'<span class="nlprec-chip">{rating:.1f}/5</span>' if rating and rating > 0 else '<span class="nlprec-chip">—/5</span>'
    )
    match_badge = f'<span class="nlprec-chip accent">Match {match_pct}%</span>'

    is_saved = title in saved_titles

    with st.container(border=True):
        st.markdown('<div class="nlprec-course">', unsafe_allow_html=True)
        # Header: title + save button
        top_l, top_r = st.columns([8, 2], vertical_alignment="top")
        with top_l:
            st.markdown(
                f'<div style="font-weight:700; font-size:15.5px; line-height:1.4; margin-bottom:12px;">'
                f'<a href="{url}" target="_blank" style="color: var(--text-primary); text-decoration:none;">'
                f'{_truncate(title, 85)}'
                f'</a></div>',
                unsafe_allow_html=True,
            )

        with top_r:
            if show_save:
                btn_label = "Saved" if is_saved else "Save"
                btn_type = "secondary" if is_saved else "primary"
                if st.button(btn_label, key=f"{key_prefix}_save_{index}_{title[:18]}", use_container_width=True, type=btn_type):
                    profile = st.session_state.profile
                    if is_saved:
                        profile = remove_course(profile, title)
                    else:
                        meta = {
                            "url": url,
                            "description": str(row.get("description", "")),
                            "difficulty": difficulty,
                            "source": source,
                            "rating": rating,
                        }
                        profile = save_course(profile, title, metadata=meta)
                        bt.log_save(profile.get("username", "guest"), title)
                    save_profile(profile)
                    st.session_state.profile = profile
                    st.rerun()

        # Metadata chips in organized groups
        st.markdown(
            f'<div style="display:flex; flex-wrap:wrap; gap:7px; margin-bottom:12px;">'
            f'{src_badge}{diff_badge}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="display:flex; flex-wrap:wrap; gap:7px; margin-bottom:14px;">'
            f'{dur_badge}{rating_badge}{price_badge}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Description
        if desc:
            st.markdown(
                f'<div class="nlprec-subtitle" style="margin-bottom:14px; line-height:1.5;">{desc}</div>',
                unsafe_allow_html=True,
            )

        # Footer: match score + action button
        footer_l, footer_r = st.columns([1, 1], vertical_alignment="center")
        with footer_l:
            st.markdown(match_badge, unsafe_allow_html=True)
        with footer_r:
            st.markdown(
                f'<div style="display:flex; justify-content:flex-end;">'
                f'<a class="nlprec-linkbtn" href="{url}" target="_blank">Open course</a>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)


def _normalize_saved_courses(profile: dict) -> tuple[list[dict], list[str]]:
    saved = profile.get("saved_courses", []) or []
    entries: list[dict] = []
    for item in saved:
        if isinstance(item, str):
            title = item
            entries.append({"title": title})
        elif isinstance(item, dict):
            title = item.get("title") or item.get("course_title") or ""
            if title:
                e = {"title": title}
                e.update(item)
                entries.append(e)
    titles = [e["title"] for e in entries if e.get("title")]
    return entries, titles


def _app_header(title: str, subtitle: str | None = None):
    st.markdown(f'<div class="nlprec-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="nlprec-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def _render_sidebar() -> tuple[str, dict, dict]:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 10px 10px 14px 10px;">
              <div style="font-weight: 800; font-size: 18px; letter-spacing: -0.02em; color: var(--text-primary);">NLPRec</div>
              <div class="nlprec-muted" style="margin-top: 4px;">Course intelligence and recommendation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            ["Discover", "Saved", "Model comparison", "Performance"],
            key="nav_page",
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Profile**")
        username = st.text_input("Username", value=st.session_state.profile.get("username", "guest"), key="username_input")
        if username != st.session_state.profile.get("username"):
            st.session_state.profile = load_profile(username)

        profile = st.session_state.profile
        stats = get_stats(profile)

        st.divider()
        st.markdown("**Data**")
        scrape_info = get_last_scrape_info()
        if scrape_info.get("exists"):
            st.markdown(
                f'<div style="display:flex; flex-wrap:wrap; gap:8px;">'
                f'<span class="nlprec-chip accent">{scrape_info.get("count", 0)} indexed</span>'
                f'<span class="nlprec-chip">Updated {scrape_info.get("last_updated", "")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No indexed dataset found yet.")

        with st.expander("Scraper", expanded=False):
            coursera_lim = st.slider("Coursera", 50, 300, 100, 50)
            mit_ocw_lim = st.slider("MIT OCW", 20, 200, 80, 20)
            include_fcc = st.checkbox("Include freeCodeCamp", value=True)
            include_khan = st.checkbox("Include Khan Academy", value=True)

            if st.button("Fetch new data", use_container_width=True):
                prog_bar = st.progress(0)
                status_txt = st.empty()
                log_area = st.empty()
                scrape_log: list[str] = []

                def _progress_cb(msg, pct):
                    prog_bar.progress(pct / 100)
                    status_txt.caption(msg)
                    scrape_log.append(f"[{pct:3d}%] {msg}")
                    log_area.code("\n".join(scrape_log[-6:]), language=None)

                with st.spinner("Updating dataset and model..."):
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
                            _progress_cb("Rebuilding vectorizer...", 95)
                            build_and_save_tfidf(df_new)
                            invalidate_cache()
                            prog_bar.progress(1.0)
                            status_txt.success("Update complete")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Scraper error: {e}")

        st.divider()
        st.markdown("**Insights**")
        m1, m2 = st.columns(2)
        m1.metric("Searches", stats.get("total_searches", 0))
        m2.metric("Saved", stats.get("saved_courses", 0))

        if st.button("Clear history", use_container_width=True):
            profile = clear_history(profile)
            save_profile(profile)
            st.session_state.profile = profile
            st.rerun()

    return page, profile, stats


def _run_live_search(query_text: str, top_n: int, difficulty: str, prog_bar=None, status_txt=None, original_query=None):
    profile = st.session_state.profile

    def _live_prog(msg, pct):
        if prog_bar:
            prog_bar.progress(pct / 100)
        if status_txt:
            status_txt.caption(msg)

    raw_results, query_info = search_courses_live(
        query_text,
        top_n=top_n,
        difficulty_filter=difficulty,
        progress_callback=_live_prog,
    )
    df_live = results_to_df(raw_results)

    # If this was a personalized/enriched query, clear the display_correction
    # to avoid confusing the user with the enriched query terms
    if original_query and original_query != query_text:
        # This was enriched, don't show the enrichment to the user
        query_info['display_correction'] = None

    # Log behaviour and persist profile using the ORIGINAL query
    bt.log_query(profile.get("username", "guest"), original_query or query_text, difficulty)
    profile = log_search(profile, original_query or query_text, difficulty)
    save_profile(profile)
    st.session_state.profile = profile

    st.session_state.live_results = df_live
    st.session_state.live_query_info = query_info
    st.session_state.live_page = 0


def _render_discover(profile: dict):
    _app_header(
        "Discover",
        "Search the live web for courses and save the best matches.",
    )

    saved_entries, saved_titles = _normalize_saved_courses(profile)

    # Horizontal filter bar (better alignment + less "left column" clutter)
    with st.container(border=True):
        st.markdown("**Filters**")

        difficulties = get_difficulties()
        default_diff = profile.get("preferred_difficulty", "All")
        diff_idx = difficulties.index(default_diff) if default_diff in difficulties else 0

        sources_list = get_sources()

        r1 = st.columns([1.15, 1.15, 1.35, 1.0], gap="large")
        with r1[0]:
            difficulty = st.selectbox("Difficulty", difficulties, index=diff_idx, key="flt_difficulty")
        with r1[1]:
            source_filter = st.selectbox("Platform", sources_list, index=0, key="flt_platform")
        with r1[2]:
            price_sel = st.radio(
                "Price",
                ["All", "Free", "Paid"],
                horizontal=True,
                key="flt_price",
            )
        with r1[3]:
            personalize = st.toggle("Personalize", value=False, key="flt_personalize")

        r2 = st.columns([1.4, 1.2, 1.4], gap="large")
        with r2[0]:
            min_rating = st.slider("Minimum rating", 0.0, 5.0, 0.0, 0.5, key="flt_min_rating")
        with r2[1]:
            top_n = st.select_slider("Results", options=[30, 60, 90, 120], value=60, key="flt_topn")
        with r2[2]:
            st.caption("Free to audit means course access is free; certificate may be paid.")

    with st.container(border=True):
        st.markdown("**Search**")
        
        # Initialize search query value if not present
        if "search_query_value" not in st.session_state:
            st.session_state.search_query_value = ""
        
        q = st.text_input(
            "Search",
            value=st.session_state.search_query_value,
            key="main_query",
            placeholder="Try: data structures, NLP, system design, React, machine learning",
            label_visibility="collapsed",
        )

        btn_l, btn_r = st.columns([1, 1])
        with btn_l:
            search_clicked = st.button("Search", type="primary", use_container_width=True)
        with btn_r:
            clear_clicked = st.button("Clear", use_container_width=True)

        if clear_clicked:
            st.session_state.search_query_value = ""
            st.session_state.live_results = pd.DataFrame()
            st.session_state.live_query_info = {}
            st.session_state.live_page = 0
            st.rerun()

        # Dynamic suggestions based on query/profile
        current_q = (st.session_state.get("main_query", "") or "").strip()
        if current_q != st.session_state.get("last_query", ""):
            st.session_state.last_query = current_q
            st.session_state.dynamic_suggestions = generate_suggestions(
                current_q or "machine learning",
                st.session_state.profile,
                n=4,
            )

        suggestions = st.session_state.dynamic_suggestions or generate_suggestions(
            "machine learning", st.session_state.profile, n=4
        )

        st.markdown("<div class='nlprec-muted' style='margin-top: 10px;'>Suggested</div>", unsafe_allow_html=True)
        sug_cols = st.columns(4)
        triggered_preset = None
        for i, preset in enumerate((suggestions or [])[:4]):
            with sug_cols[i]:
                if st.button(_truncate(preset, 28), key=f"sug_{i}", use_container_width=True, type="secondary"):
                    triggered_preset = preset

        trending = get_trending_chips(4)
        if trending:
            st.markdown("<div class='nlprec-muted' style='margin-top: 10px;'>Trending</div>", unsafe_allow_html=True)
            tr_cols = st.columns(4)
            for i, chip in enumerate(trending[:4]):
                with tr_cols[i]:
                    if st.button(_truncate(chip, 28), key=f"tr_{i}", use_container_width=True, type="secondary"):
                        triggered_preset = chip

        effective_query = (triggered_preset or q or "").strip()

        # Stage search to enable skeleton loading
        if (search_clicked or triggered_preset) and effective_query:
            # Store the original query for display, then enrich if personalization is on
            original_query = effective_query
            if personalize:
                effective_query = enrich_query(profile, effective_query)
            # Pass both original and enriched query
            st.session_state.pending_search_query = effective_query
            st.session_state.pending_search_original = original_query
            st.rerun()
        elif (search_clicked or triggered_preset) and not effective_query:
            st.warning("Enter a topic to search.")

        # Pending search run (shows skeleton first, then executes)
        pending = st.session_state.get("pending_search_query")
        if pending:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            # Progress indicator at the top
            prog_placeholder = st.empty()
            status_placeholder = st.empty()
            with prog_placeholder.container():
                prog_bar = st.progress(0)
            with status_placeholder.container():
                status_txt = st.empty()
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            _render_skeleton_grid(total_cards=9, cols=1)
            try:
                original = st.session_state.get("pending_search_original", pending)
                _run_live_search(pending, top_n=top_n, difficulty=difficulty, prog_bar=prog_bar, status_txt=status_txt, original_query=original)
                st.session_state.last_search_error = None
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.session_state.last_search_error = {"message": str(e), "details": error_details}
                st.session_state.live_results = pd.DataFrame()
                st.session_state.live_query_info = {}
            finally:
                st.session_state.pending_search_query = None
                st.session_state.pending_search_original = None
                prog_placeholder.empty()
                status_placeholder.empty()
                st.rerun()

        # Results
        df_live = st.session_state.live_results
        query_info = st.session_state.live_query_info

        if not df_live.empty and "price" not in df_live.columns:
            df_live["price"] = "Free*"
        if not df_live.empty and "duration" not in df_live.columns:
            df_live["duration"] = ""

        correction = query_info.get("display_correction") if isinstance(query_info, dict) else None
        if correction:
            st.info(f"Interpreted as: {correction}")
        
        # Display any search errors
        if st.session_state.last_search_error:
            err = st.session_state.last_search_error
            st.error(f"Search failed: {err['message']}")
            with st.expander("Technical details"):
                st.code(err['details'])

        if df_live.empty:
            st.warning("No courses found. Try different keywords or filters.")
            return

        # Apply local filters (price/source/min_rating)
        display_df = df_live.copy()
        if source_filter and source_filter != "All" and "source" in display_df.columns:
            display_df = display_df[display_df["source"] == source_filter]
        if min_rating and min_rating > 0 and "rating" in display_df.columns:
            display_df = display_df[display_df["rating"] >= min_rating]

        if price_sel == "Free":
            display_df = display_df[display_df["price"].isin(["Free", "Free*"])]
        elif price_sel == "Paid":
            display_df = display_df[display_df["price"] == "Paid"]

        total = len(display_df)
        if total == 0:
            st.warning("No courses match the current filters.")
            return

        PAGE_SIZE = 10
        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        current_pg = min(st.session_state.get("live_page", 0), total_pages - 1)

        top_a, top_b, top_c = st.columns([2, 3, 2], vertical_alignment="center")
        with top_a:
            st.markdown(f'<div class="nlprec-muted">{total} results</div>', unsafe_allow_html=True)
        with top_c:
            nav_l, nav_r = st.columns(2)
            with nav_l:
                if st.button("Prev", disabled=(current_pg == 0), use_container_width=True):
                    st.session_state.live_page = current_pg - 1
                    st.rerun()
            with nav_r:
                if st.button("Next", disabled=(current_pg >= total_pages - 1), use_container_width=True):
                    st.session_state.live_page = current_pg + 1
                    st.rerun()
        with top_b:
            st.markdown(
                f'<div class="nlprec-muted" style="text-align:center;">Page {current_pg + 1} of {total_pages}</div>',
                unsafe_allow_html=True,
            )

        start_idx = current_pg * PAGE_SIZE
        end_idx = start_idx + PAGE_SIZE
        page_df = display_df.iloc[start_idx:end_idx].reset_index(drop=True)

        for i, (_, row) in enumerate(page_df.iterrows(), start=start_idx + 1):
            render_course_card(row, index=i, saved_titles=saved_titles, key_prefix="live")
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


def _render_saved(profile: dict):
    saved_entries, saved_titles = _normalize_saved_courses(profile)
    _app_header("Saved", "Courses you have bookmarked.")

    if not saved_entries:
        st.info("No saved courses yet.")
        return

    # Backfill metadata from local TF-IDF dataset if we only have titles
    courses_df = None
    try:
        from vectorizer import load_tfidf_model
        _, _, courses_df = load_tfidf_model()
    except Exception:
        courses_df = None

    for idx, entry in enumerate(saved_entries, start=1):
        title = entry.get("title", "")
        if not title:
            continue

        row_data = {
            "course_title": title,
            "url": entry.get("url", "#"),
            "description": entry.get("description", ""),
            "difficulty": entry.get("difficulty", "Intermediate"),
            "source": entry.get("source", ""),
            "rating": entry.get("rating", 0.0),
            "duration": entry.get("duration", ""),
            "price": entry.get("price", ""),
            "similarity_score": 0.0,
        }
        if (not row_data.get("description") or row_data.get("url") in ("", "#")) and courses_df is not None:
            hit = courses_df[courses_df.get("course_title") == title]
            if not hit.empty:
                r = hit.iloc[0]
                row_data.update({
                    "url": r.get("url", row_data.get("url", "#")),
                    "description": r.get("description", row_data.get("description", "")),
                    "difficulty": r.get("difficulty", row_data.get("difficulty", "Intermediate")),
                    "source": r.get("source", row_data.get("source", "")),
                    "rating": r.get("rating", row_data.get("rating", 0.0)),
                })

        render_course_card(row_data, index=idx, saved_titles=saved_titles, key_prefix="saved")
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


def _render_model_comparison(profile: dict):
    _app_header("Model comparison", "Compare semantic ranking against keyword matching.")

    with st.container(border=True):
        cmp_query = st.text_input(
            "Query",
            placeholder="Try: machine learning for beginners without math",
            key="cmp_query",
            label_visibility="collapsed",
        )
        run = st.button("Compare", type="primary")

    if run and (cmp_query or "").strip():
        st.session_state.pending_compare_query = cmp_query.strip()
        st.rerun()
    if run and not (cmp_query or "").strip():
        st.warning("Enter a query to compare.")

    pending = st.session_state.get("pending_compare_query")
    if pending:
        _render_skeleton_grid(total_cards=6, cols=1)
        try:
            nlp_res = recommend(pending, top_n=10, difficulty_filter="All")
            kw_res = keyword_search(pending, top_n=10, difficulty_filter="All")
            st.session_state.last_results_nlp = nlp_res
            st.session_state.last_results_kw = kw_res
        except Exception as e:
            st.error(f"Comparison failed: {e}")
        finally:
            st.session_state.pending_compare_query = None
            st.rerun()

    nlp_res = st.session_state.last_results_nlp
    kw_res = st.session_state.last_results_kw
    if nlp_res.empty and kw_res.empty:
        st.caption("Run a comparison to see results.")
        return

    saved_entries, saved_titles = _normalize_saved_courses(profile)
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("**NLP semantic**")
        st.caption("TF-IDF cosine similarity")
        for i, (_, row) in enumerate(nlp_res.iterrows(), start=1):
            render_course_card(row, index=i, saved_titles=saved_titles, key_prefix="cmp_nlp")

    with col_b:
        st.markdown("**Keyword baseline**")
        st.caption("Token overlap matching")
        for i, (_, row) in enumerate(kw_res.iterrows(), start=1):
            render_course_card(row, index=i, saved_titles=saved_titles, key_prefix="cmp_kw")


def _render_performance():
    _app_header("Performance", "Evaluation metrics and research plots.")

    with st.container(border=True):
        k_val = st.slider("Top-K", 3, 10, 5, key="eval_k")
        run_eval_btn = st.button("Run evaluation", type="primary")

    if not run_eval_btn and st.session_state.eval_results is None and os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                st.session_state.eval_results = json.load(f)
        except Exception:
            st.session_state.eval_results = None

    if run_eval_btn:
        with st.spinner(f"Evaluating at K={k_val}..."):
            nlp_eval, base_eval = run_evaluation(k=k_val, save=True)
            st.session_state.eval_results = {"nlp": nlp_eval, "baseline": base_eval}

    ev = st.session_state.eval_results
    if not ev:
        st.caption("Run evaluation to compute metrics.")
        return

    nlp_ev = ev["nlp"]
    base_ev = ev["baseline"]
    k_used = nlp_ev["k"]

    c1, c2, c3 = st.columns(3, gap="large")
    for col, metric, label in [
        (c1, "precision_mean", f"Precision@{k_used}"),
        (c2, "recall_mean", f"Recall@{k_used}"),
        (c3, "f1_mean", f"F1@{k_used}"),
    ]:
        nlp_val = nlp_ev[metric]
        base_val = base_ev[metric]
        delta = nlp_val - base_val
        col.metric(f"NLP {label}", f"{nlp_val:.4f}", delta=f"{delta:+.4f} vs baseline")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    pcol, rcol = st.columns(2, gap="large")
    with pcol:
        with st.container(border=True):
            st.markdown("**Bar comparison**")
            fig = plot_comparison(nlp_ev, base_ev, save=True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    with rcol:
        with st.container(border=True):
            st.markdown("**Radar**")
            fig2 = plot_metric_radar(nlp_ev, base_ev, save=True)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    with st.container(border=True):
        st.markdown("**Per-query heatmap**")
        fig3 = plot_per_query_heatmap(nlp_ev, save=True)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    with st.container(border=True):
        st.markdown("**Per-query results**")
        pq_df = pd.DataFrame(nlp_ev["per_query"])[["query", "precision", "recall", "f1"]]
        pq_df.columns = ["Query", "Precision", "Recall", "F1"]
        st.dataframe(pq_df, use_container_width=True)


# ── App entry ───────────────────────────────────────────────────────────────
page, profile, stats = _render_sidebar()

if page == "Discover":
    _render_discover(profile)
elif page == "Saved":
    _render_saved(profile)
elif page == "Model comparison":
    _render_model_comparison(profile)
elif page == "Performance":
    _render_performance()
