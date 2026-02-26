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
from user_profile  import (load_profile, save_profile, log_search,
                            save_course, remove_course, enrich_query,
                            get_stats, clear_history)
from evaluation    import (run_evaluation, plot_comparison,
                            plot_per_query_heatmap, plot_metric_radar,
                            RESULTS_PATH)
from scraper       import scrape_all, get_last_scrape_info
from vectorizer    import build_and_save_tfidf
from live_search   import search_courses_live, results_to_df, PLATFORM_COLORS as _LIVE_PLATFORM_COLORS

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.05rem;
        margin-top: 0;
    }
    /* Course cards */
    .course-card {
        background: #f8f9ff;
        border-left: 5px solid #667eea;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .course-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-beginner     { background: #d4edda; color: #155724; }
    .badge-intermediate { background: #fff3cd; color: #856404; }
    .badge-advanced     { background: #f8d7da; color: #721c24; }
    .similarity-bar-fill {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    /* Sidebar */
    .sidebar-section { 
        background: #f0f2ff; 
        padding: 12px; 
        border-radius: 8px; 
        margin-bottom: 12px;
    }
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

_init_session()


# ── Difficulty badge helper ────────────────────────────────────────────────────
def _difficulty_badge(level: str) -> str:
    cls = {
        "Beginner":     "badge-beginner",
        "Intermediate": "badge-intermediate",
        "Advanced":     "badge-advanced",
    }.get(level, "badge-beginner")
    return f'<span class="badge {cls}">{level}</span>'


# ── Source badge helper ────────────────────────────────────────────────────────
_SOURCE_COLORS = {
    "Coursera":     ("#0056D2", "#fff"),
    "MIT OCW":      ("#8A0000", "#fff"),
    "freeCodeCamp": ("#0a0a23", "#99c9ff"),
    "Khan Academy": ("#14BF96", "#fff"),
}
# Merge in live-search platform colours (live_search has 30+ platforms)
_ALL_SOURCE_COLORS = {**_LIVE_PLATFORM_COLORS, **_SOURCE_COLORS}

def _source_badge(source: str) -> str:
    pair = _ALL_SOURCE_COLORS.get(source)
    if pair:
        bg, fg = pair
    else:
        bg, fg = "#6c757d", "#fff"
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 9px;'
        f'border-radius:10px;font-size:0.73rem;font-weight:600;'
        f'margin-right:5px">{source}</span>'
    )


# ── Render a single course card ───────────────────────────────────────────────
def render_course_card(row, index: int, saved_titles: list, show_save: bool = True):
    score_pct  = min(int(row["similarity_score"] * 100 / max(row["similarity_score"], 0.001)), 100)
    diff_badge = _difficulty_badge(row["difficulty"])
    src_badge  = _source_badge(row.get("source", "")) if row.get("source") else ""
    rating_val = float(row.get("rating", 0) or 0)
    stars      = "★" * int(rating_val) + "☆" * (5 - int(rating_val))
    rating_str = f"({rating_val}/5)" if rating_val > 0 else "(unrated)"

    st.markdown(f"""
    <div class="course-card">
        <div class="course-title">#{index} &nbsp; {row['course_title']}</div>
        <div style="margin:6px 0">
            {src_badge}{diff_badge}
            <span style="color:#f59e0b; font-size:0.9rem">{stars}</span>
            <span style="color:#6c757d; font-size:0.85rem"> &nbsp;{rating_str}</span>
            &nbsp;&nbsp;
            <span style="color:#667eea; font-size:0.85rem; font-weight:600">
                Similarity: {row['similarity_score']:.4f}
            </span>
        </div>
        <div style="background:#e9ecef; border-radius:4px; height:8px; margin:6px 0 8px 0">
            <div class="similarity-bar-fill" style="width:{score_pct}%"></div>
        </div>
        <div style="color:#495057; font-size:0.88rem; margin-bottom:6px">
            {row['description'][:220]}{'…' if len(str(row['description'])) > 220 else ''}
        </div>
        <div style="color:#6c757d; font-size:0.82rem">
            <b>Skills:</b> {row['skills']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 5])
    with col1:
        if show_save:
            is_saved = row["course_title"] in saved_titles
            btn_label = "★ Saved" if is_saved else "☆ Save"
            if st.button(btn_label, key=f"save_{index}_{row['course_title'][:15]}"):
                profile = st.session_state.profile
                if is_saved:
                    profile = remove_course(profile, row["course_title"])
                    st.toast(f"Removed: {row['course_title']}")
                else:
                    profile = save_course(profile, row["course_title"])
                    st.toast(f"Saved: {row['course_title']}")
                save_profile(profile)
                st.session_state.profile = profile
                st.rerun()
    with col2:
        st.markdown(f"[🔗 View Course]({row['url']})", unsafe_allow_html=False)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎓 NLPRec")
    st.markdown("*Intelligent Course Recommendations*")
    st.divider()

    # ── Live Data Section ─────────────────────────────────────────────────────
    st.markdown("### 🌐 Live Course Data")

    scrape_info = get_last_scrape_info()
    if scrape_info["exists"]:
        st.success(
            f"**{scrape_info['count']} courses** loaded  \n"
            f"Updated: {scrape_info['last_updated']}"
        )
        if scrape_info["sources"]:
            for src, cnt in scrape_info["sources"].items():
                src_bg, src_fg = _SOURCE_COLORS.get(src, ("#6c757d", "#fff"))
                st.markdown(
                    f'<span style="background:{src_bg};color:{src_fg};'
                    f'padding:1px 8px;border-radius:8px;font-size:0.75rem'
                    f';font-weight:600">{src}</span> {cnt} courses',
                    unsafe_allow_html=True,
                )
    else:
        st.warning("No course data yet. Fetch live courses below.")

    st.markdown("**Fetch Settings:**")
    coursera_lim  = st.slider("Coursera courses",   50, 300, 100, 50)
    mit_ocw_lim  = st.slider("MIT OCW courses",    20, 200,  80, 20)
    include_fcc  = st.checkbox("Include freeCodeCamp", value=True)
    include_khan = st.checkbox("Include Khan Academy", value=True)

    fetch_btn = st.button("🔄 Fetch Live Courses", type="primary", use_container_width=True)

    if fetch_btn:
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
                    st.error("Scraper returned no courses. Check your internet connection.")
                else:
                    # Rebuild TF-IDF with fresh data
                    _progress_cb("Building NLP model on fresh data …", 95)
                    build_and_save_tfidf(df_new)
                    invalidate_cache()          # clear in-memory model cache
                    st.session_state.scrape_log = scrape_log
                    prog_bar.progress(1.0)
                    status_txt.success(
                        f"✅ Done! {len(df_new)} courses fetched & NLP model rebuilt."
                    )
                    st.rerun()
            except Exception as e:
                st.error(f"Scraper error: {e}")

    if st.session_state.scrape_log:
        with st.expander("Last fetch log"):
            st.code("\n".join(st.session_state.scrape_log), language=None)

    st.divider()

    # ── User section ──────────────────────────────────────────────────────────
    username = st.text_input("Your name (to save history)", value="guest", key="username_input")
    if username != st.session_state.profile.get("username"):
        st.session_state.profile = load_profile(username)

    profile = st.session_state.profile
    stats   = get_stats(profile)

    st.markdown("### ⚙️ Preferences")
    difficulties = get_difficulties()
    default_diff = profile.get("preferred_difficulty", "All")
    diff_idx     = difficulties.index(default_diff) if default_diff in difficulties else 0
    difficulty   = st.selectbox("Difficulty Level", difficulties, index=diff_idx)

    st.markdown("**💰 Price**")
    price_filter = st.radio(
        "price_filter",
        ["All", "Free", "Paid"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
    )

    sources_list = get_sources()
    source_filter = st.selectbox("Platform", sources_list, index=0)

    min_rating   = st.slider("Minimum Rating ★", 0.0, 5.0, 0.0, 0.1)
    top_n        = st.select_slider("Results to fetch", options=[10, 20, 30, 40, 50], value=30)
    personalize  = st.toggle("Personalize with history", value=True)

    st.divider()
    st.markdown("### 📊 Your Stats")
    st.metric("Total Searches",  stats["total_searches"])
    st.metric("Saved Courses",   stats["saved_courses"])
    if stats["last_search"]:
        st.caption(f"Last: *{stats['last_search'][:40]}…*" if len(stats["last_search"]) > 40
                   else f"Last: *{stats['last_search']}*")

    st.divider()
    if st.button("🗑 Clear History"):
        profile = clear_history(profile)
        save_profile(profile)
        st.session_state.profile = profile
        st.success("History cleared!")
        st.rerun()

    if stats["top_interests"]:
        st.markdown("**Top Interests:**")
        st.markdown(" · ".join(f"`{i}`" for i in stats["top_interests"][-8:]))


# ════════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — Tabs
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">🎓 NLPRec</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent Course Recommendation using Natural Language Processing</p>',
            unsafe_allow_html=True)
st.divider()

tab_rec, tab_compare, tab_eval, tab_saved, tab_about = st.tabs([
    "🔍 Recommend",
    "⚖️ NLP vs Keyword",
    "📊 Evaluation",
    "🔖 Saved Courses",
    "ℹ️ About",
])

# ── TAB 1: Recommendations ────────────────────────────────────────────────────
with tab_rec:

    st.markdown("### 🌐 Real-Time Internet Course Search")
    st.caption(
        "Scans the entire internet — Coursera, Udemy, YouTube, edX, "
        "LinkedIn Learning, DataCamp, freeCodeCamp, MIT OCW, Harvard, "
        "Stanford, Pluralsight, Codecademy … and any site that teaches your topic. "
        "Handles typos, abbreviations, and natural language automatically."
    )

    query = st.text_area(
        "Ask about anything you want to learn",
        placeholder=(
            "e.g.  mchine lerning for beginers  |  "
            "bro i wanna build web apps  |  "
            "pythn django REST api  |  "
            "learn guitar from scratch"
        ),
        height=100,
        key="main_query",
    )

    col_btn, col_hint = st.columns([1, 3])
    with col_btn:
        search_clicked = st.button("🌐 Search Internet", type="primary", use_container_width=True)
    with col_hint:
        st.caption("💡 Enter ANY topic — type naturally, with or without typos, slang, or abbreviations.")

    # Quick-prompt chips
    st.markdown("**Try these:**")
    q_cols = st.columns(4)
    presets = [
        "how to make a game in Unity",
        "blockchain and Web3 development",
        "machine learning without math background",
        "learn guitar from scratch",
    ]
    triggered_preset = None
    for i, preset in enumerate(presets):
        with q_cols[i]:
            if st.button(preset, key=f"preset_{i}", use_container_width=True):
                triggered_preset = preset

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
                    price_filter=price_filter,
                    progress_callback=_live_prog,
                )
                df_live = results_to_df(raw_results)
                prog_bar.progress(1.0)
                status_txt.empty()

                profile = log_search(profile, effective_query, difficulty)
                save_profile(profile)
                st.session_state.profile         = profile
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

    # "Did you mean / Interpreted as" banner
    correction = query_info.get("display_correction")
    if correction:
        st.info(f"🔍 **Interpreted as:** {correction}", icon="💡")

    if not df_live.empty:
        PAGE_SIZE   = 10
        total       = len(df_live)
        total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        current_pg  = st.session_state.live_page  # 0-indexed

        st.success(
            f"Found **{total}** courses — Page **{current_pg + 1}** of **{total_pages}**"
        )

        # ── Pagination controls (top) ──────────────────────────────────────
        pg_cols = st.columns([1, 6, 1])
        with pg_cols[0]:
            if st.button("◀ Prev", disabled=(current_pg == 0), use_container_width=True):
                st.session_state.live_page -= 1
                st.rerun()
        with pg_cols[1]:
            # Page number pills
            pill_cols = st.columns(min(total_pages, 8))
            for pg_i in range(min(total_pages, 8)):
                with pill_cols[pg_i]:
                    label = f"**{pg_i+1}**" if pg_i == current_pg else str(pg_i + 1)
                    if st.button(label, key=f"pgbtn_{pg_i}", use_container_width=True):
                        st.session_state.live_page = pg_i
                        st.rerun()
        with pg_cols[2]:
            if st.button("Next ▶", disabled=(current_pg >= total_pages - 1), use_container_width=True):
                st.session_state.live_page += 1
                st.rerun()

        st.divider()

        # ── Courses for this page ─────────────────────────────────────────
        start_idx = current_pg * PAGE_SIZE
        end_idx   = start_idx + PAGE_SIZE
        page_df   = df_live.iloc[start_idx:end_idx]
        saved_titles = st.session_state.profile.get("saved_courses", [])

        for pos, (_, row) in enumerate(page_df.iterrows(), start=start_idx + 1):
            diff_badge  = _difficulty_badge(row["difficulty"])
            src_badge   = _source_badge(row.get("source", "")) if row.get("source") else ""
            price_val   = row.get("price", "Free*")
            price_color = {
                "Free":  ("#d4edda", "#155724"),
                "Free*": ("#cce5ff", "#004085"),
                "Paid":  ("#ffe8cc", "#7a3e00"),
            }.get(price_val, ("#e2e3e5", "#383d41"))
            price_label = {"Free": "✓ Free", "Free*": "◑ Free to Audit", "Paid": "$ Paid"}.get(price_val, price_val)
            price_badge = (
                f'<span class="badge" style="background:{price_color[0]};color:{price_color[1]};'
                f'border:1px solid {price_color[1]}22">'
                f'{price_label}</span>'
            )
            score      = float(row.get("similarity_score", 0))
            score_pct  = min(int(score * 100 / max(score, 0.001)), 100)
            title_disp = str(row["course_title"])[:90] + ("…" if len(str(row["course_title"])) > 90 else "")
            desc_disp  = str(row["description"])[:150] + ("…" if len(str(row["description"])) > 150 else "")

            st.markdown(
                f"""
                <div class="course-card" style="padding:10px 16px;margin-bottom:8px">
                    <div style="display:flex;align-items:baseline;gap:10px">
                        <span style="font-size:1.15rem;font-weight:800;color:#667eea;min-width:32px">#{pos}</span>
                        <span class="course-title" style="font-size:1rem">{title_disp}</span>
                    </div>
                    <div style="margin:5px 0 4px 42px">
                        {price_badge}{src_badge}{diff_badge}
                        <span style="color:#667eea;font-size:0.8rem;font-weight:600;margin-left:8px">Score: {score:.4f}</span>
                    </div>
                    <div style="color:#495057;font-size:0.85rem;margin:0 0 4px 42px">{desc_disp}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            lnk_col, save_col = st.columns([5, 1])
            with lnk_col:
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[🔗 Open Course]({row['url']})")
            with save_col:
                is_saved   = row["course_title"] in saved_titles
                btn_label  = "★ Saved" if is_saved else "☆ Save"
                if st.button(btn_label, key=f"save_live_{pos}_{str(row['course_title'])[:10]}"):
                    profile = st.session_state.profile
                    if is_saved:
                        profile = remove_course(profile, row["course_title"])
                        st.toast(f"Removed: {row['course_title'][:40]}")
                    else:
                        profile = save_course(profile, row["course_title"])
                        st.toast(f"Saved: {row['course_title'][:40]}")
                    save_profile(profile)
                    st.session_state.profile = profile
                    st.rerun()

        # ── Pagination controls (bottom) ───────────────────────────────────
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
                        f'<div class="course-card"><div class="course-title">'
                        f'#{int(row["rank"])} {row["course_title"]}</div>'
                        f'{diff_badge} &nbsp; Sim: <b>{row["similarity_score"]:.4f}</b> &nbsp; '
                        f'★{row["rating"]}</div>',
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
                        f'<div class="course-card" style="border-left-color:#DD8452">'
                        f'<div class="course-title">#{int(row["rank"])} {row["course_title"]}</div>'
                        f'{diff_badge} &nbsp; Matches: <b>{int(row["similarity_score"])}</b> &nbsp; '
                        f'★{row["rating"]}</div>',
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
    saved   = profile.get("saved_courses", [])

    st.markdown(f"### 🔖 Your Saved Courses  ({len(saved)})")

    if not saved:
        st.info("No saved courses yet. Hit ☆ Save on any recommendation to bookmark it here.")
    else:
        from vectorizer import load_tfidf_model
        _, _, courses_df = load_tfidf_model()

        for title in saved:
            row = courses_df[courses_df["course_title"] == title]
            if row.empty:
                continue
            row = row.iloc[0]
            badge = _difficulty_badge(row["difficulty"])
            st.markdown(
                f'<div class="course-card"><div class="course-title">{row["course_title"]}</div>'
                f'{badge} &nbsp; ★{row["rating"]}<br>'
                f'<span style="font-size:0.85rem;color:#495057">{row["description"][:150]}…</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            if st.button(f"Remove: {title[:40]}", key=f"rm_{title[:20]}"):
                profile = remove_course(profile, title)
                save_profile(profile)
                st.session_state.profile = profile
                st.rerun()


# ── TAB 5: About ──────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
## NLPRec — Intelligent Course Recommendation System

### What is NLPRec?
NLPRec is an AI-powered course recommendation engine that understands **natural language queries**
and matches them semantically to course descriptions using NLP techniques.

### How It Works

| Step | Module | What Happens |
|------|--------|-------------|
| 1 | `text_preprocessing.py` | Lowercasing → Tokenization → Stopword removal → Lemmatization |
| 2 | `vectorizer.py` | TF-IDF vectorization of corpus (courses) |
| 3 | `recommender.py` | Cosine similarity between user query and course vectors |
| 4 | `user_profile.py` | Query logging + personalization enrichment |
| 5 | `evaluation.py` | Precision@K, Recall@K, F1@K vs keyword baseline |

### NLP Pipeline
```
User Query
   ↓ Lowercase
   ↓ Remove punctuation/numbers
   ↓ Tokenize (NLTK punkt)
   ↓ Remove stopwords
   ↓ Lemmatize (WordNetLemmatizer)
   ↓ TF-IDF Vector
   ↓ Cosine Similarity with Course Corpus
   ↓ Ranked Recommendations
```

### Tech Stack
- **Python 3.10+**
- **scikit-learn** — TF-IDF, cosine similarity
- **NLTK** — tokenization, stopwords, lemmatization
- **pandas / numpy** — data processing
- **Streamlit** — web interface
- **matplotlib / seaborn / plotly** — evaluation charts

### Research Contribution
The evaluation module compares this NLP semantic approach against a keyword-matching baseline,
producing measurable improvements in **Precision@K**, **Recall@K**, and **F1@K** — the exact
comparison required for conference / journal publication.

---
*Built as part of Final Year Project — B.E. Computer Engineering*
""")
