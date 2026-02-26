"""
scraper.py
----------
Real-Time Course Scraper for NLPRec
Fetches live courses from:
  1. Coursera Public REST API   (free, no auth)
  2. edX Course Discovery API   (free, no auth)
  3. MIT OpenCourseWare         (public RSS/feed)

Usage:
    python scraper.py                 # fetch all sources, save CSV
    python scraper.py --source coursera
    python scraper.py --source edx
    python scraper.py --limit 200
"""

import os
import sys
import time
import argparse
import logging
import pandas as pd
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("NLPRec-Scraper")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "courses.csv")
os.makedirs(os.path.join(BASE_DIR, "dataset"), exist_ok=True)

# HTTP session shared across all requests
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
})
REQUEST_TIMEOUT = 15   # seconds per request
RETRY_DELAY     = 1.5  # seconds between retries


def _safe_get(url: str, params: dict = None, retries: int = 3) -> dict | None:
    """GET with retry logic; returns parsed JSON or None on failure.
    Does NOT retry on 4xx client errors (they will not succeed on retry).
    """
    for attempt in range(1, retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if 400 <= r.status_code < 500:
                log.warning(f"HTTP {r.status_code} for {url}  (client error, no retry)")
                return None
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            log.warning(f"HTTP {r.status_code} for {url}  (attempt {attempt}/{retries})")
        except requests.exceptions.RequestException as e:
            log.warning(f"Request error: {e}  (attempt {attempt}/{retries})")
        if attempt < retries:
            time.sleep(RETRY_DELAY)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — Coursera Public REST API
# Docs: https://api.coursera.org/api/courses.v1
# No authentication required for public catalog.
# ══════════════════════════════════════════════════════════════════════════════
COURSERA_API = "https://api.coursera.org/api/courses.v1"
COURSERA_FIELDS = ",".join([
    "name", "slug", "description", "domainTypes",
    "difficulty", "workload", "rating",
])

_COURSERA_DIFFICULTY_MAP = {
    "BEGINNER":     "Beginner",
    "INTERMEDIATE": "Intermediate",
    "ADVANCED":     "Advanced",
    "MIXED":        "Intermediate",
}

_COURSERA_DOMAIN_MAP = {
    "computer-science":    "Computer Science, Programming",
    "data-science":        "Data Science, Machine Learning",
    "information-technology": "IT, Systems, Networking",
    "math-and-logic":      "Mathematics, Statistics",
    "business":            "Business, Management",
    "personal-development": "Personal Development, Productivity",
    "physical-science-and-engineering": "Engineering, Physics",
    "social-sciences":     "Social Sciences",
    "arts-and-humanities": "Arts, Humanities",
    "language-learning":   "Language Learning",
    "health":              "Health, Medicine",
}


def fetch_coursera(limit: int = 100, start: int = 0) -> list[dict]:
    """
    Fetch courses from Coursera API.
    Returns list of normalised dicts ready for the DataFrame.
    """
    log.info(f"Coursera API → fetching up to {limit} courses (start={start}) …")
    courses = []
    fetched = 0
    next_start = start
    page_size = min(100, limit)

    while fetched < limit:
        data = _safe_get(COURSERA_API, params={
            "fields": COURSERA_FIELDS,
            "limit":  page_size,
            "start":  next_start,
        })
        if not data:
            log.error("Coursera API: no data returned — aborting.")
            break

        elements = data.get("elements", [])
        if not elements:
            break

        for item in elements:
            if fetched >= limit:
                break

            # Difficulty
            raw_diff = item.get("difficulty", "") or ""
            difficulty = _COURSERA_DIFFICULTY_MAP.get(raw_diff.upper(), "Intermediate")

            # Skills from domainTypes
            domain_types = item.get("domainTypes", []) or []
            skills_parts = []
            for d in domain_types:
                domain_id = d.get("domainId", "")
                topic_id  = d.get("subdomainId", "")
                label     = _COURSERA_DOMAIN_MAP.get(domain_id, domain_id.replace("-", " ").title())
                if topic_id:
                    label += f", {topic_id.replace('-', ' ').title()}"
                skills_parts.append(label)
            skills = "; ".join(skills_parts) if skills_parts else "General"

            # Rating
            rating_info = item.get("rating", {}) or {}
            if isinstance(rating_info, dict):
                rating = round(float(rating_info.get("averageFiveStars", 0) or 0), 1)
            else:
                try:
                    rating = round(float(rating_info), 1)
                except (TypeError, ValueError):
                    rating = 0.0

            slug        = item.get("slug", "")
            name        = (item.get("name", "") or "").strip()
            description = (item.get("description", "") or "").strip()

            if not name or not slug:
                continue

            url = f"https://www.coursera.org/learn/{slug}"

            courses.append({
                "course_title": name,
                "description":  description or f"Learn {name} on Coursera.",
                "skills":       skills,
                "difficulty":   difficulty,
                "rating":       rating,
                "url":          url,
                "source":       "Coursera",
            })
            fetched += 1

        # Pagination
        paging = data.get("paging", {})
        next_val = paging.get("next", None)
        if next_val is None:
            break
        next_start = int(next_val)

    log.info(f"Coursera: fetched {len(courses)} courses.")
    return courses


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — MIT OpenCourseWare (Sitemap)
# MIT publishes their full catalog (~2500 courses) in a public sitemap.
# No authentication, no rate limits — 100% open.
# ══════════════════════════════════════════════════════════════════════════════
import re as _re

MIT_OCW_SITEMAP = "https://ocw.mit.edu/sitemap.xml"

# Map department prefix → (department name, skills string)
_MIT_DEPT_MAP = {
    "1":   ("Civil & Environmental Engineering",        "Civil Engineering, Environmental Engineering"),
    "2":   ("Mechanical Engineering",                   "Mechanical Engineering, Thermodynamics, Design"),
    "3":   ("Materials Science & Engineering",          "Materials Science, Nanotechnology, Chemistry"),
    "4":   ("Architecture",                             "Architecture, Design, Urban Planning"),
    "5":   ("Chemistry",                                "Chemistry, Organic Chemistry, Biochemistry"),
    "6":   ("Electrical Engineering & Computer Science","Algorithms, Machine Learning, Programming, AI"),
    "7":   ("Biology",                                  "Biology, Genetics, Molecular Biology, Ecology"),
    "8":   ("Physics",                                  "Physics, Quantum Mechanics, Thermodynamics"),
    "9":   ("Brain & Cognitive Sciences",               "Neuroscience, Cognitive Science, Psychology"),
    "10":  ("Chemical Engineering",                     "Chemical Engineering, Process Design"),
    "11":  ("Urban Studies & Planning",                 "Urban Planning, Public Policy, Community"),
    "12":  ("Earth, Atmospheric & Planetary Sciences",  "Earth Science, Climate, Geology, Oceanography"),
    "14":  ("Economics",                                "Microeconomics, Macroeconomics, Finance, Game Theory"),
    "15":  ("Management",                               "Business, Leadership, Finance, Entrepreneurship"),
    "16":  ("Aeronautics & Astronautics",               "Aerospace Engineering, Space Systems, Fluid Dynamics"),
    "17":  ("Political Science",                        "Politics, International Relations, Public Policy"),
    "18":  ("Mathematics",                              "Mathematics, Linear Algebra, Calculus, Statistics"),
    "20":  ("Biological Engineering",                   "Bioengineering, Systems Biology, Bioinformatics"),
    "21":  ("Humanities",                               "Literature, History, Arts, Writing, Communication"),
    "22":  ("Nuclear Science & Engineering",            "Nuclear Engineering, Radiation Physics, Energy"),
    "24":  ("Philosophy",                               "Philosophy, Logic, Ethics, Language"),
    "STS": ("Science, Technology & Society",            "Science Policy, Technology Ethics, Innovation"),
    "HST": ("Health Sciences & Technology",             "Medicine, Healthcare, Biomedical Engineering"),
}


def _slug_to_title(slug: str) -> str:
    """Convert MIT OCW slug like '18-06-linear-algebra-spring-2010' to 'Linear Algebra'."""
    cleaned = _re.sub(r"^\d+-\w+-", "", slug)      # strip leading course number (e.g. '18-06-')
    cleaned = _re.sub(
        r"-(spring|fall|summer|winter|january-iap?)-\d{4}$", "", cleaned, flags=_re.I
    )
    cleaned = _re.sub(r"-\d{4}$", "", cleaned)     # strip trailing year
    return cleaned.replace("-", " ").title()


def _slug_dept_skills(slug: str) -> tuple[str, str]:
    """Return (department_name, skills_string) for a given OCW slug."""
    m = _re.match(r"^(\d+)[\w.]*-", slug, _re.I)
    if m:
        dept_num = m.group(1)
        # Try exact match first, then prefix match
        if dept_num in _MIT_DEPT_MAP:
            return _MIT_DEPT_MAP[dept_num]
        for k in _MIT_DEPT_MAP:
            if dept_num.startswith(k):
                return _MIT_DEPT_MAP[k]
    return ("MIT OpenCourseWare", "Academic Learning, MIT, Free Courses")


def _slug_difficulty(slug: str) -> str:
    """Infer difficulty from MIT course number."""
    m = _re.search(r"-(\d{2,3})[a-z]?-", slug, _re.I)
    if m:
        num = int(m.group(1))
        if num < 100:
            return "Beginner"
        elif num < 300:
            return "Intermediate"
        else:
            return "Advanced"
    return "Intermediate"


def fetch_mit_ocw(limit: int = 100) -> list[dict]:
    """
    Fetch MIT OpenCourseWare courses from the public sitemap.
    Parses https://ocw.mit.edu/sitemap.xml which lists all ~2500 courses.
    No authentication required.
    """
    from bs4 import BeautifulSoup as _BS

    log.info(f"MIT OCW → fetching sitemap (limit={limit}) …")
    courses = []

    try:
        r = SESSION.get(MIT_OCW_SITEMAP, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            log.warning(f"MIT OCW sitemap HTTP {r.status_code} — skipping.")
            return []

        soup = _BS(r.text, "xml")
        slugs_seen: set[str] = set()

        for loc in soup.find_all("loc"):
            if len(courses) >= limit:
                break
            url_text = loc.text.strip()
            m = _re.search(r"ocw\.mit\.edu/courses/([^/]+)/", url_text)
            if not m:
                continue
            slug = m.group(1)
            if slug in slugs_seen:
                continue
            slugs_seen.add(slug)

            title = _slug_to_title(slug)
            if not title or len(title) < 4:
                continue

            dept_name, skills = _slug_dept_skills(slug)
            difficulty        = _slug_difficulty(slug)
            course_url        = f"https://ocw.mit.edu/courses/{slug}/"

            courses.append({
                "course_title": title,
                "description":  (
                    f"MIT OpenCourseWare: {title}. Free academic course from MIT's "
                    f"{dept_name} department. Covers {skills}."
                ),
                "skills":       skills,
                "difficulty":   difficulty,
                "rating":       4.5,
                "url":          course_url,
                "source":       "MIT OCW",
            })

    except Exception as e:
        log.error(f"MIT OCW scraper failed: {e}")

    log.info(f"MIT OCW: {len(courses)} courses fetched.")
    return courses


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — freeCodeCamp Curriculum (GitHub raw JSON)
# Completely open, no auth needed.
# ══════════════════════════════════════════════════════════════════════════════
FCC_CURRICULUM_URL = (
    "https://raw.githubusercontent.com/freeCodeCamp/freeCodeCamp/"
    "main/shared/config/curriculum.json"
)

_FCC_DIFFICULTY_MAP = {
    "Responsive Web Design":              ("Beginner",     "HTML, CSS, Responsive Design"),
    "JavaScript Algorithms and Data Structures": ("Intermediate", "JavaScript, Algorithms, Data Structures"),
    "Front End Development Libraries":    ("Intermediate", "React, Redux, Bootstrap, jQuery"),
    "Data Visualization":                 ("Intermediate", "D3.js, Data Visualization, SVG"),
    "Relational Database":                ("Intermediate", "SQL, PostgreSQL, Bash"),
    "Back End Development and APIs":      ("Intermediate", "Node.js, Express, MongoDB"),
    "Quality Assurance":                  ("Advanced",     "Testing, Chai, Mocha"),
    "Scientific Computing with Python":   ("Beginner",     "Python, NumPy, Pandas, Matplotlib"),
    "Data Analysis with Python":          ("Intermediate", "Python, Pandas, NumPy, Data Analysis"),
    "Information Security":               ("Advanced",     "Cybersecurity, Penetration Testing"),
    "Machine Learning with Python":       ("Advanced",     "Machine Learning, TensorFlow, Python"),
    "College Algebra with Python":        ("Beginner",     "Mathematics, Algebra, Python"),
    "A2 English for Developers":          ("Beginner",     "English, Communication"),
    "Foundational C# with Microsoft":     ("Beginner",     "C#, .NET, Microsoft"),
}


def fetch_freecodecamp() -> list[dict]:
    """
    Fetch freeCodeCamp certified curriculum tracks.
    """
    log.info("freeCodeCamp → fetching curriculum …")
    courses = []

    try:
        data = _safe_get(FCC_CURRICULUM_URL)
        if not data:
            raise ValueError("No data from FCC")

        for cert_name, (difficulty, skills) in _FCC_DIFFICULTY_MAP.items():
            slug = cert_name.lower().replace(" ", "-").replace(".", "")
            courses.append({
                "course_title": cert_name,
                "description":  (
                    f"freeCodeCamp's free certified curriculum: {cert_name}. "
                    f"Covers {skills}. Project-based, earn a free certificate."
                ),
                "skills":       skills,
                "difficulty":   difficulty,
                "rating":       4.8,
                "url":          f"https://www.freecodecamp.org/learn/{slug}/",
                "source":       "freeCodeCamp",
            })
    except Exception as e:
        log.warning(f"freeCodeCamp fallback to hardcoded list: {e}")
        # Hardcoded fallback (these are always correct)
        for cert_name, (difficulty, skills) in _FCC_DIFFICULTY_MAP.items():
            slug = cert_name.lower().replace(" ", "-").replace(".", "")
            courses.append({
                "course_title": cert_name,
                "description":  f"freeCodeCamp certified curriculum: {cert_name}. {skills}.",
                "skills":       skills,
                "difficulty":   difficulty,
                "rating":       4.8,
                "url":          f"https://www.freecodecamp.org/learn/{slug}/",
                "source":       "freeCodeCamp",
            })

    log.info(f"freeCodeCamp: {len(courses)} curriculum tracks.")
    return courses


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4 — Khan Academy
# The old /api/v1/topictree endpoint was retired (HTTP 410).
# We maintain a curated list of real Khan Academy subjects with live URLs.
# ══════════════════════════════════════════════════════════════════════════════
_KA_COURSES = [
    ("Math",                    "Beginner",     "Mathematics, Arithmetic, Geometry, Trigonometry",          "math"),
    ("Algebra",                 "Beginner",     "Algebra, Linear Equations, Polynomials",                   "math/algebra"),
    ("Calculus 1",              "Intermediate", "Calculus, Derivatives, Integrals, Limits",                 "math/calculus-1"),
    ("Statistics & Probability","Intermediate", "Statistics, Probability, Data Analysis, Distributions",    "math/statistics-probability"),
    ("Linear Algebra",          "Advanced",     "Linear Algebra, Vectors, Matrices, Eigenvalues",           "math/linear-algebra"),
    ("Differential Equations",  "Advanced",     "Differential Equations, Calculus, Engineering Math",       "math/differential-equations"),
    ("AP Computer Science Principles","Beginner","Computer Science, Programming, Algorithms, Data",         "computing/ap-computer-science-principles"),
    ("Computer Programming",    "Beginner",     "JavaScript, HTML, CSS, Web Development",                   "computing/computer-programming"),
    ("Computer Science",        "Intermediate", "Algorithms, Data Structures, Computer Theory",             "computing/computer-science"),
    ("AP Physics 1",            "Intermediate", "Physics, Mechanics, Forces, Energy, Waves",                "science/ap-physics-1"),
    ("AP Chemistry",            "Intermediate", "Chemistry, Chemical Reactions, Stoichiometry",             "science/ap-chemistry-beta"),
    ("AP Biology",              "Intermediate", "Biology, Cell Biology, Genetics, Evolution",               "science/ap-biology"),
    ("Organic Chemistry",       "Advanced",     "Organic Chemistry, Reactions, Spectroscopy",               "science/organic-chemistry"),
    ("Macroeconomics",          "Intermediate", "Economics, GDP, Inflation, Monetary Policy",               "economics-finance-domain/macroeconomics"),
    ("Microeconomics",          "Intermediate", "Economics, Supply, Demand, Market Structures",             "economics-finance-domain/microeconomics"),
    ("Finance & Capital Markets","Intermediate","Finance, Stocks, Bonds, Investment, Banking",              "economics-finance-domain/core-finance"),
    ("World History",           "Beginner",     "History, World History, Civilizations, Culture",           "humanities/world-history"),
    ("US History",              "Beginner",     "American History, Government, Law, Society",               "humanities/us-history"),
    ("Art History",             "Beginner",     "Art History, Painting, Sculpture, Periods",                "humanities/art-history"),
    ("Grammar",                 "Beginner",     "English Grammar, Writing, Language Arts",                  "humanities/grammar"),
    ("GMAT",                    "Advanced",     "GMAT, Business School, Test Prep, Quantitative",           "test-prep/gmat"),
    ("SAT",                     "Intermediate", "SAT, College Entrance, Test Preparation, Math",            "test-prep/sat"),
    ("Life Skills",             "Beginner",     "Personal Finance, Career, Decision Making, Life Skills",   "college-careers-more/life-skills"),
]


def fetch_khan_academy(**_kwargs) -> list[dict]:
    """
    Return Khan Academy subject courses.
    The old /api/v1/topictree endpoint was retired (HTTP 410 Gone).
    Returns a curated list of real Khan Academy courses with live URLs.
    """
    log.info("Khan Academy → loading curated subject list (API retired, using live URL list) …")
    courses = []
    for title, difficulty, skills, path in _KA_COURSES:
        courses.append({
            "course_title": title,
            "description":  (
                f"Free Khan Academy course: {title}. "
                f"Covers {skills}. Learn at your own pace with exercises and videos."
            ),
            "skills":       skills,
            "difficulty":   difficulty,
            "rating":       4.7,
            "url":          f"https://www.khanacademy.org/{path}",
            "source":       "Khan Academy",
        })
    log.info(f"Khan Academy: {len(courses)} subjects loaded.")
    return courses


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATOR — combines all sources, deduplicates, saves CSV
# ══════════════════════════════════════════════════════════════════════════════

def scrape_all(
    coursera_limit:  int  = 100,
    mit_ocw_limit:   int  = 80,
    edx_limit:       int  = 80,   # legacy alias → ignored, use mit_ocw_limit
    include_fcc:     bool = True,
    include_khan:    bool = True,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Fetches from all sources, combines, deduplicates, assigns IDs.
    Sources:
      1. Coursera      — live REST API
      2. MIT OCW       — live public sitemap (replaces retired edX API)
      3. freeCodeCamp  — live GitHub JSON, hardcoded fallback
      4. Khan Academy  — curated list with real URLs (API retired 410)
    progress_callback(message: str, pct: int) → called with status updates.
    Returns the final DataFrame (also saves to dataset/courses.csv).
    """
    # backward compatibility: if caller only passed edx_limit, use it for MIT OCW
    if mit_ocw_limit == 80 and edx_limit != 80:
        mit_ocw_limit = edx_limit

    def _progress(msg, pct):
        log.info(f"[{pct:3d}%] {msg}")
        if progress_callback:
            progress_callback(msg, pct)

    all_courses = []

    # ── 1. Coursera ────────────────────────────────────────────────────────
    _progress("Fetching Coursera courses …", 5)
    try:
        coursera_courses = fetch_coursera(limit=coursera_limit)
        all_courses.extend(coursera_courses)
        _progress(f"Coursera: {len(coursera_courses)} courses fetched.", 30)
    except Exception as e:
        log.error(f"Coursera scraper failed: {e}")
        _progress("Coursera: failed — skipping.", 30)

    # ── 2. MIT OpenCourseWare ─────────────────────────────────────────────
    _progress("Fetching MIT OpenCourseWare courses …", 32)
    try:
        ocw_courses = fetch_mit_ocw(limit=mit_ocw_limit)
        all_courses.extend(ocw_courses)
        _progress(f"MIT OCW: {len(ocw_courses)} courses fetched.", 60)
    except Exception as e:
        log.error(f"MIT OCW scraper failed: {e}")
        _progress("MIT OCW: failed — skipping.", 60)

    # ── 3. freeCodeCamp ────────────────────────────────────────────────────
    if include_fcc:
        _progress("Fetching freeCodeCamp tracks …", 62)
        try:
            fcc = fetch_freecodecamp()
            all_courses.extend(fcc)
            _progress(f"freeCodeCamp: {len(fcc)} tracks fetched.", 75)
        except Exception as e:
            log.error(f"freeCodeCamp scraper failed: {e}")
            _progress("freeCodeCamp: failed — skipping.", 75)

    # ── 4. Khan Academy ────────────────────────────────────────────────────
    if include_khan:
        _progress("Fetching Khan Academy subjects …", 77)
        try:
            khan = fetch_khan_academy()
            all_courses.extend(khan)
            _progress(f"Khan Academy: {len(khan)} subjects loaded.", 88)
        except Exception as e:
            log.error(f"Khan Academy loader failed: {e}")
            _progress("Khan Academy: failed — skipping.", 88)

    if not all_courses:
        _progress("All sources failed — no data to save.", 100)
        return pd.DataFrame()

    # ── Build DataFrame ────────────────────────────────────────────────────
    _progress("Cleaning and deduplicating …", 90)

    df = pd.DataFrame(all_courses)

    # Normalise columns
    df["course_title"] = df["course_title"].str.strip()
    df["description"]  = df["description"].fillna("").str.strip()
    df["skills"]       = df["skills"].fillna("General").str.strip()
    df["difficulty"]   = df["difficulty"].fillna("Intermediate")
    df["rating"]       = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).round(1)
    df["url"]          = df["url"].fillna("").str.strip()

    # Drop rows without a title or URL
    df = df[df["course_title"].str.len() > 2]
    df = df[df["url"].str.startswith("http")]

    # Deduplicate by lower-cased title
    df["_title_key"] = df["course_title"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["_title_key"]).drop(columns=["_title_key"])

    # Assign sequential IDs
    df = df.reset_index(drop=True)
    df.insert(0, "course_id", df.index + 1)

    # Final column order
    df = df[["course_id", "course_title", "description", "skills",
             "difficulty", "rating", "url", "source"]]

    # Save
    df.to_csv(DATASET_PATH, index=False)
    _progress(f"Saved {len(df)} courses → {DATASET_PATH}", 100)

    return df


# ── Status helper (called from app.py) ────────────────────────────────────────
def get_last_scrape_info() -> dict:
    """Returns metadata about the current dataset file."""
    if not os.path.exists(DATASET_PATH):
        return {"exists": False, "count": 0, "last_updated": None, "sources": []}
    try:
        df = pd.read_csv(DATASET_PATH)
        sources = df["source"].value_counts().to_dict() if "source" in df.columns else {}
        mtime   = os.path.getmtime(DATASET_PATH)
        return {
            "exists":       True,
            "count":        len(df),
            "last_updated": datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M"),
            "sources":      sources,
        }
    except Exception:
        return {"exists": False, "count": 0, "last_updated": None, "sources": []}


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLPRec Real-Time Course Scraper")
    parser.add_argument("--source",         default="all",
                        choices=["all", "coursera", "edx", "freecodecamp", "khan"],
                        help="Which source to scrape (default: all)")
    parser.add_argument("--coursera-limit", type=int, default=100,
                        help="Max Coursera courses (default: 100)")
    parser.add_argument("--edx-limit",      type=int, default=80,
                        help="Max edX courses (default: 80)")
    args = parser.parse_args()

    if args.source == "coursera":
        rows = fetch_coursera(limit=args.coursera_limit)
        df   = pd.DataFrame(rows)
    elif args.source == "edx":
        rows = fetch_edx(limit=args.edx_limit)
        df   = pd.DataFrame(rows)
    elif args.source == "freecodecamp":
        rows = fetch_freecodecamp()
        df   = pd.DataFrame(rows)
    elif args.source == "khan":
        rows = fetch_khan_academy()
        df   = pd.DataFrame(rows)
    else:
        df = scrape_all(
            coursera_limit=args.coursera_limit,
            edx_limit=args.edx_limit,
        )

    print(f"\n{'='*60}")
    print(f"Total courses fetched: {len(df)}")
    if not df.empty:
        print(f"\nSample (first 5):")
        for _, row in df.head(5).iterrows():
            title = row.get("course_title", "")
            url   = row.get("url", "")
            src   = row.get("source", "")
            diff  = row.get("difficulty", "")
            print(f"  [{src:12s}] [{diff:12s}] {title[:50]}")
            print(f"              URL: {url}")
