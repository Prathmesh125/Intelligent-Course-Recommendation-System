"""
query_engine.py
---------------
Intelligent Query Understanding Engine for NLPRec
==================================================
Understands free-form user input regardless of:
  - Spelling errors ("mchine lerning", "pythn", "javascrpt")
  - Casual wording ("i wanna learn", "teach me", "how do i")
  - Abbreviations ("ML", "AI", "DS", "FE", "BE", "DL", "CV", "NLP")
  - Slang & informal ("bro i have no idea about coding but wanna build apps")
  - Mixed languages/intent ("blockchain se paise kamana", "ai kya hota hai")
  - Difficulty signals ("total noob", "already know basics", "expert level")

Pipeline:
  1. Normalize (lowercase, strip noise)
  2. Expand abbreviations & slang
  3. Spell-correct (only non-tech words; tech dict preserved)
  4. Extract intent: WHAT to learn, at WHAT level
  5. Expand with related terms for better search coverage
  6. Return structured result + cleaned search query
"""

import re
import logging
from functools import lru_cache

log = logging.getLogger("NLPRec-QueryEngine")

# ═══════════════════════════════════════════════════════════════════════════════
# TECH + COURSE VOCABULARY  (never spell-correct these)
# ═══════════════════════════════════════════════════════════════════════════════
_TECH_VOCAB = {
    # Programming languages
    "python", "javascript", "typescript", "java", "cpp", "c++", "csharp", "c#",
    "rust", "golang", "go", "swift", "kotlin", "php", "ruby", "scala", "r",
    "matlab", "perl", "haskell", "elixir", "clojure", "dart", "flutter",
    "lua", "zig", "solidity", "vyper", "move",
    # Common words that spellchecker gets wrong
    "apps", "app", "coding", "code", "coder", "build", "building",
    "science", "financial", "analytics", "analysis", "visualization",
    "beginner", "beginners", "intermediate", "advanced", "basics",
    "tutorial", "tutorials", "course", "courses", "learn", "learning",
    "project", "projects", "portfolio", "career", "job", "interview",
    # Web / Frontend
    "html", "css", "react", "reactjs", "vue", "vuejs", "angular", "svelte",
    "nextjs", "next", "nuxt", "gatsby", "webpack", "vite", "tailwind",
    "bootstrap", "jquery", "redux", "graphql", "rest", "api", "json", "xml",
    # Backend / Infra
    "nodejs", "node", "express", "django", "flask", "fastapi", "spring",
    "rails", "laravel", "dotnet", "aspnet", "grpc", "kafka", "rabbitmq",
    "redis", "nginx", "apache",
    # Databases
    "sql", "mysql", "postgresql", "postgres", "mongodb", "dynamodb",
    "cassandra", "elasticsearch", "sqlite", "prisma", "orm",
    # Cloud / DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "k8s", "terraform",
    "ansible", "jenkins", "cicd", "devops", "mlops", "linux", "bash",
    "git", "github", "gitlab",
    # AI / ML / Data
    "ml", "ai", "nlp", "cv", "dl", "rl", "llm", "gpt", "bert", "transformer",
    "tensorflow", "pytorch", "keras", "sklearn", "scikit",
    "pandas", "numpy", "matplotlib", "seaborn", "scipy", "statsmodels",
    "xgboost", "lightgbm", "catboost", "huggingface", "langchain",
    "openai", "stable", "diffusion", "midjourney",
    # Domains
    "blockchain", "web3", "defi", "nft", "crypto", "ethereum", "bitcoin",
    "solana", "polygon", "hardhat", "truffle",
    "cybersecurity", "pentest", "ctf", "oscp", "siem",
    "unity", "unreal", "godot", "gamedev", "blender", "threejs",
    "figma", "ux", "ui", "uiux", "canva", "adobe", "photoshop",
    "excel", "powerbi", "tableau", "looker", "dbt",
    # Certifications
    "aws", "gcp", "azure", "comptia", "cisco", "ccna", "pmp", "scrum",
    "agile", "itil",
}

# ═══════════════════════════════════════════════════════════════════════════════
# ABBREVIATIONS & SLANG EXPANSION TABLE
# ═══════════════════════════════════════════════════════════════════════════════
_EXPAND = {
    # Common abbreviations
    r"\bml\b":    "machine learning",
    r"\bai\b":    "artificial intelligence",
    r"\bdl\b":    "deep learning",
    r"\bnlp\b":   "natural language processing",
    r"\bcv\b":    "computer vision",
    r"\brl\b":    "reinforcement learning",
    r"\bds\b":    "data science",
    r"\bda\b":    "data analysis",
    r"\bde\b":    "data engineering",
    r"\bfe\b":    "frontend development",
    r"\bbe\b":    "backend development",
    r"\bfs\b":    "fullstack development",
    r"\bswe\b":   "software engineering",
    r"\bcs\b":    "computer science",
    r"\boop\b":   "object oriented programming",
    r"\bfp\b":    "functional programming",
    r"\bdsa\b":   "data structures and algorithms",
    r"\bos\b":    "operating systems",
    r"\bdbms\b":  "database management systems",
    r"\bcn\b":    "computer networks",
    r"\bse\b":    "software engineering",
    r"\bqa\b":    "quality assurance testing",
    r"\bpm\b":    "project management",
    r"\bui\b":    "user interface design",
    r"\bux\b":    "user experience design",
    r"\buiux\b":  "user interface and user experience design",
    r"\bba\b":    "business analysis",
    # Slang / informal
    r"\bwanna\b":         "want to",
    r"\bgonna\b":         "going to",
    r"\bgotta\b":         "got to",
    r"\bdunno\b":         "do not know",
    r"\bdontknow\b":      "do not know",
    r"\bngl\b":           "honestly",
    r"\bimo\b":           "in my opinion",
    r"\btbh\b":           "to be honest",
    r"\basap\b":          "as soon as possible",
    r"\bbtw\b":           "by the way",
    r"\bfyi\b":           "for your information",
    r"\btldr\b":          "summary",
    r"\bidk\b":           "i do not know",
    r"\bidc\b":           "i do not care",
    r"\bbro\b":           "",
    r"\bdude\b":          "",
    r"\bmate\b":          "",
    r"\bplz\b":           "please",
    r"\bpls\b":           "please",
    r"\bu\b":             "you",
    r"\bur\b":            "your",
    r"\br\b":             "are",
    r"\bw/o\b":           "without",
    r"\bw/\b":            "with",
    r"\bsth\b":           "something",
    r"\bsmth\b":          "something",
    r"\bbasically\b":     "",
    r"\bliterally\b":     "",
    r"\bactually\b":      "",
    r"\bjust\b":          "",
    r"\blike\b":          "",
    # Difficulty synonyms
    r"\btotal noob\b":    "absolute beginner",
    r"\bcomplete noob\b": "absolute beginner",
    r"\bnewbie\b":        "beginner",
    r"\bnoob\b":          "beginner",
    r"\bnewb\b":          "beginner",
    r"\bnewcomer\b":      "beginner",
    r"\bfrom zero\b":     "from scratch beginner",
    r"\bfrom scratch\b":  "beginner from scratch",
    r"\bzero to hero\b":  "beginner to advanced",
    r"\bintermediate\b":  "intermediate",
    r"\badvanced\b":      "advanced",
    r"\bpro\b":           "advanced professional",
    r"\bexpert\b":        "advanced expert",
    # Common misspellings that spell-check misses
    r"\bpythn\b":         "python",
    r"\bpyton\b":         "python",
    r"\bpyhton\b":        "python",
    r"\bpiton\b":         "python",
    r"\bmchine\b":        "machine",
    r"\bmachin\b":        "machine",
    r"\blerning\b":       "learning",
    r"\blernin\b":        "learning",
    r"\bblokchain\b":     "blockchain",
    r"\bblockchian\b":    "blockchain",
    r"\bblockchian\b":    "blockchain",
    r"\bjavascirpt\b":    "javascript",
    r"\bjavascritp\b":    "javascript",
    r"\bjavascrit\b":     "javascript",
    r"\bjsavscript\b":    "javascript",
    r"\bjscript\b":       "javascript",
    r"\breact js\b":      "reactjs react",
    r"\bnode js\b":       "nodejs node",
    r"\bnext js\b":       "nextjs next",
    r"\bvue js\b":        "vuejs vue",
    r"\bdeep lerning\b":  "deep learning",
    r"\bneural nework\b": "neural network",
    r"\bdatastructure\b": "data structures",
    r"\bsience\b":        "science",
    r"\bscince\b":        "science",
    r"\bscienece\b":      "science",
    r"\bsciece\b":        "science",
    r"\bsceince\b":       "science",
    r"\bapps\b":          "apps",
    r"\bapp\b":           "app",
    r"\banalatics\b":     "analytics",
    r"\banalyics\b":      "analytics",
    r"\bvisualizaton\b":  "visualization",
    r"\bvisulaization\b": "visualization",
    r"\bmatematics\b":    "mathematics",
    r"\bmathamatics\b":   "mathematics",
    r"\bmaths\b":         "mathematics",
    r"\bmath\b":          "mathematics",
    r"\bwithout maths\b": "without mathematics",
    r"\bno math\b":       "without mathematics",
    r"\bno maths\b":      "without mathematics",
    r"\balgo\b":          "algorithms",
    r"\balgos\b":         "algorithms",
    r"\bcyber sec\b":     "cybersecurity",
    r"\bcybersec\b":      "cybersecurity",
    r"\bgame dev\b":      "game development",
    r"\bwebdev\b":        "web development",
    r"\bmobdev\b":        "mobile development",
    r"\bappdev\b":        "app development",
    r"\bfullstack\b":     "full stack development",
    r"\bfull-stack\b":    "full stack development",
    r"\bfront-end\b":     "frontend development",
    r"\bback-end\b":      "backend development",
}

# ═══════════════════════════════════════════════════════════════════════════════
# DIFFICULTY SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════
_BEGINNER_SIGNALS = re.compile(
    r"\b(beginner|begin|start|starter|newbie|noob|novice|first time|"
    r"never|intro|introduction|basics|fundamental|zero|scratch|fresh|"
    r"new to|getting started|no experience|no background|450|101)\b", re.I
)
_ADVANCED_SIGNALS = re.compile(
    r"\b(advanced|expert|senior|professional|master|mastery|deep|"
    r"production|enterprise|architect|research|phd|graduate|"
    r"already know|have experience|familiar with|experienced)\b", re.I
)

# ═══════════════════════════════════════════════════════════════════════════════
# INTENT EXTRACTION — strip "I want to learn", "teach me about", etc.
# ═══════════════════════════════════════════════════════════════════════════════
# These phrases indicate intent but add noise to the actual query topic.
_INTENT_NOISE = re.compile(
    r"^(i want to (learn|study|understand|know|explore|get into|start|master|practice)\s*|"
    r"teach me (about\s*|how to\s*)?|"
    r"how (do i|can i|to)\s*(learn\s*)?|"
    r"help me (learn|understand|with)\s*|"
    r"i (need|would like) (to learn|a course on|courses on|course for)\s*|"
    r"looking for (a course|courses|tutorials?) (on|about|for|in)\s*|"
    r"(best\s*)?(courses?|tutorials?|resources?) (on|for|about|in|to learn)\s*|"
    r"(find|show|give|suggest|recommend) (me\s*)?(a\s*)?(course|tutorial|resource|lesson)s? (on|for|about|in|to)?\s*|"
    r"i (am|'m|am a) (beginner|noob|newbie|new) (in|at|to|with)?\s*|"
    r"(complete|absolute|total|utter)? beginner (in|at|to|with)?\s*|"
    r"i have no (idea|background|experience|knowledge) (about|in|on)?\s*|"
    r"(please|pls|plz)\s*|"
    r"can you\s*)",
    re.I,
)

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY ENRICHMENT MAP — topic → related search terms
# ═══════════════════════════════════════════════════════════════════════════════
_TOPIC_EXPAND = {
    "machine learning":           ["ML course", "scikit-learn", "tensorflow pytorch"],
    "deep learning":              ["neural networks", "CNN RNN transformer", "pytorch tensorflow"],
    "data science":               ["python pandas numpy statistics", "data analysis visualization"],
    "artificial intelligence":    ["AI machine learning deep learning", "neural networks GPT"],
    "natural language processing":["NLP text classification BERT", "transformers huggingface"],
    "computer vision":            ["image recognition CNN", "opencv pytorch torchvision"],
    "web development":            ["HTML CSS javascript", "react nodejs fullstack"],
    "blockchain":                 ["ethereum solidity smart contracts", "web3 defi"],
    "cybersecurity":              ["ethical hacking pentest", "network security CTF"],
    "game development":           ["unity unreal godot", "game design programming"],
    "mobile development":         ["flutter react native iOS android", "mobile app"],
    "data structures":            ["algorithms DSA", "leetcode competitive programming"],
    "python":                     ["python programming", "python projects tutorial"],
    "javascript":                 ["javascript ES6 modern", "web development"],
    "cloud":                      ["AWS GCP Azure cloud computing", "devops kubernetes"],
    "devops":                     ["docker kubernetes CI/CD", "linux bash automation"],
    "sql":                        ["database queries", "postgresql mysql data analysis"],
    "finance":                    ["financial modeling", "investment trading valuation"],
    "excel":                      ["microsoft excel data analysis", "spreadsheets formulas"],
    "ux design":                  ["user experience figma wireframing", "product design"],
    "music":                      ["music production", "learn instrument theory DAW"],
    "photography":                ["camera photography editing", "lightroom photoshop"],
    "cooking":                    ["culinary arts cooking techniques", "chef skills"],
    "public speaking":            ["communication presentation skills", "toastmasters"],
    "fitness":                    ["workout training health", "nutrition exercise"],
    "business":                   ["entrepreneurship startup management", "business strategy"],
    "marketing":                  ["digital marketing SEO social media", "google ads"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# SPELL CORRECTION (only for non-tech words)
# ═══════════════════════════════════════════════════════════════════════════════
@lru_cache(maxsize=1)
def _get_spellchecker():
    """Lazy-load SpellChecker, add all tech vocab so they're never corrected."""
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker(distance=1)
        # Collect ALL words to protect in one batch (avoids O(n²) dict rebuilds)
        extra: set[str] = set(_TECH_VOCAB)
        for term in _TECH_VOCAB:
            for word in term.split():
                if len(word) > 2:
                    extra.add(word)
        spell.word_frequency.load_words(list(extra))   # single rebuild
        return spell
    except ImportError:
        return None


def _spell_correct(text: str) -> str:
    """Correct misspelled non-tech words. Returns corrected text."""
    spell = _get_spellchecker()
    if not spell:
        return text

    words = text.split()
    result = []
    for word in words:
        clean = re.sub(r"[^a-z]", "", word.lower())
        # Skip: short words, numbers, tech vocab, already correct
        if len(clean) <= 2 or clean in _TECH_VOCAB or spell.known([clean]):
            result.append(word)
        else:
            correction = spell.correction(clean)
            if correction and correction != clean and len(correction) > 2:
                # Only use correction if it's not wildly different (edit distance check)
                # Don't correct if the clean word looks like a tech term
                result.append(correction)
            else:
                result.append(word)
    return " ".join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN: understand_query
# ═══════════════════════════════════════════════════════════════════════════════
def understand_query(raw_query: str) -> dict:
    """
    Full pipeline: take raw messy user input → return structured understanding.

    Returns dict:
        {
          "original":    str,   # original raw query
          "corrected":   str,   # after spell-fix + slang expansion
          "topic":       str,   # core topic extracted
          "difficulty":  str,   # "Beginner" | "Intermediate" | "Advanced" | "All"
          "search_queries": list[str],  # 2-3 optimised queries to send to search engine
          "display_correction": str | None  # message to show user if we corrected something
        }
    """
    original = raw_query.strip()
    text = original.lower()

    # ── Step 1: Remove punctuation noise (keep alphanumeric + spaces + /)
    text = re.sub(r"[^\w\s/'+#.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # ── Step 2: Expand abbreviations & slang
    for pattern, replacement in _EXPAND.items():
        text = re.sub(pattern, replacement, text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()

    # ── Step 3: Spell-correct non-tech words
    corrected = _spell_correct(text)
    corrected = re.sub(r"\s+", " ", corrected).strip()

    # ── Step 4: Extract difficulty signal
    difficulty = "All"
    if _BEGINNER_SIGNALS.search(corrected):
        difficulty = "Beginner"
    elif _ADVANCED_SIGNALS.search(corrected):
        difficulty = "Advanced"
    else:
        difficulty = "All"  # let the search engine find all levels

    # ── Step 5: Strip intent phrases → extract core topic
    topic = corrected
    # Apply intent noise removal iteratively (it may need multiple passes)
    for _ in range(3):
        new_topic = _INTENT_NOISE.sub("", topic).strip()
        if new_topic == topic:
            break
        topic = new_topic
    topic = re.sub(r"\s+", " ", topic).strip()

    # ── Step 6: Remove level words from topic (they'll be added back to queries)
    topic_clean = re.sub(
        r"\b(beginner|intermediate|advanced|expert|professional|basic|"
        r"fundamentals?|introduction|intro)\b",
        "", topic, flags=re.I
    ).strip()
    topic_clean = re.sub(r"\s+", " ", topic_clean).strip() or topic
    # Strip leading prepositions left over from intent extraction ("to ...", "about ...")
    topic_clean = re.sub(r"^(to|about|on|for|in|with|of|the)\s+", "", topic_clean, flags=re.I).strip()
    topic_clean = topic_clean or topic

    # ── Step 7: Find related terms for this topic
    expansion_terms = []
    for key, terms in _TOPIC_EXPAND.items():
        if key in topic_clean.lower():
            expansion_terms.extend(terms)
            break

    # ── Step 8: Build enriched search queries for DuckDuckGo
    level_suffix = {
        "Beginner":     "for beginners",
        "Advanced":     "advanced",
        "Intermediate": "intermediate",
    }.get(difficulty, "")

    base = topic_clean
    search_queries = [f"{base} course {level_suffix}".strip()]
    if expansion_terms:
        for et in expansion_terms[:2]:
            search_queries.append(f"{base} {et} tutorial".strip())
    search_queries.append(f"learn {base} online {level_suffix}".strip())
    search_queries.append(f"best {base} course free certificate".strip())

    # ── Step 9: Did we change anything meaningful?
    display_correction = None
    original_words = set(original.lower().split())
    corrected_words = set(corrected.split())
    changed = corrected_words - original_words - {"", " "}
    # Only show correction if substantial changes were made
    if changed and corrected.lower().strip() != original.lower().strip():
        if len(changed) >= 1 and len(original) > 3:
            display_correction = corrected

    return {
        "original":           original,
        "corrected":          corrected,
        "topic":              topic_clean,
        "difficulty":         difficulty,
        "search_queries":     search_queries,
        "display_correction": display_correction,
    }


if __name__ == "__main__":
    import sys, json
    tests = sys.argv[1:] or [
        "mchine lerning for beginers",
        "i wanna lern blockchain devlopment",
        "teach me pythn from scratck",
        "advanced deep lerning reserch",
        "javascrpt reakt beginer project",
        "how do i get into cybersec with no background",
        "bro i have no idea about coding but wanna build apps",
        "data sience witout maths background",
        "show me unity game dev tutorials",
        "I want to learn to cook Italian food",
    ]
    for q in tests:
        result = understand_query(q)
        print(f"\nIN:  {result['original']}")
        print(f"FIX: {result['corrected']}")
        print(f"TOPIC: {result['topic']}  |  LEVEL: {result['difficulty']}")
        print(f"SEARCHES: {result['search_queries'][:2]}")
        if result["display_correction"]:
            print(f"SHOW USER: 'Did you mean: {result['display_correction']}?'")
