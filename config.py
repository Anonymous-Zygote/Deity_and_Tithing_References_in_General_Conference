"""
Configuration for the Tithing Discourse Analysis pipeline.
Covers General Conference talks from 1971 to present (modern era).
"""

from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
TALKS_DIR  = DATA_DIR / "talks"        # one JSON per talk
PROC_DIR   = DATA_DIR / "processed"
OUT_DIR    = BASE_DIR / "output"
FIG_DIR    = OUT_DIR  / "figures"
TAB_DIR    = OUT_DIR  / "tables"

for _d in [RAW_DIR, TALKS_DIR, PROC_DIR, FIG_DIR, TAB_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Scraping ─────────────────────────────────────────────────────────────────
START_YEAR = 1971
END_YEAR   = 2025          # inclusive; April 2026 may be incomplete
MONTHS     = [4, 10]       # April and October general conferences
LANG       = "eng"
BASE_URL   = "https://www.churchofjesuschrist.org"
CONTENT_API = (
    f"{BASE_URL}/study/api/v3/language-pages/type/content"
)
REQUEST_DELAY = 1.2        # seconds between HTTP requests (be polite)
REQUEST_TIMEOUT = 30       # seconds
MAX_RETRIES = 3

# ── NLP ──────────────────────────────────────────────────────────────────────
# Sentence-transformer model for semantic similarity
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Tithing keyword seed set (used for both keyword filter and embedding anchor)
TITHING_KEYWORDS = [
    "tithing", "tithes", "tithe", "tenth", "one-tenth",
    "fast offering", "law of tithing", "full tithe",
    "temporal law", "storehouse", "windows of heaven",
    "pour out a blessing", "devour for your sakes",
]

# Cosine-similarity threshold for embedding-based tithing detection
EMBEDDING_THRESHOLD = 0.35

# ── Topic Modeling ────────────────────────────────────────────────────────────
N_TOPICS        = 30        # number of topics for NMF/LDA
MAX_FEATURES    = 8000      # vocabulary size
MIN_DF          = 5         # min document frequency for a term
MAX_DF          = 0.85      # max document frequency (% of docs)
N_TOP_WORDS     = 15        # words per topic to display
TOPIC_MODEL     = "nmf"     # "nmf" | "lda"

# Minimum keyword hits required for a talk to enter the tithing SUB-corpus
# topic model.  Talks caught only by embedding similarity (kw_hits=0) are
# too loosely related and pollute the sub-corpus with off-topic themes.
MIN_KW_HITS_SUBCORPUS = 2

# Additional stop words applied ONLY to the tithing sub-corpus vectorizer.
# These suppress boilerplate from auditing reports and budget summaries that
# get pulled in through loose keyword matches on "fund", "expenditure" etc.
TITHING_EXTRA_STOP = {
    # Auditing / financial report boilerplate
    "auditing", "audit", "auditor", "audited", "committee",
    "expenditure", "budget", "accounting", "financial",
    "asset", "accordance", "procedure", "certified",
    "maintained", "account", "december", "ended",
    "controlled", "received", "contributed", "contribution",
    # Church-wide statistical report boilerplate
    "stake", "ward", "branch", "district", "unit",
    "membership", "statistic", "report", "issued",
    "record", "total", "number",
}

# ── Temporal ─────────────────────────────────────────────────────────────────
DECADES = list(range(1970, 2030, 10))   # decade bucket boundaries
