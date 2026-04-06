"""
02_preprocess.py
================
Load all scraped talk JSON files, clean the text, and produce a single
master DataFrame saved to data/processed/talks_master.parquet
(and .csv for convenience).

Cleaning steps
--------------
1. Remove HTML entities and residual markup.
2. Normalize whitespace and strip non-ASCII punctuation.
3. Sentence-tokenize (NLTK punkt) and word-tokenize for topic modelling.
4. Lower-case, remove stopwords, lemmatize for the bag-of-words column.

Run
---
    python 02_preprocess.py
"""

import json
import re
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

import nltk
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import TALKS_DIR, PROC_DIR

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

STOPWORDS   = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()

# Extra domain-specific stopwords (common LDS filler phrases)
EXTRA_STOP = {
    "said", "also", "would", "us", "may", "one", "many", "even",
    "come", "go", "know", "say", "see", "new", "like", "great",
    "church", "latter-day", "latter", "saints", "day", "lds",
    "brother", "sister", "president", "elder", "apostle",
}
STOPWORDS |= EXTRA_STOP

# ── Text cleaning ─────────────────────────────────────────────────────────────
_HTML_TAG    = re.compile(r"<[^>]+>")
_ENTITY      = re.compile(r"&[a-z]+;|&#\d+;")
_WHITESPACE  = re.compile(r"\s+")
_NON_ASCII   = re.compile(r"[^\x00-\x7F]")
_PUNCT_ONLY  = re.compile(r"[^a-zA-Z0-9\s\-']")


def clean_text(raw: str) -> str:
    """Produce readable clean plain text."""
    text = _HTML_TAG.sub(" ", raw)
    text = _ENTITY.sub(" ", text)
    text = _NON_ASCII.sub(" ", text)   # simple ASCII-safe pass
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def tokenize_for_bow(text: str) -> list[str]:
    """
    Tokenize, lower-case, remove stopwords, lemmatize.
    Returns a flat list of lemma strings for bag-of-words / topic modelling.
    """
    words = word_tokenize(text.lower())
    tokens = []
    for w in words:
        if not w.isalpha():        # drop numbers and punctuation
            continue
        if len(w) < 3:
            continue
        if w in STOPWORDS:
            continue
        lemma = LEMMATIZER.lemmatize(w)
        tokens.append(lemma)
    return tokens


def load_talk(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    talk_files = sorted(TALKS_DIR.glob("*.json"))
    if not talk_files:
        print("No talk files found. Run 01_scrape_talks.py first.")
        return

    print(f"Loading {len(talk_files)} talk files…")
    records = []

    for path in tqdm(talk_files, desc="Preprocessing", unit="talk"):
        talk = load_talk(path)
        if talk is None:
            continue

        raw_text = talk.get("text", "") or " ".join(talk.get("paragraphs", []))
        if len(raw_text.split()) < 50:
            continue  # skip stubs / session headers / prayers with no body

        clean   = clean_text(raw_text)
        bow     = tokenize_for_bow(clean)
        bow_str = " ".join(bow)

        records.append({
            "uri":       talk.get("uri", ""),
            "year":      int(talk.get("year", 0)),
            "month":     int(talk.get("month", 0)),
            "speaker":   talk.get("speaker", "").strip(),
            "title":     talk.get("title", "").strip(),
            "word_count": talk.get("word_count", len(raw_text.split())),
            "text_clean":  clean,
            "text_bow":    bow_str,
            "n_tokens_bow": len(bow),
        })

    df = pd.DataFrame(records)

    # ── Derived columns ───────────────────────────────────────────────────────
    df["decade"] = (df["year"] // 10 * 10).astype(int)
    df["conference_id"] = (
        df["year"].astype(str) + "_" +
        df["month"].apply(lambda m: f"{int(m):02d}")
    )

    # Sort chronologically
    df.sort_values(["year", "month", "uri"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save
    out_parquet = PROC_DIR / "talks_master.parquet"
    out_csv     = PROC_DIR / "talks_master.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    print(f"\nSaved {len(df):,} talks to:")
    print(f"  {out_parquet}")
    print(f"  {out_csv}")
    print(f"\nYears covered: {df['year'].min()}–{df['year'].max()}")
    print(f"Unique speakers: {df['speaker'].nunique():,}")
    print(f"Total word count: {df['word_count'].sum():,}")


if __name__ == "__main__":
    main()
