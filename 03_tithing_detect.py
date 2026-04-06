"""
03_tithing_detect.py
====================
Identify which talks address tithing and measure the *degree* to which
each talk engages with the topic using two complementary methods:

Method A – Keyword matching
    Binary flag + keyword hit count + keyword density (hits / 1000 words).

Method B – Sentence-embedding cosine similarity
    Embed each talk (mean of paragraph embeddings) and compute cosine
    similarity to a set of anchor sentences about tithing.  Talks whose
    max or mean similarity exceeds EMBEDDING_THRESHOLD are flagged.

Combined signal
    A talk is labelled `tithing_talk` if EITHER method flags it.
    A continuous `tithing_score` = (keyword_density + embed_sim) / 2
    is computed for temporal trend analysis.

Segment-level analysis
    For talks flagged as tithing talks, extract the specific paragraphs
    that most strongly reference tithing (useful for discourse analysis).

Output
------
data/processed/tithing_flags.parquet   – per-talk flags & scores
data/processed/tithing_paragraphs.parquet – per-paragraph similarity scores
                                            (tithing talks only)

Run
---
    python 03_tithing_detect.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROC_DIR, EMBEDDING_MODEL,
    TITHING_KEYWORDS, EMBEDDING_THRESHOLD,
)

# ── Anchor sentences for embedding similarity ─────────────────────────────────
# These capture the semantic space of tithing discourse in LDS context.
ANCHOR_SENTENCES = [
    "We are commanded to pay tithing, giving one tenth of our income to the Lord.",
    "The law of tithing requires members to give ten percent of their increase.",
    "By paying tithing we open the windows of heaven and receive blessings.",
    "Tithing funds are sacred and used to build temples and meetinghouses.",
    "The Lord promises that if we pay our tithes and offerings he will rebuke the devourer.",
    "A full tithe is ten percent of one's annual income.",
    "Fast offerings and tithing are the financial laws of the gospel.",
    "The storehouse is replenished through the tithes of faithful members.",
    "Malachi 3 teaches that failing to pay tithing is robbing God.",
    "Temporal blessings and spiritual blessings come from obedience to the law of tithing.",
]

# ── Keyword helpers ───────────────────────────────────────────────────────────
_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in TITHING_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def keyword_hits(text: str) -> int:
    return len(_KEYWORD_RE.findall(text))


def keyword_density(hits: int, word_count: int) -> float:
    """Hits per 1,000 words."""
    return (hits / max(word_count, 1)) * 1000


# ── Embedding helpers ─────────────────────────────────────────────────────────

def mean_sim_to_anchors(
    talk_embedding: np.ndarray,
    anchor_embeddings: np.ndarray,
) -> float:
    """Mean cosine similarity from one talk embedding to all anchors."""
    talk_vec = talk_embedding.reshape(1, -1)
    sims = cosine_similarity(talk_vec, anchor_embeddings)[0]
    return float(sims.mean())


def max_sim_to_anchors(
    talk_embedding: np.ndarray,
    anchor_embeddings: np.ndarray,
) -> float:
    talk_vec = talk_embedding.reshape(1, -1)
    sims = cosine_similarity(talk_vec, anchor_embeddings)[0]
    return float(sims.max())


# ── Paragraph-level scoring ───────────────────────────────────────────────────

def score_paragraphs(
    uri: str,
    paragraphs: list[str],
    model: SentenceTransformer,
    anchor_embeddings: np.ndarray,
) -> pd.DataFrame:
    """Return a DataFrame of paragraphs with their max anchor similarity."""
    if not paragraphs:
        return pd.DataFrame()
    embs = model.encode(paragraphs, convert_to_numpy=True, show_progress_bar=False)
    sims = cosine_similarity(embs, anchor_embeddings).max(axis=1)
    kw   = [keyword_hits(p) for p in paragraphs]
    return pd.DataFrame({
        "uri":         uri,
        "para_idx":    range(len(paragraphs)),
        "paragraph":   paragraphs,
        "kw_hits":     kw,
        "embed_sim":   sims,
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load master
    master_path = PROC_DIR / "talks_master.parquet"
    if not master_path.exists():
        print("Run 02_preprocess.py first.")
        return

    df = pd.read_parquet(master_path)
    print(f"Loaded {len(df):,} talks.")

    # Load original talk JSON for paragraph-level analysis
    from config import TALKS_DIR
    import json

    talk_paragraphs: dict[str, list[str]] = {}
    for row in df.itertuples():
        slug = row.uri.strip("/").replace("/", "_")
        p = TALKS_DIR / f"{slug}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            talk_paragraphs[row.uri] = d.get("paragraphs", [])

    # ── Method A: Keyword ─────────────────────────────────────────────────────
    print("Method A: keyword matching…")
    df["kw_hits"]     = df["text_clean"].apply(keyword_hits)
    df["kw_density"]  = df.apply(
        lambda r: keyword_density(r["kw_hits"], r["word_count"]), axis=1
    )
    df["kw_flag"]     = df["kw_hits"] > 0

    # ── Method B: Sentence embeddings ─────────────────────────────────────────
    print(f"Method B: loading embedding model '{EMBEDDING_MODEL}'…")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("  Encoding anchor sentences…")
    anchor_embeddings = model.encode(
        ANCHOR_SENTENCES, convert_to_numpy=True, show_progress_bar=False
    )

    print(f"  Encoding {len(df):,} talks (this may take several minutes)…")
    talk_texts = df["text_clean"].tolist()
    talk_embeddings = model.encode(
        talk_texts,
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=True,
    )

    df["embed_mean_sim"] = [
        mean_sim_to_anchors(e, anchor_embeddings) for e in talk_embeddings
    ]
    df["embed_max_sim"]  = [
        max_sim_to_anchors(e, anchor_embeddings) for e in talk_embeddings
    ]
    df["embed_flag"] = df["embed_max_sim"] >= EMBEDDING_THRESHOLD

    # ── Combined label & score ────────────────────────────────────────────────
    df["tithing_talk"] = df["kw_flag"] | df["embed_flag"]

    # Normalized keyword density (scale to ~0-1 range via log)
    max_density = df["kw_density"].max() or 1.0
    df["kw_score_norm"] = df["kw_density"] / max_density

    df["tithing_score"] = (df["kw_score_norm"] + df["embed_mean_sim"]) / 2

    # ── Paragraph-level scoring (tithing talks only) ──────────────────────────
    print("Paragraph-level analysis for tithing talks…")
    tithing_uris = df.loc[df["tithing_talk"], "uri"].tolist()
    para_records = []

    for uri in tqdm(tithing_uris, desc="Paragraphs", unit="talk"):
        paras = talk_paragraphs.get(uri, [])
        if paras:
            pdf = score_paragraphs(uri, paras, model, anchor_embeddings)
            para_records.append(pdf)

    para_df = pd.concat(para_records, ignore_index=True) if para_records else pd.DataFrame()

    # ── Save ──────────────────────────────────────────────────────────────────
    flags_cols = [
        "uri", "year", "month", "decade", "conference_id",
        "speaker", "title", "word_count",
        "kw_hits", "kw_density", "kw_flag",
        "embed_mean_sim", "embed_max_sim", "embed_flag",
        "tithing_talk", "tithing_score",
    ]
    flags_df = df[flags_cols].copy()
    flags_df.to_parquet(PROC_DIR / "tithing_flags.parquet", index=False)
    flags_df.to_csv(PROC_DIR / "tithing_flags.csv", index=False)
    if not para_df.empty:
        para_df.to_parquet(PROC_DIR / "tithing_paragraphs.parquet", index=False)
        para_df.to_csv(PROC_DIR / "tithing_paragraphs.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total   = len(df)
    n_tithing = df["tithing_talk"].sum()
    print(f"\nResults:")
    print(f"  Total talks:          {n_total:,}")
    print(f"  Tithing talks (any):  {n_tithing:,}  ({n_tithing/n_total*100:.1f}%)")
    print(f"    Keyword only:       {(df['kw_flag'] & ~df['embed_flag']).sum():,}")
    print(f"    Embedding only:     {(df['embed_flag'] & ~df['kw_flag']).sum():,}")
    print(f"    Both methods:       {(df['kw_flag'] & df['embed_flag']).sum():,}")
    print(f"\nSaved to {PROC_DIR}/tithing_flags.parquet")


if __name__ == "__main__":
    main()
