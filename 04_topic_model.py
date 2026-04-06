"""
04_topic_model.py
=================
Run unsupervised topic modelling on the full corpus AND on the tithing
sub-corpus to discover interpretable thematic clusters.

Two models are trained:
  1. Full corpus  – N_TOPICS topics to understand the overall discourse
                    landscape of General Conference.
  2. Tithing corpus – N_TOPICS_TITHING topics to discover how tithing
                      rhetoric subdivides into sub-themes (blessings,
                      obedience, sacrifice, financial stewardship, etc.).

Both NMF (fast, interpretable) and document-term matrix statistics are
saved so the visualiser can render topic heatmaps over time.

Output
------
data/processed/topic_model_full.pkl          – fitted sklearn model + vectorizer
data/processed/topic_model_tithing.pkl       – same, tithing subset
data/processed/doc_topics_full.parquet       – per-doc topic weights (full)
data/processed/doc_topics_tithing.parquet    – per-doc topic weights (tithing)
data/tables/topic_labels_full.csv            – top-N words per topic
data/tables/topic_labels_tithing.csv         – top-N words per topic (tithing)

Run
---
    python 04_topic_model.py
"""

import pickle
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROC_DIR, TAB_DIR,
    N_TOPICS, MAX_FEATURES, MIN_DF, MAX_DF, N_TOP_WORDS, TOPIC_MODEL,
    MIN_KW_HITS_SUBCORPUS, TITHING_EXTRA_STOP,
)

N_TOPICS_TITHING = 15   # fewer topics for the sub-corpus


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(n_topics: int, use_nmf: bool = True):
    if use_nmf:
        return NMF(
            n_components=n_topics,
            random_state=42,
            max_iter=400,
            init="nndsvda",
            solver="mu",
            beta_loss="frobenius",
        )
    else:
        return LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=30,
            learning_method="online",
            n_jobs=-1,
        )


def build_vectorizer(use_nmf: bool = True, extra_stop: set | None = None):
    """TF-IDF for NMF; raw counts for LDA.

    extra_stop: additional stop words merged on top of the sklearn default.
    """
    stop_words: str | list | None = "english"
    if extra_stop:
        # sklearn only accepts 'english' or an explicit list
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop_words = list(set(ENGLISH_STOP_WORDS) | extra_stop)

    common = dict(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        stop_words=stop_words,
    )
    if use_nmf:
        return TfidfVectorizer(sublinear_tf=True, **common)
    else:
        return CountVectorizer(**common)


# ── Fit + label topics ────────────────────────────────────────────────────────

def fit_topics(
    texts: list[str],
    n_topics: int,
    use_nmf: bool,
    label: str,
    extra_stop: set | None = None,
) -> tuple:
    print(f"  Fitting {label} model ({n_topics} topics, {'NMF' if use_nmf else 'LDA'})…")
    vec   = build_vectorizer(use_nmf, extra_stop=extra_stop)
    dtm   = vec.fit_transform(texts)
    model = build_model(n_topics, use_nmf)
    W     = model.fit_transform(dtm)   # doc-topic matrix
    return vec, model, W


def top_words_per_topic(
    model,
    feature_names: list[str],
    n_words: int = N_TOP_WORDS,
) -> pd.DataFrame:
    rows = []
    for topic_idx, topic_vec in enumerate(model.components_):
        top_idx   = topic_vec.argsort()[::-1][:n_words]
        top_terms = [feature_names[i] for i in top_idx]
        rows.append({
            "topic_id":  topic_idx,
            "top_words": ", ".join(top_terms),
            **{f"word_{i+1}": t for i, t in enumerate(top_terms)},
        })
    return pd.DataFrame(rows)


def doc_topic_frame(
    W: np.ndarray,
    meta_df: pd.DataFrame,
    n_topics: int,
) -> pd.DataFrame:
    """Combine per-doc metadata with normalised topic weights."""
    W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-9)
    topic_cols = {f"topic_{i}": W_norm[:, i] for i in range(n_topics)}
    df = meta_df.copy().reset_index(drop=True)
    for col, vals in topic_cols.items():
        df[col] = vals
    df["dominant_topic"] = W_norm.argmax(axis=1)
    df["topic_entropy"]  = -(
        W_norm * np.log(W_norm + 1e-12)
    ).sum(axis=1)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    master_path = PROC_DIR / "talks_master.parquet"
    flags_path  = PROC_DIR / "tithing_flags.parquet"

    if not master_path.exists():
        print("Run 02_preprocess.py first.")
        return
    if not flags_path.exists():
        print("Run 03_tithing_detect.py first.")
        return

    master = pd.read_parquet(master_path)
    flags  = pd.read_parquet(flags_path)

    # Merge flags into master
    master = master.merge(
        flags[["uri", "tithing_talk", "tithing_score",
               "kw_hits", "kw_density", "embed_mean_sim"]],
        on="uri", how="left",
    )
    master["tithing_talk"] = master["tithing_talk"].fillna(False)

    use_nmf = TOPIC_MODEL.lower() == "nmf"

    # ── 1. Full corpus model ──────────────────────────────────────────────────
    print("\n=== Full corpus ===")
    full_texts = master["text_bow"].tolist()
    meta_cols  = ["uri", "year", "month", "decade", "speaker", "title",
                  "word_count", "tithing_talk", "tithing_score"]

    vec_full, model_full, W_full = fit_topics(
        full_texts, N_TOPICS, use_nmf, "full corpus"
    )
    topic_labels_full = top_words_per_topic(
        model_full, vec_full.get_feature_names_out().tolist()
    )
    doc_topic_full = doc_topic_frame(W_full, master[meta_cols], N_TOPICS)

    # Save
    topic_labels_full.to_csv(TAB_DIR / "topic_labels_full.csv", index=False)
    doc_topic_full.to_parquet(PROC_DIR / "doc_topics_full.parquet", index=False)
    with open(PROC_DIR / "topic_model_full.pkl", "wb") as f:
        pickle.dump({"vec": vec_full, "model": model_full}, f)

    print(f"  Full corpus: {len(full_texts):,} docs -> {N_TOPICS} topics")

    # ── 2. Tithing sub-corpus model ───────────────────────────────────────────
    # Only include talks with genuine keyword presence (>= MIN_KW_HITS_SUBCORPUS).
    # Talks flagged purely by embedding similarity tend to be off-topic and
    # pollute the sub-corpus with auditing reports, welfare talks, etc.
    print("\n=== Tithing sub-corpus ===")
    tithing_mask   = master["kw_hits"] >= MIN_KW_HITS_SUBCORPUS
    tithing_subset = master.loc[tithing_mask].copy()
    print(f"  Sub-corpus: {tithing_mask.sum():,} talks with >= {MIN_KW_HITS_SUBCORPUS} keyword hits "
          f"(excluded {(master['tithing_talk'] & ~tithing_mask).sum():,} embedding-only talks)")

    if len(tithing_subset) < 20:
        print(f"  Only {len(tithing_subset)} tithing talks – skipping sub-corpus model.")
    else:
        tithing_texts = tithing_subset["text_bow"].tolist()
        vec_tith, model_tith, W_tith = fit_topics(
            tithing_texts, N_TOPICS_TITHING, use_nmf, "tithing sub-corpus",
            extra_stop=TITHING_EXTRA_STOP,
        )
        topic_labels_tith = top_words_per_topic(
            model_tith, vec_tith.get_feature_names_out().tolist()
        )
        doc_topic_tith = doc_topic_frame(
            W_tith, tithing_subset[meta_cols], N_TOPICS_TITHING
        )

        topic_labels_tith.to_csv(TAB_DIR / "topic_labels_tithing.csv", index=False)
        doc_topic_tith.to_parquet(PROC_DIR / "doc_topics_tithing.parquet", index=False)
        with open(PROC_DIR / "topic_model_tithing.pkl", "wb") as f:
            pickle.dump({"vec": vec_tith, "model": model_tith}, f)

        print(f"  Tithing corpus: {len(tithing_texts):,} docs -> {N_TOPICS_TITHING} topics")

    # ── Print topic summaries ─────────────────────────────────────────────────
    print("\n── Top words per full-corpus topic ──────────────────────────────")
    for _, row in topic_labels_full.iterrows():
        print(f"  Topic {row['topic_id']:2d}: {row['top_words']}")

    print("\nDone. See output/tables/ for topic word lists.")


if __name__ == "__main__":
    main()
