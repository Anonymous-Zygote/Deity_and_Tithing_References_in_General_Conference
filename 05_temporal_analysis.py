"""
05_temporal_analysis.py
=======================
Aggregate the tithing detection & topic modelling results across time to
answer the central research question:

  "How has discourse on tithing in General Conference changed over time?"

Metrics computed at year AND decade granularity:

  - tithing_talk_count        – absolute number of tithing talks
  - total_talk_count          – total talks (all topics)
  - tithing_fraction          – tithing talks / total talks (%)
  - mean_tithing_score        – average combined tithing_score
  - mean_kw_density           – average keyword density (hits/1000 words)
  - mean_embed_sim            – average embedding similarity to anchors
  - topic_share_{i}           – mean weight of each tithing sub-topic
  - dominant_topic_mode       – most common dominant topic
  - unique_speakers           – number of unique speakers addressing tithing
  - rolling_fraction_{n}      – {n}-conference rolling mean of fraction

Statistical tests
-----------------
  - Mann-Kendall trend test on yearly tithing_fraction (trend direction)
  - Pearson correlation between year and mean_tithing_score
  - Decade-wise ANOVA on tithing_score (are decades significantly different?)

Output
------
data/processed/temporal_year.parquet
data/processed/temporal_decade.parquet
data/tables/trend_stats.csv

Run
---
    python 05_temporal_analysis.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import PROC_DIR, TAB_DIR

warnings.filterwarnings("ignore")


# ── Mann–Kendall trend test (pure Python, no pymannkendall dep) ───────────────

def mann_kendall(x: np.ndarray) -> dict:
    """Compute Mann–Kendall S statistic, p-value, and Kendall's tau."""
    n = len(x)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = x[j] - x[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance of S
    var_s = n * (n - 1) * (2 * n + 5) / 18

    # Continuity-corrected z
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    return {"S": s, "z": z, "p_value": p, "tau": tau,
            "trend": "increasing" if tau > 0 else "decreasing" if tau < 0 else "no trend"}


# ── Temporal aggregation ──────────────────────────────────────────────────────

def aggregate_by_period(
    flags_df: pd.DataFrame,
    doc_topic_tith: pd.DataFrame | None,
    period_col: str,
) -> pd.DataFrame:
    """Aggregate tithing metrics by year or decade."""

    # Base stats from tithing flags
    agg = (
        flags_df
        .groupby(period_col)
        .agg(
            total_talks      = ("uri", "count"),
            tithing_talks    = ("tithing_talk", "sum"),
            mean_tith_score  = ("tithing_score", "mean"),
            mean_kw_density  = ("kw_density", "mean"),
            mean_embed_sim   = ("embed_mean_sim", "mean"),
            unique_speakers  = ("speaker", "nunique"),
        )
        .reset_index()
    )
    agg["tithing_fraction"] = agg["tithing_talks"] / agg["total_talks"]

    # Add tithing sub-topic shares (if model was run)
    if doc_topic_tith is not None and not doc_topic_tith.empty:
        topic_cols = [c for c in doc_topic_tith.columns if c.startswith("topic_")]
        if period_col in doc_topic_tith.columns:
            topic_agg = (
                doc_topic_tith
                .groupby(period_col)[topic_cols]
                .mean()
                .reset_index()
            )
            agg = agg.merge(topic_agg, on=period_col, how="left")

    agg.sort_values(period_col, inplace=True)
    agg.reset_index(drop=True, inplace=True)
    return agg


def add_rolling(df: pd.DataFrame, col: str, windows: list[int], period: str) -> pd.DataFrame:
    """Add rolling means of `col` for given window sizes."""
    df = df.copy()
    for w in windows:
        df[f"{col}_roll{w}"] = df[col].rolling(w, center=True, min_periods=1).mean()
    return df


# ── Statistical tests ─────────────────────────────────────────────────────────

def run_stats(year_df: pd.DataFrame, decade_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # 1. Mann–Kendall on yearly tithing fraction
    mk = mann_kendall(year_df["tithing_fraction"].values)
    rows.append({
        "test": "Mann-Kendall (yearly fraction)",
        "statistic": mk["tau"],
        "p_value": mk["p_value"],
        "interpretation": (
            f"Kendall's tau = {mk['tau']:.3f}; "
            f"trend = {mk['trend']}; "
            f"p = {mk['p_value']:.4f}"
        ),
    })

    # 2. Pearson correlation: year vs mean tithing score
    r, p = stats.pearsonr(year_df["year"], year_df["mean_tith_score"])
    rows.append({
        "test": "Pearson r (year vs mean_tithing_score)",
        "statistic": r,
        "p_value": p,
        "interpretation": f"r = {r:.3f}, p = {p:.4f}",
    })

    # 3. One-way ANOVA across decades on mean tithing score
    decade_groups = [
        year_df.loc[year_df["year"] // 10 * 10 == d, "mean_tith_score"].values
        for d in decade_df["decade"]
        if len(year_df.loc[year_df["year"] // 10 * 10 == d]) > 1
    ]
    if len(decade_groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*decade_groups)
        rows.append({
            "test": "One-way ANOVA (decades vs mean_tithing_score)",
            "statistic": f_stat,
            "p_value": p_anova,
            "interpretation": f"F = {f_stat:.3f}, p = {p_anova:.4f}",
        })

    # 4. Linear regression slope (year vs tithing fraction)
    slope, intercept, r_val, p_val, std_err = stats.linregress(
        year_df["year"], year_df["tithing_fraction"]
    )
    rows.append({
        "test": "Linear regression (year → tithing_fraction)",
        "statistic": slope,
        "p_value": p_val,
        "interpretation": (
            f"slope = {slope:.5f} fraction/year, "
            f"R² = {r_val**2:.3f}, p = {p_val:.4f}"
        ),
    })

    return pd.DataFrame(rows)


# ── Top tithing speakers ──────────────────────────────────────────────────────

def top_speakers(flags_df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    tdf = flags_df[flags_df["tithing_talk"]].copy()
    spk = (
        tdf.groupby("speaker")
        .agg(
            tithing_talk_count = ("uri", "count"),
            year_first  = ("year", "min"),
            year_last   = ("year", "max"),
            mean_score  = ("tithing_score", "mean"),
            mean_kw     = ("kw_density", "mean"),
        )
        .reset_index()
        .sort_values("tithing_talk_count", ascending=False)
        .head(n)
    )
    return spk


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    flags_path = PROC_DIR / "tithing_flags.parquet"
    tith_doc_path = PROC_DIR / "doc_topics_tithing.parquet"

    if not flags_path.exists():
        print("Run 03_tithing_detect.py first.")
        return

    flags_df = pd.read_parquet(flags_path)
    doc_topic_tith = (
        pd.read_parquet(tith_doc_path)
        if tith_doc_path.exists() else None
    )

    print(f"Loaded {len(flags_df):,} talks, "
          f"{flags_df['tithing_talk'].sum():,} tithing talks.")

    # ── Year-level aggregation ────────────────────────────────────────────────
    year_df = aggregate_by_period(flags_df, doc_topic_tith, "year")
    year_df = add_rolling(year_df, "tithing_fraction", [3, 5, 10], "year")
    year_df = add_rolling(year_df, "mean_tith_score",  [3, 5], "year")

    # ── Decade-level aggregation ──────────────────────────────────────────────
    decade_df = aggregate_by_period(flags_df, doc_topic_tith, "decade")

    # ── Statistical tests ─────────────────────────────────────────────────────
    stat_df = run_stats(year_df, decade_df)

    # ── Top speakers ──────────────────────────────────────────────────────────
    speakers_df = top_speakers(flags_df)

    # ── Save ──────────────────────────────────────────────────────────────────
    year_df.to_parquet(PROC_DIR / "temporal_year.parquet", index=False)
    year_df.to_csv(PROC_DIR / "temporal_year.csv", index=False)

    decade_df.to_parquet(PROC_DIR / "temporal_decade.parquet", index=False)
    decade_df.to_csv(PROC_DIR / "temporal_decade.csv", index=False)

    stat_df.to_csv(TAB_DIR / "trend_stats.csv", index=False)
    speakers_df.to_csv(TAB_DIR / "top_speakers.csv", index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Trend statistics ────────────────────────────────────────────")
    for _, row in stat_df.iterrows():
        print(f"  {row['test']}")
        print(f"    {row['interpretation']}")

    print("\n── Top 10 tithing speakers ─────────────────────────────────────")
    print(speakers_df.head(10).to_string(index=False))

    print(f"\nSaved temporal summaries to {PROC_DIR}/")


if __name__ == "__main__":
    main()
