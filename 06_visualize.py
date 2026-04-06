"""
06_visualize.py
===============
Generate all figures for the tithing discourse analysis.

Figures produced
----------------
1.  tithing_fraction_timeseries.html   – yearly fraction of tithing talks
                                         (with rolling average)
2.  tithing_score_timeseries.html      – mean tithing score over time
3.  tithing_counts_bar.html            – stacked bar: tithing vs other talks/year
4.  decade_comparison.html             – grouped bar chart by decade
5.  embedding_vs_keyword.html          – scatter: embed_sim vs kw_density
6.  tithing_topics_heatmap.html        – tithing sub-topic weights over time
7.  speaker_bubble.html               – speaker vs # tithing talks (bubble chart)
8.  wordcloud_tithing.png              – word cloud of tithing corpus
9.  wordcloud_alltalks.png             – word cloud of all talks (contrast)
10. topic_barchart_full.html           – top words per full-corpus topic
11. topic_barchart_tithing.html        – top words per tithing sub-topic

All HTML figures are interactive Plotly charts.
Static PNG figures are saved with matplotlib/wordcloud.

Run
---
    python 06_visualize.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import PROC_DIR, TAB_DIR, FIG_DIR

warnings.filterwarnings("ignore")

# ── Plotly ────────────────────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold
TITHING_COLOR  = "#2563EB"   # blue
OTHER_COLOR    = "#D1D5DB"   # grey
ACCENT_COLOR   = "#F59E0B"   # amber


# ── Helper: save Plotly fig ───────────────────────────────────────────────────

def save_fig(fig: go.Figure, name: str, height: int = 550):
    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", size=13),
        margin=dict(t=60, b=50, l=60, r=30),
    )
    path = FIG_DIR / f"{name}.html"
    fig.write_html(str(path))
    print(f"  Saved {path.name}")


# ── 1. Tithing fraction timeseries ────────────────────────────────────────────

def plot_fraction_timeseries(year_df: pd.DataFrame):
    fig = go.Figure()

    # Raw yearly fraction
    fig.add_trace(go.Scatter(
        x=year_df["year"],
        y=year_df["tithing_fraction"] * 100,
        mode="lines+markers",
        name="Yearly fraction",
        line=dict(color=OTHER_COLOR, width=1),
        marker=dict(size=5),
        opacity=0.6,
    ))

    # 5-conference rolling average
    if "tithing_fraction_roll5" in year_df:
        fig.add_trace(go.Scatter(
            x=year_df["year"],
            y=year_df["tithing_fraction_roll5"] * 100,
            mode="lines",
            name="5-yr rolling avg",
            line=dict(color=TITHING_COLOR, width=3),
        ))

    # 10-conference rolling average
    if "tithing_fraction_roll10" in year_df:
        fig.add_trace(go.Scatter(
            x=year_df["year"],
            y=year_df["tithing_fraction_roll10"] * 100,
            mode="lines",
            name="10-yr rolling avg",
            line=dict(color=ACCENT_COLOR, width=2, dash="dash"),
        ))

    fig.update_layout(
        title="Fraction of General Conference Talks Addressing Tithing (1971–2025)",
        xaxis_title="Year",
        yaxis_title="% of talks",
        legend=dict(orientation="h", y=1.05),
    )
    save_fig(fig, "tithing_fraction_timeseries")


# ── 2. Mean tithing score timeseries ─────────────────────────────────────────

def plot_score_timeseries(year_df: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=year_df["year"],
        y=year_df["mean_tith_score"],
        mode="lines+markers",
        name="Annual mean",
        line=dict(color=OTHER_COLOR, width=1),
        marker=dict(size=5),
        opacity=0.6,
    ))

    if "mean_tith_score_roll5" in year_df:
        fig.add_trace(go.Scatter(
            x=year_df["year"],
            y=year_df["mean_tith_score_roll5"],
            mode="lines",
            name="5-yr rolling avg",
            line=dict(color=TITHING_COLOR, width=3),
        ))

    fig.update_layout(
        title="Mean Tithing Relevance Score Over Time",
        xaxis_title="Year",
        yaxis_title="Tithing score (0–1)",
    )
    save_fig(fig, "tithing_score_timeseries")


# ── 3. Stacked bar: tithing vs other ─────────────────────────────────────────

def plot_counts_bar(year_df: pd.DataFrame):
    year_df = year_df.copy()
    year_df["other_talks"] = year_df["total_talks"] - year_df["tithing_talks"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=year_df["year"],
        y=year_df["tithing_talks"],
        name="Tithing talks",
        marker_color=TITHING_COLOR,
    ))
    fig.add_trace(go.Bar(
        x=year_df["year"],
        y=year_df["other_talks"],
        name="Other talks",
        marker_color=OTHER_COLOR,
    ))

    fig.update_layout(
        barmode="stack",
        title="General Conference Talk Count by Year",
        xaxis_title="Year",
        yaxis_title="Number of talks",
        legend=dict(orientation="h", y=1.05),
    )
    save_fig(fig, "tithing_counts_bar")


# ── 4. Decade comparison bar chart ───────────────────────────────────────────

def plot_decade_comparison(decade_df: pd.DataFrame):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Tithing Fraction by Decade", "Mean Tithing Score by Decade"],
    )

    decade_labels = [str(d) + "s" for d in decade_df["decade"]]

    fig.add_trace(
        go.Bar(
            x=decade_labels,
            y=decade_df["tithing_fraction"] * 100,
            marker_color=TITHING_COLOR,
            name="Fraction (%)",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=decade_labels,
            y=decade_df["mean_tith_score"],
            marker_color=ACCENT_COLOR,
            name="Mean score",
        ),
        row=1, col=2,
    )

    fig.update_layout(
        title="Decade-Level Tithing Discourse Summary",
        showlegend=False,
    )
    fig.update_yaxes(title_text="% of talks", row=1, col=1)
    fig.update_yaxes(title_text="Score (0–1)", row=1, col=2)
    save_fig(fig, "decade_comparison", height=500)


# ── 5. Embedding vs keyword scatter ──────────────────────────────────────────

def plot_embed_vs_kw(flags_df: pd.DataFrame):
    plot_df = flags_df.sample(min(3000, len(flags_df)), random_state=42)

    fig = px.scatter(
        plot_df,
        x="kw_density",
        y="embed_mean_sim",
        color="tithing_talk",
        color_discrete_map={True: TITHING_COLOR, False: OTHER_COLOR},
        hover_data=["speaker", "title", "year"],
        labels={
            "kw_density": "Keyword density (hits/1000 words)",
            "embed_mean_sim": "Embedding similarity to tithing anchors",
            "tithing_talk": "Tithing talk",
        },
        title="Keyword Density vs Semantic Similarity (sample of talks)",
        opacity=0.6,
    )
    save_fig(fig, "embedding_vs_keyword")


# ── 6. Topic-over-time heatmap (tithing sub-topics) ──────────────────────────

def plot_topic_heatmap(doc_topic_tith: pd.DataFrame, topic_labels: pd.DataFrame):
    topic_cols = sorted(
        [c for c in doc_topic_tith.columns if c.startswith("topic_") and c.split("_")[1].isdigit()],
        key=lambda c: int(c.split("_")[1]),
    )
    if not topic_cols:
        print("  No tithing topic columns – skipping heatmap.")
        return

    year_topic = (
        doc_topic_tith.groupby("year")[topic_cols].mean().reset_index()
    )

    # Create short labels from top-3 words
    def short_label(tid):
        matching = topic_labels[topic_labels["topic_id"] == tid]
        if matching.empty:
            return f"T{tid}"
        words = matching.iloc[0]["top_words"].split(", ")[:3]
        return " / ".join(words)

    z_data = year_topic[topic_cols].values.T
    y_labels = [short_label(int(c.split("_")[1])) for c in topic_cols]

    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=year_topic["year"].tolist(),
        y=y_labels,
        colorscale="Blues",
        colorbar=dict(title="Topic weight"),
    ))
    fig.update_layout(
        title="Tithing Sub-Topic Prevalence Over Time",
        xaxis_title="Year",
        yaxis_title="Topic",
        yaxis=dict(autorange="reversed"),
    )
    save_fig(fig, "tithing_topics_heatmap", height=600)


# ── 7. Speaker bubble chart ───────────────────────────────────────────────────

def plot_speaker_bubble(speakers_df: pd.DataFrame):
    fig = px.scatter(
        speakers_df.head(25),
        x="year_first",
        y="tithing_talk_count",
        size="tithing_talk_count",
        color="mean_kw",
        text="speaker",
        hover_data=["year_first", "year_last", "tithing_talk_count", "mean_kw"],
        color_continuous_scale="Blues",
        labels={
            "year_first": "First tithing talk (year)",
            "tithing_talk_count": "# tithing talks",
            "mean_kw": "Avg keyword density",
        },
        title="Top Speakers on Tithing in General Conference",
    )
    fig.update_traces(textposition="top center", textfont=dict(size=10))
    save_fig(fig, "speaker_bubble", height=600)


# ── 8 & 9. Word clouds ────────────────────────────────────────────────────────

def plot_wordclouds(master_df: pd.DataFrame, flags_df: pd.DataFrame):
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except ImportError:
        print("  wordcloud not installed – skipping word clouds.")
        return

    def make_wc(text: str, title: str, out_name: str):
        wc = WordCloud(
            width=1200, height=600,
            background_color="white",
            max_words=200,
            colormap="Blues",
            collocations=False,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=16, pad=12)
        fig.savefig(FIG_DIR / f"{out_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_name}.png")

    # Merge flags into master
    merged = master_df.merge(flags_df[["uri", "tithing_talk"]], on="uri", how="left")
    merged["tithing_talk"] = merged["tithing_talk"].fillna(False)

    tithing_text = " ".join(merged.loc[merged["tithing_talk"], "text_bow"].dropna())
    all_text     = " ".join(merged["text_bow"].dropna())

    if tithing_text.strip():
        make_wc(tithing_text, "Most Frequent Terms in Tithing Talks", "wordcloud_tithing")
    if all_text.strip():
        make_wc(all_text, "Most Frequent Terms – All General Conference Talks", "wordcloud_alltalks")


# ── 10 & 11. Topic bar charts ─────────────────────────────────────────────────

def plot_topic_barchart(topic_labels: pd.DataFrame, name: str, title: str):
    if topic_labels.empty:
        return

    word_cols = [c for c in topic_labels.columns if c.startswith("word_")][:8]
    if not word_cols:
        return

    fig = go.Figure()
    for _, row in topic_labels.iterrows():
        words = [row[c] for c in word_cols if pd.notna(row.get(c))]
        fig.add_trace(go.Bar(
            name=f"T{int(row['topic_id'])}",
            x=words,
            y=[1] * len(words),
            text=words,
            textposition="outside",
            showlegend=True,
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Term",
        yaxis_visible=False,
        barmode="group",
        height=500,
    )
    save_fig(fig, name, height=500)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load data
    year_path       = PROC_DIR / "temporal_year.parquet"
    decade_path     = PROC_DIR / "temporal_decade.parquet"
    flags_path      = PROC_DIR / "tithing_flags.parquet"
    master_path     = PROC_DIR / "talks_master.parquet"
    tith_doc_path   = PROC_DIR / "doc_topics_tithing.parquet"
    speakers_path   = TAB_DIR  / "top_speakers.csv"
    tith_labels_path = TAB_DIR / "topic_labels_tithing.csv"
    full_labels_path = TAB_DIR / "topic_labels_full.csv"

    missing = [p for p in [year_path, decade_path, flags_path, master_path]
               if not p.exists()]
    if missing:
        print("Missing data files. Run earlier pipeline steps first:")
        for m in missing:
            print(f"  {m.name}")
        return

    year_df     = pd.read_parquet(year_path)
    decade_df   = pd.read_parquet(decade_path)
    flags_df    = pd.read_parquet(flags_path)
    master_df   = pd.read_parquet(master_path)

    doc_topic_tith  = pd.read_parquet(tith_doc_path)  if tith_doc_path.exists()  else pd.DataFrame()
    speakers_df     = pd.read_csv(speakers_path)       if speakers_path.exists()  else pd.DataFrame()
    tith_labels     = pd.read_csv(tith_labels_path)    if tith_labels_path.exists() else pd.DataFrame()
    full_labels     = pd.read_csv(full_labels_path)    if full_labels_path.exists() else pd.DataFrame()

    print("Generating figures…")

    plot_fraction_timeseries(year_df)
    plot_score_timeseries(year_df)
    plot_counts_bar(year_df)
    plot_decade_comparison(decade_df)
    plot_embed_vs_kw(flags_df)

    if not doc_topic_tith.empty and not tith_labels.empty:
        plot_topic_heatmap(doc_topic_tith, tith_labels)

    if not speakers_df.empty:
        plot_speaker_bubble(speakers_df)

    plot_wordclouds(master_df, flags_df)

    if not full_labels.empty:
        plot_topic_barchart(
            full_labels.head(15),
            "topic_barchart_full",
            "Top Words per Full-Corpus Topic",
        )
    if not tith_labels.empty:
        plot_topic_barchart(
            tith_labels,
            "topic_barchart_tithing",
            "Top Words per Tithing Sub-Topic",
        )

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
