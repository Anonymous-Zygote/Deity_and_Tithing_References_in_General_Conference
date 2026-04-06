"""
Analysis of terminology shifts: "God" vs "Heavenly Father" variants
across LDS General Conference talks (1971-2025)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import re
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
PROC_DIR = Path('data/processed')
FIG_DIR = Path('output/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load corpus
print("Loading talks corpus...")
talks = pd.read_parquet(PROC_DIR / 'talks_master.parquet')
print(f"Loaded {len(talks)} talks")

# Define terminology patterns (case-insensitive)
GOD_PATTERN = r'\bG[Oo][Dd]\b'
HF_PATTERNS = [
    r'\bheavenly\s+father\b',
    r'\bfather\s+in\s+heaven\b',
    r'\bheavenly\s+parents?\b',
    r'\breternal\s+father\b',
    r'\bcelestial\s+parents?\b',
    r'\bcelestial\s+father\b',
]
HF_PATTERN = '|'.join(f'({p})' for p in HF_PATTERNS)

print("\nCounting mentions...")

def count_mentions(text):
    """Count matches for a given pattern, normalized per 1000 words"""
    if pd.isna(text) or not text:
        return 0
    text_str = str(text).lower()
    return len(re.findall(GOD_PATTERN, text_str, re.IGNORECASE))

def count_hf_mentions(text):
    """Count heavenly father variants"""
    if pd.isna(text) or not text:
        return 0
    text_str = str(text).lower()
    matches = re.findall(HF_PATTERN, text_str, re.IGNORECASE)
    return len(matches)

talks['god_count'] = talks['text_clean'].apply(count_mentions)
talks['hf_count'] = talks['text_clean'].apply(count_hf_mentions)
talks['total_words'] = talks['text_clean'].str.split().str.len()

# Per-1000-words normalization
talks['god_per_1000'] = (talks['god_count'] / talks['total_words'] * 1000).fillna(0)
talks['hf_per_1000'] = (talks['hf_count'] / talks['total_words'] * 1000).fillna(0)

print(f"God mentions: {talks['god_count'].sum()} total, {talks['god_per_1000'].mean():.2f} per 1000 words avg")
print(f"Heavenly Father mentions: {talks['hf_count'].sum()} total, {talks['hf_per_1000'].mean():.2f} per 1000 words avg")

# Aggregate by conference (year + month pair)
conf_stats = (talks.groupby(['year', 'month', 'conference_id'])
              .agg(
                  talk_count=('uri', 'count'),
                  god_total=('god_count', 'sum'),
                  hf_total=('hf_count', 'sum'),
                  god_per_1000_mean=('god_per_1000', 'mean'),
                  hf_per_1000_mean=('hf_per_1000', 'mean'),
                  word_count=('total_words', 'sum'),
              )
              .reset_index())

# Create fractional year for plotting (e.g., 1971.333 for April, 1971.833 for October)
conf_stats['conf_year'] = conf_stats['year'] + (conf_stats['month'] - 1) / 12
conf_stats['conf_label'] = conf_stats['year'].astype(str) + '-' + conf_stats['month'].astype(str).str.zfill(2)
conf_stats['god_pct_of_both'] = (conf_stats['god_total'] / 
                                  (conf_stats['god_total'] + conf_stats['hf_total']) * 100)

# Calculate 5-year rolling average (centered window)
conf_stats['god_pct_rolling_5yr'] = conf_stats['god_pct_of_both'].rolling(window=5, center=True, min_periods=3).mean()

print("\nConference-level aggregates (first 10):")
print(conf_stats[['conf_label', 'talk_count', 'god_total', 'hf_total', 'god_pct_of_both']].head(10).to_string(index=False))

# ============================================================================
# FIGURE 1: Absolute Counts Over Time (Dual Line Chart)
# ============================================================================
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=conf_stats['conf_year'],
    y=conf_stats['god_total'],
    mode='lines+markers',
    name='God',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=5),
))

fig1.add_trace(go.Scatter(
    x=conf_stats['conf_year'],
    y=conf_stats['hf_total'],
    mode='lines+markers',
    name='Heavenly Father (variants)',
    line=dict(color='#ff7f0e', width=2),
    marker=dict(size=5),
))

fig1.update_layout(
    title='Absolute Frequency: "God" vs "Heavenly Father" Terminology<br><sub>LDS General Conference Talks, per Conference (1971–2025)</sub>',
    xaxis_title='Year',
    yaxis_title='Total Mentions per Conference',
    hovermode='x unified',
    template='plotly_white',
    font=dict(size=11),
    height=500,
    showlegend=True,
    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
)

fig1.write_html(FIG_DIR / 'figure_01_deity_absolute_counts.html', include_plotlyjs='cdn')
print(f"\nSaved Figure 1: {FIG_DIR / 'figure_01_deity_absolute_counts.html'}")

# ============================================================================
# FIGURE 2: Proportional Frequency (% of "God" among both terms)
# ============================================================================
fig2 = go.Figure()

# Raw data
fig2.add_trace(go.Scatter(
    x=conf_stats['conf_year'],
    y=conf_stats['god_pct_of_both'],
    mode='lines+markers',
    name='% "God" (raw)',
    line=dict(color='rgba(31, 119, 180, 0.4)', width=1),
    marker=dict(size=3),
    fill='tozeroy',
    fillcolor='rgba(31, 119, 180, 0.1)',
))

# 5-year rolling average
fig2.add_trace(go.Scatter(
    x=conf_stats['conf_year'],
    y=conf_stats['god_pct_rolling_5yr'],
    mode='lines',
    name='5-Year Rolling Average',
    line=dict(color='#1f77b4', width=3, dash='solid'),
))

fig2.update_layout(
    title='Proportional Shift: "God" as % of Both Terms<br><sub>Higher values indicate increased "God" usage relative to "Heavenly Father"</sub>',
    xaxis_title='Year',
    yaxis_title='% "God" (of "God" + "Heavenly Father" mentions)',
    hovermode='x',
    template='plotly_white',
    font=dict(size=11),
    height=500,
    showlegend=True,
    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
    yaxis=dict(range=[0, 100]),
)

# Add reference line at 50%
fig2.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Equal split", annotation_position="right")

fig2.write_html(FIG_DIR / 'figure_02_deity_proportional_shift.html', include_plotlyjs='cdn')
print(f"Saved Figure 2: {FIG_DIR / 'figure_02_deity_proportional_shift.html'}")

# ============================================================================
# FIGURE 3: Per-1000-Word Normalized Rates (Line Chart)
# ============================================================================
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=conf_stats['conf_year'],
    y=conf_stats['god_per_1000_mean'],
    mode='lines+markers',
    name='God (per 1000 words)',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=5),
))

fig3.add_trace(go.Scatter(
    x=conf_stats['conf_year'],
    y=conf_stats['hf_per_1000_mean'],
    mode='lines+markers',
    name='Heavenly Father variants (per 1000 words)',
    line=dict(color='#ff7f0e', width=2),
    marker=dict(size=5),
))

fig3.update_layout(
    title='Normalized Frequency Trends: Per 1,000 Words<br><sub>Controls for differences in talk length; shows actual emphasis in discourse</sub>',
    xaxis_title='Year',
    yaxis_title='Mentions per 1,000 Words',
    hovermode='x unified',
    template='plotly_white',
    font=dict(size=11),
    height=500,
    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
)

fig3.write_html(FIG_DIR / 'figure_03_deity_normalized_rates.html', include_plotlyjs='cdn')
print(f"Saved Figure 3: {FIG_DIR / 'figure_03_deity_normalized_rates.html'}")

# ============================================================================
# FIGURE 4: Divergence/Convergence (100% Stacked Area)
# ============================================================================
fig4 = go.Figure()

fig4.add_trace(go.Scatter(
    x=conf_stats['conf_year'],
    y=conf_stats['god_pct_of_both'],
    mode='lines',
    name='"God"',
    fill='tonexty' if False else 'tozeroy',
    line=dict(width=0),
    fillcolor='#1f77b4',
))

fig4.add_trace(go.Scatter(
    x=conf_stats['conf_year'],
    y=100 - conf_stats['god_pct_of_both'],
    mode='lines',
    name='"Heavenly Father" variants',
    fill='tonexty',
    line=dict(width=0),
    fillcolor='#ff7f0e',
))

fig4.update_layout(
    title='Share of Deity References: "God" vs "Heavenly Father"<br><sub>Stacked 100%; shows shifting preference in terminology over time</sub>',
    xaxis_title='Year',
    yaxis_title='Share (%)',
    hovermode='x unified',
    template='plotly_white',
    font=dict(size=11),
    height=500,
    yaxis=dict(range=[0, 100]),
    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
)

fig4.write_html(FIG_DIR / 'figure_04_deity_stacked_area.html', include_plotlyjs='cdn')
print(f"Saved Figure 4: {FIG_DIR / 'figure_04_deity_stacked_area.html'}")

# ============================================================================
# Summary statistics to CSV
# ============================================================================
summary_df = conf_stats[['conf_label', 'talk_count', 'god_total', 'hf_total', 
                          'god_pct_of_both', 'god_per_1000_mean', 'hf_per_1000_mean']].copy()
summary_df.columns = ['Conference', 'Talks', 'God_Total', 'HeavenlyFather_Total', 
                      'God_Pct', 'God_Per1000Words', 'HF_Per1000Words']
summary_df.to_csv(Path('output/tables') / 'deity_terminology_summary.csv', index=False)
print(f"\nSaved summary: output/tables/deity_terminology_summary.csv")

print("\n" + "="*70)
print("DEITY TERMINOLOGY ANALYSIS COMPLETE")
print("="*70)
print(f"4 interactive figures generated in output/figures/")
print(f"Summary table saved to output/tables/deity_terminology_summary.csv")
