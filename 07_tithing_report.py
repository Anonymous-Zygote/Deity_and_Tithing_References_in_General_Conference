"""
07_tithing_report.py
====================
Produces four focused, publication-quality figures that directly answer
the four research questions about tithing discourse in General Conference:

  Figure 1 – tithing_core_vs_mention.html
    "How often is tithing the CORE message vs. just mentioned?"
    Two lines: fraction of talks where tithing is the core focus
    (kw_hits >= CORE_FOCUS_THRESHOLD) vs. at least mentioned (kw_hits >= 1).

  Figure 2 – tithing_explicit_frequency.html
    "How often is 'tithe' / 'tithing' explicitly referenced, and how has
    that changed over time?"
    Stacked bar (mentioned / not-mentioned) per year with a rolling-average
    trend line.

  Figure 3 – tithing_associated_themes.html
    "What themes do speakers consistently pair with tithing?"
    Full-corpus topic model: computes lift = P(topic | tithing talk) /
    P(topic | any talk) for each of the 30 topics and shows which topics
    are most over-represented in tithing talks relative to the baseline.

  Figure 4 – tithing_speaker_profiles.html
    "Which leaders are most vs. least outspoken about tithing?"
    Among speakers with enough total talks to compare (>= MIN_TALKS),
    shows (a) raw count of tithing talks and (b) rate = tithing talks /
    total talks, so volume and emphasis can be distinguished.

Run
---
    python 07_tithing_report.py
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from config import PROC_DIR, TAB_DIR, FIG_DIR

# ── Constants ─────────────────────────────────────────────────────────────────
# A talk is "core focused" on tithing when the keyword appears >= this many times.
# At 5+ explicit mentions the speaker has devoted significant air-time to tithing.
CORE_FOCUS_THRESHOLD = 5

# Minimum total talks for a speaker to appear in the speaker profile.
MIN_TALKS = 15

# Full-corpus topic IDs that are procedural / administrative boilerplate
# (sustaining votes, audit reports, statistical reports).  Excluded from
# the associated-themes chart because they don't represent discourse choices.
PROCEDURAL_TOPICS = {1, 2, 7, 10, 11, 13, 20}

ROLLING = 5          # year window for rolling average
PALETTE  = px.colors.qualitative.Bold
BLUE     = '#2563EB'
AMBER    = '#F59E0B'
GREEN    = '#10B981'
GREY     = '#D1D5DB'
RED      = '#EF4444'
WHITE_TEMPLATE = 'plotly_white'


def save(fig: go.Figure, name: str, height: int = 560):
    fig.update_layout(
        height=height,
        template=WHITE_TEMPLATE,
        font=dict(family='Inter, Arial, sans-serif', size=13),
        margin=dict(t=70, b=55, l=65, r=30),
    )
    path = FIG_DIR / f'{name}.html'
    fig.write_html(str(path))
    print(f'  Saved {path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – Core focus vs. mentioned
# ─────────────────────────────────────────────────────────────────────────────

def fig1_core_vs_mention(flags: pd.DataFrame):
    """
    Two lines per year:
      - Fraction of talks where tithing is 'mentioned' (kw_hits >= 1)
      - Fraction where tithing is the 'core focus' (kw_hits >= CORE_FOCUS_THRESHOLD)
    """
    year = (
        flags.groupby('year')
        .agg(
            total    =('uri', 'count'),
            mentioned=('kw_hits', lambda s: (s >= 1).sum()),
            focused  =('kw_hits', lambda s: (s >= CORE_FOCUS_THRESHOLD).sum()),
        )
        .reset_index()
    )
    year['frac_mentioned'] = year['mentioned'] / year['total']
    year['frac_focused']   = year['focused']   / year['total']

    # Rolling averages
    roll = lambda s: s.rolling(ROLLING, center=True, min_periods=1).mean()
    year['roll_mentioned'] = roll(year['frac_mentioned'])
    year['roll_focused']   = roll(year['frac_focused'])

    fig = go.Figure()

    # Raw yearly dots (faint)
    fig.add_trace(go.Scatter(
        x=year['year'], y=year['frac_mentioned'] * 100,
        mode='markers', name='Mentioned (yearly)',
        marker=dict(color=BLUE, size=5, opacity=0.35),
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=year['year'], y=year['frac_focused'] * 100,
        mode='markers', name=f'Core focus (yearly, ≥{CORE_FOCUS_THRESHOLD} hits)',
        marker=dict(color=AMBER, size=5, opacity=0.35),
        showlegend=True,
    ))

    # Rolling trend lines
    fig.add_trace(go.Scatter(
        x=year['year'], y=year['roll_mentioned'] * 100,
        mode='lines', name=f'Mentioned ({ROLLING}-yr avg)',
        line=dict(color=BLUE, width=3),
    ))
    fig.add_trace(go.Scatter(
        x=year['year'], y=year['roll_focused'] * 100,
        mode='lines', name=f'Core focus ({ROLLING}-yr avg)',
        line=dict(color=AMBER, width=3),
    ))

    # Annotation: definition of "core focus"
    fig.add_annotation(
        x=0.01, y=0.97, xref='paper', yref='paper',
        text=f'"Core focus" = talk mentions tithing >= {CORE_FOCUS_THRESHOLD} times',
        showarrow=False, align='left',
        font=dict(size=11, color='#6B7280'),
        bgcolor='rgba(255,255,255,0.7)',
    )

    fig.update_layout(
        title='Tithing as Core Message vs. Passing Reference (1971-2025)',
        xaxis=dict(title='Year', dtick=5, tickmode='linear', tick0=1971),
        yaxis_title='% of General Conference talks',
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.18),
        margin=dict(t=60, b=100, l=65, r=30),
    )
    save(fig, 'tithing_core_vs_mention')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – Explicit mention frequency over time
# ─────────────────────────────────────────────────────────────────────────────

def fig2_explicit_frequency(flags: pd.DataFrame):
    """
    Stacked bar per year: talks mentioning 'tithing'/'tithe' vs. talks that do not.
    Overlay: 5-year rolling fraction (trend line).
    Annotations: Ensign Peak founding (1997), scandal public (2019), SEC settlement (2023).
    Background: NBER recession periods shaded yellow.
    """
    year = (
        flags.groupby('year')
        .agg(
            total    =('uri', 'count'),
            mentioned=('kw_hits', lambda s: (s >= 1).sum()),
        )
        .reset_index()
    )
    year['not_mentioned']  = year['total'] - year['mentioned']
    year['frac']           = year['mentioned'] / year['total']
    year['roll_frac']      = year['frac'].rolling(ROLLING, center=True, min_periods=1).mean()

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    # ── NBER recession shading (yellow) ───────────────────────────────────────
    # Each tuple: (start_year_frac, end_year_frac, label)
    RECESSIONS = [
        (1973 + 10/12, 1975 + 3/12,  '1973-75 recession'),
        (1980 + 0/12,  1980 + 7/12,  '1980 recession'),
        (1981 + 6/12,  1982 + 11/12, '1981-82 recession'),
        (1990 + 6/12,  1991 + 3/12,  '1990-91 recession'),
        (2001 + 2/12,  2001 + 11/12, '2001 recession'),
        (2007 + 11/12, 2009 + 6/12,  'Great Recession'),
        (2020 + 1/12,  2020 + 4/12,  'COVID-19 recession'),
    ]
    for x0, x1, _ in RECESSIONS:
        fig.add_shape(
            type='rect',
            xref='x', yref='paper',
            x0=x0, x1=x1,
            y0=0, y1=1,
            fillcolor='rgba(253, 224, 71, 0.45)',
            line_width=0,
            layer='below',
        )
    # Single "recession" legend entry via a dummy scatter
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=14, color='rgba(253, 224, 71, 0.7)', symbol='square'),
        name='US recession (NBER)',
        showlegend=True,
    ), secondary_y=False)

    # ── Bars and trend line ───────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=year['year'], y=year['mentioned'],
        name='Mentions "tithe" / "tithing"',
        marker_color=BLUE, opacity=0.85,
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=year['year'], y=year['not_mentioned'],
        name='No explicit mention',
        marker_color=GREY, opacity=0.5,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=year['year'], y=year['roll_frac'] * 100,
        mode='lines', name=f'{ROLLING}-yr rolling % mentioning tithing',
        line=dict(color=AMBER, width=3),
    ), secondary_y=True)

    # ── Ensign Peak key events (vertical lines) ────────────────────────────────
    EVENTS = [
        (1997 + 9/12,  'EP Advisors\nincorporated\n(Sep 1997)',  0.92),
        (2019 + 11/12, 'Scandal public\n(Dec 2019)',             0.78),
        (2023 + 1/12,  'SEC settlement\n(Feb 2023)',             0.62),
    ]
    for x_pos, label, y_anchor in EVENTS:
        fig.add_vline(
            x=x_pos,
            line=dict(color='rgba(220,38,38,0.85)', width=2, dash='dash'),
            layer='above',
        )
        fig.add_annotation(
            x=x_pos,
            y=y_anchor,
            xref='x', yref='paper',
            text=label,
            showarrow=True,
            arrowhead=2,
            arrowsize=0.8,
            arrowcolor='rgba(220,38,38,0.85)',
            ax=28, ay=0,
            font=dict(size=10, color='rgba(185,28,28,1)'),
            bgcolor='rgba(255,255,255,0.82)',
            bordercolor='rgba(220,38,38,0.5)',
            borderwidth=1,
            align='left',
        )

    fig.update_layout(
        barmode='stack',
        title='Explicit Tithing References per General Conference Year (1971-2025)',
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.20),
        margin=dict(t=60, b=130, l=65, r=30),
    )
    fig.update_xaxes(title_text='Year', dtick=5, tickmode='linear', tick0=1971)
    fig.update_yaxes(title_text='Number of talks', secondary_y=False)
    fig.update_yaxes(title_text='% of talks mentioning tithing', secondary_y=True,
                     ticksuffix='%', showgrid=False)
    save(fig, 'tithing_explicit_frequency')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – Co-occurring themes (mean topic weight comparison)
# ─────────────────────────────────────────────────────────────────────────────

def _topic_label(topic_labels: pd.DataFrame, tid: int, n_words: int = 4) -> str:
    """Return a short readable label from the topic_labels CSV."""
    row = topic_labels[topic_labels['topic_id'] == tid]
    if row.empty:
        return f'T{tid:02d}'
    words = [str(row.iloc[0].get(f'word_{i}', '')) for i in range(1, n_words + 1)]
    return ' / '.join(w for w in words if w and w != 'nan')


def fig3_cooccurrence_themes(flags: pd.DataFrame, doc_topics: pd.DataFrame,
                              topic_labels: pd.DataFrame):
    """
    Uses continuous topic WEIGHTS (not just the single dominant topic) to measure
    which themes genuinely co-occur with tithing inside the same talk.

    For each topic T:
      mean_weight_tithing     = average weight of T across tithing talks
      mean_weight_non_tithing = average weight of T across non-tithing talks
      ratio = mean_weight_tithing / mean_weight_non_tithing

    Ratio > 1 means the theme is more present in talks that also discuss tithing.
    Shows the 12 most co-occurring themes AND the 5 most avoided themes
    (topics that are significantly LESS present when tithing is discussed).
    """
    topic_cols = sorted(
        [c for c in doc_topics.columns if c.startswith('topic_') and c[6:].isdigit()],
        key=lambda c: int(c.split('_')[1])
    )

    merged = flags[['uri', 'kw_hits']].merge(
        doc_topics[['uri'] + topic_cols], on='uri', how='inner'
    )

    tithing     = merged[merged['kw_hits'] >= 1]
    non_tithing = merged[merged['kw_hits'] == 0]

    rows = []
    for col in topic_cols:
        tid = int(col.split('_')[1])
        if tid in PROCEDURAL_TOPICS:
            continue
        mean_t  = tithing[col].mean()
        mean_nt = non_tithing[col].mean()
        if mean_nt < 1e-9:
            continue
        ratio = mean_t / mean_nt
        diff  = mean_t - mean_nt          # absolute shift in weight
        rows.append({
            'topic_id':          tid,
            'label':             _topic_label(topic_labels, tid),
            'ratio':             ratio,
            'diff':              diff,
            'mean_tithing_pct':  mean_t  * 100,
            'mean_baseline_pct': mean_nt * 100,
        })

    df = pd.DataFrame(rows).sort_values('ratio', ascending=False)

    # Top 12 positively correlated + bottom 5 avoided
    top    = df.head(12)
    bottom = df.tail(5).sort_values('ratio', ascending=False)
    combined = pd.concat([bottom, top]).drop_duplicates('topic_id').sort_values('ratio')

    bar_colors = [
        BLUE  if r > 1.05 else
        RED   if r < 0.95 else
        GREY
        for r in combined['ratio']
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=combined['ratio'],
        y=combined['label'],
        orientation='h',
        marker_color=bar_colors,
        text=[f'{r:.2f}x' for r in combined['ratio']],
        textposition='outside',
        customdata=combined[['mean_tithing_pct', 'mean_baseline_pct', 'diff']].values,
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Ratio: %{x:.2f}x<br>'
            'Avg weight in tithing talks: %{customdata[0]:.2f}%<br>'
            'Avg weight in other talks:   %{customdata[1]:.2f}%<br>'
            'Absolute difference: +%{customdata[2]:.3f}<extra></extra>'
        ),
    ))

    fig.add_vline(x=1.0, line_dash='dash', line_color='#9CA3AF',
                  annotation_text='No difference (1.0x)',
                  annotation_position='top right')

    x_max = max(combined['ratio'].max() * 1.18, 1.5)
    fig.update_layout(
        title=('Themes That Co-Occur With Tithing (vs. Talks That Do Not Mention It)<br>'
               '<sup>Ratio of mean topic weight: tithing talks / non-tithing talks '
               '| Blue = more common with tithing | Red = less common</sup>'),
        xaxis=dict(title='Mean topic weight ratio (tithing / non-tithing)',
                   range=[0, x_max]),
        yaxis_title='',
    )
    save(fig, 'tithing_associated_themes', height=600)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – Speaker profiles
# ─────────────────────────────────────────────────────────────────────────────

def fig4_speaker_profiles(flags: pd.DataFrame):
    """
    Two panels:
      Left  – Top 20 speakers by raw count of tithing talks (volume)
      Right – Top 20 speakers by tithing rate (tithing talks / total talks),
              restricted to speakers with >= MIN_TALKS total talks so that
              a one-talk speaker with one tithing talk doesn't top the list.

    Colour encodes mean keyword density (how heavily each speaker's tithing
    talks reference tithing, not just whether they mention it).
    """
    total_per_speaker = flags.groupby('speaker').size().rename('total_talks')
    tithing_per_speaker = (
        flags[flags['kw_hits'] >= 1]
        .groupby('speaker')
        .agg(
            tithing_count=('uri', 'count'),
            mean_kw_density=('kw_density', 'mean'),
            mean_embed_sim=('embed_mean_sim', 'mean'),
            first_year=('year', 'min'),
            last_year=('year', 'max'),
        )
        .reset_index()
    )
    df = tithing_per_speaker.merge(total_per_speaker, on='speaker')
    df['tithing_rate'] = df['tithing_count'] / df['total_talks']

    # Qualified speakers for rate chart
    qualified = df[df['total_talks'] >= MIN_TALKS].copy()

    top_count = df.sort_values('tithing_count', ascending=False).head(20)
    top_rate  = qualified.sort_values('tithing_rate', ascending=False).head(20)
    # Sort ascending so most impactful is at top of horizontal bar
    top_count = top_count.sort_values('tithing_count')
    top_rate  = top_rate.sort_values('tithing_rate')

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'By Total Tithing Talks (Volume)',
            f'By Tithing Rate (>= {MIN_TALKS} total talks)',
        ],
        horizontal_spacing=0.32,
    )

    # --- Left: count ---
    fig.add_trace(go.Bar(
        x=top_count['tithing_count'],
        y=top_count['speaker'],
        orientation='h',
        marker=dict(
            color=top_count['mean_kw_density'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title='Avg KW<br>density', x=0.30, thickness=12, len=0.7),
        ),
        customdata=top_count[['total_talks', 'tithing_rate', 'mean_kw_density',
                               'first_year', 'last_year']].values,
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Tithing talks: %{x}<br>'
            'Total talks: %{customdata[0]}<br>'
            'Tithing rate: %{customdata[1]:.1%}<br>'
            'Avg KW density: %{customdata[2]:.2f}/1k words<br>'
            'Active: %{customdata[3]}–%{customdata[4]}<extra></extra>'
        ),
        name='Volume',
    ), row=1, col=1)

    # --- Right: rate ---
    fig.add_trace(go.Bar(
        x=top_rate['tithing_rate'] * 100,
        y=top_rate['speaker'],
        orientation='h',
        marker=dict(
            color=top_rate['mean_kw_density'],
            colorscale='Oranges',
            showscale=True,
            colorbar=dict(title='Avg KW<br>density', x=1.02, thickness=12, len=0.7),
        ),
        customdata=top_rate[['tithing_count', 'total_talks', 'mean_kw_density',
                              'first_year', 'last_year']].values,
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Rate: %{x:.1f}% of talks mention tithing<br>'
            'Tithing talks: %{customdata[0]}<br>'
            'Total talks: %{customdata[1]}<br>'
            'Avg KW density: %{customdata[2]:.2f}/1k words<br>'
            'Active: %{customdata[3]}–%{customdata[4]}<extra></extra>'
        ),
        name='Rate',
    ), row=1, col=2)

    fig.update_xaxes(title_text='Number of tithing talks', row=1, col=1)
    fig.update_xaxes(title_text='% of talks mentioning tithing', row=1, col=2,
                     ticksuffix='%')

    fig.update_layout(
        title='General Authority Tithing Profiles<br>'
              '<sup>Left: who spoke most often about tithing | '
              'Right: who emphasised tithing most relative to their total talks</sup>',
        showlegend=False,
        height=620,
    )
    save(fig, 'tithing_speaker_profiles', height=620)

    # Also save a "least outspoken" chart – bottom 20 by rate (qualified speakers)
    bottom_rate = qualified.sort_values('tithing_rate').head(20)
    bottom_rate = bottom_rate.sort_values('tithing_rate', ascending=False)

    fig2 = go.Figure(go.Bar(
        x=bottom_rate['tithing_rate'] * 100,
        y=bottom_rate['speaker'],
        orientation='h',
        marker=dict(color=GREY),
        customdata=bottom_rate[['tithing_count', 'total_talks', 'first_year', 'last_year']].values,
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Rate: %{x:.1f}%<br>'
            'Tithing talks: %{customdata[0]} / %{customdata[1]} total<br>'
            'Active: %{customdata[2]}–%{customdata[3]}<extra></extra>'
        ),
    ))
    fig2.update_layout(
        title=f'Leaders Who Rarely Spoke About Tithing<br>'
              f'<sup>Speakers with >= {MIN_TALKS} total talks, ranked by tithing mention rate (lowest first)</sup>',
        xaxis_title='% of talks mentioning tithing',
        xaxis=dict(ticksuffix='%'),
        height=560,
    )
    save(fig2, 'tithing_speaker_least', height=560)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – Speaker × Theme heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig5_speaker_theme_heatmap(flags: pd.DataFrame, doc_topics: pd.DataFrame,
                                topic_labels: pd.DataFrame):
    """
    Heatmap: rows = top speakers (sorted by tithing rate, most focused at top),
             columns = full-corpus topics (sorted by tithing co-occurrence ratio,
                       most tithing-correlated on the left).
    Cell value = mean topic weight for that speaker across ALL their talks.

    Tithing-correlated topics will be bright on the left for speakers who
    discuss tithing frequently, and dim for those who avoid it.
    A second heatmap panel shows the same speakers but only for their
    tithing talks, so you can see how their topic mix shifts.
    """
    topic_cols = sorted(
        [c for c in doc_topics.columns if c.startswith('topic_') and c[6:].isdigit()],
        key=lambda c: int(c.split('_')[1])
    )
    content_cols = [c for c in topic_cols
                    if int(c.split('_')[1]) not in PROCEDURAL_TOPICS]

    # Merge flags onto doc_topics
    merged = doc_topics[['uri', 'speaker'] + content_cols].merge(
        flags[['uri', 'kw_hits']], on='uri', how='left'
    )
    merged['kw_hits'] = merged['kw_hits'].fillna(0)

    # ── Speaker selection: qualified speakers, sorted by tithing rate ──────
    total_per_spk  = merged.groupby('speaker').size().rename('total')
    tithing_per_spk = (merged[merged['kw_hits'] >= 1]
                       .groupby('speaker').size().rename('tithing'))
    spk_stats = pd.concat([total_per_spk, tithing_per_spk], axis=1).fillna(0)
    spk_stats['rate'] = spk_stats['tithing'] / spk_stats['total']
    qualified_spk = (spk_stats[spk_stats['total'] >= MIN_TALKS]
                     .sort_values('rate', ascending=False)
                     .head(40)
                     .index.tolist())

    # ── Topic ordering: sort by co-occurrence ratio (tithing / non-tithing) ─
    t_talks  = merged[merged['kw_hits'] >= 1]
    nt_talks = merged[merged['kw_hits'] == 0]
    ratios   = {}
    for col in content_cols:
        nt_mean = nt_talks[col].mean()
        ratios[col] = t_talks[col].mean() / nt_mean if nt_mean > 1e-9 else 1.0
    ordered_cols = sorted(content_cols, key=lambda c: ratios[c], reverse=True)

    # ── Build per-speaker mean weight matrix ─────────────────────────────────
    subset = merged[merged['speaker'].isin(qualified_spk)]

    def build_matrix(df, speakers, cols):
        rows = []
        for spk in speakers:
            spk_df = df[df['speaker'] == spk]
            rows.append(spk_df[cols].mean().values)
        return np.array(rows)

    # Panel A: all talks
    mat_all = build_matrix(subset, qualified_spk, ordered_cols)
    # Panel B: only tithing talks (NaN row if speaker has no tithing talks)
    t_subset = merged[(merged['speaker'].isin(qualified_spk)) & (merged['kw_hits'] >= 1)]
    mat_tithing_rows = []
    for spk in qualified_spk:
        spk_df = t_subset[t_subset['speaker'] == spk]
        if len(spk_df) == 0:
            mat_tithing_rows.append(np.full(len(ordered_cols), np.nan))
        else:
            mat_tithing_rows.append(spk_df[ordered_cols].mean().values)
    mat_tithing = np.array(mat_tithing_rows)

    # ── Column labels ──────────────────────────────────────────────────────
    col_labels = [_topic_label(topic_labels, int(c.split('_')[1]), n_words=3)
                  for c in ordered_cols]

    # ── Tithing-rate annotation for speaker axis ───────────────────────────
    spk_labels = [
        f"{spk}  ({spk_stats.loc[spk,'rate']:.0%})"
        if spk in spk_stats.index else spk
        for spk in qualified_spk
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'All talks (topic fingerprint)',
            'Tithing talks only (topic shift)',
        ],
        horizontal_spacing=0.06,
    )

    common_heatmap = dict(
        x=col_labels,
        colorscale='YlOrRd',
        zmin=0, zmax=0.12,
        xgap=1, ygap=1,
    )

    fig.add_trace(go.Heatmap(
        **common_heatmap,
        z=mat_all,
        y=spk_labels,
        name='All talks',
        colorbar=dict(title='Mean<br>weight', x=0.44, thickness=12, len=0.8),
        hovertemplate='Speaker: %{y}<br>Topic: %{x}<br>Mean weight: %{z:.3f}<extra></extra>',
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        **common_heatmap,
        z=mat_tithing,
        y=spk_labels,
        name='Tithing talks',
        colorbar=dict(title='Mean<br>weight', x=1.01, thickness=12, len=0.8),
        hovertemplate='Speaker: %{y}<br>Topic: %{x}<br>Mean weight (tithing talks): %{z:.3f}<extra></extra>',
    ), row=1, col=2)

    # Mark the first few columns (highest co-occurrence) with a shape
    n_mark = min(5, len(ordered_cols))
    for col_idx in range(n_mark):
        for panel_col in [1, 2]:
            fig.add_shape(
                type='rect',
                x0=col_idx - 0.5, x1=col_idx + 0.5,
                y0=-0.5, y1=len(qualified_spk) - 0.5,
                xref=f'x{"" if panel_col==1 else "2"}',
                yref=f'y{"" if panel_col==1 else "2"}',
                line=dict(color='#1D4ED8', width=1.5),
                fillcolor='rgba(0,0,0,0)',
            )

    fig.add_annotation(
        x=0.5, y=1.06, xref='paper', yref='paper',
        text='Blue outlines = topics most correlated with tithing | '
             'Speakers sorted by tithing mention rate (highest at top) | '
             f'Min {MIN_TALKS} total talks',
        showarrow=False, font=dict(size=10, color='#6B7280'),
    )

    # (heatmap replaced by fig5_speaker_topic_profile below)
    pass


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – Speaker topic profiles (stacked bar, replaces heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def fig5_speaker_topic_profile(flags: pd.DataFrame, doc_topics: pd.DataFrame,
                                topic_labels: pd.DataFrame):
    """
    For each qualified speaker (>= MIN_TALKS total talks), show a horizontal
    100 %-stacked bar where each segment is the share of their talks where a
    given topic is dominant.

    Only the 10 most common content topics (across all speakers) are shown
    individually; the rest are collapsed into 'Other'.

    Speakers are sorted top-to-bottom by their tithing-talk fraction so
    heavy tithing speakers appear at the top.
    """
    # ── Identify the top-10 content topics by frequency in corpus ────────────
    content_mask = ~doc_topics['dominant_topic'].isin(PROCEDURAL_TOPICS)
    top_topics = (
        doc_topics[content_mask]['dominant_topic']
        .value_counts()
        .head(10)
        .index.tolist()
    )

    # ── Speaker selection: qualified, sorted by tithing rate ─────────────────
    total_spk   = doc_topics.groupby('speaker').size().rename('total')
    tithing_spk = (
        doc_topics.merge(flags[['uri', 'kw_hits']], on='uri', how='left')
        .query('kw_hits >= 1')
        .groupby('speaker').size().rename('tithing')
    )
    spk_stats = pd.concat([total_spk, tithing_spk], axis=1).fillna(0)
    spk_stats['rate'] = spk_stats['tithing'] / spk_stats['total']
    qualified = (
        spk_stats[spk_stats['total'] >= MIN_TALKS]
        .sort_values('rate', ascending=True)   # ascending = bottom of chart is lowest
        .head(40)
    )
    speakers = qualified.index.tolist()

    # ── Build topic-share matrix ──────────────────────────────────────────────
    merged = doc_topics[doc_topics['speaker'].isin(speakers)].copy()
    merged['topic_bucket'] = merged['dominant_topic'].apply(
        lambda t: t if t in top_topics else -1   # -1 = Other
    )

    rows = []
    for spk in speakers:
        spk_df = merged[merged['speaker'] == spk]
        n = len(spk_df)
        for tid in top_topics:
            rows.append({
                'speaker': spk,
                'topic_id': tid,
                'share': (spk_df['topic_bucket'] == tid).sum() / n * 100,
            })
        rows.append({
            'speaker': spk,
            'topic_id': -1,
            'share': (spk_df['topic_bucket'] == -1).sum() / n * 100,
        })

    matrix = pd.DataFrame(rows)

    # ── Colour palette ────────────────────────────────────────────────────────
    topic_ids_ordered = top_topics + [-1]
    color_seq = px.colors.qualitative.Plotly + px.colors.qualitative.Bold
    id_to_color = {tid: color_seq[i % len(color_seq)]
                   for i, tid in enumerate(topic_ids_ordered)}
    id_to_color[-1] = '#D1D5DB'   # grey for Other

    def topic_short_label(tid):
        if tid == -1:
            return 'Other'
        row = topic_labels[topic_labels['topic_id'] == tid]
        if row.empty:
            return f'T{tid:02d}'
        words = [str(row.iloc[0].get(f'word_{i}', '')) for i in range(1, 4)]
        return ' / '.join(w for w in words if w and w != 'nan')

    # ── Speaker y-axis labels (name + tithing %) ──────────────────────────────
    spk_labels = [
        f"{spk}  ({qualified.loc[spk, 'rate']:.0%} tithing)"
        for spk in speakers
    ]

    fig = go.Figure()
    for tid in topic_ids_ordered:
        subset = matrix[matrix['topic_id'] == tid]
        # align to speakers list order
        share_by_spk = subset.set_index('speaker')['share'].reindex(speakers).fillna(0)
        fig.add_trace(go.Bar(
            x=share_by_spk.values,
            y=spk_labels,
            orientation='h',
            name=topic_short_label(tid),
            marker_color=id_to_color[tid],
            hovertemplate='<b>%{y}</b><br>' + topic_short_label(tid) +
                          ': %{x:.1f}% of talks<extra></extra>',
        ))

    fig.update_layout(
        barmode='stack',
        title=(
            'Speaker Topic Profiles — What Does Each Leader Speak About?<br>'
            '<sup>100% stacked bar: share of each speaker\'s talks where a topic is dominant | '
            f'Sorted by tithing mention rate | Min {MIN_TALKS} total talks</sup>'
        ),
        xaxis=dict(title='% of talks', ticksuffix='%', range=[0, 100]),
        yaxis=dict(title=''),
        legend=dict(
            orientation='h', x=0.5, xanchor='center', y=-0.18,
            title='Dominant topic',
        ),
        margin=dict(t=80, b=140, l=230, r=30),
        height=max(560, len(speakers) * 24 + 200),
    )
    save(fig, 'tithing_speaker_topic_profile',
         height=max(560, len(speakers) * 24 + 200))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 – Recession vs. tithing frequency
# ─────────────────────────────────────────────────────────────────────────────

# NBER recession periods (start_year_frac, end_year_frac) since 1971
_RECESSION_RANGES = [
    (1973 + 10/12, 1975 +  3/12),
    (1980 +  0/12, 1980 +  7/12),
    (1981 +  6/12, 1982 + 11/12),
    (1990 +  6/12, 1991 +  3/12),
    (2001 +  2/12, 2001 + 11/12),
    (2007 + 11/12, 2009 +  6/12),
    (2020 +  1/12, 2020 +  4/12),
]

def _recession_overlap(year: int) -> float:
    """Return the fraction of year `year` that falls inside an NBER recession (0–1)."""
    total = 0.0
    for start, end in _RECESSION_RANGES:
        overlap_start = max(start, year)
        overlap_end   = min(end,   year + 1)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return min(total, 1.0)


def fig6_recession_tithing(flags: pd.DataFrame):
    """
    Two-panel figure examining whether tithing discourse tracks US recessions.

    Panel A – Scatter: each year as a point, x = year, y = % of talks
      mentioning tithing.  Points in recession years are red; expansion years
      are blue.  Separate least-squares trend lines for each group.  A
      two-sample t-test p-value is annotated.

    Panel B – Lead / lag bar chart: mean tithing % at t-2, t-1, t=0 (first
      year of recession), t+1, t+2, compared to the overall mean.  Shows
      whether tithing discourse rises *before*, *during* or *after* a
      recession.
    """
    from scipy import stats as sp_stats

    # ── Year-level stats ──────────────────────────────────────────────────────
    yr = (
        flags.groupby('year')
        .agg(total=('uri', 'count'),
             mentioned=('kw_hits', lambda s: (s >= 1).sum()))
        .reset_index()
    )
    yr['pct'] = yr['mentioned'] / yr['total'] * 100
    yr['recession_frac'] = yr['year'].apply(_recession_overlap)
    yr['in_recession']   = yr['recession_frac'] > 0   # any overlap = recession year

    rec  = yr[yr['in_recession']]['pct']
    exp  = yr[~yr['in_recession']]['pct']
    t_stat, p_val = sp_stats.ttest_ind(rec, exp, equal_var=False)
    p_str = f'p = {p_val:.3f}' if p_val >= 0.001 else 'p < 0.001'

    # ── Trend lines ───────────────────────────────────────────────────────────
    def trend(df_sub):
        x = df_sub['year'].values
        y = df_sub['pct'].values
        m, b, *_ = sp_stats.linregress(x, y)
        x_line = np.array([x.min(), x.max()])
        return x_line, m * x_line + b

    rec_x, rec_y = trend(yr[yr['in_recession']])
    exp_x, exp_y = trend(yr[~yr['in_recession']])

    # ── Panel B: lead/lag ─────────────────────────────────────────────────────
    # Identify the first year of each recession
    recession_start_years = []
    for start, end in _RECESSION_RANGES:
        start_yr = int(start)
        if yr['year'].min() <= start_yr <= yr['year'].max():
            recession_start_years.append(start_yr)

    overall_mean = yr['pct'].mean()
    lags = range(-2, 3)   # t-2 to t+2
    lag_means = {}
    for lag in lags:
        pcts = []
        for ry in recession_start_years:
            target = ry + lag
            row = yr[yr['year'] == target]
            if not row.empty:
                pcts.append(row.iloc[0]['pct'])
        lag_means[lag] = np.mean(pcts) if pcts else np.nan

    lag_labels  = ['t-2', 't-1', 't=0\n(recession\nstart)', 't+1', 't+2']
    lag_values  = [lag_means[l] for l in lags]
    lag_diffs   = [v - overall_mean if not np.isnan(v) else 0 for v in lag_values]
    lag_colors  = [RED if d > 0 else BLUE for d in lag_diffs]

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Tithing Talk % — Recession vs. Expansion Years',
            'Lead / Lag Around Recession Start',
        ],
        horizontal_spacing=0.14,
    )

    # Panel A – scatter
    for label, mask, color in [
        ('Expansion year', ~yr['in_recession'], BLUE),
        ('Recession year', yr['in_recession'],  RED),
    ]:
        sub = yr[mask]
        fig.add_trace(go.Scatter(
            x=sub['year'], y=sub['pct'],
            mode='markers',
            name=label,
            marker=dict(color=color, size=8, opacity=0.8,
                        line=dict(color='white', width=0.5)),
            hovertemplate='%{x}: %{y:.1f}%<extra>' + label + '</extra>',
        ), row=1, col=1)

    # Trend lines
    fig.add_trace(go.Scatter(
        x=exp_x, y=exp_y, mode='lines',
        line=dict(color=BLUE, width=2, dash='dot'),
        name='Expansion trend', showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=rec_x, y=rec_y, mode='lines',
        line=dict(color=RED, width=2, dash='dot'),
        name='Recession trend', showlegend=False,
    ), row=1, col=1)

    # Mean lines
    fig.add_hline(y=rec.mean(),  line_dash='dash', line_color=RED,
                  annotation_text=f'Recession mean {rec.mean():.1f}%',
                  annotation_position='top right', row=1, col=1)
    fig.add_hline(y=exp.mean(), line_dash='dash', line_color=BLUE,
                  annotation_text=f'Expansion mean {exp.mean():.1f}%',
                  annotation_position='bottom right', row=1, col=1)

    # t-test annotation
    direction = 'higher' if rec.mean() > exp.mean() else 'lower'
    sig = 'significant' if p_val < 0.05 else 'not significant'
    fig.add_annotation(
        x=0.01, y=0.04, xref='paper', yref='paper',
        text=(f'Welch t-test: recession % is {direction} ({sig})<br>'
              f't = {t_stat:.2f}, {p_str}'),
        showarrow=False, align='left',
        font=dict(size=11),
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='#9CA3AF', borderwidth=1,
    )

    # Panel B – bar
    fig.add_trace(go.Bar(
        x=lag_labels,
        y=lag_diffs,
        marker_color=lag_colors,
        text=[f'{v:.1f}%' for v in lag_values],
        textposition='outside',
        name='Tithing % vs. overall mean',
        hovertemplate='%{x}<br>Tithing %: %{text}<br>vs. mean: %{y:+.2f}pp<extra></extra>',
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash='solid', line_color='#9CA3AF',
                  line_width=1, row=1, col=2)
    fig.add_annotation(
        x=1.0, y=0.04, xref='paper', yref='paper',
        text=f'Overall mean: {overall_mean:.1f}%',
        showarrow=False, align='right',
        font=dict(size=10, color='#6B7280'),
    )

    fig.update_xaxes(title_text='Year', dtick=10, row=1, col=1)
    fig.update_yaxes(title_text='% of talks mentioning tithing',
                     ticksuffix='%', row=1, col=1)
    fig.update_xaxes(title_text='Years relative to recession start', row=1, col=2)
    fig.update_yaxes(title_text='Difference from overall mean (pp)',
                     ticksuffix='pp', row=1, col=2)

    fig.update_layout(
        title=('Recession Context and Tithing Discourse in General Conference<br>'
               '<sup>Left: each year as a point, colored by NBER recession status | '
               'Right: average tithing talk % around the start of each recession</sup>'),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.15),
        margin=dict(t=80, b=100, l=70, r=30),
    )
    save(fig, 'tithing_recession_analysis')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    flags_path        = PROC_DIR / 'tithing_flags.parquet'
    doc_topics_path   = PROC_DIR / 'doc_topics_full.parquet'
    topic_labels_path = TAB_DIR  / 'topic_labels_full.csv'

    for p in [flags_path, doc_topics_path, topic_labels_path]:
        if not p.exists():
            print(f'Missing: {p.name} -- run earlier pipeline steps first.')
            return

    flags        = pd.read_parquet(flags_path)
    doc_topics   = pd.read_parquet(doc_topics_path)
    topic_labels = pd.read_csv(topic_labels_path)

    print(f'Loaded {len(flags):,} talks, '
          f'{(flags.kw_hits >= 1).sum():,} mention tithing, '
          f'{(flags.kw_hits >= CORE_FOCUS_THRESHOLD).sum():,} core-focus.')

    print('\nGenerating Figure 1: Core focus vs. mentioned...')
    fig1_core_vs_mention(flags)

    print('Generating Figure 2: Explicit frequency over time...')
    fig2_explicit_frequency(flags)

    print('Generating Figure 3: Co-occurring themes...')
    fig3_cooccurrence_themes(flags, doc_topics, topic_labels)

    print('Generating Figure 4: Speaker profiles...')
    fig4_speaker_profiles(flags)

    print('Generating Figure 5: Speaker topic profiles (stacked bar)...')
    fig5_speaker_topic_profile(flags, doc_topics, topic_labels)

    print('Generating Figure 6: Recession vs. tithing analysis...')
    fig6_recession_tithing(flags)

    print(f'\nAll figures saved to {FIG_DIR}/')


if __name__ == '__main__':
    main()
