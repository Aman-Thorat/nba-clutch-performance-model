# ============================================================
# CLUTCH PLAYER PERFORMANCE MODEL
# Streamlit Dashboard
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 72
plt.rcParams['savefig.dpi'] = 72

# Override PIL decompression bomb limit
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Removes the size limit entirely

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="NBA Clutch Performance Model",
    page_icon="🏀",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("shots_engineered.csv")
    return df

@st.cache_data
def load_clutch_summary():
    df = pd.read_csv("shots_engineered.csv")

    # Rebuild clutch summary for all seasons
    clutch = df[df['is_clutch'] == 1].copy()

    summary = (
        clutch.groupby(['player', 'season'])
        .agg(
            clutch_attempts = ('shot_made', 'count'),
            actual_makes    = ('shot_made', 'sum'),
        )
        .reset_index()
    )
    summary['actual_fg_pct'] = (
        summary['actual_makes'] /
        summary['clutch_attempts'] * 100
    )
    return summary

df = load_data()
clutch_summary = load_clutch_summary()

# ── Header ────────────────────────────────────────────────
st.title("🏀 NBA Clutch Player Performance Model")
st.markdown(
    "Identifying truly clutch NBA players using machine learning. "
    "A shot is **clutch** if it occurs in Q4/OT, ≤5 minutes remaining, "
    "and the score margin is ≤5 points."
)
st.markdown("---")

# ── Sidebar filters ───────────────────────────────────────
st.sidebar.title("🔧 Filters")

seasons = sorted(df['season'].unique())
selected_season = st.sidebar.selectbox(
    "Select Season", options=['All'] + seasons
)

min_attempts = st.sidebar.slider(
    "Minimum Clutch Attempts",
    min_value=5,
    max_value=50,
    value=15,
    step=5
)

# ── Key Metrics Row ───────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

filter_df = df if selected_season == 'All' else df[df['season'] == selected_season]
clutch_df = filter_df[filter_df['is_clutch'] == 1]

col1.metric("Total Shots Analyzed",
            f"{len(filter_df):,}")
col2.metric("Clutch Shots",
            f"{len(clutch_df):,}",
            f"{len(clutch_df)/len(filter_df):.1%} of total")
col3.metric("Overall FG%",
            f"{filter_df['shot_made'].mean():.1%}")
col4.metric("Clutch FG%",
            f"{clutch_df['shot_made'].mean():.1%}",
            f"{(clutch_df['shot_made'].mean() - filter_df['shot_made'].mean())*100:.1f}% vs normal")

st.markdown("---")

# ── Tab layout ────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Clutch Rankings",
    "🔍 Player Deep Dive",
    "📈 Season Trends"
])

# ════════════════════════════════════════════════════════
# TAB 1: Clutch Rankings
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Clutch Player Rankings")
    st.markdown(
        "Players ranked by **Clutch FG%** with minimum attempt threshold applied."
    )

    # Build ranking table
    season_filter = (
        clutch_summary if selected_season == 'All'
        else clutch_summary[clutch_summary['season'] == selected_season]
    )

    if selected_season == 'All':
        ranking = (
            season_filter.groupby('player')
            .agg(
                clutch_attempts = ('clutch_attempts', 'sum'),
                actual_makes    = ('actual_makes', 'sum')
            )
            .reset_index()
        )
        ranking['actual_fg_pct'] = (
            ranking['actual_makes'] /
            ranking['clutch_attempts'] * 100
        )
    else:
        ranking = season_filter.copy()

    # Apply minimum attempts filter
    ranking = ranking[
        ranking['clutch_attempts'] >= min_attempts
    ].sort_values('actual_fg_pct', ascending=False).reset_index(drop=True)
    ranking.index += 1  # Start rank at 1

    # Top and bottom charts side by side
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**🟢 Top 15 Clutch Performers**")
        top15 = ranking.head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(top15['player'], top15['actual_fg_pct'],
                       color='#2ecc71', edgecolor='white')
        ax.axvline(x=clutch_df['shot_made'].mean()*100,
                   color='gray', linestyle='--', linewidth=1,
                   label=f'Avg: {clutch_df["shot_made"].mean()*100:.1f}%')
        ax.set_xlabel('Clutch FG%')
        ax.set_title('Most Clutch Players', fontweight='bold')
        ax.invert_yaxis()
        ax.legend(fontsize=9)
        for bar, (_, row) in zip(bars, top15.iterrows()):
            ax.text(bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height()/2,
                    f"{row['actual_fg_pct']:.1f}%",
                    va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**🔴 Bottom 15 Clutch Performers**")
        bottom15 = ranking.tail(15).sort_values('actual_fg_pct')
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(bottom15['player'], bottom15['actual_fg_pct'],
                       color='#e74c3c', edgecolor='white')
        ax.axvline(x=clutch_df['shot_made'].mean()*100,
                   color='gray', linestyle='--', linewidth=1,
                   label=f'Avg: {clutch_df["shot_made"].mean()*100:.1f}%')
        ax.set_xlabel('Clutch FG%')
        ax.set_title('Least Clutch Players', fontweight='bold')
        ax.invert_yaxis()
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Full table
    st.markdown("**📋 Full Clutch Rankings Table**")
    display_cols = ['player', 'clutch_attempts',
                    'actual_makes', 'actual_fg_pct']
    st.dataframe(
        ranking[display_cols].rename(columns={
            'player': 'Player',
            'clutch_attempts': 'Clutch Attempts',
            'actual_makes': 'Clutch Makes',
            'actual_fg_pct': 'Clutch FG%'
        }).style.format({'Clutch FG%': '{:.1f}%'}),
        width='stretch'
    )

# ════════════════════════════════════════════════════════
# TAB 2: Player Deep Dive
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Player Deep Dive")

    # Player selector
    all_players = sorted(df['player'].unique())
    selected_player = st.selectbox("Select a Player", all_players,
                                   index=all_players.index('A. Reaves')
                                   if 'A. Reaves' in all_players else 0)

    player_df = df[df['player'] == selected_player]
    player_clutch = player_df[player_df['is_clutch'] == 1]

    # Player metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Attempts", f"{len(player_df):,}")
    c2.metric("Overall FG%", f"{player_df['shot_made'].mean():.1%}")
    c3.metric("Clutch Attempts", f"{len(player_clutch):,}")
    c4.metric("Clutch FG%",
              f"{player_clutch['shot_made'].mean():.1%}" if len(player_clutch) > 0 else "N/A",
              f"{(player_clutch['shot_made'].mean() - player_df['shot_made'].mean())*100:.1f}% vs their own avg"
              if len(player_clutch) > 0 else "")

    # Season-by-season breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{selected_player} — Clutch FG% by Season**")
        season_stats = (
            player_clutch.groupby('season')['shot_made']
            .agg(['mean', 'count'])
            .reset_index()
        )
        season_stats.columns = ['Season', 'FG%', 'Attempts']
        season_stats['FG%'] *= 100

        if len(season_stats) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(season_stats['Season'],
                          season_stats['FG%'],
                          color='#3498db', edgecolor='white')
            ax.axhline(y=player_df['shot_made'].mean()*100,
                       color='orange', linestyle='--',
                       label='Overall FG%')
            ax.set_ylabel('Clutch FG%')
            ax.set_ylim(0, 100)
            ax.set_title(f'{selected_player}: Clutch FG% by Season',
                         fontweight='bold')
            ax.legend()
            for bar, att in zip(bars, season_stats['Attempts']):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 1,
                        f'n={att}', ha='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Not enough clutch data for this player.")

    with col2:
        st.markdown(f"**{selected_player} — Shot Distance Distribution**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(player_df[player_df['shot_made']==0]['dist'],
                bins=20, alpha=0.6, color='#e74c3c', label='Missed',
                density=True)
        ax.hist(player_df[player_df['shot_made']==1]['dist'],
                bins=20, alpha=0.6, color='#2ecc71', label='Made',
                density=True)
        ax.set_xlabel('Shot Distance (feet)')
        ax.set_ylabel('Density')
        ax.set_title(f'{selected_player}: Shot Distance', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════
# TAB 3: Season Trends
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("Season-Level Trends")

    season_trends = (
        df.groupby('season')
        .agg(
            total_shots   = ('shot_made', 'count'),
            overall_fg    = ('shot_made', 'mean'),
            clutch_shots  = ('is_clutch', 'sum'),
        )
        .reset_index()
    )
    clutch_fg_by_season = (
        df[df['is_clutch']==1]
        .groupby('season')['shot_made']
        .mean()
        .reset_index()
        .rename(columns={'shot_made': 'clutch_fg'})
    )
    season_trends = season_trends.merge(clutch_fg_by_season, on='season')
    season_trends['overall_fg'] *= 100
    season_trends['clutch_fg']  *= 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Overall vs Clutch FG% by Season**")
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(season_trends))
        ax.plot(season_trends['season'], season_trends['overall_fg'],
                marker='o', color='#4C72B0', linewidth=2,
                label='Overall FG%')
        ax.plot(season_trends['season'], season_trends['clutch_fg'],
                marker='s', color='#DD8452', linewidth=2,
                label='Clutch FG%')
        ax.fill_between(season_trends['season'],
                        season_trends['overall_fg'],
                        season_trends['clutch_fg'],
                        alpha=0.1, color='red')
        ax.set_ylabel('FG%')
        ax.set_title('FG% Trends: Overall vs Clutch', fontweight='bold')
        ax.legend()
        ax.set_ylim(40, 55)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Clutch Shot Volume by Season**")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(season_trends['season'], season_trends['clutch_shots'],
               color='#DD8452', edgecolor='white')
        ax.set_ylabel('Number of Clutch Shots')
        ax.set_title('Clutch Shot Volume per Season', fontweight='bold')
        for i, (_, row) in enumerate(season_trends.iterrows()):
            ax.text(i, row['clutch_shots'] + 50,
                    f"{int(row['clutch_shots']):,}",
                    ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("**Season Summary Table**")
    st.dataframe(
        season_trends.rename(columns={
            'season': 'Season',
            'total_shots': 'Total Shots',
            'overall_fg': 'Overall FG%',
            'clutch_shots': 'Clutch Shots',
            'clutch_fg': 'Clutch FG%'
        }).style.format({
            'Overall FG%': '{:.1f}%',
            'Clutch FG%': '{:.1f}%',
            'Total Shots': '{:,.0f}',
            'Clutch Shots': '{:,.0f}'
        }),
        width='stretch'
    )

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**Data:** NBA Play-by-Play 2019–2023 | "
    "**Model:** Random Forest Classifier | "
    "**Built with:** Python, scikit-learn, Streamlit"
)