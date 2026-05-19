"""
app.py

Streamlit dashboard for News Pulse.

Four tabs:
  1. Segment Overview   -- top 5 most central articles per segment
  2. Recommendations    -- paste a URL, get top 5 semantically similar articles
  3. Entity Trends      -- search an entity name, see weekly mention volume over time
  4. Top Sources        -- which outlets publish the most per segment

Run with:
  streamlit run src/dashboard/app.py

Why cache_resource vs cache_data?
  @st.cache_resource:  for objects that are expensive to create and should
                       live for the entire session -- the Recommender holds
                       90,000 embeddings in memory. Creating it on every
                       interaction would take 10+ seconds each time.

  @st.cache_data:      for query results. These are cached by their input
                       arguments, so the same query only hits MySQL once
                       per session. The cache is invalidated automatically
                       if the arguments change (e.g. a different entity name).
"""

import os
import logging
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# --- Project root and credentials ---
# Walk up from app.py's location until environment.yml is found.
# This works whether you run from the project root or the src/dashboard folder.
def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "environment.yml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root.")

import sys
PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

PROJECT_ROOT = _find_project_root()
load_dotenv(PROJECT_ROOT / ".env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME", "news_pulse")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page config -- must be the first Streamlit call in the script
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="News Pulse",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Database engine
# ---------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    """
    Creates a single SQLAlchemy engine reused across the session.
    cache_resource means this runs once, not on every interaction.
    quote_plus handles the $ and @ characters in the password.
    """
    return create_engine(
        f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASS)}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True
    )


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading recommendation engine...")
def get_recommender():
    """
    Loads the Recommender once per session.
    The Recommender object holds 90,000 embeddings in memory.
    Without cache_resource, it would reload on every user interaction.
    """
    from src.pipeline.recommend import Recommender
    return Recommender()


# ---------------------------------------------------------------------------
# Data queries
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_segments() -> list[str]:
    """Returns all segment names from dim_segment, sorted alphabetically."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT segment_name FROM dim_segment ORDER BY segment_name")
        )
        return [r[0] for r in result if r[0] != "General"]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_top_sources(segment: str, n: int = 10) -> pd.DataFrame:
    """
    Returns the top-N sources by article count for a given segment.

    Why the two-step join?
      fact_articles stores segment_id and source_id as foreign keys.
      We need to join dim_segment to filter by name and dim_source to
      get the outlet name. The COUNT is on fact_articles rows.
    """
    engine = get_engine()
    query = text("""
        SELECT
            ds.source_name,
            COUNT(fa.article_id) AS article_count
        FROM fact_articles fa
        JOIN dim_source  ds  ON fa.source_id  = ds.source_id
        JOIN dim_segment seg ON fa.segment_id = seg.segment_id
        WHERE seg.segment_name = :segment
        GROUP BY ds.source_name
        ORDER BY article_count DESC
        LIMIT :n
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"segment": segment, "n": n})
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_entity_trends(entity_name: str) -> pd.DataFrame:
    """
    Returns weekly article mention counts for a given entity name.

    Why split into two queries?
      fact_entity_mentions has 64M rows. A single JOIN across the full
      table times out even with indexes. The pattern that works is:
        1. Look up entity_id from dim_entity (tiny table, instant).
        2. Query fact_entity_mentions scoped to that single entity_id,
           joining only to fact_articles and dim_date for the time axis.
      This uses the idx_article_id index properly and returns in under 1s.

    Returns a DataFrame with columns: week_start, mention_count.
    week_start is the Monday of each ISO week.
    """
    engine = get_engine()

    # Step 1: resolve entity_id
    id_query = text("""
        SELECT entity_id
        FROM dim_entity
        WHERE LOWER(entity_name) = LOWER(:name)
        LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(id_query, {"name": entity_name}).fetchone()

    if result is None:
        return pd.DataFrame(columns=["week_start", "mention_count"])

    entity_id = result[0]

    # Step 2: count weekly mentions scoped to this entity_id
    trend_query = text("""
        SELECT
            dd.year,
            dd.week,
            COUNT(fem.article_id) AS mention_count
        FROM fact_entity_mentions fem
        JOIN fact_articles fa ON fem.article_id = fa.article_id
        JOIN dim_date      dd ON fa.date_id      = dd.date_id
        WHERE fem.entity_id = :entity_id
        GROUP BY dd.year, dd.week
        ORDER BY dd.year, dd.week
    """)
    with engine.connect() as conn:
        df = pd.read_sql(trend_query, conn, params={"entity_id": entity_id})

    if df.empty:
        return pd.DataFrame(columns=["week_start", "mention_count"])

    # Convert year + ISO week to a proper date (Monday of that week)
    # so Plotly can render a continuous time axis
    df["week_start"] = pd.to_datetime(
        df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u"
    )
    return df[["week_start", "mention_count"]].sort_values("week_start")


@st.cache_data(ttl=3600, show_spinner=False)
def search_entity_names(partial: str, limit: int = 10) -> list[str]:
    """
    Returns entity names that start with the search string.
    Used to populate the autocomplete suggestions in the entity trends tab.
    LIKE with a trailing wildcard uses the index on entity_name.
    """
    if len(partial.strip()) < 2:
        return []
    engine = get_engine()
    query = text("""
        SELECT entity_name
        FROM dim_entity
        WHERE entity_name LIKE :pattern
        ORDER BY entity_name
        LIMIT :limit
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"pattern": f"{partial}%", "limit": limit})
        return [r[0] for r in result]


@st.cache_data(ttl=3600, show_spinner=False)
def search_articles_by_entity(entity_name: str, limit: int = 20) -> pd.DataFrame:
    """
    Returns articles that mention a given entity, ordered by most recent first.

    Why two queries?
      Same pattern as fetch_entity_trends: resolve entity_id first from the
      small dim_entity table, then scope the fact_entity_mentions query to
      that single ID. Avoids a full-table scan on 64M rows.

    Returns columns: article_id, url, source_name, segment_name, seendate.
    """
    engine = get_engine()

    id_query = text("""
        SELECT entity_id
        FROM dim_entity
        WHERE LOWER(entity_name) = LOWER(:name)
        LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(id_query, {"name": entity_name}).fetchone()

    if result is None:
        return pd.DataFrame(columns=["article_id", "url", "source_name",
                                     "segment_name", "seendate"])

    entity_id = result[0]

    articles_query = text("""
        SELECT
            fa.article_id,
            fa.url,
            ds.source_name,
            seg.segment_name,
            fa.seendate
        FROM fact_entity_mentions fem
        JOIN fact_articles fa  ON fem.article_id  = fa.article_id
        JOIN dim_source    ds  ON fa.source_id     = ds.source_id
        JOIN dim_segment   seg ON fa.segment_id    = seg.segment_id
        WHERE fem.entity_id = :entity_id
        ORDER BY fa.seendate DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(articles_query, conn,
                         params={"entity_id": entity_id, "limit": limit})
    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("News Pulse")
    st.caption("Live news intelligence from GDELT")
    st.divider()

    segments = fetch_segments()
    selected_segment = st.selectbox(
        "Segment filter",
        options=segments,
        index=0,
        help="Filters the Segment Overview and Top Sources tabs."
    )

    st.divider()
    st.caption("Warehouse: 11.9M+ articles across 7 segments")
    st.caption("Updated every 15 minutes via GDELT")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Segment Overview",
    "Recommendations",
    "Entity Trends",
    "Top Sources",
])


# ---------------------------------------------------------------------------
# Tab 1: Segment Overview
# ---------------------------------------------------------------------------
with tab1:
    st.header(f"Top stories: {selected_segment}")
    st.caption(
        "The five most topically central articles in this segment, "
        "ranked by average semantic similarity to other articles in the segment. "
        "Central articles represent what the segment is predominantly covering."
    )

    rec = get_recommender()

    with st.spinner("Finding central articles..."):
        central = rec.recommend_by_segment(selected_segment, n=5)

    if central.empty:
        st.warning(f"No articles found for '{selected_segment}' in the current index.")
    else:
        for _, row in central.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**{row['source_name']}**")
                    st.markdown(f"[{row['url']}]({row['url']})")
                    st.caption(
                        f"Published: {pd.to_datetime(row['seendate']).strftime('%d %b %Y, %H:%M')}"
                    )
                with col2:
                    st.metric(
                        label="Centrality",
                        value=f"{row['centrality_score']:.3f}",
                        help="Average cosine similarity to other articles in this segment. "
                             "Higher means more representative of the segment's current coverage."
                    )


# ---------------------------------------------------------------------------
# Tab 2: Recommendations
# ---------------------------------------------------------------------------
with tab2:
    st.header("Article recommendations")
    st.caption(
        "Search by topic or person to find a starting article, "
        "then get the five most semantically similar articles from the warehouse."
    )

    n_results = st.slider("Number of results", min_value=1, max_value=10, value=5)

    st.divider()

    # --- Step 1: search by entity to find a starting article ---
    st.subheader("Step 1 — Find a starting article")
    search_input = st.text_input(
        "Search by topic, person, or organisation",
        placeholder="Manchester United, Elon Musk, Apple...",
        help="Type at least 2 characters. The search matches entity names "
             "extracted from article metadata."
    )

    selected_article_id = None
    selected_url        = None

    if search_input and len(search_input.strip()) >= 2:
        suggestions = search_entity_names(search_input.strip())

        if not suggestions:
            st.info("No matching entities found. Try a different spelling.")
        else:
            chosen_entity = st.selectbox(
                "Select entity",
                options=suggestions,
                help="Choose the exact entity to search for."
            )

            with st.spinner(f"Loading articles mentioning {chosen_entity}..."):
                articles_df = search_articles_by_entity(chosen_entity, limit=20)

            if articles_df.empty:
                st.warning(f"No articles found for '{chosen_entity}'.")
            else:
                # Build a readable label for each article so the selectbox
                # shows something meaningful instead of a raw URL.
                articles_df["label"] = (
                    articles_df["source_name"]
                    + "  |  "
                    + articles_df["segment_name"]
                    + "  |  "
                    + pd.to_datetime(articles_df["seendate"]).dt.strftime("%d %b %Y")
                )

                chosen_label = st.selectbox(
                    f"Select an article ({len(articles_df)} found)",
                    options=articles_df["label"].tolist(),
                    help="Pick the article you want recommendations for."
                )

                chosen_row = articles_df[
                    articles_df["label"] == chosen_label
                ].iloc[0]

                selected_article_id = int(chosen_row["article_id"])
                selected_url        = chosen_row["url"]

                st.caption(f"URL: {selected_url}")

    # --- Step 2: run recommendations on the chosen article ---
    if selected_article_id is not None:
        st.divider()
        st.subheader("Step 2 — Similar articles")

        rec = get_recommender()
        with st.spinner("Finding similar articles..."):
            results = rec.recommend(article_id=selected_article_id, n=n_results)

        if results.empty:
            st.warning(
                "Article not found in the recommendation index. "
                "It may have been ingested after the last cache build. "
                "To rebuild: `from src.pipeline.recommend import Recommender; "
                "Recommender(force_rebuild=True)`"
            )
        else:
            st.success(f"Top {len(results)} recommendations")
            for _, row in results.iterrows():
                with st.container(border=True):
                    col1, col2, col3 = st.columns([4, 2, 1])
                    with col1:
                        st.markdown(f"**{row['source_name']}**")
                        st.markdown(f"[{row['url']}]({row['url']})")
                    with col2:
                        st.caption(row["segment_name"])
                        st.caption(
                            pd.to_datetime(row["seendate"]).strftime("%d %b %Y")
                        )
                    with col3:
                        st.metric(
                            label="Similarity",
                            value=f"{row['similarity_score']:.3f}"
                        )


# ---------------------------------------------------------------------------
# Tab 3: Entity Trends
# ---------------------------------------------------------------------------
with tab3:
    st.header("Entity trends over time")
    st.caption(
        "Search for a person or organisation to see how many articles "
        "mentioned them per week across the full dataset period."
    )

    entity_input = st.text_input(
        "Entity name",
        placeholder="Joe Biden, Reuters, Apple...",
        help="Partial matches are supported. The search is case-insensitive."
    )

    if entity_input and len(entity_input.strip()) >= 2:
        # Show matching entity names so the user can confirm spelling
        suggestions = search_entity_names(entity_input.strip())
        if suggestions:
            chosen = st.selectbox(
                "Select exact entity",
                options=suggestions,
                help="Choose the exact entity name to plot."
            )

            with st.spinner(f"Loading trend for {chosen}..."):
                trend_df = fetch_entity_trends(chosen)

            if trend_df.empty:
                st.warning(f"No mention data found for '{chosen}'.")
            else:
                total_mentions = trend_df["mention_count"].sum()
                peak_week = trend_df.loc[
                    trend_df["mention_count"].idxmax(), "week_start"
                ]
                peak_count = trend_df["mention_count"].max()

                col1, col2, col3 = st.columns(3)
                col1.metric("Total mentions", f"{total_mentions:,}")
                col2.metric("Peak week", peak_week.strftime("%d %b %Y"))
                col3.metric("Peak mentions", f"{peak_count:,}")

                fig = px.area(
                    trend_df,
                    x="week_start",
                    y="mention_count",
                    labels={
                        "week_start":    "Week",
                        "mention_count": "Article mentions"
                    },
                    title=f"Weekly mentions: {chosen}",
                    template="plotly_white",
                )
                fig.update_traces(
                    line_color="#2563EB",
                    fillcolor="rgba(37, 99, 235, 0.12)"
                )
                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title="Mentions per week",
                    hovermode="x unified",
                    margin=dict(t=40, b=20, l=0, r=0),
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No entities found matching that name. Try a different spelling.")

    elif entity_input and len(entity_input.strip()) < 2:
        st.info("Type at least 2 characters to search.")


# ---------------------------------------------------------------------------
# Tab 4: Top Sources
# ---------------------------------------------------------------------------
with tab4:
    st.header(f"Top sources: {selected_segment}")
    st.caption(
        "The ten outlets publishing the most articles in this segment. "
        "Use the segment filter in the sidebar to switch segments."
    )

    with st.spinner("Loading source data..."):
        sources_df = fetch_top_sources(selected_segment, n=10)

    if sources_df.empty:
        st.warning(f"No source data found for '{selected_segment}'.")
    else:
        fig = px.bar(
            sources_df.sort_values("article_count"),
            x="article_count",
            y="source_name",
            orientation="h",
            labels={
                "article_count": "Articles published",
                "source_name":   "Source"
            },
            title=f"Top 10 sources in {selected_segment}",
            template="plotly_white",
        )
        fig.update_traces(marker_color="#2563EB")
        fig.update_layout(
            yaxis_title=None,
            xaxis_title="Articles published",
            margin=dict(t=40, b=20, l=0, r=0),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            sources_df.rename(columns={
                "source_name":   "Source",
                "article_count": "Articles"
            }),
            use_container_width=True,
            hide_index=True,
        )
