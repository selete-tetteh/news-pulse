"""
load.py

Writes transformed article and entity data into the news_pulse warehouse.

Takes the two outputs from transform.py:
  - df_articles:    one row per article with segment, sentiment, and metadata
  - entity_records: flat list of entity dicts with url as the join key

Loading order respects foreign key constraints:
  1. dim_source      -- must exist before fact_articles references it
  2. dim_date        -- looked up by date, must be pre-populated
  3. dim_segment     -- looked up by segment name, seeded at schema creation
  4. fact_articles   -- inserted after all dimension lookups succeed
  5. dim_entity      -- upserted before fact_entity_mentions references it
  6. fact_entity_mentions -- inserted last

Duplicate articles (same URL) are skipped via INSERT IGNORE.

Usage:
  from src.pipeline.load import load

  load(df_articles, entity_records)
"""

import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from dotenv import load_dotenv
from pathlib import Path

log = logging.getLogger(__name__)

# --- Project root and credentials ---
def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "environment.yml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root — environment.yml not found.")

PROJECT_ROOT = find_project_root()
load_dotenv(PROJECT_ROOT / ".env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASSWORD")
if DB_PASS is None:
    raise ValueError("DB_PASSWORD not found in .env — check your .env file exists and is populated")
DB_NAME = os.getenv("DB_NAME", "news_pulse")

if DB_PASS is None:
    raise ValueError("DB_PASSWORD not found in .env — check your .env file exists and is populated")


def get_engine():
    password = quote_plus(DB_PASS) # pyright: ignore[reportArgumentType, reportCallIssue]
    return create_engine(
        f"mysql+pymysql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True  # verifies connection is alive before using it
    )


# --- Dimension loaders ---

def _upsert_sources(conn, source_names: list[str]) -> dict[str, int]:
    """
    Inserts any new source names into dim_source and returns
    a mapping of source_name -> source_id for all sources in the batch.
    INSERT IGNORE skips sources that already exist.
    """
    if not source_names:
        return {}

    # Insert new sources
    for name in set(source_names):
        conn.execute(
            text("INSERT IGNORE INTO dim_source (source_name) VALUES (:name)"),
            {"name": name}
        )

    # Fetch IDs for all sources in this batch
    placeholders = ", ".join([f":s{i}" for i in range(len(source_names))])
    params = {f"s{i}": name for i, name in enumerate(set(source_names))}
    result = conn.execute(
        text(f"SELECT source_name, source_id FROM dim_source WHERE source_name IN ({placeholders})"),
        params
    )
    return {row.source_name: row.source_id for row in result}


def _lookup_date_ids(conn, dates: pd.Series) -> dict:
    """
    Looks up date_id values from dim_date for a Series of datetimes.
    date_id is stored as YYYYMMDD integer.
    Dates not found in dim_date are excluded — this should not happen
    if populate_dim_date.py was run and the date range is covered.
    """
    date_ints = dates.dropna().dt.strftime("%Y%m%d").astype(int).unique().tolist()
    if not date_ints:
        return {}

    placeholders = ", ".join([f":d{i}" for i in range(len(date_ints))])
    params = {f"d{i}": d for i, d in enumerate(date_ints)}
    result = conn.execute(
        text(f"SELECT date_id, full_date FROM dim_date WHERE date_id IN ({placeholders})"),
        params
    )
    return {str(row.full_date): row.date_id for row in result}


def _lookup_segment_ids(conn) -> dict[str, int]:
    """
    Loads the full segment name -> segment_id mapping from dim_segment.
    This is a small table (7 rows) so we load it all at once.
    """
    result = conn.execute(text("SELECT segment_id, segment_name FROM dim_segment"))
    return {row.segment_name: row.segment_id for row in result}


# --- Fact loaders ---

def _insert_articles(conn, df: pd.DataFrame,
                     source_map: dict, date_map: dict,
                     segment_map: dict) -> dict[str, int]:
    """
    Inserts articles into fact_articles after resolving all foreign keys.
    Returns a mapping of url -> article_id for use when loading entity mentions.
    Rows where any foreign key lookup fails are skipped with a warning.
    """
    inserted = 0
    skipped  = 0
    url_to_article_id = {}

    for _, row in df.iterrows():
        source_id  = source_map.get(row.get("source_name"))
        date_key   = str(row["seendate"].date()) if pd.notna(row.get("seendate")) else None
        date_id    = date_map.get(date_key)
        segment_id = segment_map.get(row.get("segment_name"),
                                     segment_map.get("General"))

        if not all([source_id, date_id, segment_id]):
            skipped += 1
            continue

        result = conn.execute(text("""
            INSERT IGNORE INTO fact_articles
                (source_id, date_id, segment_id, url, seendate,
                 sentiment_score, sentiment_label, language)
            VALUES
                (:source_id, :date_id, :segment_id, :url, :seendate,
                 :sentiment_score, :sentiment_label, :language)
        """), {
            "source_id":       source_id,
            "date_id":         date_id,
            "segment_id":      segment_id,
            "url":             str(row.get("url", "")),
            "seendate":        row.get("seendate"),
            "sentiment_score": row.get("sentiment_score"),
            "sentiment_label": row.get("sentiment_label"),
            "language":        row.get("language", "English"),
        })

        if result.rowcount > 0:
            inserted += 1
            # Fetch the article_id for the entity mentions join
            id_result = conn.execute(
                text("SELECT article_id FROM fact_articles WHERE url = :url"),
                {"url": str(row.get("url", ""))}
            )
            id_row = id_result.fetchone()
            if id_row:
                url_to_article_id[str(row.get("url", ""))] = id_row.article_id

    log.info(f"fact_articles — inserted: {inserted}, skipped: {skipped}")
    return url_to_article_id


def _insert_entities(conn, entity_records: list[dict],
                     url_to_article_id: dict) -> None:
    """
    Upserts entities into dim_entity, then inserts mention records
    into fact_entity_mentions using the url -> article_id mapping.
    """
    if not entity_records:
        log.info("No entity records to load")
        return

    # Upsert all unique entities into dim_entity
    unique_entities = {
        (e["entity_name"], e["entity_type"])
        for e in entity_records
    }

    for name, etype in unique_entities:
        conn.execute(text("""
            INSERT IGNORE INTO dim_entity (entity_name, entity_type)
            VALUES (:name, :etype)
        """), {"name": name, "etype": etype})

    # Build entity lookup map
    result = conn.execute(text("SELECT entity_id, entity_name, entity_type FROM dim_entity"))
    entity_map = {(row.entity_name, row.entity_type): row.entity_id for row in result}

    # Insert mention records
    inserted = 0
    skipped  = 0

    for record in entity_records:
        article_id = url_to_article_id.get(record["url"])
        entity_id  = entity_map.get((record["entity_name"], record["entity_type"]))

        if not article_id or not entity_id:
            skipped += 1
            continue

        conn.execute(text("""
            INSERT IGNORE INTO fact_entity_mentions (article_id, entity_id)
            VALUES (:article_id, :entity_id)
        """), {"article_id": article_id, "entity_id": entity_id})
        inserted += 1

    log.info(f"fact_entity_mentions — inserted: {inserted}, skipped: {skipped}")


# --- Main load function ---

def load(df_articles: pd.DataFrame, entity_records: list[dict]) -> None:
    """
    Loads transformed articles and entity records into the warehouse.
    All inserts run inside a single transaction — if anything fails,
    the entire batch is rolled back so the warehouse stays consistent.
    """
    if df_articles.empty:
        log.warning("load() received an empty DataFrame — nothing to load")
        return

    log.info(f"Loading {len(df_articles)} articles and {len(entity_records)} entity mentions...")

    engine = get_engine()

    with engine.begin() as conn:
        source_names = df_articles["source_name"].dropna().unique().tolist()
        source_map   = _upsert_sources(conn, source_names)
        date_map     = _lookup_date_ids(conn, df_articles["seendate"])
        segment_map  = _lookup_segment_ids(conn)

        url_to_article_id = _insert_articles(
            conn, df_articles, source_map, date_map, segment_map
        )
        _insert_entities(conn, entity_records, url_to_article_id)

    log.info("Load complete")


# --- Quick test when run directly ---
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    from src.pipeline.ingest import fetch_latest
    from src.pipeline.transform import transform

    log.info("Running full pipeline test: ingest -> transform -> load")
    df_raw                    = fetch_latest()
    df_articles, entity_records = transform(df_raw)
    load(df_articles, entity_records)

    engine = get_engine()
    with engine.connect() as conn:
        articles = conn.execute(text("SELECT COUNT(*) FROM fact_articles")).scalar()
        entities = conn.execute(text("SELECT COUNT(*) FROM dim_entity")).scalar()
        mentions = conn.execute(text("SELECT COUNT(*) FROM fact_entity_mentions")).scalar()
        sources  = conn.execute(text("SELECT COUNT(*) FROM dim_source")).scalar()

    print(f"\nWarehouse counts after load:")
    print(f"  dim_source:             {sources} rows")
    print(f"  fact_articles:          {articles} rows")
    print(f"  dim_entity:             {entities} rows")
    print(f"  fact_entity_mentions:   {mentions} rows")