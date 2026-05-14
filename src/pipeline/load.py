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

Performance design:
  Rows are inserted in bulk using executemany() and committed in batches
  of BATCH_SIZE rows. This replaces the original row-by-row loop which
  stalled on large datasets by issuing millions of individual round trips
  to MySQL inside a single transaction.

  Batched commits mean:
    - Memory usage stays flat regardless of dataset size
    - A failure only loses the current batch, not hours of work
    - Progress is visible in the logs as each batch completes

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

# How many rows to insert and commit at a time.
# 5,000 is a safe default — large enough to be fast, small enough
# that MySQL does not run out of memory on the batch.
BATCH_SIZE = 5_000


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


def get_engine():
    password = quote_plus(DB_PASS)  # pyright: ignore[reportArgumentType, reportCallIssue]
    return create_engine(
        f"mysql+pymysql://{DB_USER}:{password}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True  # verifies connection is alive before using it
    )


# --- Dimension loaders ---

def _upsert_sources(engine, source_names: list[str]) -> dict[str, int]:
    """
    Inserts any new source names into dim_source and returns
    a mapping of source_name -> source_id for all sources in the batch.
    INSERT IGNORE skips sources that already exist.

    Uses a single bulk insert rather than one INSERT per source name.
    """
    if not source_names:
        return {}

    unique_names = list(set(source_names))

    with engine.begin() as conn:
        # Bulk insert all new sources in one statement
        conn.execute(
            text("INSERT IGNORE INTO dim_source (source_name) VALUES (:name)"),
            [{"name": name} for name in unique_names]
        )

        # Fetch IDs for all sources in this batch
        placeholders = ", ".join([f":s{i}" for i in range(len(unique_names))])
        params = {f"s{i}": name for i, name in enumerate(unique_names)}
        result = conn.execute(
            text(f"SELECT source_name, source_id FROM dim_source WHERE source_name IN ({placeholders})"),
            params
        )
        return {row.source_name: row.source_id for row in result}


def _lookup_date_ids(engine, dates: pd.Series) -> dict:
    """
    Looks up date_id values from dim_date for a Series of datetimes.
    date_id is stored as YYYYMMDD integer.
    """
    date_ints = dates.dropna().dt.strftime("%Y%m%d").astype(int).unique().tolist()
    if not date_ints:
        return {}

    with engine.connect() as conn:
        placeholders = ", ".join([f":d{i}" for i in range(len(date_ints))])
        params = {f"d{i}": d for i, d in enumerate(date_ints)}
        result = conn.execute(
            text(f"SELECT date_id, full_date FROM dim_date WHERE date_id IN ({placeholders})"),
            params
        )
        return {str(row.full_date): row.date_id for row in result}


def _lookup_segment_ids(engine) -> dict[str, int]:
    """
    Loads the full segment name -> segment_id mapping from dim_segment.
    Small table (8 rows) so we load it all at once.
    """
    with engine.connect() as conn:
        result = conn.execute(text("SELECT segment_id, segment_name FROM dim_segment"))
        return {row.segment_name: row.segment_id for row in result}


# --- Fact loaders ---

def _insert_articles(engine, df: pd.DataFrame,
                     source_map: dict, date_map: dict,
                     segment_map: dict) -> dict[str, int]:
    """
    Inserts articles into fact_articles in batches of BATCH_SIZE rows.

    Each batch is committed independently. This keeps memory usage flat
    and means a failure only rolls back the current batch.

    Returns a url -> article_id mapping built from all inserted rows,
    used when loading entity mentions.
    """
    # Build the full list of row dicts upfront, resolving all foreign keys.
    # Rows where any foreign key lookup fails are skipped.
    rows = []
    skipped = 0

    for _, row in df.iterrows():
        source_id  = source_map.get(row.get("source_name"))
        date_key   = str(row["seendate"].date()) if pd.notna(row.get("seendate")) else None
        date_id    = date_map.get(date_key)
        segment_id = segment_map.get(row.get("segment_name"),
                                     segment_map.get("General"))

        if not all([source_id, date_id, segment_id]):
            skipped += 1
            continue

        rows.append({
            "source_id":       source_id,
            "date_id":         date_id,
            "segment_id":      segment_id,
            "url":             str(row.get("url", "")),
            "seendate":        row.get("seendate"),
            "sentiment_score": row.get("sentiment_score"),
            "sentiment_label": row.get("sentiment_label"),
            "language":        row.get("language", "English"),
        })

    if skipped:
        log.warning(f"fact_articles — skipped {skipped} rows (missing foreign key lookup)")

    # Insert in batches and commit after each batch
    inserted = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i: i + BATCH_SIZE]
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT IGNORE INTO fact_articles
                    (source_id, date_id, segment_id, url, seendate,
                     sentiment_score, sentiment_label, language)
                VALUES
                    (:source_id, :date_id, :segment_id, :url, :seendate,
                     :sentiment_score, :sentiment_label, :language)
            """), batch)
        inserted += len(batch)
        log.info(f"fact_articles — committed batch {i // BATCH_SIZE + 1} "
                 f"({inserted}/{len(rows)} rows)")

    log.info(f"fact_articles — total inserted: {inserted}, skipped: {skipped}")

    # Build url -> article_id map by fetching all URLs we just inserted.
    # Done in one query rather than one SELECT per row.
    url_to_article_id = {}
    urls = [r["url"] for r in rows]

    for i in range(0, len(urls), BATCH_SIZE):
        url_batch = urls[i: i + BATCH_SIZE]
        placeholders = ", ".join([f":u{j}" for j in range(len(url_batch))])
        params = {f"u{j}": url for j, url in enumerate(url_batch)}
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT article_id, url FROM fact_articles WHERE url IN ({placeholders})"),
                params
            )
            for row in result:
                url_to_article_id[row.url] = row.article_id

    return url_to_article_id


def _insert_entities(engine, entity_records: list[dict],
                     url_to_article_id: dict) -> None:
    """
    Upserts entities into dim_entity in bulk, then inserts mention records
    into fact_entity_mentions in batches of BATCH_SIZE rows.
    """
    if not entity_records:
        log.info("No entity records to load")
        return

    # Bulk upsert all unique entities in one pass
    unique_entities = list({
        (e["entity_name"], e["entity_type"])
        for e in entity_records
    })

    for i in range(0, len(unique_entities), BATCH_SIZE):
        batch = unique_entities[i: i + BATCH_SIZE]
        with engine.begin() as conn:
            conn.execute(
                text("INSERT IGNORE INTO dim_entity (entity_name, entity_type) VALUES (:name, :etype)"),
                [{"name": name, "etype": etype} for name, etype in batch]
            )

    # Load full entity map after all upserts are committed
    entity_map = {}
    with engine.connect() as conn:
        result = conn.execute(text("SELECT entity_id, entity_name, entity_type FROM dim_entity"))
        entity_map = {(row.entity_name, row.entity_type): row.entity_id for row in result}

    # Build mention rows, resolving article_id and entity_id
    mention_rows = []
    skipped = 0

    for record in entity_records:
        article_id = url_to_article_id.get(record["url"])
        entity_id  = entity_map.get((record["entity_name"], record["entity_type"]))

        if not article_id or not entity_id:
            skipped += 1
            continue

        mention_rows.append({
            "article_id": article_id,
            "entity_id":  entity_id,
        })

    if skipped:
        log.warning(f"fact_entity_mentions — skipped {skipped} rows (unresolved foreign key)")

    # Insert mentions in batches
    inserted = 0
    for i in range(0, len(mention_rows), BATCH_SIZE):
        batch = mention_rows[i: i + BATCH_SIZE]
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT IGNORE INTO fact_entity_mentions (article_id, entity_id)
                VALUES (:article_id, :entity_id)
            """), batch)
        inserted += len(batch)
        log.info(f"fact_entity_mentions — committed batch {i // BATCH_SIZE + 1} "
                 f"({inserted}/{len(mention_rows)} rows)")

    log.info(f"fact_entity_mentions — total inserted: {inserted}, skipped: {skipped}")


# --- Main load function ---

def load(df_articles: pd.DataFrame, entity_records: list[dict]) -> None:
    """
    Loads transformed articles and entity records into the warehouse.

    Dimension lookups are resolved upfront. Fact table inserts are committed
    in batches of BATCH_SIZE rows so memory usage stays flat and progress
    is visible in the logs throughout the run.
    """
    if df_articles.empty:
        log.warning("load() received an empty DataFrame — nothing to load")
        return

    log.info(f"Loading {len(df_articles)} articles and {len(entity_records)} entity mentions...")

    engine = get_engine()

    # Resolve all dimension lookups before touching fact tables
    source_names = df_articles["source_name"].dropna().unique().tolist()
    source_map   = _upsert_sources(engine, source_names)
    date_map     = _lookup_date_ids(engine, df_articles["seendate"])
    segment_map  = _lookup_segment_ids(engine)

    url_to_article_id = _insert_articles(
        engine, df_articles, source_map, date_map, segment_map
    )
    _insert_entities(engine, entity_records, url_to_article_id)

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
    df_raw                      = fetch_latest()
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
