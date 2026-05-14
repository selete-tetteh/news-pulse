"""
transform.py

Takes the raw DataFrame from ingest.py and adds three things:
  1. Segment routing  — maps GDELT theme tags to one of seven segments
  2. Sentiment scoring — VADER compound score and label on the article title
  3. Entity extraction — cleans and structures GDELT's persons and
                         organizations fields into a list of named entities

These three outputs are what the warehouse needs. Nothing here touches
the database — transform.py only produces a clean DataFrame. load.py
handles the database writes.

Usage:
  from src.pipeline.transform import transform

  df_raw       = fetch_latest()
  df_transformed, entities = transform(df_raw)
"""

import logging
import re
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger(__name__)

# --- Load NLP tools once at import time ---
# Loading spaCy and VADER inside the transform function would reload
# them on every call. Loading at module level means they are
# initialised once and reused for the lifetime of the pipeline run.
try:
    nlp = spacy.load("en_core_web_sm")
    log.info("spaCy model loaded")
except OSError:
    raise OSError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )

vader = SentimentIntensityAnalyzer()
log.info("VADER initialised")


# --- Segment routing ---

# Segment map mirrors the seed data in dim_segment.
# Keys are segment names; values are the GDELT theme tag prefixes
# that map to that segment. Order matters — the first match wins,
# so more specific segments should come before general ones.
SEGMENT_MAP = {
    "Politics and Government":   ["POLITICS", "GOV", "ELECTIONS", "LEADER"],
    "Business and Markets":      ["ECON", "BUSINESS", "MARKETS", "FINANCE", "TRADE"],
    "Technology":                ["TECH", "CYBER", "AI", "INTERNET", "INNOVATION"],
    "Sports":                    ["SPORTS", "SPORT"],
    "Entertainment and Culture": ["ENTERTAIN", "CULTURE", "ARTS", "MEDIA"],
    "Science and Health":        ["HEALTH", "SCIENCE", "MEDICAL", "ENV"],
    "Crime and Justice":         ["CRIME", "LEGAL", "JUSTICE", "LAW"],
}

FALLBACK_SEGMENT = "General"


def _route_segment(themes_str: str) -> str:
    """
    Takes a raw GDELT themes string (semicolon-separated theme tags)
    and returns the name of the best matching segment.

    GDELT themes look like: 'TAX_FNCACT;CRIME;ECON_POVERTY;LEADER'
    We check whether any tag starts with a known segment prefix.
    First match wins. If no match, returns FALLBACK_SEGMENT.
    """
    if pd.isna(themes_str) or str(themes_str).strip() == "":
        return FALLBACK_SEGMENT

    # Normalise to uppercase and split on semicolons
    tags = [t.strip().upper() for t in str(themes_str).split(";")]

    for segment, prefixes in SEGMENT_MAP.items():
        for tag in tags:
            for prefix in prefixes:
                if tag.startswith(prefix):
                    return segment

    return FALLBACK_SEGMENT


# --- Sentiment scoring ---

def _score_sentiment(title: str) -> tuple[float, str]:
    """
    Runs VADER sentiment analysis on the article title.
    Returns (compound_score, label).

    Compound score ranges from -1.0 to 1.0.
    Thresholds follow VADER's recommended cutoffs:
      >= 0.05  -> POSITIVE
      <= -0.05 -> NEGATIVE
      else     -> NEUTRAL
    """
    if pd.isna(title) or str(title).strip() == "":
        return 0.0, "NEUTRAL"

    scores = vader.polarity_scores(str(title))
    compound = round(scores["compound"], 4)

    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return compound, label


# --- Entity extraction ---

def _extract_entities(persons_str: str, orgs_str: str) -> list[dict]:
    """
    Cleans and structures GDELT's persons and organizations fields
    into a list of entity dicts ready for dim_entity and
    fact_entity_mentions.

    GDELT persons format:  'john smith,1;jane doe,2'
    GDELT orgs format:     'united nations,1;bbc,3'

    The trailing ',N' is a position indicator — we strip it.
    Entity type follows spaCy conventions: PERSON or ORG.
    """
    entities = []

    def parse_gdelt_field(raw: str, entity_type: str):
        if pd.isna(raw) or str(raw).strip() == "":
            return
        for item in str(raw).split(";"):
            # Strip the trailing position number
            name = re.sub(r",\d+$", "", item.strip()).strip()
            if name and len(name) > 1:
                entities.append({
                    "entity_name": name.title(),  # normalise to title case
                    "entity_type": entity_type
                })

    parse_gdelt_field(persons_str, "PERSON")
    parse_gdelt_field(orgs_str, "ORG")

    # Deduplicate within this article
    seen = set()
    unique = []
    for e in entities:
        key = (e["entity_name"], e["entity_type"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


# --- Main transform function ---

def transform(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Applies segment routing, sentiment scoring, and entity extraction
    to a raw DataFrame from ingest.py.

    Returns:
        df_articles: one row per article with all warehouse fields populated
        entity_records: flat list of dicts for entity loading, each containing
                        url (to join back to the article) plus entity fields
    """
    if df_raw.empty:
        log.warning("transform() received an empty DataFrame — nothing to process")
        return df_raw, []

    log.info(f"Transforming {len(df_raw)} articles...")

    df = df_raw.copy()

    # 1. Segment routing
    df["segment_name"] = df["themes"].apply(_route_segment)
    log.info(f"Segment distribution:\n{df['segment_name'].value_counts().to_string()}")

    # 2. Sentiment scoring
    # Source name is the closest thing to a title in the GKG feed.
    # Full article titles require scraping — out of scope for this pipeline.
    sentiment_results = df["source_name"].apply(_score_sentiment)
    df["sentiment_score"] = sentiment_results.apply(lambda x: x[0])
    df["sentiment_label"] = sentiment_results.apply(lambda x: x[1])
    
    # 3. Deduplicate on URL before loading.
    # GDELT occasionally republishes the same article URL across multiple
    # 15-minute windows. Deduplicating here means the database-level
    # INSERT IGNORE acts as a second line of defence, not the first.
    df = df.drop_duplicates(subset=["url"], keep="first")
    log.info(f"After deduplication: {len(df)} articles (duplicates removed at transform stage)")

    # 4. Entity extraction — build flat list with url as the join key
    entity_records = []
    persons_col = "persons" if "persons" in df.columns else None
    orgs_col    = "organizations" if "organizations" in df.columns else None

    for _, row in df.iterrows():
        persons_val = row[persons_col] if persons_col else ""
        orgs_val    = row[orgs_col]    if orgs_col    else ""
        entities    = _extract_entities(persons_val, orgs_val)
        for e in entities:
            e["url"] = row["url"]
        entity_records.extend(entities)

    log.info(f"Extracted {len(entity_records)} entity mentions across {len(df)} articles")
    

    # Keep only the columns the warehouse needs
    warehouse_cols = [
        "seendate", "source_name", "url",
        "segment_name", "sentiment_score", "sentiment_label", "language"
    ]
    df_articles = df[[c for c in warehouse_cols if c in df.columns]].copy()

    log.info("Transform complete")
    return df_articles, entity_records


# --- Quick test when run directly ---
if __name__ == "__main__":
    from src.pipeline.ingest import fetch_latest

    log.info("Fetching latest articles for transform test...")
    df_raw = fetch_latest()

    df_articles, entity_records = transform(df_raw)

    print(f"\nArticles: {len(df_articles)}")
    print(f"Entity mentions: {len(entity_records)}")
    print(f"\nSample articles:\n{df_articles[['seendate','source_name','segment_name','sentiment_label']].head(5).to_string()}")
    print(f"\nSample entities:\n{pd.DataFrame(entity_records).head(5).to_string()}")