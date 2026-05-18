"""
recommend.py

Production recommendation engine for News Pulse.

Given an article_id or URL, returns the top-N most similar articles
from the warehouse using sentence transformer embeddings.

Why sentence transformers over TF-IDF?
  Notebook 03 evaluated both systems on 600 articles across 6 segments.
  The sentence transformer (all-MiniLM-L6-v2) achieved overall P@5 of
  0.878 vs TF-IDF's 0.657 — a 22 percentage point gap consistent across
  every segment. TF-IDF also produced erratic similarity scores (median
  0.34, wide spread). Sentence transformers produced tighter, more
  reliable confidence (median 0.67). TF-IDF is documented as the
  baseline in notebook 03 but is not used in production.

How it works:
  1. All articles in the warehouse are loaded with their feature strings
     (source name + segment + entity names).
  2. Feature strings are encoded into 384-dimensional vectors using
     all-MiniLM-L6-v2.
  3. Embeddings are cached to disk so subsequent calls skip re-encoding.
  4. For a query article, cosine similarity is computed against all
     stored embeddings.
  5. Top-N results are returned after deduplicating by URL.

Embedding cache:
  Encoding 11M+ articles on every call is not feasible. The cache stores
  embeddings as a numpy .npz file alongside a metadata CSV so article IDs
  and embeddings stay in sync. The cache is rebuilt when stale (older than
  CACHE_MAX_AGE_HOURS) or when force_rebuild=True is passed.

  Cache files are written to data/processed/ which is in .gitignore.
  They are never committed.

Usage:
  from src.pipeline.recommend import Recommender

  rec = Recommender()
  results = rec.recommend(article_id=12345, n=5)
  results = rec.recommend(url="https://reuters.com/...", n=5)

  # Force a full cache rebuild (e.g. after a large backfill)
  rec = Recommender(force_rebuild=True)
"""

import logging
import os
import time
from pathlib import Path
from urllib.parse import quote_plus

# Load .env and set HF offline flags BEFORE importing sentence_transformers.
# sentence_transformers / huggingface_hub read these env vars at import time,
# so setting them afterwards (or letting load_dotenv run later in the file)
# has no effect — the library has already decided whether to go online.
from dotenv import load_dotenv

def _find_project_root_early() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "environment.yml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root — environment.yml not found.")

load_dotenv(_find_project_root_early() / ".env")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Cache expires after this many hours. Set conservatively — the live
# pipeline adds ~15k articles/day, so a 24-hour cache loses at most one
# day of new articles from recommendations. Rebuild manually after a
# backfill using force_rebuild=True.
CACHE_MAX_AGE_HOURS = 24

# Sentence transformer model. all-MiniLM-L6-v2 is small (80MB), fast on
# CPU, and benchmarks well for semantic similarity on short text.
# Changing this requires a full cache rebuild.
ST_MODEL_NAME = "all-MiniLM-L6-v2"

# How many articles to encode per batch. 64 is a safe default for CPU.
# Increase to 128 if you have a GPU available.
ENCODE_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Project root and credentials
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "environment.yml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root — environment.yml not found.")


PROJECT_ROOT = _find_project_root()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASSWORD")
if DB_PASS is None:
    raise ValueError("DB_PASSWORD not found in .env")
DB_NAME = os.getenv("DB_NAME", "news_pulse")

CACHE_DIR          = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_PATH    = CACHE_DIR / "embeddings.npz"
METADATA_PATH      = CACHE_DIR / "embeddings_meta.csv"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def _get_engine():
    return create_engine(
        f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True
    )


# ---------------------------------------------------------------------------
# Feature string builder
# ---------------------------------------------------------------------------
# Mirrors the logic in notebook 03 exactly. Any change here must be
# reflected in the notebook and vice versa — the feature string definition
# is the contract between the two.

def _build_feature_string(source_name: str, segment_name: str,
                           entities: str) -> str:
    """
    Builds a synthetic feature string for one article.

    Format: source_name segment_name entity1 entity2 ...

    Multi-word terms have spaces replaced with underscores so the
    sentence transformer treats them as single semantic units rather
    than splitting them into individual words.
    """
    import re
    parts = []

    if pd.notna(source_name) and str(source_name).strip():
        parts.append(re.sub(r"[^\w]", "_", str(source_name).strip().lower()))

    if pd.notna(segment_name) and str(segment_name).strip():
        parts.append(str(segment_name).replace(" ", "_"))

    if pd.notna(entities) and str(entities).strip():
        for name in str(entities).split(" "):
            clean = name.strip()
            if clean:
                parts.append(clean.replace(" ", "_"))

    return " ".join(parts) if parts else "unknown"


# ---------------------------------------------------------------------------
# Warehouse loader
# ---------------------------------------------------------------------------

def _load_articles_from_warehouse(engine) -> pd.DataFrame:
    """
    Loads a stratified sample of articles from the warehouse and fetches
    their entities in scoped chunks.

    Why stratified?
      A plain LIMIT pulls rows in insertion order, which skews toward
      whichever segment dominated early ingestion. Stratification ensures
      all six segments are represented equally in the embedding index.

    Why chunk the entity fetch?
      fact_entity_mentions has 64M rows. A single JOIN across all article
      IDs hits the full table even with an index. Chunking to 1,000 IDs
      per query scopes each hit to a small slice — confirmed at 0.48s per
      chunk in testing.
    """
    log.info("Loading articles from warehouse...")

    ARTICLES_PER_SEGMENT = 15_000
    ENTITY_CHUNK_SIZE    = 1_000

    # Step 1: load articles per segment separately so each segment
    # gets equal representation regardless of insertion order.
    segments_query = text("""
        SELECT segment_name FROM dim_segment
        WHERE segment_name != 'General'
    """)

    with engine.connect() as conn:
        segments = [r[0] for r in conn.execute(segments_query)]

    frames = []
    for segment in segments:
        seg_query = text("""
            SELECT
                fa.article_id,
                fa.url,
                fa.seendate,
                ds_src.source_name,
                ds_seg.segment_name
            FROM fact_articles fa
            JOIN dim_source  ds_src ON fa.source_id  = ds_src.source_id
            JOIN dim_segment ds_seg ON fa.segment_id = ds_seg.segment_id
            WHERE ds_seg.segment_name = :seg
            LIMIT :lim
        """)
        with engine.connect() as conn:
            df_seg = pd.read_sql(seg_query, conn,
                                 params={"seg": segment, "lim": ARTICLES_PER_SEGMENT})
        frames.append(df_seg)
        log.info(f"  {segment}: {len(df_seg):,} articles")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["url"], keep="first")
    log.info(f"Articles loaded: {len(df):,} after deduplication")

    # Step 2: fetch entities in chunks of ENTITY_CHUNK_SIZE.
    article_ids = df["article_id"].tolist()
    entity_rows = []

    log.info(f"Fetching entities in chunks of {ENTITY_CHUNK_SIZE}...")
    total_chunks = (len(article_ids) + ENTITY_CHUNK_SIZE - 1) // ENTITY_CHUNK_SIZE

    for i in range(0, len(article_ids), ENTITY_CHUNK_SIZE):
        chunk_ids   = article_ids[i: i + ENTITY_CHUNK_SIZE]
        placeholders = ", ".join([f":id{j}" for j in range(len(chunk_ids))])
        params       = {f"id{j}": v for j, v in enumerate(chunk_ids)}

        entity_query = text(f"""
            SELECT
                fem.article_id,
                GROUP_CONCAT(
                    DISTINCT de.entity_name
                    ORDER BY de.entity_name
                    SEPARATOR ' '
                ) AS entities
            FROM fact_entity_mentions fem
            JOIN dim_entity de ON fem.entity_id = de.entity_id
            WHERE fem.article_id IN ({placeholders})
            GROUP BY fem.article_id
        """)

        with engine.connect() as conn:
            chunk_df = pd.read_sql(entity_query, conn, params=params)
        entity_rows.append(chunk_df)

        chunk_num = i // ENTITY_CHUNK_SIZE + 1
        if chunk_num % 20 == 0 or chunk_num == total_chunks:
            log.info(f"  Entity fetch: {chunk_num}/{total_chunks} chunks")

    df_entities = pd.concat(entity_rows, ignore_index=True) if entity_rows else \
                  pd.DataFrame(columns=["article_id", "entities"])

    df = df.merge(df_entities, on="article_id", how="left")
    df["entities"] = df["entities"].fillna("")

    log.info(f"Loaded {len(df):,} articles with entities")
    return df


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _cache_is_fresh() -> bool:
    """
    Returns True if both cache files exist and are younger than
    CACHE_MAX_AGE_HOURS. Returns False if either is missing or stale.
    """
    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        return False
    age_hours = (time.time() - EMBEDDINGS_PATH.stat().st_mtime) / 3600
    return age_hours < CACHE_MAX_AGE_HOURS


def _save_cache(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    """
    Saves embeddings and metadata to disk.

    embeddings.npz holds the float32 matrix.
    embeddings_meta.csv holds article_id, url, source_name,
    segment_name, seendate — enough to return useful results without
    a second database query.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(str(EMBEDDINGS_PATH), embeddings=embeddings)

    meta_cols = ["article_id", "url", "source_name", "segment_name", "seendate"]
    df[meta_cols].to_csv(METADATA_PATH, index=False)

    log.info(f"Cache saved — {len(df):,} articles, "
             f"embeddings shape: {embeddings.shape}")


def _load_cache() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Loads embeddings and metadata from disk.
    Raises FileNotFoundError if either file is missing.
    """
    log.info("Loading embeddings from cache...")
    embeddings = np.load(str(EMBEDDINGS_PATH))["embeddings"]
    df_meta    = pd.read_csv(METADATA_PATH)
    log.info(f"Cache loaded — {len(df_meta):,} articles, "
             f"embeddings shape: {embeddings.shape}")
    return df_meta, embeddings


# ---------------------------------------------------------------------------
# Recommender class
# ---------------------------------------------------------------------------

class Recommender:
    """
    Sentence transformer recommendation engine for News Pulse.

    Initialisation loads or builds the embedding cache. After that,
    recommend() calls are fast — just a cosine similarity computation
    against the in-memory embedding matrix.

    Args:
        force_rebuild: if True, ignore the cache and rebuild from scratch.
                       Use this after a large backfill to include new articles.
    """

    def __init__(self, force_rebuild: bool = False):
        self._engine = _get_engine()
        self._model  = None  # loaded lazily — only needed for (re)building cache
        self._meta   = None  # DataFrame: article_id, url, source_name, segment_name, seendate
        self._emb    = None  # numpy array: (n_articles, 384)

        if force_rebuild or not _cache_is_fresh():
            self._build_cache()
        else:
            self._meta, self._emb = _load_cache()

        log.info(
            f"Recommender ready — {len(self._meta):,} articles indexed, "
            f"embedding dim: {self._emb.shape[1]}"
        )

    # --- Cache builder ---

    def _build_cache(self) -> None:
        """
        Loads all articles from the warehouse, builds feature strings,
        encodes them, and saves the cache to disk.
        """
        log.info("Building embedding cache from warehouse...")
        start = time.time()

        df = _load_articles_from_warehouse(self._engine)

        df["feature_string"] = df.apply(
            lambda row: _build_feature_string(
                row["source_name"], row["segment_name"], row["entities"]
            ),
            axis=1
        )

        log.info(f"Encoding {len(df):,} feature strings "
                 f"(model: {ST_MODEL_NAME})...")

        if self._model is None:
            log.info("Loading sentence transformer model...")
            self._model = SentenceTransformer(ST_MODEL_NAME)

        embeddings = self._model.encode(
            df["feature_string"].tolist(),
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        _save_cache(df, embeddings)

        self._meta = df[["article_id", "url", "source_name",
                          "segment_name", "seendate"]].reset_index(drop=True)
        self._emb  = embeddings

        elapsed = round(time.time() - start, 1)
        log.info(f"Cache build complete in {elapsed}s")

    # --- Core recommendation logic ---

    def recommend(self, article_id: int | None = None,
                  url: str | None = None,
                  n: int = 5) -> pd.DataFrame:
        """
        Returns the top-N most similar articles to the query article.

        Pass either article_id (int) or url (str). If both are provided,
        article_id takes precedence.

        Args:
            article_id: the warehouse article_id of the query article.
            url:        the URL of the query article.
            n:          number of recommendations to return. Default 5.

        Returns:
            DataFrame with columns:
                rank, article_id, url, source_name, segment_name,
                seendate, similarity_score

            Returns an empty DataFrame if the query article is not found
            in the index. This can happen if the article was ingested after
            the last cache build — caller should handle gracefully.
        """
        if article_id is None and url is None:
            raise ValueError("Provide either article_id or url.")

        # Locate query article in metadata
        if article_id is not None:
            mask = self._meta["article_id"] == article_id
        else:
            mask = self._meta["url"] == url

        matches = self._meta[mask]

        if matches.empty:
            log.warning(
                f"Article not found in index — "
                f"{'article_id=' + str(article_id) if article_id else 'url=' + str(url)}. "
                f"Cache may be stale. Try Recommender(force_rebuild=True)."
            )
            return pd.DataFrame()

        # Use the first match if multiple rows share the same URL
        # (near-duplicate records — noted in notebook 03 findings)
        query_idx = matches.index[0]
        query_vec = self._emb[query_idx].reshape(1, -1)

        # Compute cosine similarity against all embeddings
        scores = cosine_similarity(query_vec, self._emb).flatten()
        scores[query_idx] = 0  # exclude self

        # Rank all articles
        ranked_indices = scores.argsort()[::-1]
        ranked_meta    = self._meta.iloc[ranked_indices].copy()
        ranked_meta["similarity_score"] = scores[ranked_indices].round(4)

        # Deduplicate by URL before returning.
        # The warehouse can contain near-duplicate records from the same
        # source URL ingested across multiple pipeline runs. Without this
        # step, the same article can appear multiple times in the top-N,
        # which is a poor user experience.
        ranked_meta = ranked_meta.drop_duplicates(subset=["url"], keep="first")

        # Return top N
        results = ranked_meta.head(n).copy()
        results.insert(0, "rank", range(1, len(results) + 1))
        results = results.reset_index(drop=True)

        return results[["rank", "article_id", "url", "source_name",
                         "segment_name", "seendate", "similarity_score"]]

    # --- Convenience helpers ---

    def recommend_by_segment(self, segment_name: str,
                              n: int = 5) -> pd.DataFrame:
        """
        Returns the top-N most topically central articles in a segment.

        Capped at 2,000 articles for the similarity computation to keep
        runtime and memory usage flat. For segments with more than 2,000
        articles, a random sample of 2,000 is used.
        """
        seg_mask    = self._meta["segment_name"] == segment_name
        seg_indices = self._meta[seg_mask].index.tolist()

        if not seg_indices:
            log.warning(f"Segment '{segment_name}' not found in index.")
            return pd.DataFrame()

        # Cap at 2,000 to keep the similarity matrix fast
        CAP = 2_000
        if len(seg_indices) > CAP:
            rng        = np.random.default_rng(42)
            seg_indices = rng.choice(seg_indices, size=CAP, replace=False).tolist()

        seg_emb    = self._emb[seg_indices]
        sim_matrix = cosine_similarity(seg_emb)
        np.fill_diagonal(sim_matrix, 0)

        mean_scores = sim_matrix.mean(axis=1)
        top_within  = np.argsort(mean_scores)[::-1][:n]
        top_global  = [seg_indices[i] for i in top_within]

        results = self._meta.iloc[top_global].copy()
        results["centrality_score"] = mean_scores[top_within].round(4)
        results.insert(0, "rank", range(1, len(results) + 1))

        return results[["rank", "article_id", "url", "source_name",
                         "segment_name", "seendate", "centrality_score"]]

    @property
    def index_size(self) -> int:
        """Number of articles currently in the embedding index."""
        return len(self._meta)

    @property
    def segments(self) -> list[str]:
        """Unique segment names in the current index."""
        return sorted(self._meta["segment_name"].unique().tolist())


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    print("Initialising recommender (first run will build cache)...")
    rec = Recommender()

    print(f"\nIndex size:  {rec.index_size:,} articles")
    print(f"Segments:    {rec.segments}")

    # Test recommend() on the first article in the index
    test_id  = int(rec._meta["article_id"].iloc[0])
    test_seg = rec._meta["segment_name"].iloc[0]

    print(f"\nTest query — article_id: {test_id}, segment: {test_seg}")
    recs = rec.recommend(article_id=test_id, n=5)
    print(recs[["rank", "source_name", "segment_name",
                "similarity_score"]].to_string(index=False))

    # Test recommend_by_segment()
    print(f"\nTop 5 central articles in '{test_seg}':")
    central = rec.recommend_by_segment(test_seg, n=5)
    print(central[["rank", "source_name", "seendate",
                   "centrality_score"]].to_string(index=False))
