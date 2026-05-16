"""
ingest.py

Fetches raw article data from the GDELT 2.0 Global Knowledge Graph (GKG).

Two modes:
  - Live:     fetches the single most recent 15-minute update file.
  - Backfill: fetches all update files within a specified date range.

GDELT publishes a master file list at a fixed URL. Every 15 minutes a new
line is appended pointing to the latest GKG file. Each file is a compressed
CSV covering that window. This script reads the master list, identifies the
relevant files, downloads and decompresses them, and returns a clean DataFrame.

Master list caching:
  The master file list is 386,000+ lines and takes ~2.5 minutes to download.
  For a live pipeline running every 15 minutes, downloading the full list on
  every run adds unacceptable overhead. The list is cached locally and reused
  if it is less than CACHE_MAX_AGE_HOURS old. If the cache is stale or missing,
  a full download runs and the result is saved to disk.

  The cache is safe because the master list is append-only — lines are never
  edited or removed, only added. Reading a cached version never misses updates
  that existed at cache time.

Usage:
  from src.pipeline.ingest import fetch_latest, fetch_backfill

  df_live     = fetch_latest()
  df_backfill = fetch_backfill("2024-01-01", "2024-01-07")
"""

import io
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# --- Project root and logging ---
def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "environment.yml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate project root — environment.yml not found.")

PROJECT_ROOT = find_project_root()
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# --- GDELT config ---
# The master file list is a plain text file where each line contains
# the size, MD5 hash, and URL of one 15-minute GKG update file.
GDELT_MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"

# Cache location and expiry.
# data/processed/ is already in .gitignore so the cache is never committed.
# 6 hours is a safe expiry — long enough to avoid redundant downloads during
# a backfill session, short enough that the live pipeline always has fresh data.
MASTER_LIST_CACHE = PROJECT_ROOT / "data" / "processed" / "gdelt_master_cache.csv"
CACHE_MAX_AGE_HOURS = 6

# GKG columns we keep. The full GKG has 27 columns — most are not
# relevant for this project. We keep only what the warehouse needs.
KEEP_COLS = {
    "DATE":                 "seendate",
    "SourceCommonName":     "source_name",
    "DocumentIdentifier":   "url",
    "Themes":               "themes",
    "Locations":            "locations",
    "Persons":              "persons",
    "Organizations":        "organizations",
    "SharingImage":         "image_url",
    "Extras":               "extras",
    "TranslationInfo":      "language_info",
}

# Full GKG column names in order (GDELT GKG 2.0 spec)
GKG_COLUMNS = [
    "GKGRECORDID", "DATE", "SourceCollectionIdentifier", "SourceCommonName",
    "DocumentIdentifier", "Counts", "V2Counts", "Themes", "V2Themes",
    "Locations", "V2Locations", "Persons", "V2Persons", "Organizations",
    "V2Organizations", "V2Tone", "Dates", "GCAM", "SharingImage",
    "RelatedImages", "SocialImageEmbeds", "SocialVideoEmbeds", "Quotations",
    "AllNames", "Amounts", "TranslationInfo", "Extras"
]


# --- Core fetch functions ---

def _fetch_master_list(force_cache: bool = False) -> pd.DataFrame:
    """
    Returns the GDELT master file list as a DataFrame.

    Checks for a local cache first. If the cache exists and is less than
    CACHE_MAX_AGE_HOURS old, it is loaded from disk — no network request.
    If the cache is stale or missing, a full download runs and the result
    is saved to disk for future calls.

    Args:
        force_cache: if True, skip the age check and load from disk regardless
                     of how old the cache is. Used by backfill runs to prevent
                     a mid-run re-download when a long backfill crosses the
                     6-hour expiry threshold.

    Why cache?
      The master list is ~386,000 lines and takes ~2.5 minutes to download.
      For a live pipeline running every 15 minutes, this overhead is
      unacceptable. The list is append-only, so a cached version is always
      a valid subset of the current list — it never contains incorrect data,
      only potentially missing the most recent entries.
    """
    if MASTER_LIST_CACHE.exists():
        if force_cache:
            log.info("Loading master list from cache (force_cache=True — age check skipped)")
            df = pd.read_csv(MASTER_LIST_CACHE, parse_dates=["file_datetime"])
            log.info(f"Master list loaded from cache — {len(df)} GKG files")
            return df

        age_seconds = time.time() - MASTER_LIST_CACHE.stat().st_mtime
        age_hours   = age_seconds / 3600

        if age_hours < CACHE_MAX_AGE_HOURS:
            log.info(
                f"Loading master list from cache "
                f"(age: {age_hours:.1f}h, expires in {CACHE_MAX_AGE_HOURS - age_hours:.1f}h)"
            )
            df = pd.read_csv(MASTER_LIST_CACHE, parse_dates=["file_datetime"])
            log.info(f"Master list loaded from cache — {len(df)} GKG files")
            return df
        else:
            log.info(f"Master list cache is stale ({age_hours:.1f}h old) — refreshing")
    else:
        log.info("No master list cache found — downloading")

    # Full download
    log.info("Fetching GDELT master file list...")
    response = requests.get(GDELT_MASTER_URL, timeout=60)
    response.raise_for_status()

    lines = response.text.strip().split("\n")
    records = []
    for line in lines:
        parts = line.strip().split(" ")
        if len(parts) == 3 and "gkg.csv.zip" in parts[2]:
            records.append({
                "size": int(parts[0]),
                "md5":  parts[1],
                "url":  parts[2]
            })

    df = pd.DataFrame(records)

    df["file_datetime"] = pd.to_datetime(
        df["url"].str.extract(r"(\d{14})")[0],
        format="%Y%m%d%H%M%S"
    )

    MASTER_LIST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_LIST_CACHE, index=False)
    log.info(f"Master list saved to cache — {len(df)} GKG files")

    return df


def _download_and_parse_gkg(url: str) -> pd.DataFrame:
    """
    Downloads one GDELT GKG zip file, decompresses it in memory,
    and returns a DataFrame with only the columns we need.

    Files are decompressed in memory rather than saved to disk.
    Raw GDELT files are excluded from Git and not persisted locally
    unless explicitly saved by the caller.
    """
    log.info(f"Downloading: {url.split('/')[-1]}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=GKG_COLUMNS,
                on_bad_lines="skip",
                low_memory=False
            )

    # Keep only the columns the warehouse needs
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].rename(columns=KEEP_COLS)

    # Parse seendate to datetime
    if "seendate" in df.columns:
        df["seendate"] = pd.to_datetime(
            df["seendate"], format="%Y%m%d%H%M%S", errors="coerce"
        )

    # Extract language from TranslationInfo column.
    # English articles have no TranslationInfo entry — null means English.
    # Translated articles carry a srclang tag.
    if "language_info" in df.columns:
        df["language"] = df["language_info"].apply(
            lambda x: "English" if pd.isna(x) or str(x).strip() == ""
            else str(x).split(";")[0].replace("srclang=", "")
        )
        df.drop(columns=["language_info"], inplace=True)
    else:
        df["language"] = "English"

    # Filter to English only — NLP models are English-language
    df = df[df["language"] == "English"].copy()

    # Drop rows with no URL — these cannot be stored or referenced
    df = df[df["url"].notna() & (df["url"].str.strip() != "")].copy()

    df.reset_index(drop=True, inplace=True)
    return df


def fetch_latest() -> pd.DataFrame:
    """
    Fetches the single most recent GDELT GKG update file.
    Used by the live pipeline scheduler.
    """
    master = _fetch_master_list()
    latest_url = master.sort_values("file_datetime").iloc[-1]["url"]
    return _download_and_parse_gkg(latest_url)


def fetch_backfill(start_date: str, end_date: str, max_workers: int = 8,
                   force_cache: bool = False) -> pd.DataFrame:
    """
    Fetches all GDELT GKG update files within a date range and
    concatenates them into a single DataFrame.

    Files are downloaded in parallel using ThreadPoolExecutor.
    Most of the time in a sequential fetch is network waiting, not
    computation. Parallelising the downloads means multiple files are
    in-flight at once, significantly reducing total runtime.

    The master list is loaded once per call — when the backfill runs
    day-by-day from run_pipeline.py, the cache means subsequent days
    skip the download entirely.

    Args:
        start_date:  inclusive start date, format 'YYYY-MM-DD'
        end_date:    inclusive end date, format 'YYYY-MM-DD'
        max_workers: number of parallel download threads.
                     8 is a safe default. Drop to 4 if GDELT starts
                     returning errors or skipping files at high volume.
        force_cache: passed through to _fetch_master_list(). Set to True
                     during long backfill runs to prevent mid-run cache expiry.

    Returns:
        DataFrame containing all articles from the date range.
    """
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date) + timedelta(days=1)

    master = _fetch_master_list(force_cache=force_cache)
    files_in_range = master[
        (master["file_datetime"] >= start) &
        (master["file_datetime"] <  end)
    ].sort_values("file_datetime")

    if files_in_range.empty:
        log.warning(f"No GDELT files found between {start_date} and {end_date}")
        return pd.DataFrame()

    urls = files_in_range["url"].tolist()
    log.info(f"Backfill: {len(urls)} files to fetch ({start_date} to {end_date})")

    frames    = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_and_parse_gkg, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                df = future.result()
                frames.append(df)
                completed += 1
                if completed % 50 == 0:
                    log.info(f"  Progress: {completed}/{len(urls)} files fetched")
            except Exception as e:
                log.warning(f"  Skipped {url} — {e}")

    if not frames:
        log.warning("Backfill completed with no data retrieved.")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    log.info(f"Backfill complete — {len(result)} total articles")
    return result


# --- Quick test when run directly ---
if __name__ == "__main__":
    log.info("Running live fetch test...")
    df = fetch_latest()
    log.info(f"Fetched {len(df)} articles")
    log.info(f"Columns: {list(df.columns)}")
    log.info(f"Sample:\n{df[['seendate','source_name','url']].head(3).to_string()}")
