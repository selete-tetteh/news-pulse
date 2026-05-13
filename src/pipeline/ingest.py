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

Usage:
  from src.pipeline.ingest import fetch_latest, fetch_backfill

  df_live     = fetch_latest()
  df_backfill = fetch_backfill("2024-01-01", "2024-01-07")
"""

import os
import io
import logging
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus
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

# GKG columns we keep. The full GKG has 27 columns — most are not
# relevant for this project. We keep only what the warehouse needs.
KEEP_COLS = {
    "DATE":       "seendate",
    "SourceCommonName": "source_name",
    "DocumentIdentifier": "url",
    "Themes":     "themes",
    "Locations":  "locations",
    "Persons":    "persons",
    "Organizations": "organizations",
    "SharingImage": "image_url",
    "Extras":     "extras",
    "TranslationInfo": "language_info",
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

def _fetch_master_list() -> pd.DataFrame:
    """
    Downloads the GDELT master file list and returns it as a DataFrame.
    Each row is one 15-minute update file with its size, hash, and URL.
    """
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

    # Extract the datetime from the filename.
    # GDELT filenames follow the pattern: YYYYMMDDHHMMSS.gkg.csv.zip
    df["file_datetime"] = pd.to_datetime(
        df["url"].str.extract(r"(\d{14})")[0],
        format="%Y%m%d%H%M%S"
    )

    log.info(f"Master list loaded — {len(df)} GKG files found")
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


def fetch_backfill(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches all GDELT GKG update files within a date range and
    concatenates them into a single DataFrame.

    Args:
        start_date: inclusive start date, format 'YYYY-MM-DD'
        end_date:   inclusive end date, format 'YYYY-MM-DD'

    Returns:
        DataFrame containing all articles from the date range.
    """
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date) + timedelta(days=1)  # make end inclusive

    master = _fetch_master_list()
    files_in_range = master[
        (master["file_datetime"] >= start) &
        (master["file_datetime"] <  end)
    ].sort_values("file_datetime")

    if files_in_range.empty:
        log.warning(f"No GDELT files found between {start_date} and {end_date}")
        return pd.DataFrame()

    log.info(f"Backfill: {len(files_in_range)} files to fetch ({start_date} to {end_date})")

    frames = []
    for i, row in files_in_range.iterrows():
        try:
            df = _download_and_parse_gkg(row["url"])
            frames.append(df)
            log.info(f"  Fetched {len(df)} articles — {row['file_datetime']}")
        except Exception as e:
            log.warning(f"  Skipped {row['url']} — {e}")
            continue

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