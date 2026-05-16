"""
run_pipeline.py

Orchestrates the full ingest -> transform -> load sequence with one command.
Sends a Telegram alert after every run — success, failure, or empty.

Two modes:
  --mode live      Fetches the single most recent GDELT 15-minute update.
                   Used by the scheduler for continuous live updates.

  --mode backfill  Fetches GDELT files within a date range, processing one
                   day at a time to keep memory usage flat.
                   Requires --start and --end arguments.

Why process backfill day by day?
  A naive backfill fetches all files in the range into one DataFrame before
  transforming and loading. For large ranges (3 months = ~11 million articles),
  this exhausts available RAM and the OS kills the process before anything
  is written to the database. Processing one day at a time (~130k articles)
  keeps each batch well within memory limits. Each day's data is transformed,
  loaded, and discarded before the next day begins.

Why an orchestrator?
  The three pipeline modules (ingest, transform, load) each do one job.
  This script connects them in the right order, adds error handling around
  the full sequence, and produces a summary log after every run.
  The scheduler calls this script rather than managing the module chain itself.

Usage:
  python -m src.pipeline.run_pipeline --mode live

  python -m src.pipeline.run_pipeline --mode backfill --start 2025-02-07 --end 2025-05-06

  # Dry run — fetch and transform only, skip the database write.
  # Useful for verifying the pipeline produces sensible output before
  # committing to a full backfill load.
  python -m src.pipeline.run_pipeline --mode live --dry-run
"""

import argparse
import logging
import sys
import time
from datetime import timedelta

import pandas as pd

from src.pipeline.ingest import fetch_latest, fetch_backfill
from src.pipeline.transform import transform
from src.pipeline.load import load
from src.pipeline.notify import send_alert

# ---------------------------------------------------------------------------
# Logging
# A single logger for this script. The modules write their own log lines
# so the full picture is visible in one stream when running from the terminal.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run(mode: str, start_date: str | None = None, end_date: str | None = None,
        dry_run: bool = False) -> dict:
    """
    Runs the full pipeline in the specified mode.

    Returns a summary dict with counts and timing so the caller
    (or the scheduler) can log or act on the outcome.

    Args:
        mode:       'live' or 'backfill'
        start_date: required when mode='backfill', format 'YYYY-MM-DD'
        end_date:   required when mode='backfill', format 'YYYY-MM-DD'
        dry_run:    if True, skips the load step. Useful for testing.

    Returns:
        dict with keys: mode, articles_fetched, entities_extracted,
                        loaded, duration_seconds, status, error
    """
    summary = {
        "mode":               mode,
        "articles_fetched":   0,
        "entities_extracted": 0,
        "loaded":             False,
        "duration_seconds":   0.0,
        "status":             "pending",
        "error":              None,
    }

    start_time = time.time()

    try:
        log.info(f"Pipeline starting — mode: {mode}")

        # -------------------------------------------------------------------
        # Live mode — single 15-minute update
        # -------------------------------------------------------------------
        if mode == "live":
            log.info("Fetching latest GDELT update...")
            df_raw = fetch_latest()

            if df_raw.empty:
                log.warning("Ingest returned zero articles. Nothing to process.")
                summary["status"] = "empty"
                return summary

            summary["articles_fetched"] = len(df_raw)
            log.info(f"Ingest complete — {len(df_raw)} articles fetched")

            log.info("Transforming articles...")
            df_articles, entity_records = transform(df_raw)
            summary["entities_extracted"] = len(entity_records)

            if dry_run:
                log.info("Dry run — skipping load step. No data written to warehouse.")
            else:
                log.info("Loading into warehouse...")
                load(df_articles, entity_records)
                summary["loaded"] = True
                log.info("Load complete")

            summary["status"] = "success"

        # -------------------------------------------------------------------
        # Backfill mode — process one day at a time
        #
        # Why day by day?
        #   Loading 3 months of data (11M articles) as a single DataFrame
        #   exhausts RAM — the OS killed the process before load() ran.
        #   Each daily batch is ~130k articles, safely within memory limits.
        #   The batch is discarded after load() commits, so memory stays flat
        #   across the full backfill run.
        # -------------------------------------------------------------------
        elif mode == "backfill":
            if not start_date or not end_date:
                raise ValueError(
                    "Backfill mode requires --start and --end arguments. "
                    "Example: --start 2025-02-07 --end 2025-05-06"
                )

            current        = pd.to_datetime(start_date)
            end            = pd.to_datetime(end_date)
            total_articles = 0
            total_entities = 0
            day_number     = 0
            days_skipped   = 0

            log.info(f"Starting backfill: {start_date} to {end_date}")
            log.info("Processing one day at a time to keep memory usage flat")

            # Download the master list once before the day loop begins.
            # This writes a fresh cache to disk. Every fetch_backfill() call
            # inside the loop then uses force_cache=True, which skips the age
            # check entirely. A long backfill (10+ hours) would otherwise hit
            # the 6-hour expiry mid-run and re-download the full 386k-line list.
            log.info("Pre-fetching master list before backfill loop...")
            from src.pipeline.ingest import _fetch_master_list
            _fetch_master_list(force_cache=False)  # fresh download, resets cache timestamp
            log.info("Master list ready. Starting day loop with force_cache=True.")

            while current <= end:
                day_str = current.strftime("%Y-%m-%d")
                day_number += 1

                log.info(f"--- Day {day_number}: {day_str} ---")

                try:
                    df_raw = fetch_backfill(day_str, day_str, force_cache=True)

                    if df_raw.empty:
                        log.warning(f"No data returned for {day_str} — skipping")
                        days_skipped += 1
                        current += timedelta(days=1)
                        continue

                    df_articles, entity_records = transform(df_raw)

                    if dry_run:
                        log.info(
                            f"Dry run — {len(df_articles)} articles, "
                            f"{len(entity_records)} entities (not written)"
                        )
                    else:
                        load(df_articles, entity_records)
                        summary["loaded"] = True

                    total_articles += len(df_articles)
                    total_entities += len(entity_records)

                    log.info(
                        f"Day {day_str} complete — "
                        f"{len(df_articles)} articles, {len(entity_records)} entities "
                        f"| Running total: {total_articles:,} articles"
                    )

                except Exception as day_error:
                    log.error(f"Day {day_str} failed: {day_error} — continuing to next day")
                    days_skipped += 1

                current += timedelta(days=1)

            summary["articles_fetched"]   = total_articles
            summary["entities_extracted"] = total_entities
            summary["status"]             = "success"

            log.info(
                f"Backfill complete — {total_articles:,} articles across "
                f"{day_number - days_skipped} days ({days_skipped} days skipped)"
            )

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'live' or 'backfill'.")

    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        summary["status"] = "failed"
        summary["error"]  = str(e)

    finally:
        summary["duration_seconds"] = round(time.time() - start_time, 2)

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="News Pulse pipeline orchestrator — runs ingest, transform, and load.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline.run_pipeline --mode live
  python -m src.pipeline.run_pipeline --mode backfill --start 2025-02-07 --end 2025-05-06
  python -m src.pipeline.run_pipeline --mode live --dry-run
        """
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["live", "backfill"],
        help="'live' fetches the latest 15-minute GDELT update. "
             "'backfill' fetches a full date range, one day at a time."
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date for backfill mode. Format: YYYY-MM-DD"
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date for backfill mode (inclusive). Format: YYYY-MM-DD"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and transform only — skip the database write. "
             "Useful for verifying output before a large backfill load."
    )

    args = parser.parse_args()

    summary = run(
        mode=args.mode,
        start_date=args.start,
        end_date=args.end,
        dry_run=args.dry_run
    )

    # Print a clean summary at the end of every run.
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    print(f"  Mode:               {summary['mode']}")
    print(f"  Status:             {summary['status'].upper()}")
    print(f"  Articles fetched:   {summary['articles_fetched']:,}")
    print(f"  Entities extracted: {summary['entities_extracted']:,}")
    print(f"  Written to DB:      {'Yes' if summary['loaded'] else 'No'}")
    print(f"  Duration:           {summary['duration_seconds']}s")
    if summary["error"]:
        print(f"  Error:              {summary['error']}")
    print("=" * 50 + "\n")

    # Send Telegram alert — fires for every status: success, failed, empty
    send_alert(summary)

    # Exit with a non-zero code on failure so schedulers can detect it.
    sys.exit(0 if summary["status"] in ("success", "empty") else 1)


if __name__ == "__main__":
    main()
