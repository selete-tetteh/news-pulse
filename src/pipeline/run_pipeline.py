"""
run_pipeline.py

Orchestrates the full ingest -> transform -> load sequence with one command.
Sends a Telegram alert after every run — success, failure, or empty.

Two modes:
  --mode live      Fetches the single most recent GDELT 15-minute update.
                   Used by the scheduler for continuous live updates.

  --mode backfill  Fetches all GDELT files within a date range.
                   Used once to seed the warehouse with historical data.
                   Requires --start and --end arguments.

Why an orchestrator?
  The three pipeline modules (ingest, transform, load) each do one job.
  This script connects them in the right order, adds error handling around
  the full sequence, and produces a summary log after every run.
  The scheduler calls this script rather than managing the module chain itself.

Usage:
  python -m src.pipeline.run_pipeline --mode live

  python -m src.pipeline.run_pipeline --mode backfill --start 2025-05-06 --end 2025-05-12

  # Dry run — fetch and transform only, skip the database write.
  # Useful for verifying the pipeline produces sensible output before
  # committing to a full backfill load.
  python -m src.pipeline.run_pipeline --mode live --dry-run
"""

import argparse
import logging
import sys
import time

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
        # --- Step 1: Ingest ---
        log.info(f"Pipeline starting — mode: {mode}")

        if mode == "live":
            log.info("Fetching latest GDELT update...")
            df_raw = fetch_latest()

        elif mode == "backfill":
            if not start_date or not end_date:
                raise ValueError(
                    "Backfill mode requires --start and --end arguments. "
                    "Example: --start 2025-05-06 --end 2025-05-12"
                )
            log.info(f"Starting backfill: {start_date} to {end_date}")
            df_raw = fetch_backfill(start_date, end_date)

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'live' or 'backfill'.")

        if df_raw.empty:
            log.warning("Ingest returned zero articles. Nothing to process.")
            summary["status"] = "empty"
            return summary

        summary["articles_fetched"] = len(df_raw)
        log.info(f"Ingest complete — {len(df_raw)} articles fetched")

        # --- Step 2: Transform ---
        log.info("Transforming articles...")
        df_articles, entity_records = transform(df_raw)
        summary["entities_extracted"] = len(entity_records)
        log.info(f"Transform complete — {len(df_articles)} articles, "
                 f"{len(entity_records)} entity mentions")

        # --- Step 3: Load ---
        if dry_run:
            log.info("Dry run — skipping load step. No data written to warehouse.")
            summary["loaded"] = False
        else:
            log.info("Loading into warehouse...")
            load(df_articles, entity_records)
            summary["loaded"] = True
            log.info("Load complete")

        summary["status"] = "success"

    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        summary["status"] = "failed"
        summary["error"] = str(e)

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
  python -m src.pipeline.run_pipeline --mode backfill --start 2025-05-06 --end 2025-05-12
  python -m src.pipeline.run_pipeline --mode live --dry-run
        """
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["live", "backfill"],
        help="'live' fetches the latest 15-minute GDELT update. "
             "'backfill' fetches a full date range."
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
    print(f"  Articles fetched:   {summary['articles_fetched']}")
    print(f"  Entities extracted: {summary['entities_extracted']}")
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
