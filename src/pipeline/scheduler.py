"""
scheduler.py

Runs the live pipeline every 15 minutes using APScheduler.

Start with:
    python -m src.pipeline.scheduler

Stop with Ctrl+C in the same terminal, or run:
    bash scripts/kill_scheduler.sh

How it works:
    APScheduler is a Python library that fires a function on a fixed
    interval. Here we fire run_pipeline.run(mode="live") every 15 minutes.
    That function fetches the latest GDELT update, transforms it, and loads
    it into the warehouse — the same sequence as running the pipeline manually,
    just automated.

    The scheduler also fires once immediately on startup so the warehouse
    is updated the moment you start it, rather than waiting up to 15 minutes
    for the first interval to elapse.

Why APScheduler and not cron?
    cron works well for scripts that run and exit. APScheduler keeps a Python
    process alive and manages the schedule internally, which means it has access
    to the full pipeline context (imports, logging, environment variables) without
    any extra shell configuration. For a project already written in Python, this
    is the simpler choice.

    For OS-managed scheduling that survives reboots without a running process,
    see launchd/com.newspulse.pipeline.plist. That is the macOS production approach.

Scheduler behaviour:
    - misfire_grace_time: if a run is delayed (e.g. the previous run was still
      going), APScheduler will still execute it if the delay is under this
      threshold. Set to 60 seconds — enough to absorb minor delays without
      stacking up missed runs.
    - max_instances: only one pipeline run at a time. If a run takes longer
      than 15 minutes (rare but possible on slow networks), the next scheduled
      run is skipped rather than stacking on top of the running one.
    - coalesce: if multiple runs were missed while the scheduler was paused,
      only one catch-up run fires rather than all missed runs at once.
"""

import logging
import signal
import sys
import time

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

from src.pipeline.run_pipeline import run

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scheduler config
# ---------------------------------------------------------------------------
INTERVAL_MINUTES = 15


# ---------------------------------------------------------------------------
# Job function
# ---------------------------------------------------------------------------

def pipeline_job():
    """
    Called by APScheduler every INTERVAL_MINUTES.
    Runs the full live pipeline: ingest -> transform -> load.
    The summary is logged here. Telegram alert is sent inside run_pipeline.run().
    """
    log.info("Scheduler firing — starting live pipeline run")
    summary = run(mode="live")
    log.info(
        f"Run complete — status: {summary['status'].upper()} | "
        f"articles: {summary['articles_fetched']} | "
        f"entities: {summary['entities_extracted']} | "
        f"duration: {summary['duration_seconds']}s"
    )


# ---------------------------------------------------------------------------
# Event listener
# ---------------------------------------------------------------------------

def on_job_event(event):
    """
    Fires after every job execution. Logs whether the job succeeded or
    raised an exception. APScheduler catches exceptions inside jobs so the
    scheduler itself keeps running — this listener surfaces them in the logs.
    """
    if event.exception:
        log.error(f"Scheduled job raised an exception: {event.exception}")
    else:
        log.info("Scheduled job finished without errors")


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

def handle_shutdown(signum, frame):
    """
    Handles SIGTERM (sent by kill_scheduler.sh or the OS) cleanly.
    Ctrl+C sends SIGINT which BlockingScheduler already handles.
    This handler ensures SIGTERM also shuts down gracefully.
    """
    log.info("Shutdown signal received — stopping scheduler")
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_shutdown)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    log.info("News Pulse scheduler starting")
    log.info(f"Pipeline will run every {INTERVAL_MINUTES} minutes")
    log.info("Press Ctrl+C to stop, or run: bash scripts/kill_scheduler.sh")

    scheduler = BlockingScheduler(timezone="UTC")

    scheduler.add_job(
        pipeline_job,
        trigger="interval",
        minutes=INTERVAL_MINUTES,
        id="live_pipeline",
        name="News Pulse live pipeline",
        misfire_grace_time=60,   # fire late runs if delay is under 60s
        max_instances=1,         # never run two pipeline instances at once
        coalesce=True,           # if multiple runs were missed, fire once only
    )

    scheduler.add_listener(on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    # Fire once immediately on startup so the warehouse is updated right away
    # rather than waiting for the first 15-minute interval to elapse.
    log.info("Firing initial run on startup...")
    pipeline_job()
    log.info(f"Initial run complete. Next scheduled run in {INTERVAL_MINUTES} minutes.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")


if __name__ == "__main__":
    main()
