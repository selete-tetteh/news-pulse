#!/usr/bin/env bash
# kill_scheduler.sh
#
# Stops a running News Pulse scheduler process.
#
# Usage:
#   bash scripts/kill_scheduler.sh
#
# How it works:
#   Searches all running processes for the scheduler module path,
#   extracts the process ID (PID), and sends SIGTERM to it.
#   SIGTERM asks the process to shut down cleanly — the scheduler's
#   signal handler logs the shutdown before exiting.
#
#   If no scheduler process is found, prints a message and exits.
#   If multiple matches are found (unlikely), kills all of them.

set -euo pipefail

MATCH="src.pipeline.scheduler"

# pgrep -f searches full command lines, not just process names.
# This matches "python -m src.pipeline.scheduler" exactly.
PIDS=$(pgrep -f "$MATCH" || true)

if [ -z "$PIDS" ]; then
    echo "No running scheduler process found."
    exit 0
fi

echo "Found scheduler process(es): $PIDS"
echo "Sending shutdown signal..."

# kill sends SIGTERM by default — the scheduler handles this gracefully.
# We do not use SIGKILL (kill -9) because that would not let the scheduler
# log the shutdown or finish any in-progress work.
kill $PIDS

echo "Scheduler stopped."
