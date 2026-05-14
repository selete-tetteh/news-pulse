"""
notify.py

Sends pipeline run alerts to a Telegram chat.

Called by run_pipeline.py after every run — success, failure, or empty.
Failures include the error message. Completions include article and entity
counts. Empty runs are flagged separately so you know GDELT returned nothing
rather than the pipeline breaking.

Why Telegram?
  The Telegram Bot API is a simple HTTP endpoint — no SDK required, no
  paid tier, no app that needs to be open. A bot token and a chat ID are
  the only credentials needed. Both are stored in .env and never committed.

Setup (one-time):
  1. Message @BotFather on Telegram -> /newbot -> follow prompts -> copy token
  2. Start a conversation with your new bot
  3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates -> copy chat id
  4. Add to .env:
       TELEGRAM_BOT_TOKEN=your_token_here
       TELEGRAM_CHAT_ID=your_chat_id_here

Usage:
  from src.pipeline.notify import send_alert
  send_alert(summary)   # summary dict from run_pipeline.run()
"""

import os
import logging
import requests
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

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


# --- Message builder ---

def _build_message(summary: dict) -> str:
    """
    Builds the Telegram message text from a pipeline summary dict.

    Status icons make it immediately obvious what happened without
    having to read the full message at a glance.
    """
    status = summary.get("status", "unknown")
    mode   = summary.get("mode", "unknown")

    if status == "success":
        icon = "✅"
        headline = f"{icon} Pipeline completed — {mode} run"
    elif status == "failed":
        icon = "❌"
        headline = f"{icon} Pipeline FAILED — {mode} run"
    elif status == "empty":
        icon = "⚠️"
        headline = f"{icon} Pipeline returned no data — {mode} run"
    else:
        icon = "❓"
        headline = f"{icon} Pipeline status unknown — {mode} run"

    lines = [headline, ""]

    lines.append(f"Articles fetched:   {summary.get('articles_fetched', 0)}")
    lines.append(f"Entities extracted: {summary.get('entities_extracted', 0)}")
    lines.append(f"Written to DB:      {'Yes' if summary.get('loaded') else 'No'}")
    lines.append(f"Duration:           {summary.get('duration_seconds', 0)}s")

    if summary.get("error"):
        lines.append("")
        lines.append(f"Error: {summary['error']}")

    return "\n".join(lines)


# --- Sender ---

def send_alert(summary: dict) -> None:
    """
    Sends a pipeline summary alert to the configured Telegram chat.

    Fails silently with a log warning if credentials are missing or
    the request fails. A notification failure should never crash the
    pipeline itself.

    Args:
        summary: the dict returned by run_pipeline.run()
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning(
            "Telegram credentials not found in .env — skipping alert. "
            "Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env to enable alerts."
        )
        return

    message = _build_message(summary)

    try:
        response = requests.post(
            TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN),
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text":    message,
            },
            timeout=10
        )
        response.raise_for_status()
        log.info("Telegram alert sent")

    except requests.exceptions.RequestException as e:
        # Log the failure but do not raise — a failed notification
        # should never cause the pipeline run to be marked as failed.
        log.warning(f"Telegram alert failed to send: {e}")


# --- Quick test when run directly ---

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    test_summary = {
        "mode":               "live",
        "status":             "success",
        "articles_fetched":   1308,
        "entities_extracted": 7758,
        "loaded":             True,
        "duration_seconds":   78.04,
        "error":              None,
    }

    print("Sending test alert...")
    send_alert(test_summary)
    print("Done. Check your Telegram.")
