"""
NIH paper ingestion scheduler.
Runs the pipeline once on startup, then repeats every INGEST_INTERVAL_HOURS hours.
Default: 168 hours (weekly).
"""
import os
import sys
import time
import logging
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ingestion] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

INTERVAL_HOURS = int(os.getenv("INGEST_INTERVAL_HOURS", "168"))


def run_once():
    log.info("Starting NIH paper ingestion pipeline...")
    try:
        from app.core.nih_paper_downloader import run_pipeline
        run_pipeline(force_recreate=False)
        log.info("Pipeline completed successfully.")
    except Exception:
        log.error("Pipeline failed:", exc_info=True)


if __name__ == "__main__":
    log.info(f"Ingestion scheduler started — interval: every {INTERVAL_HOURS}h ({INTERVAL_HOURS // 24}d).")
    while True:
        run_once()
        next_run = datetime.now() + timedelta(hours=INTERVAL_HOURS)
        log.info(f"Next run at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(INTERVAL_HOURS * 3600)
