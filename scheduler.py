"""Market-hours scheduler for the trading agent. Respects main.DRY_RUN."""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date, datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

from execute import TRADE_LOG_PATH, count_open_option_positions
from main import DRY_RUN, format_summary_line, run_once

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
LOG_PATH = PROJECT_ROOT / "scheduler.log"

load_dotenv(ENV_PATH)

# Day-one live paper: most liquid names only.
# Full list to restore later: ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "IWM"]
SYMBOLS = ["SPY", "QQQ", "IWM"]
INTERVAL_MINUTES = 15
MAX_RUNS_PER_DAY = 20  # one run = one symbol pipeline execution

logger = logging.getLogger("scheduler")


def setup_logging() -> None:
    """Log to console and a rotating scheduler.log next to this script."""
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        return

    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def _log(message: str) -> None:
    """Write to the rotating file and stdout. Never waits for input."""
    logger.info(message)


def _trading_client() -> TradingClient:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret:
        raise ValueError("Missing Alpaca API credentials in .env.")
    return TradingClient(api_key, secret, paper=True)


def _get_clock():
    return _trading_client().get_clock()


def run_cycle() -> int:
    """Run one scheduler cycle. Returns how many symbol pipelines were attempted.

    If the market is closed, skips all data/trade work and returns 0.
    """
    now = datetime.now(timezone.utc).isoformat()
    clock = _get_clock()
    status = "OPEN" if clock.is_open else "CLOSED"
    _log(f"======== CYCLE {now} ========")
    _log(f"Market: {status}")
    _log(f"DRY_RUN: {DRY_RUN}")
    _log(f"Symbols: {', '.join(SYMBOLS)}")

    if not clock.is_open:
        _log(f"Market closed, skipping run. Next open: {clock.next_open}")
        return 0

    open_positions = count_open_option_positions()
    _log(f"Open option positions (fresh): {open_positions}")

    attempted = 0
    for symbol in SYMBOLS:
        attempted += 1
        try:
            summary = run_once(
                symbol,
                current_open_positions=open_positions,
                verbose=False,
            )
            _log(summary["line"])
        except Exception as exc:
            error = str(exc).replace("\n", " ")
            _log(format_summary_line(symbol, error=error))
    return attempted


def main() -> None:
    setup_logging()
    mode = "DRY RUN (no orders)" if DRY_RUN else "LIVE PAPER ORDERS"
    _log("Scheduler starting.")
    _log(f"Mode: {mode}")
    _log(f"Env file: {ENV_PATH}")
    _log(f"Log file: {LOG_PATH}")
    _log(f"Trade log: {TRADE_LOG_PATH}")
    _log(f"Interval: {INTERVAL_MINUTES} minutes")
    _log(f"Max runs per day (all symbols combined): {MAX_RUNS_PER_DAY}")

    runs_today = 0
    runs_date = date.today()

    try:
        while True:
            today = date.today()
            if today != runs_date:
                runs_today = 0
                runs_date = today
                _log(f"New day {runs_date.isoformat()}: run counter reset.")

            if runs_today >= MAX_RUNS_PER_DAY:
                _log(
                    f"MAX_RUNS_PER_DAY ({MAX_RUNS_PER_DAY}) reached. "
                    "Stopping for the day. Will resume tomorrow."
                )
            else:
                attempted = run_cycle()
                runs_today += attempted
                _log(f"Runs today: {runs_today}/{MAX_RUNS_PER_DAY}")
                if runs_today >= MAX_RUNS_PER_DAY:
                    _log(
                        f"MAX_RUNS_PER_DAY ({MAX_RUNS_PER_DAY}) reached. "
                        "No more symbol runs until tomorrow."
                    )

            _log(f"Sleeping {INTERVAL_MINUTES} minutes...")
            time.sleep(INTERVAL_MINUTES * 60)
    except KeyboardInterrupt:
        _log("Scheduler stopped")


if __name__ == "__main__":
    main()
