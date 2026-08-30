"""Market-hours scheduler for the trading agent. Respects main.DRY_RUN."""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timezone

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

from execute import count_open_option_positions
from main import DRY_RUN, format_summary_line, run_once

load_dotenv()

# Day-one live paper: most liquid names only.
# Full list to restore later: ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "IWM"]
SYMBOLS = ["SPY", "QQQ", "IWM"]
INTERVAL_MINUTES = 15
MAX_RUNS_PER_DAY = 20  # one run = one symbol pipeline execution


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
    print(f"\n======== CYCLE {now} ========")
    print(f"Market: {status}")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"Symbols: {', '.join(SYMBOLS)}")

    if not clock.is_open:
        print(
            f"Market closed, skipping run. Next open: {clock.next_open}"
        )
        return 0

    open_positions = count_open_option_positions()
    print(f"Open option positions (fresh): {open_positions}")

    attempted = 0
    for symbol in SYMBOLS:
        attempted += 1
        try:
            summary = run_once(
                symbol,
                current_open_positions=open_positions,
                verbose=False,
            )
            print(summary["line"])
        except Exception as exc:
            error = str(exc).replace("\n", " ")
            print(format_summary_line(symbol, error=error))
    return attempted


def main() -> None:
    mode = "DRY RUN (no orders)" if DRY_RUN else "LIVE PAPER ORDERS"
    print("Scheduler starting.")
    print(f"Mode: {mode}")
    print(f"Interval: {INTERVAL_MINUTES} minutes")
    print(f"Max runs per day (all symbols combined): {MAX_RUNS_PER_DAY}")
    print("Ctrl+C to stop.")

    runs_today = 0
    runs_date = date.today()

    try:
        while True:
            today = date.today()
            if today != runs_date:
                runs_today = 0
                runs_date = today
                print(f"New day {runs_date.isoformat()}: run counter reset.")

            if runs_today >= MAX_RUNS_PER_DAY:
                print(
                    f"MAX_RUNS_PER_DAY ({MAX_RUNS_PER_DAY}) reached. "
                    "Stopping for the day. Will resume tomorrow."
                )
            else:
                attempted = run_cycle()
                runs_today += attempted
                print(f"Runs today: {runs_today}/{MAX_RUNS_PER_DAY}")
                if runs_today >= MAX_RUNS_PER_DAY:
                    print(
                        f"MAX_RUNS_PER_DAY ({MAX_RUNS_PER_DAY}) reached. "
                        "No more symbol runs until tomorrow."
                    )

            print(f"Sleeping {INTERVAL_MINUTES} minutes...")
            time.sleep(INTERVAL_MINUTES * 60)
    except KeyboardInterrupt:
        print("\nScheduler stopped")


if __name__ == "__main__":
    main()
