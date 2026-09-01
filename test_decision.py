"""Dry-run test for the LLM decision layer. Does not place any orders."""

from __future__ import annotations

import json
import os
import sys

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

from data import get_option_chain, get_price_history
from vol_signal import compute_iv_rv_spread, compute_realized_volatility
from decision import allowed_actions, decide_trade

load_dotenv()

SYMBOL = "SPY"
LOOKBACK_DAYS = 30
RV_WINDOW = 20
DTE_RANGE = (20, 45)


def _account_equity() -> float:
    """Paper-account equity from Alpaca."""
    api_key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret:
        raise ValueError("Missing Alpaca API credentials in .env.")
    account = TradingClient(api_key, secret, paper=True).get_account()
    return float(account.equity)


def main() -> None:
    print(f"Dry run: computing SPY signal, then asking the LLM to decide.")
    print("No orders will be placed.\n")

    try:
        price_df = get_price_history(SYMBOL, lookback_days=LOOKBACK_DAYS)
        option_chain_df = get_option_chain(SYMBOL)
        realized_vol = compute_realized_volatility(price_df, window=RV_WINDOW)
        spread, avg_iv, rv = compute_iv_rv_spread(
            option_chain_df,
            realized_vol,
            target_dte_range=DTE_RANGE,
        )
        equity = _account_equity()
    except Exception as exc:
        print(
            "Could not build the signal / account inputs. "
            f"The market may be closed or data may be unavailable.\nDetails: {exc}"
        )
        sys.exit(1)

    permitted = allowed_actions(spread)
    print(f"Symbol:            {SYMBOL}")
    print(f"Account equity:    ${equity:,.2f}")
    print(f"Realized vol:      {rv:.4f}")
    print(f"Average IV:        {avg_iv:.4f}")
    print(f"IV/RV spread:      {spread:.4f}")
    print(f"Allowed actions:   {permitted}")
    print()

    try:
        decision = decide_trade(
            iv_rv_spread=spread,
            avg_iv=avg_iv,
            realized_vol=rv,
            option_chain_df=option_chain_df,
            account_equity=equity,
            symbol=SYMBOL,
        )
    except Exception as exc:
        print(f"Decision layer failed.\nDetails: {exc}")
        sys.exit(1)

    print("LLM decision (dry run, no order placed):")
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
