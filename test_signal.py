"""Standalone test for the IV/RV signal layer. Change SYMBOL to try another ticker."""

from __future__ import annotations

import sys

from data import get_option_chain, get_price_history
from vol_signal import compute_iv_rv_spread, compute_realized_volatility

# Easy to change for a different underlying.
SYMBOL = "SPY"
LOOKBACK_DAYS = 30
RV_WINDOW = 20
DTE_RANGE = (20, 45)


def main() -> None:
    print(f"Computing IV/RV signal for {SYMBOL}...")
    try:
        price_df = get_price_history(SYMBOL, lookback_days=LOOKBACK_DAYS)
        option_chain_df = get_option_chain(SYMBOL)
        realized_vol = compute_realized_volatility(price_df, window=RV_WINDOW)
        spread, avg_iv, rv = compute_iv_rv_spread(
            option_chain_df,
            realized_vol,
            target_dte_range=DTE_RANGE,
        )
    except Exception as exc:
        print(
            "Could not compute the signal. The market may be closed or "
            f"data may be unavailable.\nDetails: {exc}"
        )
        sys.exit(1)

    print(f"Contracts used (full fetched chain): {len(option_chain_df)}")
    print(f"Daily bars used: {len(price_df)}")
    print(f"Realized volatility ({RV_WINDOW}d, annualized): {rv:.4f}")
    print(f"Average IV ({DTE_RANGE[0]}-{DTE_RANGE[1]} DTE): {avg_iv:.4f}")
    print(f"IV/RV spread (avg IV - realized vol): {spread:.4f}")


if __name__ == "__main__":
    main()
