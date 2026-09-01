"""Strategy and signal logic."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def rolling_realized_volatility(price_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualized rolling realized vol: std(log returns) * sqrt(252)."""
    if price_df is None or price_df.empty:
        raise ValueError("price_df is empty; cannot compute realized volatility.")
    if "close" not in price_df.columns:
        raise ValueError("price_df must include a 'close' column.")

    closes = price_df["close"].astype(float)
    log_returns = np.log(closes / closes.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(252)


def compute_realized_volatility(price_df: pd.DataFrame, window: int = 20) -> float:
    """Annualized historical realized vol from daily log returns.

    Formula: std(log returns) * sqrt(252), using a rolling window.
    Returns the most recent value.
    """
    rolling_vol = rolling_realized_volatility(price_df, window=window)
    latest = rolling_vol.dropna()
    if latest.empty:
        raise ValueError("Realized volatility could not be computed from the price series.")
    if len(price_df.dropna(subset=["close"])) < window + 1:
        raise ValueError(
            f"Need at least {window + 1} daily closes for a {window}-day window; "
            f"got {len(price_df.dropna(subset=['close']))}."
        )
    return float(latest.iloc[-1])


def compute_iv_rv_spread(
    option_chain_df: pd.DataFrame,
    realized_vol: float,
    target_dte_range: tuple[int, int] = (20, 45),
) -> tuple[float, float, float]:
    """Compare average implied vol to realized vol in a DTE window.

    Filters the chain to `target_dte_range`, then averages implied volatility.
    If delta is available, near-the-money contracts (|delta| ≈ 0.50) are
    weighted more heavily.

    Returns:
        (spread, avg_iv, realized_vol) as decimals.
        Example: 0.05 means IV is 5 percentage points above realized vol.
    """
    if option_chain_df is None or option_chain_df.empty:
        raise ValueError("option_chain_df is empty; cannot compute IV/RV spread.")
    required = {"expiration", "implied_volatility"}
    missing = required - set(option_chain_df.columns)
    if missing:
        raise ValueError(f"option_chain_df missing columns: {sorted(missing)}")

    min_dte, max_dte = target_dte_range
    today = date.today()
    expirations = pd.to_datetime(option_chain_df["expiration"]).dt.date
    dte = expirations.map(lambda exp: (exp - today).days)
    filtered = option_chain_df.loc[(dte >= min_dte) & (dte <= max_dte)].copy()
    filtered = filtered.dropna(subset=["implied_volatility"])

    if filtered.empty:
        raise ValueError(
            f"No option contracts with implied vol in the {min_dte}-{max_dte} DTE window."
        )

    if "delta" in filtered.columns and filtered["delta"].notna().any():
        atm = filtered.dropna(subset=["delta"])
        # Weight peaks at |delta| = 0.50 (ATM) and falls off as strikes move away.
        distance = (atm["delta"].abs() - 0.50).abs()
        weights = 1.0 / (distance + 0.05)
        avg_iv = float(np.average(atm["implied_volatility"].astype(float), weights=weights))
    else:
        avg_iv = float(filtered["implied_volatility"].astype(float).mean())

    spread = avg_iv - realized_vol
    return spread, avg_iv, float(realized_vol)
