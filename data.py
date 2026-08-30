"""Market and option data fetching."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed, OptionsFeed
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# OCC option symbols end with YYMMDD + C/P + 8-digit strike (strike * 1000).
_OCC_SUFFIX = re.compile(r"^(?P<root>.+)(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


def _load_keys() -> tuple[str, str]:
    """Return Alpaca API credentials from the environment."""
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError(
            "Missing Alpaca API credentials. "
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your .env file."
        )
    return api_key, secret_key


def _parse_occ_symbol(symbol: str) -> tuple[date, str, float]:
    """Parse expiration, call/put, and strike from an OCC option symbol."""
    match = _OCC_SUFFIX.match(symbol)
    if match is None:
        raise ValueError(f"Unrecognized option symbol format: {symbol}")
    expiration = datetime.strptime(match.group("ymd"), "%y%m%d").date()
    option_type = "call" if match.group("cp") == "C" else "put"
    strike = int(match.group("strike")) / 1000.0
    return expiration, option_type, strike


def get_price_history(symbol: str, lookback_days: int = 30) -> pd.DataFrame:
    """Pull daily OHLCV bars for `symbol` over the lookback window.

    Extra calendar days are requested so weekends and holidays still leave
    enough trading sessions for a 20-day volatility window.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    api_key, secret_key = _load_keys()
    client = StockHistoricalDataClient(api_key, secret_key)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days + 14)

    last_error: Exception | None = None
    bar_set = None
    for feed in (DataFeed.IEX, DataFeed.SIP):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                adjustment=Adjustment.ALL,
                feed=feed,
            )
            bar_set = client.get_stock_bars(request)
            if not bar_set.df.empty:
                break
        except Exception as exc:
            last_error = exc
            bar_set = None

    if bar_set is None or bar_set.df.empty:
        detail = f" Details: {last_error}" if last_error else ""
        raise RuntimeError(
            f"No daily price history returned for {symbol}. "
            "The market may be closed or the data feed may be unavailable."
            + detail
        )

    df = bar_set.df.reset_index()
    keep = [col for col in ("timestamp", "open", "high", "low", "close", "volume") if col in df.columns]
    return df[keep].sort_values("timestamp").reset_index(drop=True)


def get_long_price_history(symbol: str, lookback_years: float = 2.0) -> pd.DataFrame:
    """Pull as much daily history as the feed will give, targeting `lookback_years`.

    Uses the same client path as get_price_history. If the free tier returns a
    shorter window, the caller should report the actual span.
    """
    calendar_days = int(lookback_years * 365) + 40
    return get_price_history(symbol, lookback_days=calendar_days)


def get_option_chain(symbol: str) -> pd.DataFrame:
    """Pull the current option chain snapshot for an underlying symbol.

    Uses Alpaca's pre-computed implied volatility and greeks. Expiration is
    limited to roughly 15–60 DTE so the request stays within API page limits
    and still covers the 20–45 DTE signal window.

    Returns:
        DataFrame with strike, expiration, type, implied_volatility, delta,
        bid, and ask (plus contract symbol).
    """
    api_key, secret_key = _load_keys()
    client = OptionHistoricalDataClient(api_key, secret_key)

    today = date.today()
    exp_gte = today + timedelta(days=15)
    exp_lte = today + timedelta(days=60)

    last_error: Exception | None = None
    snapshots = None
    for feed in (OptionsFeed.INDICATIVE, OptionsFeed.OPRA):
        try:
            request = OptionChainRequest(
                underlying_symbol=symbol,
                feed=feed,
                expiration_date_gte=exp_gte,
                expiration_date_lte=exp_lte,
            )
            snapshots = client.get_option_chain(request)
            if snapshots:
                break
        except Exception as exc:
            last_error = exc
            snapshots = None

    if not snapshots:
        detail = f" Details: {last_error}" if last_error else ""
        raise RuntimeError(
            f"No option chain returned for {symbol}. "
            "The market may be closed or options data may be unavailable."
            + detail
        )

    rows: list[dict] = []
    for contract_symbol, snap in snapshots.items():
        try:
            expiration, option_type, strike = _parse_occ_symbol(contract_symbol)
        except ValueError:
            continue

        quote = snap.latest_quote
        greeks = snap.greeks
        rows.append(
            {
                "symbol": contract_symbol,
                "strike": strike,
                "expiration": expiration,
                "type": option_type,
                "implied_volatility": snap.implied_volatility,
                "delta": greeks.delta if greeks is not None else None,
                "bid": quote.bid_price if quote is not None else None,
                "ask": quote.ask_price if quote is not None else None,
            }
        )

    if not rows:
        raise RuntimeError(
            f"Option chain for {symbol} had no parseable contracts."
        )

    return pd.DataFrame(rows)
