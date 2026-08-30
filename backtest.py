"""PROXY backtest of vol mean-reversion — not options P&L.

Tests whether 20-day realized vol mean-reverts toward a 90-day baseline.
This is the core assumption behind selling premium when vol looks rich
and buying it when vol looks cheap. It does NOT simulate spreads, IV,
or fill prices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import get_long_price_history
from signal import rolling_realized_volatility

SYMBOLS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "IWM"]
SHORT_WINDOW = 20
BASELINE_WINDOW = 90
FORWARD_WINDOW = 20
THRESHOLD = 0.02
LOW_SAMPLE = 15
LOOKBACK_YEARS = 2.0


@dataclass
class SignalResult:
    direction: str  # "sell" or "buy"
    date: object
    rv_20: float
    rv_90: float
    vol_gap: float
    forward_rv: float
    vol_change: float
    win: bool


@dataclass
class SymbolReport:
    symbol: str
    bars: int
    first_date: str
    last_date: str
    years_covered: float
    sell_signals: list[SignalResult] = field(default_factory=list)
    buy_signals: list[SignalResult] = field(default_factory=list)
    error: str | None = None


def _forward_realized_vol(closes: pd.Series, start_idx: int, window: int) -> float:
    """Realized vol over the next `window` trading-day returns after start_idx."""
    segment = closes.iloc[start_idx : start_idx + window + 1].astype(float)
    log_ret = np.log(segment / segment.shift(1)).dropna()
    return float(log_ret.std() * math.sqrt(252))


def _summarize_side(signals: list[SignalResult], direction: str) -> dict:
    n = len(signals)
    if n == 0:
        return {
            "n": 0,
            "wins": 0,
            "win_rate": None,
            "avg_change_wins": None,
            "avg_change_losses": None,
            "worst_adverse": None,
        }
    wins = [s for s in signals if s.win]
    losses = [s for s in signals if not s.win]
    if direction == "sell":
        # Adverse for a premium seller: vol rose after the signal.
        worst = max(s.vol_change for s in signals)
    else:
        # Adverse for a premium buyer: vol fell after the signal.
        worst = min(s.vol_change for s in signals)
    return {
        "n": n,
        "wins": len(wins),
        "win_rate": len(wins) / n,
        "avg_change_wins": (
            float(np.mean([s.vol_change for s in wins])) if wins else None
        ),
        "avg_change_losses": (
            float(np.mean([s.vol_change for s in losses])) if losses else None
        ),
        "worst_adverse": worst,
    }


def _caveat(n: int, label: str) -> str | None:
    if n == 0:
        return f"No {label} signals in this sample."
    if n < LOW_SAMPLE:
        return (
            f"only {n} {label} signals - not statistically robust, "
            "treat as directional evidence only"
        )
    return None


def backtest_symbol(symbol: str) -> SymbolReport:
    """PROXY: vol mean-reversion on historical prices only."""
    prices = get_long_price_history(symbol, lookback_years=LOOKBACK_YEARS)
    if prices.empty or "close" not in prices.columns:
        return SymbolReport(symbol, 0, "", "", 0.0, error="no price history")

    prices = prices.sort_values("timestamp").reset_index(drop=True)
    first = pd.to_datetime(prices["timestamp"].iloc[0])
    last = pd.to_datetime(prices["timestamp"].iloc[-1])
    years = (last - first).days / 365.25
    report = SymbolReport(
        symbol=symbol,
        bars=len(prices),
        first_date=first.date().isoformat(),
        last_date=last.date().isoformat(),
        years_covered=years,
    )

    closes = prices["close"].astype(float)
    rv_20 = rolling_realized_volatility(prices, window=SHORT_WINDOW)
    rv_90 = rolling_realized_volatility(prices, window=BASELINE_WINDOW)

    last_usable = len(prices) - FORWARD_WINDOW - 1
    i = BASELINE_WINDOW  # skip the first 90 sessions used for the baseline
    while i <= last_usable:
        short_vol = rv_20.iloc[i]
        base_vol = rv_90.iloc[i]
        if pd.isna(short_vol) or pd.isna(base_vol):
            i += 1
            continue

        gap = float(short_vol) - float(base_vol)
        if abs(gap) <= THRESHOLD:
            i += 1
            continue

        direction = "sell" if gap > THRESHOLD else "buy"
        forward = _forward_realized_vol(closes, i, FORWARD_WINDOW)
        change = forward - float(short_vol)
        if direction == "sell":
            win = forward < float(short_vol)
        else:
            win = forward > float(short_vol)

        result = SignalResult(
            direction=direction,
            date=pd.to_datetime(prices["timestamp"].iloc[i]).date(),
            rv_20=float(short_vol),
            rv_90=float(base_vol),
            vol_gap=gap,
            forward_rv=forward,
            vol_change=change,
            win=win,
        )
        if direction == "sell":
            report.sell_signals.append(result)
        else:
            report.buy_signals.append(result)

        # Non-overlapping: skip the forward window so the next signal is independent.
        i += FORWARD_WINDOW

    return report


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _fmt_vol(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def _print_side(title: str, stats: dict, short_label: str | None = None) -> None:
    note = _caveat(stats["n"], short_label or title.lower())
    print(f"  {title}: {stats['n']} signals, win rate {_fmt_pct(stats['win_rate'])}")
    print(
        f"    avg forward vol change | wins: {_fmt_vol(stats['avg_change_wins'])}  "
        f"losses: {_fmt_vol(stats['avg_change_losses'])}"
    )
    print(f"    worst-case adverse vol move: {_fmt_vol(stats['worst_adverse'])}")
    if note:
        print(f"    caveat: {note}")


def _print_report(report: SymbolReport) -> None:
    print(f"\n--- {report.symbol} (PROXY, not options P&L) ---")
    if report.error:
        print(f"  ERROR: {report.error}")
        return
    print(
        f"  History: {report.bars} daily bars, {report.first_date} to "
        f"{report.last_date} ({report.years_covered:.2f} years). "
        f"Requested ~{LOOKBACK_YEARS:.0f}y; this is what the feed returned."
    )
    sell = _summarize_side(report.sell_signals, "sell")
    buy = _summarize_side(report.buy_signals, "buy")
    _print_side("Sell-premium (rv_20 > rv_90 + 0.02)", sell, "sell")
    _print_side("Buy-premium  (rv_20 < rv_90 - 0.02)", buy, "buy")


def _print_aggregate(reports: list[SymbolReport]) -> None:
    sells: list[SignalResult] = []
    buys: list[SignalResult] = []
    for report in reports:
        sells.extend(report.sell_signals)
        buys.extend(report.buy_signals)

    print("\n========== AGGREGATE ACROSS SYMBOLS (PROXY) ==========")
    sell = _summarize_side(sells, "sell")
    buy = _summarize_side(buys, "buy")
    print(f"  Symbols with data: {sum(1 for r in reports if not r.error)}/{len(reports)}")
    _print_side("Sell-premium", sell, "sell")
    _print_side("Buy-premium", buy, "buy")


def _print_caveat() -> None:
    print(
        "\n"
        "========== HONEST CAVEAT - READ THIS FOR THE PITCH ==========\n"
        "This is a PROXY test of the strategy's CORE ASSUMPTION, not a\n"
        "backtest of actual options P&L.\n"
        "\n"
        "It only asks: when 20-day realized vol is elevated or depressed\n"
        "versus a 90-day baseline, does realized vol over the next 20\n"
        "trading days tend to mean-revert? That is the premise behind\n"
        "selling premium when vol looks 'rich' and buying it when vol\n"
        "looks 'cheap'.\n"
        "\n"
        "It does NOT account for:\n"
        "  - actual option implied volatility or IV/RV spread at the time\n"
        "  - bid/ask spreads, slippage, or fill quality\n"
        "  - time decay (theta), greeks, or strike/expiration selection\n"
        "  - multi-leg vertical construction or defined-risk max loss\n"
        "\n"
        "Real paper/live P&L will differ, sometimes by a lot. Treat a\n"
        "supportive win rate as evidence the core assumption is not\n"
        "obviously wrong - not as proof the exact spread strategy is\n"
        "profitable.\n"
        "=============================================================="
    )


def main() -> None:
    print("PROXY VOL MEAN-REVERSION BACKTEST")
    print("Not an options P&L backtest. Price history only.")
    print(
        f"Signal: |rv_{SHORT_WINDOW} - rv_{BASELINE_WINDOW}| > {THRESHOLD}  |  "
        f"forward window: {FORWARD_WINDOW} trading days, non-overlapping"
    )

    reports: list[SymbolReport] = []
    for symbol in SYMBOLS:
        try:
            report = backtest_symbol(symbol)
        except Exception as exc:
            report = SymbolReport(
                symbol, 0, "", "", 0.0, error=str(exc).replace("\n", " ")
            )
        reports.append(report)
        _print_report(report)

    _print_aggregate(reports)
    _print_caveat()


if __name__ == "__main__":
    main()
