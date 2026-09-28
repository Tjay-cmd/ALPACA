"""Implied-vs-realized backtest on real implied vol: S&P 500 + VIX, 2002-2018.

backtest.py can only test realized vol against realized vol, because Alpaca
has no deep options history. This script uses the VIX (30-day implied vol on
the S&P 500) so we can test the live IV/RV rule itself, and compare it with
candidate replacement signals.

Data (no Alpaca keys needed):
  - S&P 500 daily closes 1999-2018, bundled with the `arch` package
  - VIX daily closes, CBOE history mirrored at github.com/datasets/finance-vix

P&L proxy: a 21-trading-day short variance swap, in vega-notional vol points,
    pnl = (IV^2 - RV_fwd^2) / (2 * IV)
Positive when forward realized vol comes in under the implied vol we sold.
It is convex like short options, so crash months hurt more than a plain
"did vol fall" win rate shows. Buying premium is the same trade with the sign
flipped.

Method: one decision every 21 trading days (non-overlapping holds). The
results are averaged over all 21 possible start offsets so no single calendar
alignment drives them. The sample is split at 2010/2011 and every rule is
reported on both halves; a rule counts only if it holds up in both.

Caveats: VIX is a variance-swap style index and sits roughly 1-3 vol points
above the ATM-weighted IV the live agent uses; this is one underlying (SPX);
and a variance swap is not a defined-risk vertical. Treat results as evidence
about the signal, not as spread P&L.

Usage:
    pip install arch pandas numpy
    python vrp_backtest.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from decision import DEFAULT_SPREAD_THRESHOLD
from vol_signal import DEFAULT_RICH_RATIO

VIX_URL = "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"
VIX_CACHE = Path(__file__).resolve().parent / "data_cache" / "vix-daily.csv"
HOLD_DAYS = 21  # VIX is a 30-calendar-day vol, about 21 sessions
RV_WINDOW = 20
ANNUALIZE = np.sqrt(252)
SPLIT_YEAR = 2010  # first half <= 2010, second half > 2010
NEVER = -np.inf


def load_vix() -> pd.Series:
    if not VIX_CACHE.exists():
        VIX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(VIX_URL).to_csv(VIX_CACHE, index=False)
    vix = pd.read_csv(VIX_CACHE, parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
    return vix / 100.0


def load_panel() -> pd.DataFrame:
    """Daily SPX close, VIX, trailing RV, and forward RV over the hold window."""
    try:
        from arch.data import sp500
    except ImportError as exc:
        raise SystemExit("This backtest needs the `arch` package: pip install arch") from exc

    spx = sp500.load()["Adj Close"]
    spx.index = pd.to_datetime(spx.index)
    df = pd.DataFrame({"close": spx}).join(load_vix().rename("iv"), how="inner").dropna()

    log_ret = np.log(df["close"]).diff()
    df["rv20"] = log_ret.rolling(RV_WINDOW).std() * ANNUALIZE
    # Realized vol over the next HOLD_DAYS returns (t+1 .. t+HOLD_DAYS).
    df["fwd_rv"] = log_ret[::-1].rolling(HOLD_DAYS).std()[::-1].shift(-1) * ANNUALIZE
    df["short_var_pnl"] = (df["iv"] ** 2 - df["fwd_rv"] ** 2) / (2 * df["iv"])
    # Match the other rules' start date so every row is comparable.
    return df.dropna().loc["2002":]


def rule_signals(df: pd.DataFrame) -> dict[str, tuple[pd.Series, pd.Series]]:
    """name -> (sell_mask, buy_mask). Parameters are fixed up front, not tuned."""
    spread = df["iv"] - df["rv20"]
    ratio = df["iv"] / df["rv20"]
    t = DEFAULT_SPREAD_THRESHOLD
    k = DEFAULT_RICH_RATIO
    no_buy = pd.Series(False, index=df.index)
    return {
        "Live rule: IV-RV > +/-0.02": (spread > t, spread < -t),
        "Live rule, credit side only": (spread > t, no_buy),
        "Live rule, debit side only": (spread > np.inf, spread < -t),
        f"NEW ratio IV/RV > {k}, credit only": (ratio > k, no_buy),
        "Always sell (no signal)": (spread > NEVER, no_buy),
    }


def evaluate(df: pd.DataFrame, sell: pd.Series, buy: pd.Series) -> dict[str, float]:
    rows = []
    for offset in range(HOLD_DAYS):
        idx = df.index[offset::HOLD_DAYS]
        s, b, pnl = sell.loc[idx], buy.loc[idx], df["short_var_pnl"].loc[idx]
        trade_pnl = pnl.where(s, 0.0) - pnl.where(b, 0.0)
        equity = trade_pnl.cumsum()
        sharpe = trade_pnl.mean() / trade_pnl.std() * np.sqrt(12) if trade_pnl.std() > 0 else np.nan
        rows.append(
            {
                "credits": s.sum(),
                "debits": b.sum(),
                "credit_win": (pnl[s] > 0).mean() if s.any() else np.nan,
                "debit_win": (pnl[b] < 0).mean() if b.any() else np.nan,
                "debit_avg": -pnl[b].mean() if b.any() else np.nan,
                "total": trade_pnl.sum(),
                "worst": trade_pnl.min(),
                "max_dd": (equity - equity.cummax()).min(),
                "sharpe": sharpe,
            }
        )
    return pd.DataFrame(rows).mean(skipna=True).to_dict()


def _fmt(value: float, spec: str) -> str:
    width = int("".join(ch for ch in spec.split(".")[0] if ch.isdigit()) or 0)
    return "n/a".rjust(width) if pd.isna(value) else format(value, spec)


def print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title}: {df.index[0].date()} to {df.index[-1].date()} ===")
    print(
        f"{'rule':38s} {'credits':>7s} {'win':>5s} {'debits':>6s} {'win':>5s} "
        f"{'debit avg':>9s} {'total':>7s} {'worst':>7s} {'max DD':>7s} {'Sharpe':>6s}"
    )
    for name, (sell, buy) in rule_signals(df).items():
        r = evaluate(df, sell, buy)
        print(
            f"{name:38s} {r['credits']:7.0f} {_fmt(r['credit_win'], '5.0%')} "
            f"{r['debits']:6.0f} {_fmt(r['debit_win'], '5.0%')} {_fmt(r['debit_avg'], '+9.3f')} "
            f"{r['total']:+7.2f} {r['worst']:+7.3f} {r['max_dd']:+7.3f} {_fmt(r['sharpe'], '6.2f')}"
        )


def main() -> None:
    df = load_panel()
    print("IV/RV backtest on SPX + VIX. P&L = 21-day short variance swap, vol points.")
    print("One decision per 21 sessions, averaged over all 21 start offsets.")
    print_table("Full sample", df)
    print_table("First half", df[df.index.year <= SPLIT_YEAR])
    print_table("Second half", df[df.index.year > SPLIT_YEAR])
    print(
        "\nColumns: trade counts and win rates per 21-day hold; 'debit avg' is the mean\n"
        "P&L of a debit trade; total / worst / max DD in vol points; Sharpe annualized.\n"
        "VIX runs ~1-3 pts above ATM IV, which flatters credits and penalizes debits a\n"
        "little. With ~70 trades per half, Sharpe gaps under ~0.3 are within noise."
    )


if __name__ == "__main__":
    main()
