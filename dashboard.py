"""Local Streamlit dashboard for trade_log.csv (dry-run and live paper)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

TRADE_LOG_PATH = Path(__file__).resolve().parent / "trade_log.csv"
TABLE_COLUMNS = [
    "timestamp",
    "symbol",
    "action",
    "spread_type",
    "strikes",
    "expiration",
    "contracts",
    "max_loss_dollars",
    "confidence",
    "risk_check_passed",
    "order_placed",
]
PNL_COLUMNS = ("realized_pnl", "fill_price", "close_price", "equity")


def load_trades(path: Path = TRADE_LOG_PATH) -> pd.DataFrame:
    """Load the trade log, or an empty frame if it is missing or blank."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame()
    if df.empty:
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", ascending=False, na_position="last")

    if "order_placed" in df.columns:
        df["order_placed"] = df["order_placed"].map(_as_bool)

    if "risk_check_passed" in df.columns:
        df["risk_check_passed"] = df["risk_check_passed"].map(_as_bool)

    if "short_strike" in df.columns and "long_strike" in df.columns:
        df["strikes"] = df.apply(
            lambda row: (
                f"{row['short_strike']} / {row['long_strike']}"
                if pd.notna(row.get("short_strike")) and pd.notna(row.get("long_strike"))
                else ""
            ),
            axis=1,
        )
    else:
        df["strikes"] = ""

    return df.reset_index(drop=True)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _is_present_order_id(value: object) -> bool:
    text = str(value).strip()
    return text not in {"", "nan", "None", "none"}


def detect_mode(df: pd.DataFrame) -> str:
    """Return dry_run, live, mixed, or empty from order_id values."""
    if df.empty or "order_id" not in df.columns:
        return "empty"
    ids = df["order_id"]
    is_dry = ids.astype(str).str.strip().eq("DRY_RUN")
    is_live = ids.map(_is_present_order_id) & ~is_dry
    if bool(is_live.any()) and bool(is_dry.any()):
        return "mixed"
    if bool(is_live.any()):
        return "live"
    if bool(is_dry.any()):
        return "dry_run"
    return "empty"


def _has_realized_pnl(df: pd.DataFrame) -> bool:
    """True when the log has real fill/close/P&L fields populated."""
    present = [col for col in PNL_COLUMNS if col in df.columns]
    if not present or df.empty:
        return False
    return bool(df[present].notna().any().any())


def _trade_label(row: pd.Series) -> str:
    ts = row.get("timestamp")
    ts_s = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else "unknown time"
    return f"{ts_s}  |  {row.get('symbol', '?')}  |  {row.get('action', '?')}"


def render_banner(mode: str) -> None:
    if mode == "dry_run":
        st.warning(
            "DEMO DATA - DRY RUN, NO REAL TRADES PLACED. "
            "order_id is DRY_RUN for every row."
        )
    elif mode == "live":
        st.success("LIVE PAPER TRADES — rows use real Alpaca order IDs.")
    elif mode == "mixed":
        st.warning(
            "MIXED LOG — some DRY_RUN rows and some live paper order IDs."
        )
    else:
        st.info("No trade log yet. Run the agent (dry-run or live) to populate it.")


def render_metrics(df: pd.DataFrame) -> None:
    st.subheader("Summary")
    if df.empty:
        st.caption("Metrics will appear after the first logged run.")
        return

    actions = df["action"] if "action" in df.columns else pd.Series(dtype=str)
    if "risk_check_passed" not in df.columns:
        st.error(
            "trade_log.csv is missing risk_check_passed. "
            "Run `python migrate_trade_log.py` once."
        )
        return
    passed = df["risk_check_passed"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Trades attempted", int(len(df)))
    c2.metric("Risk passed", int(passed.sum()))
    c3.metric("Risk rejected", int((~passed).sum()))
    c4.metric("credit_spread", int((actions == "credit_spread").sum()))
    c5.metric("debit_spread", int((actions == "debit_spread").sum()))
    c6.metric("no_trade", int((actions == "no_trade").sum()))


def render_table(df: pd.DataFrame) -> None:
    st.subheader("All logged trades")
    if df.empty:
        return
    cols = [col for col in TABLE_COLUMNS if col in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)


def render_symbol_chart(df: pd.DataFrame) -> None:
    st.subheader("Decisions by symbol")
    if df.empty or "symbol" not in df.columns:
        return
    counts = df["symbol"].value_counts().rename_axis("symbol").to_frame("trades")
    st.bar_chart(counts, use_container_width=True)


def render_equity_curve(df: pd.DataFrame) -> None:
    st.subheader("Equity Curve (P&L)")
    if df.empty or not _has_realized_pnl(df):
        st.info(
            "No filled trades yet — this will populate once live paper trades "
            "are placed and closed."
        )
        return

    # Future path: plot cumulative realized P&L once fill/close columns exist.
    working = df.dropna(subset=[col for col in PNL_COLUMNS if col in df.columns]).copy()
    working = working.sort_values("timestamp")
    if "realized_pnl" in working.columns:
        working["equity_curve"] = working["realized_pnl"].astype(float).cumsum()
        st.line_chart(working.set_index("timestamp")["equity_curve"], use_container_width=True)
    elif "equity" in working.columns:
        st.line_chart(working.set_index("timestamp")["equity"], use_container_width=True)
    else:
        st.info(
            "No filled trades yet — this will populate once live paper trades "
            "are placed and closed."
        )


def render_rationale(df: pd.DataFrame) -> None:
    st.subheader("Rationale viewer")
    if df.empty or "rationale" not in df.columns:
        st.caption("Select a trade here after the log has rows.")
        return

    labels = [_trade_label(row) for _, row in df.iterrows()]
    selected = st.selectbox("Select a trade", options=list(range(len(df))), format_func=lambda i: labels[i])
    row = df.iloc[int(selected)]
    st.markdown(
        f"**{row.get('symbol', '')}** · {row.get('action', '')} · "
        f"{row.get('spread_type', '')} · confidence {row.get('confidence', '')}"
    )
    st.write(row.get("rationale") or "(no rationale logged)")


def main() -> None:
    st.set_page_config(page_title="ALPACA Trade Dashboard", layout="wide")
    st.title("ALPACA Trade Dashboard")
    st.caption("Local view of trade_log.csv — dry-run today, live paper later.")

    df = load_trades()
    render_banner(detect_mode(df))

    if df.empty:
        st.info("No trades yet. Run `python main.py` or `python test_multi_symbol.py` first.")
        return

    render_metrics(df)
    st.divider()
    render_table(df)
    st.divider()
    left, right = st.columns(2)
    with left:
        render_symbol_chart(df)
    with right:
        render_equity_curve(df)
    st.divider()
    render_rationale(df)


if __name__ == "__main__":
    main()
