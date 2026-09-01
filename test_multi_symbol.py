"""Dry-run stress test across several underlyings. Never places a live order."""

from __future__ import annotations

import os
import sys

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

from data import get_option_chain, get_price_history
from vol_signal import compute_iv_rv_spread, compute_realized_volatility
from decision import allowed_actions, decide_trade
from execute import count_open_option_positions, place_spread_order
from risk import validate_and_size_trade

load_dotenv()

SYMBOLS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "IWM"]
LOOKBACK_DAYS = 30
RV_WINDOW = 20
DTE_RANGE = (20, 45)
MAX_RISK_PCT = 0.02
MAX_OPEN_POSITIONS = 3
DRY_RUN = True


def _account_equity() -> float:
    api_key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret:
        raise ValueError("Missing Alpaca API credentials in .env.")
    return float(TradingClient(api_key, secret, paper=True).get_account().equity)


def _format_line(
    symbol: str,
    rv: float | None,
    avg_iv: float | None,
    spread: float | None,
    decision_action: str | None,
    risk_label: str,
    contracts: int | None,
    permitted: list[str] | None = None,
    error: str | None = None,
) -> str:
    name = f"{symbol:<5}"
    if error:
        return f"{name} | ERROR: {error}"
    spread_s = f"{spread:+.3f}" if spread is not None else "n/a"
    rv_s = f"{rv:.3f}" if rv is not None else "n/a"
    iv_s = f"{avg_iv:.3f}" if avg_iv is not None else "n/a"
    contracts_s = str(contracts) if contracts is not None else "n/a"
    allowed_s = ",".join(permitted) if permitted else "n/a"
    return (
        f"{name} | RV: {rv_s} | IV: {iv_s} | Spread: {spread_s} | "
        f"Allowed: {allowed_s} | Decision: {decision_action} | "
        f"Risk: {risk_label} | Contracts: {contracts_s}"
    )


def run_symbol(symbol: str, equity: float, open_positions: int) -> dict:
    """Run data -> signal -> decision -> risk -> dry-run execute for one symbol."""
    print(f"\n--- {symbol} ---")
    price_df = get_price_history(symbol, lookback_days=LOOKBACK_DAYS)
    option_chain_df = get_option_chain(symbol)
    print(f"  bars={len(price_df)}  chain={len(option_chain_df)}")

    realized_vol = compute_realized_volatility(price_df, window=RV_WINDOW)
    spread, avg_iv, rv = compute_iv_rv_spread(
        option_chain_df, realized_vol, target_dte_range=DTE_RANGE
    )
    permitted = allowed_actions(spread)
    print(f"  RV={rv:.4f}  IV={avg_iv:.4f}  spread={spread:+.4f}")
    print(f"  allowed_actions={permitted}")

    decision = decide_trade(
        iv_rv_spread=spread,
        avg_iv=avg_iv,
        realized_vol=rv,
        option_chain_df=option_chain_df,
        account_equity=equity,
        symbol=symbol,
    )
    print(
        f"  LLM action={decision.get('action')}  type={decision.get('spread_type')}  "
        f"override={decision.get('direction_override')}"
    )

    validated = validate_and_size_trade(
        decision=decision,
        option_chain_df=option_chain_df,
        account_equity=equity,
        max_risk_pct=MAX_RISK_PCT,
        max_open_positions=MAX_OPEN_POSITIONS,
        current_open_positions=open_positions,
    )
    print(
        f"  risk_passed={validated.get('risk_check_passed')}  "
        f"contracts={validated.get('contracts')}  "
        f"notes={validated.get('risk_notes')}"
    )

    result = place_spread_order(validated, symbol=symbol, dry_run=DRY_RUN)
    print(f"  execute={result}")

    llm_action = decision.get("action")
    if llm_action == "no_trade":
        risk_label = "N/A"
        contracts = None
    elif validated.get("risk_check_passed"):
        risk_label = "PASS"
        contracts = validated.get("contracts")
    else:
        risk_label = "FAIL"
        contracts = None

    return {
        "symbol": symbol,
        "ok": True,
        "error": None,
        "rv": rv,
        "avg_iv": avg_iv,
        "spread": spread,
        "decision_action": llm_action,
        "risk_passed": bool(validated.get("risk_check_passed")),
        "risk_label": risk_label,
        "contracts": contracts,
        "line": _format_line(
            symbol,
            rv,
            avg_iv,
            spread,
            llm_action,
            risk_label,
            contracts,
            permitted=permitted,
        ),
    }


def main() -> None:
    print("Multi-symbol dry-run stress test. No orders will be placed.")
    print(f"Symbols: {', '.join(SYMBOLS)}")

    try:
        equity = _account_equity()
        open_positions = count_open_option_positions()
    except Exception as exc:
        print(f"Could not load account state: {exc}")
        sys.exit(1)

    print(f"Equity: ${equity:,.2f}  Open option positions: {open_positions}")

    rows: list[dict] = []
    for symbol in SYMBOLS:
        try:
            rows.append(run_symbol(symbol, equity, open_positions))
        except Exception as exc:
            error = str(exc).replace("\n", " ")
            print(f"  FAILED {symbol}: {error}")
            rows.append(
                {
                    "symbol": symbol,
                    "ok": False,
                    "error": error,
                    "rv": None,
                    "avg_iv": None,
                    "spread": None,
                    "decision_action": None,
                    "risk_passed": False,
                    "risk_label": "ERROR",
                    "contracts": None,
                    "line": _format_line(
                        symbol, None, None, None, None, "ERROR", None, error=error
                    ),
                }
            )

    print("\n========== PER-SYMBOL SUMMARY ==========")
    for row in rows:
        print(row["line"])

    succeeded = sum(1 for r in rows if r["ok"])
    errored = sum(1 for r in rows if not r["ok"])
    no_trade = sum(1 for r in rows if r["ok"] and r["decision_action"] == "no_trade")
    trade_decision = sum(
        1
        for r in rows
        if r["ok"] and r["decision_action"] in {"credit_spread", "debit_spread"}
    )
    risk_pass = sum(1 for r in rows if r["ok"] and r["risk_label"] == "PASS")
    risk_fail = sum(1 for r in rows if r["ok"] and r["risk_label"] == "FAIL")
    risk_na = sum(1 for r in rows if r["ok"] and r["risk_label"] == "N/A")

    print("\n========== TOTALS ==========")
    print(f"Ran:                {len(rows)}")
    print(f"Succeeded end-to-end: {succeeded}")
    print(f"Errored:            {errored}")
    print(f"LLM no_trade:       {no_trade}")
    print(f"LLM trade decision: {trade_decision}")
    print(f"Risk PASS:          {risk_pass}")
    print(f"Risk FAIL:          {risk_fail}")
    print(f"Risk N/A (no_trade): {risk_na}")
    print("DRY_RUN is True. No live paper order was submitted.")
    print(f"Trade log: {os.path.join(os.path.dirname(__file__), 'trade_log.csv')}")


if __name__ == "__main__":
    main()
