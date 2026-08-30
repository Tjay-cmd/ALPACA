"""Single-run agent loop: data -> signal -> decision -> risk -> execute."""

from __future__ import annotations

import json
import os
import sys

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

from data import get_option_chain, get_price_history
from signal import compute_iv_rv_spread, compute_realized_volatility
from decision import allowed_actions, decide_trade
from execute import count_open_option_positions, place_spread_order
from risk import validate_and_size_trade

load_dotenv()

# Flip to False only after you have reviewed a dry run and want paper orders.
DRY_RUN = False

SYMBOL = "SPY"
LOOKBACK_DAYS = 30
RV_WINDOW = 20
DTE_RANGE = (20, 45)
MAX_RISK_PCT = 0.02
MAX_OPEN_POSITIONS = 3


def _account_equity() -> float:
    """Paper-account equity from Alpaca."""
    api_key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret:
        raise ValueError("Missing Alpaca API credentials in .env.")
    account = TradingClient(api_key, secret, paper=True).get_account()
    return float(account.equity)


def _print_step(title: str, payload: object) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, dict):
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(payload)


def format_summary_line(
    symbol: str,
    rv: float | None = None,
    avg_iv: float | None = None,
    spread: float | None = None,
    decision_action: str | None = None,
    risk_label: str = "n/a",
    contracts: int | None = None,
    permitted: list[str] | None = None,
    error: str | None = None,
) -> str:
    """One-line result used by the scheduler and multi-symbol tests."""
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


def run_once(
    symbol: str = SYMBOL,
    current_open_positions: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run one full pass of the agent. Does not loop or schedule.

    Returns a summary dict for the scheduler. Respects DRY_RUN from this module.
    """
    mode = "DRY RUN (no orders)" if DRY_RUN else "LIVE PAPER ORDERS"
    if verbose:
        print(f"Agent starting for {symbol} - {mode}")

    if verbose:
        print("\n=== 1. Market data ===")
    price_df = get_price_history(symbol, lookback_days=LOOKBACK_DAYS)
    option_chain_df = get_option_chain(symbol)
    if verbose:
        print(f"Daily bars: {len(price_df)}")
        print(f"Option contracts fetched: {len(option_chain_df)}")

    if verbose:
        print("\n=== 2. Signal ===")
    realized_vol = compute_realized_volatility(price_df, window=RV_WINDOW)
    spread, avg_iv, rv = compute_iv_rv_spread(
        option_chain_df,
        realized_vol,
        target_dte_range=DTE_RANGE,
    )
    permitted = allowed_actions(spread)
    if verbose:
        _print_step(
            "2. Signal values",
            {
                "realized_vol": round(rv, 4),
                "avg_iv": round(avg_iv, 4),
                "iv_rv_spread": round(spread, 4),
                "allowed_actions": permitted,
            },
        )

    equity = _account_equity()
    if current_open_positions is None:
        current_open_positions = count_open_option_positions()
    if verbose:
        _print_step(
            "3. Account",
            {
                "equity": equity,
                "open_option_positions": current_open_positions,
                "max_open_positions": MAX_OPEN_POSITIONS,
                "max_risk_pct": MAX_RISK_PCT,
            },
        )

    if verbose:
        print("\n=== 4. LLM decision ===")
    decision = decide_trade(
        iv_rv_spread=spread,
        avg_iv=avg_iv,
        realized_vol=rv,
        option_chain_df=option_chain_df,
        account_equity=equity,
        symbol=symbol,
    )
    if verbose:
        _print_step("4. LLM decision", decision)

    if verbose:
        print("\n=== 5. Risk gate ===")
    validated = validate_and_size_trade(
        decision=decision,
        option_chain_df=option_chain_df,
        account_equity=equity,
        max_risk_pct=MAX_RISK_PCT,
        max_open_positions=MAX_OPEN_POSITIONS,
        current_open_positions=current_open_positions,
    )
    if verbose:
        _print_step("5. Validated trade", validated)

    if verbose:
        print("\n=== 6. Execution ===")
    result = place_spread_order(validated, symbol=symbol, dry_run=DRY_RUN)
    if verbose:
        _print_step("6. Execution result", result)
        print("\n=== Done ===")
        if DRY_RUN:
            print("DRY_RUN is True. No live paper order was submitted.")
        print(f"Trade log: {os.path.join(os.path.dirname(__file__), 'trade_log.csv')}")

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

    line = format_summary_line(
        symbol,
        rv,
        avg_iv,
        spread,
        llm_action,
        risk_label,
        contracts,
        permitted=permitted,
    )
    return {
        "symbol": symbol,
        "ok": True,
        "error": None,
        "rv": rv,
        "avg_iv": avg_iv,
        "spread": spread,
        "decision_action": llm_action,
        "risk_label": risk_label,
        "contracts": contracts,
        "allowed_actions": permitted,
        "line": line,
        "execution": result,
    }


def main() -> None:
    try:
        run_once(SYMBOL)
    except Exception as exc:
        print(
            "Agent run failed. The market may be closed or a downstream "
            f"service may be unavailable.\nDetails: {exc}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
