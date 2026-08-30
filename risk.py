"""Risk checks and position limits. Recalculates size from live quotes."""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

CREDIT_SPREADS = {"bull_put", "bear_call"}
DEBIT_SPREADS = {"bull_call", "bear_put"}
CALL_SPREADS = {"bear_call", "bull_call"}
PUT_SPREADS = {"bull_put", "bear_put"}
STRIKE_TOLERANCE = 0.011


def _copy_decision(decision: dict) -> dict[str, Any]:
    """Shallow-copy the LLM decision fields we preserve."""
    return {
        "action": decision.get("action"),
        "spread_type": decision.get("spread_type"),
        "short_strike": decision.get("short_strike"),
        "long_strike": decision.get("long_strike"),
        "expiration": decision.get("expiration"),
        "contracts": decision.get("contracts"),
        "max_loss_dollars": decision.get("max_loss_dollars"),
        "confidence": decision.get("confidence"),
        "rationale": decision.get("rationale"),
        "risk_check_passed": False,
        "risk_notes": "",
    }


def _reject(decision: dict, reason: str) -> dict[str, Any]:
    """Return a no_trade result with the rejection reason."""
    out = _copy_decision(decision)
    out["action"] = "no_trade"
    out["contracts"] = None
    out["max_loss_dollars"] = None
    out["risk_check_passed"] = False
    out["risk_notes"] = reason
    return out


def _parse_expiration(value: Any) -> date | None:
    if value is None:
        return None
    try:
        return pd.to_datetime(value).date()
    except (TypeError, ValueError):
        return None


def _lookup_leg(
    option_chain_df: pd.DataFrame,
    strike: float,
    expiration: date,
    option_type: str,
) -> pd.Series | None:
    """Find one contract matching strike, expiration, and call/put."""
    if option_chain_df is None or option_chain_df.empty:
        return None
    df = option_chain_df.copy()
    df["_exp"] = pd.to_datetime(df["expiration"]).dt.date
    matches = df.loc[
        (df["type"] == option_type)
        & (df["_exp"] == expiration)
        & ((df["strike"].astype(float) - float(strike)).abs() < STRIKE_TOLERANCE)
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def _mid(bid: float, ask: float) -> float:
    return (bid + ask) / 2.0


def validate_and_size_trade(
    decision: dict,
    option_chain_df: pd.DataFrame,
    account_equity: float,
    max_risk_pct: float = 0.02,
    max_open_positions: int = 3,
    current_open_positions: int = 0,
) -> dict[str, Any]:
    """Re-verify an LLM trade using live bid/ask and hard risk caps.

    Ignores decision["max_loss_dollars"] and decision["contracts"]. Recalculates
    both from the option chain. Clamps oversized contract counts down instead
    of rejecting. Returns the decision schema plus risk_check_passed / risk_notes.
    """
    if not decision or decision.get("action") == "no_trade":
        return _reject(decision or {}, "LLM chose no_trade; nothing to validate.")

    if current_open_positions >= max_open_positions:
        return _reject(
            decision,
            f"Open positions ({current_open_positions}) already at the cap "
            f"of {max_open_positions}.",
        )

    action = decision.get("action")
    spread_type = decision.get("spread_type")
    if action not in {"credit_spread", "debit_spread"}:
        return _reject(decision, f"Unsupported action {action!r}.")
    if spread_type not in CREDIT_SPREADS | DEBIT_SPREADS:
        return _reject(decision, f"Unsupported spread_type {spread_type!r}.")
    if action == "credit_spread" and spread_type not in CREDIT_SPREADS:
        return _reject(
            decision,
            f"{spread_type} is not a credit spread; refusing mismatched action.",
        )
    if action == "debit_spread" and spread_type not in DEBIT_SPREADS:
        return _reject(
            decision,
            f"{spread_type} is not a debit spread; refusing mismatched action.",
        )

    expiration = _parse_expiration(decision.get("expiration"))
    if expiration is None:
        return _reject(decision, "Missing or unreadable expiration.")
    if expiration <= date.today():
        return _reject(
            decision,
            f"Expiration {expiration.isoformat()} is today or in the past.",
        )

    try:
        short_strike = float(decision["short_strike"])
        long_strike = float(decision["long_strike"])
    except (TypeError, ValueError, KeyError):
        return _reject(decision, "short_strike and long_strike must be numbers.")

    width = abs(short_strike - long_strike)
    if width <= 0:
        return _reject(decision, "Strike width is zero; not a valid vertical.")

    option_type = "call" if spread_type in CALL_SPREADS else "put"
    short_leg = _lookup_leg(option_chain_df, short_strike, expiration, option_type)
    long_leg = _lookup_leg(option_chain_df, long_strike, expiration, option_type)
    if short_leg is None or long_leg is None:
        missing = []
        if short_leg is None:
            missing.append(f"short {option_type} {short_strike} {expiration}")
        if long_leg is None:
            missing.append(f"long {option_type} {long_strike} {expiration}")
        return _reject(
            decision,
            "Strikes not found in the option chain: " + "; ".join(missing) + ".",
        )

    try:
        short_bid = float(short_leg["bid"])
        short_ask = float(short_leg["ask"])
        long_bid = float(long_leg["bid"])
        long_ask = float(long_leg["ask"])
    except (TypeError, ValueError):
        return _reject(decision, "Bid/ask quotes are missing on one or both legs.")

    if min(short_bid, short_ask, long_bid, long_ask) <= 0:
        return _reject(decision, "One or both legs have non-positive bid/ask.")

    # Worst-case fills: sell the short at bid, buy the long at ask.
    conservative_credit = short_bid - long_ask
    conservative_debit = long_ask - short_bid
    mid_net = _mid(short_bid, short_ask) - _mid(long_bid, long_ask)

    if action == "credit_spread":
        net_premium = conservative_credit
        if net_premium <= 0:
            return _reject(
                decision,
                f"Conservative net credit is {net_premium:.2f}; quotes do not "
                "support a credit fill.",
            )
        max_loss_per = (width - net_premium) * 100.0
        # Alpaca mleg: negative limit_price = credit received.
        limit_price = -round(abs(mid_net), 2)
        if limit_price >= 0:
            limit_price = -round(net_premium, 2)
    else:
        net_premium = conservative_debit
        if net_premium <= 0:
            return _reject(
                decision,
                f"Conservative net debit is {net_premium:.2f}; quotes do not "
                "support a debit fill.",
            )
        max_loss_per = net_premium * 100.0
        # Alpaca mleg: positive limit_price = debit paid.
        limit_price = round(abs(mid_net), 2)
        if limit_price <= 0:
            limit_price = round(net_premium, 2)

    if max_loss_per <= 0:
        return _reject(
            decision,
            f"Recalculated max loss per contract is ${max_loss_per:.2f}, which is not usable.",
        )

    max_risk_dollars = float(account_equity) * float(max_risk_pct)
    max_contracts = math.floor(max_risk_dollars / max_loss_per)
    if max_contracts < 1:
        return _reject(
            decision,
            f"Even 1 contract (${max_loss_per:.2f} max loss) exceeds "
            f"{max_risk_pct:.0%} of equity (${max_risk_dollars:.2f}).",
        )

    notes: list[str] = []
    proposed = decision.get("contracts")
    try:
        proposed_contracts = int(proposed) if proposed is not None else max_contracts
    except (TypeError, ValueError):
        proposed_contracts = max_contracts
        notes.append("LLM contracts were unreadable; sized from the risk cap.")

    if proposed_contracts < 1:
        proposed_contracts = 1

    contracts = proposed_contracts
    if contracts > max_contracts:
        warning = (
            f"WARNING: LLM proposed {proposed_contracts} contracts; "
            f"clamped to {max_contracts} so max loss stays within "
            f"{max_risk_pct:.0%} of equity (${max_risk_dollars:.2f}). "
            f"Max loss/contract = ${max_loss_per:.2f}."
        )
        logger.warning(warning)
        print(warning)
        notes.append(warning)
        contracts = max_contracts
    else:
        notes.append(
            f"Sized {contracts} contract(s); cap was {max_contracts}. "
            f"Conservative max loss/contract = ${max_loss_per:.2f}."
        )

    total_max_loss = max_loss_per * contracts
    out = _copy_decision(decision)
    out["action"] = action
    out["spread_type"] = spread_type
    out["short_strike"] = short_strike
    out["long_strike"] = long_strike
    out["expiration"] = expiration.isoformat()
    out["contracts"] = int(contracts)
    out["max_loss_dollars"] = round(total_max_loss, 2)
    out["risk_check_passed"] = True
    out["risk_notes"] = " ".join(notes)
    out["short_symbol"] = str(short_leg["symbol"])
    out["long_symbol"] = str(long_leg["symbol"])
    out["limit_price"] = float(limit_price)
    out["max_loss_per_contract"] = round(max_loss_per, 2)
    out["net_premium"] = round(net_premium, 4)
    return out
