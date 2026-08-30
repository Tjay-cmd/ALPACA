"""Order placement and trade logging. Paper account only."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderClass, OrderSide, PositionIntent, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
from dotenv import load_dotenv

load_dotenv()

TRADE_LOG_PATH = Path(__file__).resolve().parent / "trade_log.csv"
LOG_COLUMNS = [
    "timestamp",
    "symbol",
    "action",
    "spread_type",
    "short_strike",
    "long_strike",
    "expiration",
    "contracts",
    "max_loss_dollars",
    "confidence",
    "risk_check_passed",
    "risk_notes",
    "order_placed",
    "order_id",
    "rationale",
]


def _trading_client() -> TradingClient:
    """Paper TradingClient from .env credentials."""
    api_key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret:
        raise ValueError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY in .env.")
    return TradingClient(api_key, secret, paper=True)


def count_open_option_positions() -> int:
    """Count currently open US option positions on the paper account."""
    positions = _trading_client().get_all_positions()
    count = 0
    for pos in positions:
        asset_class = pos.asset_class
        value = asset_class.value if hasattr(asset_class, "value") else str(asset_class)
        if value == AssetClass.US_OPTION.value:
            count += 1
    return count


def _infer_risk_passed(action: object, contracts: object) -> bool:
    """Same rule the dashboard used: sized credit/debit = pass, else fail."""
    try:
        sized = contracts is not None and str(contracts).strip() not in {"", "nan", "None"}
        sized = sized and float(contracts) > 0
    except (TypeError, ValueError):
        sized = False
    return str(action) in {"credit_spread", "debit_spread"} and sized


def migrate_trade_log(path: Path = TRADE_LOG_PATH) -> int:
    """Backfill risk_check_passed / risk_notes on an existing log. Idempotent."""
    if not path.exists() or path.stat().st_size == 0:
        print("No trade_log.csv to migrate.")
        return 0

    df = pd.read_csv(path)
    if df.empty:
        for col in LOG_COLUMNS:
            if col not in df.columns:
                df[col] = pd.Series(dtype=object)
        df.to_csv(path, index=False)
        print("trade_log.csv was empty. Wrote the updated header only.")
        return 0

    changed = False
    migrated = 0
    if "risk_check_passed" not in df.columns:
        df["risk_check_passed"] = [
            _infer_risk_passed(row.get("action"), row.get("contracts"))
            for _, row in df.iterrows()
        ]
        migrated = len(df)
        changed = True
    else:
        blank = df["risk_check_passed"].isna() | (
            df["risk_check_passed"].astype(str).str.strip().isin({"", "nan", "None"})
        )
        if blank.any():
            df.loc[blank, "risk_check_passed"] = [
                _infer_risk_passed(row.get("action"), row.get("contracts"))
                for _, row in df.loc[blank].iterrows()
            ]
            migrated = int(blank.sum())
            changed = True

    if "risk_notes" not in df.columns:
        df["risk_notes"] = ""
        if migrated == 0:
            migrated = len(df)
        changed = True

    if changed:
        ordered = [col for col in LOG_COLUMNS if col in df.columns]
        extras = [col for col in df.columns if col not in ordered]
        df[ordered + extras].to_csv(path, index=False)
        print(
            f"Migrated {migrated} trade_log.csv row(s): added/backfilled "
            "risk_check_passed and risk_notes."
        )
    return migrated


def _append_trade_log(row: dict[str, Any]) -> None:
    """Append one row to trade_log.csv, writing the header if needed."""
    migrate_trade_log()
    new_file = not TRADE_LOG_PATH.exists()
    with TRADE_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in LOG_COLUMNS})


def _log_attempt(
    validated_trade: dict,
    symbol: str,
    order_placed: bool,
    order_id: str | None,
) -> None:
    _append_trade_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "action": validated_trade.get("action"),
            "spread_type": validated_trade.get("spread_type"),
            "short_strike": validated_trade.get("short_strike"),
            "long_strike": validated_trade.get("long_strike"),
            "expiration": validated_trade.get("expiration"),
            "contracts": validated_trade.get("contracts"),
            "max_loss_dollars": validated_trade.get("max_loss_dollars"),
            "confidence": validated_trade.get("confidence"),
            "risk_check_passed": bool(validated_trade.get("risk_check_passed")),
            "risk_notes": validated_trade.get("risk_notes") or "",
            "order_placed": order_placed,
            "order_id": order_id or "",
            "rationale": validated_trade.get("rationale")
            or validated_trade.get("risk_notes")
            or "",
        }
    )


def place_spread_order(
    validated_trade: dict,
    symbol: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit a two-leg vertical as an Alpaca mleg limit order.

    Does not place anything when risk_check_passed is False, action is
    no_trade, or dry_run is True. Always appends a trade_log.csv row.
    """
    if (
        not validated_trade.get("risk_check_passed")
        or validated_trade.get("action") == "no_trade"
    ):
        reason = validated_trade.get("risk_notes") or "risk check failed or no_trade"
        print(f"No order placed: {reason}")
        _log_attempt(validated_trade, symbol, order_placed=False, order_id=None)
        return {"order_placed": False, "order_id": None, "error": None}

    short_symbol = validated_trade.get("short_symbol")
    long_symbol = validated_trade.get("long_symbol")
    contracts = validated_trade.get("contracts")
    limit_price = validated_trade.get("limit_price")
    if not short_symbol or not long_symbol or not contracts or limit_price is None:
        error = "Validated trade is missing OCC symbols, contracts, or limit_price."
        print(f"No order placed: {error}")
        _log_attempt(validated_trade, symbol, order_placed=False, order_id=None)
        return {"order_placed": False, "order_id": None, "error": error}

    summary = (
        f"{validated_trade.get('spread_type')} {symbol} "
        f"short {validated_trade.get('short_strike')} / "
        f"long {validated_trade.get('long_strike')} "
        f"{validated_trade.get('expiration')} x{contracts} "
        f"limit={limit_price} max_loss=${validated_trade.get('max_loss_dollars')}"
    )

    print(
        f"ABOUT TO SUBMIT: {symbol} {validated_trade.get('spread_type')} "
        f"short {validated_trade.get('short_strike')} / "
        f"long {validated_trade.get('long_strike')} "
        f"x{contracts} max_loss=${validated_trade.get('max_loss_dollars')}"
    )

    if dry_run:
        print(f"DRY RUN: would have placed {summary}")
        _log_attempt(validated_trade, symbol, order_placed=False, order_id="DRY_RUN")
        return {"order_placed": False, "order_id": None, "error": "dry_run"}

    request = LimitOrderRequest(
        qty=int(contracts),
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.MLEG,
        limit_price=float(limit_price),
        client_order_id=f"alpaca-{symbol}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        legs=[
            OptionLegRequest(
                symbol=str(short_symbol),
                ratio_qty=1,
                side=OrderSide.SELL,
                position_intent=PositionIntent.SELL_TO_OPEN,
            ),
            OptionLegRequest(
                symbol=str(long_symbol),
                ratio_qty=1,
                side=OrderSide.BUY,
                position_intent=PositionIntent.BUY_TO_OPEN,
            ),
        ],
    )

    try:
        order = _trading_client().submit_order(request)
    except Exception as exc:
        error = f"Alpaca submit_order failed: {exc}"
        print(error)
        _log_attempt(validated_trade, symbol, order_placed=False, order_id=None)
        return {"order_placed": False, "order_id": None, "error": error}

    order_id = str(getattr(order, "id", "") or "")
    print(f"Order placed: {summary} id={order_id}")
    _log_attempt(validated_trade, symbol, order_placed=True, order_id=order_id)
    return {"order_placed": True, "order_id": order_id, "error": None}
