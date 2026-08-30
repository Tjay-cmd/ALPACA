"""LLM decision layer: turn an IV/RV signal into a defined-risk options spread."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


def _import_openai_client():
    """Import OpenAI without our local signal.py shadowing the stdlib module."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path = [
        p for p in sys.path if os.path.abspath(p or os.getcwd()) != os.path.abspath(here)
    ]
    cached_signal = sys.modules.pop("signal", None)
    try:
        from openai import OpenAI as _OpenAI
        return _OpenAI
    finally:
        if here not in sys.path:
            sys.path.insert(0, here)
        if cached_signal is not None:
            sys.modules["signal"] = cached_signal


OpenAI = _import_openai_client()

load_dotenv(Path(__file__).resolve().parent / ".env")

# General-purpose instruct models (not coding specialists). Qwen2.5-7B is
# Featherless's documented default and is reliable for JSON; others are fallbacks
# if a model is at capacity.
FEATHERLESS_MODELS = (
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "NousResearch/Hermes-3-Llama-3.1-8B",
)
FEATHERLESS_MODEL = FEATHERLESS_MODELS[0]
FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"

VALID_ACTIONS = {"credit_spread", "debit_spread", "no_trade"}
VALID_SPREAD_TYPES = {"bull_put", "bear_call", "bull_call", "bear_put", None}
DEFAULT_SPREAD_THRESHOLD = 0.02


def allowed_actions(iv_rv_spread: float, threshold: float = DEFAULT_SPREAD_THRESHOLD) -> list[str]:
    """Hard direction constraint from the IV/RV spread.

    Positive spread (IV rich) may only credit or stand aside.
    Negative spread (IV cheap) may only debit or stand aside.
    A small spread is not worth trading either way.
    """
    if iv_rv_spread > threshold:
        return ["credit_spread", "no_trade"]
    if iv_rv_spread < -threshold:
        return ["debit_spread", "no_trade"]
    return ["no_trade"]

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _load_featherless_key() -> str:
    """Return the Featherless API key from the environment."""
    api_key = os.getenv("FEATHERLESS_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing FEATHERLESS_API_KEY. Add it to your .env file."
        )
    return api_key


def _get_client() -> OpenAI:
    """OpenAI-compatible client pointed at Featherless."""
    return OpenAI(base_url=FEATHERLESS_BASE_URL, api_key=_load_featherless_key())


def _estimate_spot(option_chain_df: pd.DataFrame) -> float:
    """Estimate the underlying price from the call closest to 0.50 delta."""
    if "delta" in option_chain_df.columns and "type" in option_chain_df.columns:
        calls = option_chain_df[
            (option_chain_df["type"] == "call") & option_chain_df["delta"].notna()
        ]
        if not calls.empty:
            idx = (calls["delta"].astype(float) - 0.50).abs().idxmin()
            return float(calls.loc[idx, "strike"])
    return float(option_chain_df["strike"].median())


def _summarize_near_money_chain(
    option_chain_df: pd.DataFrame,
    current_price: float,
    dte_range: tuple[int, int] = (20, 45),
    pct_band: float = 0.10,
    max_rows: int = 80,
) -> str:
    """Compact list of 20-45 DTE contracts within +/-10% of spot."""
    if option_chain_df is None or option_chain_df.empty:
        return "No option chain data available."

    today = date.today()
    df = option_chain_df.copy()
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    df["dte"] = df["expiration"].map(lambda exp: (exp - today).days)

    low = current_price * (1 - pct_band)
    high = current_price * (1 + pct_band)
    min_dte, max_dte = dte_range
    near = df.loc[
        (df["dte"] >= min_dte)
        & (df["dte"] <= max_dte)
        & (df["strike"] >= low)
        & (df["strike"] <= high)
    ].copy()

    if near.empty:
        return (
            f"No contracts in the {min_dte}-{max_dte} DTE window "
            f"within +/-10% of spot ({current_price:.2f})."
        )

    near["atm_dist"] = (near["strike"].astype(float) - current_price).abs()
    near = near.sort_values(["expiration", "atm_dist", "type", "strike"]).head(max_rows)

    lines = [
        f"Spot estimate: {current_price:.2f}",
        f"Showing up to {max_rows} contracts, {min_dte}-{max_dte} DTE, "
        f"strikes in [{low:.2f}, {high:.2f}]:",
    ]
    for _, row in near.iterrows():
        iv = row.get("implied_volatility")
        delta = row.get("delta")
        iv_s = f"{float(iv):.3f}" if pd.notna(iv) else "n/a"
        delta_s = f"{float(delta):.2f}" if pd.notna(delta) else "n/a"
        lines.append(
            f"  {row['expiration']}  {int(row['dte'])}DTE  "
            f"{str(row['type']).upper():4}  {float(row['strike']):.1f}  "
            f"IV={iv_s}  delta={delta_s}"
        )
    return "\n".join(lines)


def _build_prompt(
    symbol: str,
    iv_rv_spread: float,
    avg_iv: float,
    realized_vol: float,
    account_equity: float,
    chain_summary: str,
    permitted: list[str],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the decision call."""
    max_risk = account_equity * 0.02
    permitted_s = ", ".join(permitted)
    system_prompt = (
        "You are an options trading decision engine. Reply with a single JSON "
        "object only - no markdown, no extra text.\n"
        "Use defined-risk vertical spreads only: bull_put, bear_call, bull_call, "
        "or bear_put. Never propose naked or undefined-risk positions.\n"
        "Max risk per trade must not exceed 2% of account equity. Size the "
        "spread width and contract count so max loss stays within that cap, "
        "and state max_loss_dollars explicitly.\n"
        "no_trade is a valid, expected outcome when the edge is unclear, "
        "sizing would exceed the risk cap, or the chain is a poor fit.\n"
        f"HARD CONSTRAINT: action MUST be one of: [{permitted_s}]. "
        "Do not propose any other action.\n"
        "JSON schema:\n"
        "{\n"
        '  "action": "credit_spread" | "debit_spread" | "no_trade",\n'
        '  "spread_type": "bull_put" | "bear_call" | "bull_call" | "bear_put" | null,\n'
        '  "short_strike": float | null,\n'
        '  "long_strike": float | null,\n'
        '  "expiration": "YYYY-MM-DD" | null,\n'
        '  "contracts": int | null,\n'
        '  "max_loss_dollars": float | null,\n'
        '  "confidence": float,\n'
        '  "rationale": string\n'
        "}\n"
        "rationale must be 2-4 plain English sentences. confidence is 0 to 1."
    )
    user_prompt = (
        f"Symbol: {symbol}\n"
        f"Account equity: ${account_equity:,.2f}\n"
        f"Max risk (2% of equity): ${max_risk:,.2f}\n"
        f"IV/RV spread (avg IV - realized vol): {iv_rv_spread:.4f}\n"
        f"Average IV: {avg_iv:.4f}\n"
        f"Realized volatility: {realized_vol:.4f}\n"
        f"Allowed actions (you may ONLY choose from this list): {permitted_s}\n\n"
        f"Near-the-money option chain summary:\n{chain_summary}\n\n"
        "Pick strikes and an expiration from the summarized chain only if "
        "your action is a spread. Return only the JSON object."
    )
    return system_prompt, user_prompt


def _parse_decision_json(text: str) -> dict[str, Any]:
    """Parse the LLM reply into a dict, stripping markdown fences if present."""
    if not text or not text.strip():
        raise ValueError("LLM returned an empty response; cannot parse a decision.")

    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM returned malformed JSON. "
            f"Parser error: {exc}. Raw text: {text[:400]}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"LLM JSON must be an object, got {type(payload).__name__}.")

    action = payload.get("action")
    if action not in VALID_ACTIONS:
        raise ValueError(
            f"Invalid action {action!r}. Expected one of {sorted(VALID_ACTIONS)}."
        )

    spread_type = payload.get("spread_type")
    if spread_type not in VALID_SPREAD_TYPES:
        raise ValueError(
            f"Invalid spread_type {spread_type!r}. "
            f"Expected one of {sorted(t for t in VALID_SPREAD_TYPES if t)} or null."
        )

    if "confidence" not in payload or "rationale" not in payload:
        raise ValueError("LLM JSON is missing required fields: confidence and/or rationale.")

    return payload


def _enforce_direction_constraint(
    payload: dict[str, Any],
    permitted: list[str],
) -> dict[str, Any]:
    """Force no_trade if the LLM picked an action outside the allowed list."""
    payload["allowed_actions"] = list(permitted)
    chosen = payload.get("action")
    if chosen in permitted:
        payload["direction_override"] = False
        return payload

    original_rationale = payload.get("rationale") or ""
    payload["action"] = "no_trade"
    payload["spread_type"] = None
    payload["short_strike"] = None
    payload["long_strike"] = None
    payload["expiration"] = None
    payload["contracts"] = None
    payload["max_loss_dollars"] = None
    payload["direction_override"] = True
    payload["rationale"] = (
        f"Rejected invalid LLM action {chosen!r}; allowed actions were "
        f"{permitted}. Direction constraint overrode the decision to no_trade. "
        f"Original rationale: {original_rationale}"
    )
    return payload


def decide_trade(
    iv_rv_spread: float,
    avg_iv: float,
    realized_vol: float,
    option_chain_df: pd.DataFrame,
    account_equity: float,
    symbol: str,
) -> dict[str, Any]:
    """Ask the LLM which defined-risk spread (if any) to trade.

    Applies a hard IV/RV direction constraint before and after the LLM call.
    Returns the parsed JSON decision dict. `no_trade` is a valid outcome,
    not an error. Does not place any orders.
    """
    permitted = allowed_actions(iv_rv_spread)
    current_price = _estimate_spot(option_chain_df)
    chain_summary = _summarize_near_money_chain(option_chain_df, current_price)
    system_prompt, user_prompt = _build_prompt(
        symbol=symbol,
        iv_rv_spread=iv_rv_spread,
        avg_iv=avg_iv,
        realized_vol=realized_vol,
        account_equity=account_equity,
        chain_summary=chain_summary,
        permitted=permitted,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    client = _get_client()
    last_error: Exception | None = None

    for model in FEATHERLESS_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            last_error = exc
            continue

        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        if response.choices is None:
            err = raw.get("error") or {}
            last_error = RuntimeError(
                err.get("message")
                or f"{model} returned no choices: {raw}"
            )
            continue

        content = response.choices[0].message.content or ""
        return _enforce_direction_constraint(_parse_decision_json(content), permitted)

    raise RuntimeError(
        "Featherless request failed for all candidate models. "
        f"Last error: {last_error}"
    )
