"""Offline checks for the IV/RV ratio signal (no API keys or network)."""

from __future__ import annotations

import pytest

from decision import allowed_actions_by_ratio
from vol_signal import DEFAULT_RICH_RATIO, compute_iv_rv_ratio


def test_ratio_is_iv_over_rv():
    assert compute_iv_rv_ratio(0.23, 0.20) == pytest.approx(1.15)


def test_ratio_rejects_nonpositive_rv():
    with pytest.raises(ValueError):
        compute_iv_rv_ratio(0.20, 0.0)


def test_rich_ratio_allows_credit_only():
    assert allowed_actions_by_ratio(DEFAULT_RICH_RATIO + 0.01) == ["credit_spread", "no_trade"]


@pytest.mark.parametrize("ratio", [DEFAULT_RICH_RATIO, 1.0, 0.5])
def test_not_rich_never_trades(ratio):
    # No debit branch: cheap-looking premium lost money in vrp_backtest.py.
    assert allowed_actions_by_ratio(ratio) == ["no_trade"]


def test_ratio_scales_with_vol_level():
    # Same 2-point spread: rich at low vol, not rich at high vol.
    assert allowed_actions_by_ratio(compute_iv_rv_ratio(0.14, 0.12)) == ["credit_spread", "no_trade"]
    assert allowed_actions_by_ratio(compute_iv_rv_ratio(0.42, 0.40)) == ["no_trade"]
