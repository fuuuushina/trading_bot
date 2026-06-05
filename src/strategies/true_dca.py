"""
src/strategies/true_dca.py

True DCA strategy with fixed allocation sizing and dip enhancement.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.strategies.base import BaseStrategy, Horizon, Signal, SignalType, no_trade


class TrueDCAStrategy(BaseStrategy):
    name = "true_dca"
    horizon = Horizon.LONG_TERM

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.assets = config.get("assets", ["SPY"])
        self.buy_day = config.get("buy_day_of_month", 1)
        self.monthly_size_pct = config.get("monthly_size_pct", 0.05)
        self.dip_buy_enabled = config.get("dip_buy_enabled", True)
        self.dip_size_pct = config.get("dip_size_pct", 0.075)
        self.dip_threshold_pct = config.get("dip_threshold_pct", -0.05)
        self.max_exposure_pct = config.get("max_exposure_pct", 0.85)
        self.min_cash_reserve_pct = config.get("min_cash_reserve_pct", 0.10)

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        if df.empty or asset not in self.assets:
            return no_trade(
                self.name,
                asset,
                "1d",
                self.horizon,
                "No True DCA trade: no data or unsupported asset.",
            )

        today = df.index[-1].date()
        close = float(df["close"].iloc[-1])
        is_buy_day = today.day == self.buy_day

        recent_high = float(df["close"].rolling(20).max().iloc[-1])
        dip_pct = (close / recent_high) - 1 if recent_high > 0 else 0.0
        is_dip = self.dip_buy_enabled and dip_pct <= self.dip_threshold_pct

        if not is_buy_day and not is_dip:
            return no_trade(
                self.name,
                asset,
                "1d",
                self.horizon,
                f"No True DCA trigger today (day {today.day}, buy_day={self.buy_day}).",
                metadata={"regime": regime, "dip_pct": round(dip_pct, 4)},
            )

        trigger = "dip" if is_dip else "scheduled"
        requested_pct = self.dip_size_pct if is_dip else self.monthly_size_pct
        reason = (
            f"True DCA {trigger} buy: {dip_pct:.1%} dip." if is_dip
            else f"True DCA scheduled buy on day {self.buy_day}."
        )

        return Signal(
            strategy_name=self.name,
            asset=asset,
            timeframe="1d",
            signal=SignalType.BUY,
            confidence=0.70,
            entry_price=close,
            stop_loss=None,
            take_profit=None,
            risk_reward=None,
            horizon=self.horizon,
            reason=reason,
            metadata={
                "strategy": self.name,
                "strategy_type": "wealth",
                "sizing_mode": "fixed_allocation",
                "requested_size_pct": requested_pct,
                "min_cash_reserve_pct": self.min_cash_reserve_pct,
                "max_exposure_pct": self.max_exposure_pct,
                "trigger": trigger,
                "regime": regime,
                "dip_pct": round(dip_pct, 4),
            },
        )
