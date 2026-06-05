"""
src/strategies/tactical_dca.py

Legacy tactical DCA strategy.
This is the old DCA behavior renamed to tactical_dca.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.indicators import drawdown
from src.strategies.base import BaseStrategy, Horizon, Signal, SignalType, no_trade


class TacticalDCAStrategy(BaseStrategy):
    name = "tactical_dca"
    horizon = Horizon.LONG_TERM

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.buy_day = config.get("buy_day_of_month", 1)
        self.dip_buy_enabled = config.get("dip_buy_enabled", True)
        self.dip_threshold_pct = config.get("dip_threshold_pct", -0.05)
        self.bear_reduce_enabled = config.get("bear_reduce_enabled", True)
        self.monthly_size_pct = config.get("monthly_size_pct", 0.05)
        self.dip_size_pct = config.get("dip_size_pct", 0.075)
        self.max_exposure_pct = config.get("max_exposure_pct", 0.85)
        self.min_cash_reserve_pct = config.get("min_cash_reserve_pct", 0.10)

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        timeframe = "1d"
        buy_day = self.buy_day
        dip_enabled = self.dip_buy_enabled
        dip_threshold = self.dip_threshold_pct
        bear_reduce = self.bear_reduce_enabled

        close = df["close"]
        c = float(close.iloc[-1])
        today = df.index[-1].date()

        size_multiplier = self._regime_size_multiplier(regime)
        risk_multiplier = self._regime_risk_multiplier(regime)
        if bear_reduce and regime in ("bear_trend", "panic"):
            size_multiplier = min(size_multiplier, 0.25)

        if dip_enabled and len(close) >= 20:
            ret_5d = float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6])
            if ret_5d <= dip_threshold:
                dd = float(drawdown(close).iloc[-1])
                if dd > -0.25:
                    return Signal(
                        strategy_name=self.name,
                        asset=asset,
                        timeframe=timeframe,
                        signal=SignalType.BUY,
                        confidence=0.72,
                        entry_price=c,
                        stop_loss=None,
                        take_profit=None,
                        risk_reward=None,
                        horizon=self.horizon,
                        reason=(
                            f"Tactical DCA dip buy: {ret_5d:.1%} drop over 5 days. "
                            f"Regime={regime}, size_multiplier={size_multiplier:.2f}."
                        ),
                        metadata={
                            "strategy": self.name,
                            "strategy_type": "growth",
                            "sizing_mode": "fixed_allocation",
                            "requested_size_pct": self.dip_size_pct * size_multiplier,
                            "min_cash_reserve_pct": self.min_cash_reserve_pct,
                            "max_exposure_pct": self.max_exposure_pct,
                            "trigger": "dip",
                            "ret_5d": round(ret_5d, 4),
                            "drawdown": round(dd, 4),
                            "regime": regime,
                            "regime_strength": size_multiplier,
                            "risk_multiplier": risk_multiplier,
                            "size_multiplier": size_multiplier,
                        },
                    )

        if today.day == buy_day:
            return Signal(
                strategy_name=self.name,
                asset=asset,
                timeframe=timeframe,
                signal=SignalType.BUY,
                confidence=0.65,
                entry_price=c,
                stop_loss=None,
                take_profit=None,
                risk_reward=None,
                horizon=self.horizon,
                reason=(
                    f"Scheduled Tactical DCA buy on day {buy_day}. "
                    f"Regime={regime}, size_multiplier={size_multiplier:.2f}."
                ),
                metadata={
                    "strategy": self.name,
                    "strategy_type": "growth",
                    "sizing_mode": "fixed_allocation",
                    "requested_size_pct": self.monthly_size_pct * size_multiplier,
                    "min_cash_reserve_pct": self.min_cash_reserve_pct,
                    "max_exposure_pct": self.max_exposure_pct,
                    "trigger": "scheduled",
                    "regime": regime,
                    "regime_strength": size_multiplier,
                    "risk_multiplier": risk_multiplier,
                    "size_multiplier": size_multiplier,
                },
            )

        return no_trade(
            self.name,
            asset,
            timeframe,
            self.horizon,
            f"No Tactical DCA trigger today (day {today.day}, buy_day={buy_day}).",
        )

    @staticmethod
    def _regime_size_multiplier(regime: str) -> float:
        if regime in ("bull_trend", "euphoric", "breakout_expansion"):
            return 1.5
        if regime in ("range", "low_volatility"):
            return 1.0
        if regime in ("compression", "high_volatility"):
            return 0.5
        if regime in ("bear_trend", "panic"):
            return 0.25
        return 1.0

    @staticmethod
    def _regime_risk_multiplier(regime: str) -> float:
        if regime in ("bull_trend", "euphoric", "breakout_expansion"):
            return 1.5
        if regime in ("range", "low_volatility"):
            return 1.0
        if regime in ("compression", "high_volatility"):
            return 0.75
        if regime in ("bear_trend", "panic"):
            return 0.35
        return 1.0
