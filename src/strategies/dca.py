"""
src/strategies/dca.py

Dollar-Cost Averaging strategy for long-term ETF positions.
- Regular scheduled buys
- Enhanced buys on significant dips
- Exposure reduction in bear markets
"""
from __future__ import annotations

import pandas as pd

from src.features.indicators import drawdown, ema
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)


class DCAStrategy(BaseStrategy):
    name = "dca_etf"
    horizon = Horizon.LONG_TERM

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        cfg = self.config
        timeframe = "1d"
        buy_day = cfg.get("buy_day_of_month", 1)
        dip_enabled = cfg.get("dip_buy_enabled", True)
        dip_threshold = cfg.get("dip_threshold_pct", -0.05)
        bear_reduce = cfg.get("bear_reduce_enabled", True)

        close = df["close"]
        c = float(close.iloc[-1])
        today = df.index[-1].date()

        # ---- Bear market handling ----
        size_multiplier = self._regime_size_multiplier(regime)
        risk_multiplier = self._regime_risk_multiplier(regime)
        if bear_reduce and regime in ("bear_trend", "panic"):
            # In bear conditions, keep exposure but slow the DCA cadence.
            size_multiplier = min(size_multiplier, 0.25)

        # ---- Dip buy ----
        if dip_enabled and len(close) >= 20:
            ret_5d = float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6])
            if ret_5d <= dip_threshold:
                dd = float(drawdown(close).iloc[-1])
                if dd > -0.25:  # Not in a deep structural sell-off
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
                            f"DCA dip buy: {ret_5d:.1%} drop over 5 days. "
                            f"Regime={regime}, size_multiplier={size_multiplier:.2f}."
                        ),
                        metadata={
                            "trigger": "dip",
                            "ret_5d": round(ret_5d, 4),
                            "drawdown": round(dd, 4),
                            "regime": regime,
                            "regime_strength": size_multiplier,
                            "risk_multiplier": risk_multiplier,
                            "size_multiplier": size_multiplier,
                        },
                    )

        # ---- Regular scheduled buy ----
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
                    f"Scheduled DCA buy on day {buy_day}. "
                    f"Regime={regime}, size_multiplier={size_multiplier:.2f}."
                ),
                metadata={
                    "trigger": "scheduled",
                    "regime": regime,
                    "regime_strength": size_multiplier,
                    "risk_multiplier": risk_multiplier,
                    "size_multiplier": size_multiplier,
                },
            )

        return no_trade(
            self.name, asset, timeframe, self.horizon,
            f"No DCA trigger today (day {today.day}, buy_day={buy_day})."
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
