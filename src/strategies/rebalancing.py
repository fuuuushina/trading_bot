"""
src/strategies/rebalancing.py

Monthly portfolio rebalancing strategy for long-term allocations.
Compares current weights vs target weights and generates
BUY/SELL signals to rebalance drifted positions.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)


class RebalancingStrategy(BaseStrategy):
    name = "rebalancing"
    horizon = Horizon.LONG_TERM

    def generate_signals_for_portfolio(
        self,
        prices: dict[str, float],
        current_weights: dict[str, float],
        portfolio_state: dict,
        regime: str,
    ) -> list[Signal]:
        """
        Generate rebalancing signals for all assets in the portfolio.
        Returns a list of BUY/SELL signals — one per drifted asset.
        """
        cfg = self.config
        target_weights: dict[str, float] = cfg.get("target_weights", {})
        threshold = cfg.get("rebalance_threshold_pct", 0.05)
        regime_adjust = cfg.get("regime_adjust", True)

        # In bear or panic regimes, skip rebalancing (preserve cash)
        if regime_adjust and regime in ("panic", "bear_trend"):
            return []

        signals: list[Signal] = []

        for asset, target_w in target_weights.items():
            if asset not in prices:
                continue

            current_w = current_weights.get(asset, 0.0)
            drift = current_w - target_w

            if abs(drift) < threshold:
                continue  # Within tolerance — no action

            price = prices[asset]
            total_capital = portfolio_state.get("total_capital", 0.0)
            target_value = target_w * total_capital
            current_value = current_w * total_capital
            delta_value = target_value - current_value

            signal_type = SignalType.BUY if delta_value > 0 else SignalType.SELL
            confidence = min(0.60 + abs(drift) * 2, 0.90)

            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                timeframe="1d",
                signal=signal_type,
                confidence=round(confidence, 3),
                entry_price=price,
                stop_loss=None,
                take_profit=None,
                risk_reward=None,
                horizon=self.horizon,
                reason=(
                    f"Rebalance: current={current_w:.1%} target={target_w:.1%} "
                    f"drift={drift:+.1%} delta=${delta_value:+,.0f}"
                ),
                metadata={
                    "target_weight": target_w,
                    "current_weight": current_w,
                    "drift": round(drift, 4),
                    "delta_usd": round(delta_value, 2),
                    "regime": regime,
                },
            ))

        return signals

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        """
        Single-asset interface required by BaseStrategy.
        For rebalancing, use generate_signals_for_portfolio() instead.
        """
        cfg = self.config
        buy_day = 1  # First of month
        today = date.today()

        if today.day != buy_day:
            return no_trade(
                self.name, asset, "1d", self.horizon,
                f"Rebalancing runs on day {buy_day}. Today is day {today.day}."
            )

        return no_trade(
            self.name, asset, "1d", self.horizon,
            "Use generate_signals_for_portfolio() for rebalancing."
        )
