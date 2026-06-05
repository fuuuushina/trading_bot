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

        # ---- Bear market guard ----
        if bear_reduce and regime in ("bear_trend", "panic"):
            e200 = float(ema(close, 200).iloc[-1])
            if c < e200 * 0.95:
                return Signal(
                    strategy_name=self.name,
                    asset=asset,
                    timeframe=timeframe,
                    signal=SignalType.SELL,
                    confidence=0.80,
                    entry_price=c,
                    stop_loss=None,
                    take_profit=None,
                    risk_reward=None,
                    horizon=self.horizon,
                    reason=f"Bear market detected ({regime}). Price 5%+ below EMA200. Reducing exposure.",
                    metadata={"regime": regime, "price_vs_ema200": round(c / e200 - 1, 4)},
                )

        # ---- Dip buy ----
        if dip_enabled and len(close) >= 20:
            ret_5d = float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6])
            if ret_5d <= dip_threshold:
                dd = float(drawdown(close).iloc[-1])
                if dd > -0.25:  # Not in a deep bear market
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
                        reason=f"DCA dip buy: {ret_5d:.1%} drop over 5 days. Drawdown={dd:.1%}.",
                        metadata={"trigger": "dip", "ret_5d": round(ret_5d, 4), "drawdown": round(dd, 4)},
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
                reason=f"Scheduled DCA buy on day {buy_day} of month.",
                metadata={"trigger": "scheduled", "regime": regime},
            )

        return no_trade(
            self.name, asset, timeframe, self.horizon,
            f"No DCA trigger today (day {today.day}, buy_day={buy_day})."
        )
