"""
src/strategies/momentum.py

Multi-timeframe momentum strategy (Swing horizon).
Ranks momentum across lookback periods and enters when
momentum is consistently positive and accelerating.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.features.indicators import atr, ema, momentum_return, rsi, volume_ratio
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)


class MomentumStrategy(BaseStrategy):
    name = "momentum"
    horizon = Horizon.SWING

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        cfg = self.config
        timeframe = cfg.get("timeframe", "1d")
        lookbacks = cfg.get("lookback_returns", [5, 20, 60])
        min_score = cfg.get("min_momentum_score", 0.6)
        rsi_min = cfg.get("rsi_min", 45)
        rsi_max = cfg.get("rsi_max", 75)
        min_conf = cfg.get("min_confidence", 0.60)
        atr_sl_mult = cfg.get("atr_multiplier_sl", 2.0)
        atr_tp_mult = cfg.get("atr_multiplier_tp", 4.0)

        max_lookback = max(lookbacks)
        if len(df) < max_lookback + 20:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Insufficient data for momentum calculation.")

        close = df["close"]
        c = float(close.iloc[-1])

        # ---- Compute momentum across timeframes ----
        mom_returns: dict[int, float] = {}
        for lb in lookbacks:
            mom_returns[lb] = float(momentum_return(close, lb).iloc[-1])

        # All momentum periods positive → sustained momentum
        all_positive = all(r > 0 for r in mom_returns.values())
        # Shorter > longer → accelerating
        accelerating = mom_returns.get(5, 0) > mom_returns.get(20, 0)

        if not all_positive:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Momentum not uniformly positive: {mom_returns}")

        # ---- RSI filter — not overbought ----
        rsi_val = float(rsi(close, 14).iloc[-1])
        rsi_ok = rsi_min <= rsi_val <= rsi_max

        # ---- Trend alignment ----
        e50 = float(ema(close, 50).iloc[-1])
        e200 = float(ema(close, 200).iloc[-1])
        above_e50 = c > e50
        trend_aligned = above_e50 and e50 > e200

        # ---- Volume ----
        vol_r = float(volume_ratio(df, 20).iloc[-1])
        vol_ok = vol_r > 1.0

        # ---- ATR ----
        atr_val = float(atr(df, 14).iloc[-1])

        # ---- Composite momentum score ----
        # Normalise the 20-day return to a 0-1 score
        raw_mom_score = min(mom_returns.get(20, 0) / 0.15, 1.0)  # 15% = perfect

        score = 0.0
        reasons: list[str] = []

        if all_positive:
            score += 0.30
            reasons.append(f"All momentum periods positive {mom_returns}")
        if accelerating:
            score += 0.15
            reasons.append("Short-term > long-term momentum (accelerating)")
        if rsi_ok:
            score += 0.20
            reasons.append(f"RSI={rsi_val:.1f} in healthy range [{rsi_min}, {rsi_max}]")
        if trend_aligned:
            score += 0.20
            reasons.append("Price above EMA50 > EMA200")
        if vol_ok:
            score += 0.10
            reasons.append(f"Volume ratio={vol_r:.2f}")
        score += raw_mom_score * 0.05

        score = round(min(score, 1.0), 3)

        if score < min_conf:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Momentum score {score:.2f} < {min_conf}.")

        entry = c
        sl = self._atr_stop(entry, atr_val, 1, atr_sl_mult)
        tp = self._atr_target(entry, atr_val, 1, atr_tp_mult)
        rr = self._rr_ratio(entry, sl, tp)

        if rr is None or rr < 1.5:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr} below 1.5.")

        return Signal(
            strategy_name=self.name,
            asset=asset,
            timeframe=timeframe,
            signal=SignalType.BUY,
            confidence=score,
            entry_price=round(entry, 4),
            stop_loss=round(sl, 4),
            take_profit=round(tp, 4),
            risk_reward=rr,
            horizon=self.horizon,
            reason="; ".join(reasons),
            metadata={
                **{f"mom_{lb}d": round(r, 4) for lb, r in mom_returns.items()},
                "rsi": round(rsi_val, 2),
                "volume_ratio": round(vol_r, 3),
                "trend_aligned": trend_aligned,
                "accelerating": accelerating,
                "atr": round(atr_val, 4),
                "regime": regime,
            },
        )
