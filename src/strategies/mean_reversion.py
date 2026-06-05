"""
src/strategies/mean_reversion.py

Swing mean reversion strategy.
Signal logic:
  - Z-score of price deviates significantly from mean (|z| > entry threshold)
  - Bollinger Bands confirms extreme extension
  - RSI confirms oversold / overbought
  - Enter counter-trend, target mean reversion
"""
from __future__ import annotations

import pandas as pd

from src.features.indicators import atr, bollinger_bands, rsi, z_score
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"
    horizon = Horizon.SWING

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        cfg = self.config
        timeframe = cfg.get("timeframe", "1d")
        zscore_entry = cfg.get("zscore_entry", 2.0)
        zscore_exit = cfg.get("zscore_exit", 0.5)
        lookback = cfg.get("lookback_period", 20)
        rsi_oversold = cfg.get("rsi_oversold", 30)
        rsi_overbought = cfg.get("rsi_overbought", 70)
        min_conf = cfg.get("min_confidence", 0.60)
        atr_sl_mult = cfg.get("atr_multiplier_sl", 1.5)
        atr_tp_mult = cfg.get("atr_multiplier_tp", 2.5)

        # Block mean reversion in strong trending regimes
        if regime in ("bull_trend", "bear_trend", "breakout_expansion", "panic"):
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Mean reversion blocked in {regime} regime.")

        if len(df) < lookback + 10:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Insufficient data.")

        close = df["close"]
        c = float(close.iloc[-1])

        z = float(z_score(close, lookback).iloc[-1])
        rsi_val = float(rsi(close, 14).iloc[-1])
        bb = bollinger_bands(close, lookback, 2.0)
        bb_pct_b = float(bb["bb_pct_b"].iloc[-1])
        bb_mid = float(bb["bb_middle"].iloc[-1])
        atr_val = float(atr(df, 14).iloc[-1])

        if pd.isna(z) or pd.isna(rsi_val):
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "NaN in indicators.")

        # ---- Long signal (oversold) ----
        long_conditions = {
            "zscore_low": z < -zscore_entry,
            "rsi_oversold": rsi_val < rsi_oversold,
            "bb_oversold": bb_pct_b < 0.05,
        }
        # ---- Short signal (overbought) ----
        short_conditions = {
            "zscore_high": z > zscore_entry,
            "rsi_overbought": rsi_val > rsi_overbought,
            "bb_overbought": bb_pct_b > 0.95,
        }

        long_hits = sum(long_conditions.values())
        short_hits = sum(short_conditions.values())

        if long_hits == 0 and short_hits == 0:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"No extreme detected. z={z:.2f}, RSI={rsi_val:.1f}")

        if long_hits >= short_hits:
            direction = 1
            signal_type = SignalType.BUY
            conditions = long_conditions
        else:
            direction = -1
            signal_type = SignalType.SELL
            conditions = short_conditions

        # Confidence proportional to number of confirming conditions
        hits = max(long_hits, short_hits)
        score = 0.40 + 0.20 * hits  # 0.60 | 0.80 | 1.00

        # Penalise if z is borderline
        if abs(z) < zscore_entry + 0.3:
            score -= 0.10

        score = round(min(score, 1.0), 3)

        if score < min_conf:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Confidence {score:.2f} below minimum {min_conf}.")

        entry = c
        # Stop: beyond the current extension
        sl = self._atr_stop(entry, atr_val, direction, atr_sl_mult)
        # Target: mean (BB middle / SMA)
        tp = bb_mid  # target is the mean

        if abs(tp - entry) < abs(sl - entry) * 0.8:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Target (mean) too close — poor R:R.")

        rr = self._rr_ratio(entry, sl, tp)

        active_conditions = [k for k, v in conditions.items() if v]
        reason = (
            f"z-score={z:.2f}, RSI={rsi_val:.1f}, BB%B={bb_pct_b:.2f}. "
            f"Conditions: {', '.join(active_conditions)}"
        )

        return Signal(
            strategy_name=self.name,
            asset=asset,
            timeframe=timeframe,
            signal=signal_type,
            confidence=score,
            entry_price=round(entry, 4),
            stop_loss=round(sl, 4),
            take_profit=round(tp, 4),
            risk_reward=rr,
            horizon=self.horizon,
            reason=reason,
            metadata={
                "z_score": round(z, 4),
                "rsi": round(rsi_val, 2),
                "bb_pct_b": round(bb_pct_b, 4),
                "bb_mid": round(bb_mid, 4),
                "atr": round(atr_val, 4),
                "regime": regime,
            },
        )
