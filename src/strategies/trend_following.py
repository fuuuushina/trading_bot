"""
src/strategies/trend_following.py

Swing trend-following strategy.
Signal logic:
  - EMA 20 > EMA 50 > EMA 200  (bullish stack)
  - Price > EMA 20
  - ADX > threshold (trend strength)
  - Volume confirms (ratio above threshold)
  - MACD histogram positive and rising
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.indicators import adx, ema, macd, rsi, volume_ratio, atr
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)


class TrendFollowingStrategy(BaseStrategy):
    name = "trend_following"
    horizon = Horizon.SWING

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        cfg = self.config
        timeframe = cfg.get("timeframe", "1d")
        min_adx = cfg.get("min_adx", 25)
        vol_ratio_thresh = cfg.get("volume_confirm_ratio", 1.3)
        min_conf = cfg.get("min_confidence", 0.65)
        atr_sl_mult = cfg.get("atr_multiplier_sl", 2.0)
        atr_tp_mult = cfg.get("atr_multiplier_tp", 4.0)

        if len(df) < 210:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Insufficient data for trend following.")

        close = df["close"]
        c = float(close.iloc[-1])

        # ---- EMA stack ----
        e20 = float(ema(close, 20).iloc[-1])
        e50 = float(ema(close, 50).iloc[-1])
        e200 = float(ema(close, 200).iloc[-1])
        bull_stack = e20 > e50 > e200 and c > e20
        bear_stack = e20 < e50 < e200 and c < e20

        if not bull_stack and not bear_stack:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "No clear EMA stack alignment.")

        # ---- ADX ----
        adx_val = float(adx(df, 14).iloc[-1])
        if pd.isna(adx_val) or adx_val < min_adx:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"ADX {adx_val:.1f} below minimum {min_adx}.")

        # ---- Volume confirmation ----
        vol_r = float(volume_ratio(df, 20).iloc[-1])
        volume_ok = vol_r >= vol_ratio_thresh

        # ---- MACD ----
        macd_df = macd(close)
        macd_hist = float(macd_df["histogram"].iloc[-1])
        macd_hist_prev = float(macd_df["histogram"].iloc[-2])
        macd_rising = macd_hist > macd_hist_prev

        # ---- RSI filter (not extreme) ----
        rsi_val = float(rsi(close, 14).iloc[-1])

        # ---- ATR for stops ----
        atr_val = float(atr(df, 14).iloc[-1])

        # ---- Score / confidence ----
        score = 0.0
        reasons = []

        if bull_stack:
            score += 0.35
            reasons.append("EMA stack bullish")
        if adx_val >= min_adx:
            score += 0.25
            reasons.append(f"ADX={adx_val:.1f}")
        if volume_ok:
            score += 0.20
            reasons.append(f"Volume ratio={vol_r:.2f}")
        if macd_rising and macd_hist > 0 and bull_stack:
            score += 0.15
            reasons.append("MACD histogram rising")
        if 40 < rsi_val < 75:
            score += 0.05
            reasons.append(f"RSI={rsi_val:.1f} healthy")

        if score < min_conf:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Confidence {score:.2f} below minimum {min_conf}.")

        # ---- Build signal ----
        if bull_stack and macd_hist > 0:
            direction = 1
            signal_type = SignalType.BUY
        elif bear_stack and macd_hist < 0:
            direction = -1
            signal_type = SignalType.SELL
        else:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "EMA and MACD diverge — no signal.")

        entry = c
        sl = self._atr_stop(entry, atr_val, direction, atr_sl_mult)
        tp = self._atr_target(entry, atr_val, direction, atr_tp_mult)
        rr = self._rr_ratio(entry, sl, tp)

        if rr is None or rr < 1.5:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr} below minimum 1.5.")

        return Signal(
            strategy_name=self.name,
            asset=asset,
            timeframe=timeframe,
            signal=signal_type,
            confidence=round(score, 3),
            entry_price=round(entry, 4),
            stop_loss=round(sl, 4),
            take_profit=round(tp, 4),
            risk_reward=rr,
            horizon=self.horizon,
            reason="; ".join(reasons),
            metadata={
                "ema_20": round(e20, 4),
                "ema_50": round(e50, 4),
                "ema_200": round(e200, 4),
                "adx": round(adx_val, 2),
                "volume_ratio": round(vol_r, 3),
                "macd_hist": round(macd_hist, 6),
                "rsi": round(rsi_val, 2),
                "atr": round(atr_val, 4),
                "regime": regime,
            },
        )
