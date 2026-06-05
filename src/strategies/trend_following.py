"""
src/strategies/trend_following.py

Simplified trend-following for ETFs.

Filters (hard requirements):
  - EMA(20) > EMA(50) > EMA(200) with price above EMA(20)  [bull]
  - ADX > threshold confirms trend is real, not just drift

Scoring (soft confidence):
  EMA stack = 0.55 | ADX strength = 0.30 | RSI health = 0.15

No MACD or volume requirement — both are redundant for ETFs:
  - If EMA stack is clean, momentum is already confirmed
  - ETF volume is structurally stable; volume spikes are noise, not confirmation
"""
from __future__ import annotations

import pandas as pd

from src.features.indicators import adx, atr, ema, rsi
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
        timeframe    = cfg.get("timeframe",         "1d")
        min_adx      = cfg.get("min_adx",            25)
        min_conf     = cfg.get("min_confidence",      0.60)
        atr_sl_mult  = cfg.get("atr_multiplier_sl",   2.0)
        atr_tp_mult  = cfg.get("atr_multiplier_tp",   3.5)

        if len(df) < 210:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Insufficient data for trend following.")

        close = df["close"]
        c = float(close.iloc[-1])

        # ── Hard filter: EMA stack ─────────────────────────────────────────
        e20  = float(ema(close, 20).iloc[-1])
        e50  = float(ema(close, 50).iloc[-1])
        e200 = float(ema(close, 200).iloc[-1])

        bull_stack = e20 > e50 > e200 and c > e20
        bear_stack = e20 < e50 < e200 and c < e20

        if not bull_stack and not bear_stack:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "No clear EMA stack alignment.")

        # ── Hard filter: ADX must confirm trend strength ────────────────────
        adx_val = float(adx(df, 14).iloc[-1])
        if pd.isna(adx_val) or adx_val < min_adx:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"ADX={adx_val:.1f} below minimum {min_adx}.")

        # ── RSI — avoid extreme overbought/oversold entries ────────────────
        rsi_val = float(rsi(close, 14).iloc[-1])

        # ── ATR for stops ──────────────────────────────────────────────────
        atr_val = float(atr(df, 14).iloc[-1])

        # ── Confidence scoring ─────────────────────────────────────────────
        score = 0.0
        reasons: list[str] = []

        # EMA stack (0.55) — primary signal
        score += 0.55
        reasons.append("EMA(20)>EMA(50)>EMA(200) " + ("bull" if bull_stack else "bear"))

        # ADX strength (0.30)
        if adx_val >= 35:
            score += 0.30
            reasons.append(f"ADX={adx_val:.0f} strong")
        elif adx_val >= min_adx:
            score += 0.15
            reasons.append(f"ADX={adx_val:.0f} moderate")

        # RSI health (0.15) — reward entries in healthy territory, not stretched
        if bull_stack and 40 < rsi_val < 72:
            score += 0.15
            reasons.append(f"RSI={rsi_val:.0f} healthy")
        elif bear_stack and 28 < rsi_val < 60:
            score += 0.15
            reasons.append(f"RSI={rsi_val:.0f} healthy")

        score = round(min(score, 1.0), 3)

        if score < min_conf:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Confidence {score:.2f} below minimum {min_conf}.")

        # ── Build signal ───────────────────────────────────────────────────
        direction   = 1 if bull_stack else -1
        signal_type = SignalType.BUY if bull_stack else SignalType.SELL

        entry = c
        sl    = self._atr_stop(entry, atr_val, direction, atr_sl_mult)
        tp    = self._atr_target(entry, atr_val, direction, atr_tp_mult)
        rr    = self._rr_ratio(entry, sl, tp)

        if rr is None or rr < 1.5:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr} below minimum 1.5.")

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
            reason="; ".join(reasons),
            metadata={
                "strategy":  self.name,
                "ema_20":    round(e20,   4),
                "ema_50":    round(e50,   4),
                "ema_200":   round(e200,  4),
                "adx":       round(adx_val, 2),
                "rsi":       round(rsi_val, 2),
                "atr":       round(atr_val, 4),
                "regime":    regime,
            },
        )
