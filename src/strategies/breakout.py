"""
src/strategies/breakout.py

N-day high/low breakout aligned with the primary trend (EMA 200).

Filters (hard requirements):
  - Price breaks the N-day high (bullish) or N-day low (bearish)
  - Breakout direction aligns with EMA(200) macro trend
  - ATR is large enough (avoids low-vol fake breakouts)

Scoring:
  Breakout confirmed = 0.60 | EMA200 aligned = 0.25 | ATR valid = 0.15

Removed: volume confirmation and consolidation check — both are unreliable
for liquid ETFs and added more bugs than edge.
"""
from __future__ import annotations

import pandas as pd

from src.features.indicators import atr, ema
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)


class BreakoutStrategy(BaseStrategy):
    name = "breakout"
    horizon = Horizon.SWING

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        cfg = self.config
        timeframe   = cfg.get("timeframe",         "1d")
        lookback    = cfg.get("lookback_period",    20)
        min_atr_pct = cfg.get("min_atr_pct",        0.005)
        min_conf    = cfg.get("min_confidence",      0.70)
        atr_sl_mult = cfg.get("atr_multiplier_sl",   1.5)
        atr_tp_mult = cfg.get("atr_multiplier_tp",   3.0)

        required = lookback + 205
        if len(df) < required:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Insufficient data for breakout.")

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        c = float(close.iloc[-1])

        # ── N-day high/low on CLOSE prices (exclude current bar) ──────────
        # Using close (not the candle high/low) so the breakout level is
        # attainable — a close rarely exceeds the 20-day candle HIGH.
        window_high = float(close.iloc[-(lookback + 1):-1].max())
        window_low  = float(close.iloc[-(lookback + 1):-1].min())

        # ── ATR filter ─────────────────────────────────────────────────────
        atr_val = float(atr(df, 14).iloc[-1])
        atr_pct = atr_val / c if c > 0 else 0.0
        if atr_pct < min_atr_pct:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"ATR {atr_pct:.4f} too small — low-vol trap.")

        # ── Breakout detection ──────────────────────────────────────────────
        breakout_up   = c > window_high
        breakout_down = c < window_low

        if not breakout_up and not breakout_down:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"No breakout. Price={c:.2f}, Range=[{window_low:.2f}, {window_high:.2f}]")

        # ── EMA(200) macro trend alignment ──────────────────────────────────
        e200 = float(ema(close, 200).iloc[-1])
        trend_aligned = (breakout_up and c > e200) or (breakout_down and c < e200)

        # ── Scoring ─────────────────────────────────────────────────────────
        score    = 0.0
        reasons: list[str] = []

        # Breakout confirmed (0.60)
        score += 0.60
        lvl = window_high if breakout_up else window_low
        reasons.append(f"Price broke {lookback}d {'high' if breakout_up else 'low'} @ {lvl:.2f}")

        # EMA200 alignment (0.25)
        if trend_aligned:
            score += 0.25
            reasons.append(f"EMA200={e200:.0f} trend aligned")

        # ATR quality (0.15)
        if atr_pct >= 0.008:
            score += 0.15
            reasons.append(f"ATR={atr_pct:.3%} strong vol")
        elif atr_pct >= min_atr_pct:
            score += 0.08
            reasons.append(f"ATR={atr_pct:.3%} ok")

        score = round(min(score, 1.0), 3)

        if score < min_conf:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Confidence {score:.2f} < {min_conf}.")

        # ── False-breakout filter: require close clearly above/below level ──
        margin = window_high * 0.002
        if breakout_up and c < window_high + margin:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Marginal breakout — risk of false signal.")
        if breakout_down and c > window_low - margin:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Marginal breakdown — risk of false signal.")

        direction   = 1 if breakout_up else -1
        signal_type = SignalType.BUY if breakout_up else SignalType.SELL

        entry = c
        sl    = self._atr_stop(entry, atr_val, direction, atr_sl_mult)
        tp    = self._atr_target(entry, atr_val, direction, atr_tp_mult)
        rr    = self._rr_ratio(entry, sl, tp)

        if rr is None or rr < 1.5:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr} below 1.5.")

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
                "strategy":     self.name,
                "window_high":  round(window_high, 4),
                "window_low":   round(window_low,  4),
                "ema_200":      round(e200,         4),
                "atr":          round(atr_val,      4),
                "trend_aligned": trend_aligned,
                "regime":       regime,
            },
        )
