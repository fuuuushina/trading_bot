"""
src/strategies/ema_cross_swing.py

EMA Cross Swing — croisement EMA9/EMA21 sur barres journalières.

Logique ultra-simple et réactive :
  BUY  quand EMA9 vient de passer au-dessus de EMA21 (ou prix > EMA9 > EMA21)
  SELL quand EMA9 vient de passer en-dessous de EMA21
  Pas de filtre ADX, pas de SMA200 — se déclenche vite.

Horizon : SWING (daily bars)
Min bars : 25
"""
from __future__ import annotations

import pandas as pd

from src.features.indicators import atr, ema, rsi
from src.strategies.base import BaseStrategy, Horizon, Signal, SignalType, no_trade


class EMACrossSwingStrategy(BaseStrategy):
    name = "ema_cross_swing"
    horizon = Horizon.SWING

    def generate_signal(self, df: pd.DataFrame, asset: str, regime: str) -> Signal:
        cfg          = self.config
        timeframe    = cfg.get("timeframe",        "1d")
        atr_sl_mult  = cfg.get("atr_sl_multiplier", 2.0)
        atr_tp_mult  = cfg.get("atr_tp_multiplier", 3.0)
        min_conf     = cfg.get("min_confidence",    0.40)
        fast_p       = cfg.get("ema_fast",           9)
        slow_p       = cfg.get("ema_slow",           21)

        if len(df) < slow_p + 5:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Not enough bars ({len(df)} < {slow_p + 5}).")

        close = df["close"]
        e_fast = ema(close, fast_p)
        e_slow = ema(close, slow_p)

        ef_now  = float(e_fast.iloc[-1])
        es_now  = float(e_slow.iloc[-1])
        ef_prev = float(e_fast.iloc[-2])
        es_prev = float(e_slow.iloc[-2])
        price   = float(close.iloc[-1])
        rsi_val = float(rsi(close, 14).iloc[-1])
        atr_val = float(atr(df, 14).iloc[-1])

        if any(pd.isna(v) for v in (ef_now, es_now, ef_prev, es_prev)) or atr_val <= 0:
            return no_trade(self.name, asset, timeframe, self.horizon, "Indicator NaN.")

        # Fresh bullish cross (yesterday below, today above)
        fresh_cross_up   = ef_prev <= es_prev and ef_now > es_now
        # Continuation (EMA9 already above EMA21 and price above EMA9)
        continuation_up  = ef_now > es_now and price > ef_now

        # Fresh bearish cross
        fresh_cross_down  = ef_prev >= es_prev and ef_now < es_now
        continuation_down = ef_now < es_now and price < ef_now

        if fresh_cross_up or continuation_up:
            if regime == "panic":
                return no_trade(self.name, asset, timeframe, self.horizon, "Panic regime — no entries.")
            if rsi_val > 80:
                return no_trade(self.name, asset, timeframe, self.horizon,
                                f"RSI={rsi_val:.0f} extremely overbought.")
            confidence = 0.75 if fresh_cross_up else 0.60
            if 45 < rsi_val < 70:
                confidence += 0.05
            if confidence < min_conf:
                return no_trade(self.name, asset, timeframe, self.horizon,
                                f"Confidence {confidence:.2f} below {min_conf}.")
            sl = price - atr_sl_mult * atr_val
            tp = price + atr_tp_mult * atr_val
            rr = self._rr_ratio(price, sl, tp)
            return Signal(
                strategy_name=self.name, asset=asset, timeframe=timeframe,
                signal=SignalType.BUY, confidence=round(confidence, 3),
                entry_price=round(price, 4),
                stop_loss=round(sl, 4), take_profit=round(tp, 4),
                risk_reward=rr, horizon=self.horizon,
                reason=f"{'CROSS UP' if fresh_cross_up else 'EMA9>21'} EMA{fast_p}={ef_now:.2f} EMA{slow_p}={es_now:.2f} RSI={rsi_val:.0f}",
                metadata={"ema_fast": round(ef_now, 4), "ema_slow": round(es_now, 4),
                           "rsi": round(rsi_val, 2), "fresh_cross": fresh_cross_up},
            )

        if fresh_cross_down or continuation_down:
            if rsi_val < 20:
                return no_trade(self.name, asset, timeframe, self.horizon,
                                f"RSI={rsi_val:.0f} extremely oversold — skip short.")
            confidence = 0.72 if fresh_cross_down else 0.58
            if confidence < min_conf:
                return no_trade(self.name, asset, timeframe, self.horizon,
                                f"Confidence {confidence:.2f} below {min_conf}.")
            sl = price + atr_sl_mult * atr_val
            tp = price - atr_tp_mult * atr_val
            rr = self._rr_ratio(price, sl, tp)
            return Signal(
                strategy_name=self.name, asset=asset, timeframe=timeframe,
                signal=SignalType.SELL, confidence=round(confidence, 3),
                entry_price=round(price, 4),
                stop_loss=round(sl, 4), take_profit=round(tp, 4),
                risk_reward=rr, horizon=self.horizon,
                reason=f"{'CROSS DOWN' if fresh_cross_down else 'EMA9<21'} RSI={rsi_val:.0f}",
                metadata={"ema_fast": round(ef_now, 4), "ema_slow": round(es_now, 4),
                           "fresh_cross": fresh_cross_down},
            )

        return no_trade(self.name, asset, timeframe, self.horizon,
                        f"No EMA cross signal. EMA{fast_p}={ef_now:.2f} EMA{slow_p}={es_now:.2f}")
