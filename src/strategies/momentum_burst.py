"""
src/strategies/momentum_burst.py

Momentum Burst — stratégie agressive pour capter les hausses rapides.

Logique :
  BUY  quand :  prix > EMA20  ET  return 5j > seuil  ET  RSI 40-75
  SELL quand :  prix < EMA20  ET  return 5j < -seuil ET  RSI 25-60

Pas de filtre ADX, pas de SMA200, pas de régime strict.
Très réactif : se déclenche sur la dynamique de prix récente.

Horizon : SWING (daily bars)
Min bars : 30
"""
from __future__ import annotations

import pandas as pd

from src.features.indicators import atr, ema, rsi
from src.strategies.base import BaseStrategy, Horizon, Signal, SignalType, no_trade


class MomentumBurstStrategy(BaseStrategy):
    name = "momentum_burst"
    horizon = Horizon.SWING

    def generate_signal(self, df: pd.DataFrame, asset: str, regime: str) -> Signal:
        cfg          = self.config
        timeframe    = cfg.get("timeframe",         "1d")
        min_ret5     = cfg.get("min_return_5d",      0.01)   # +1% sur 5j minimum
        atr_sl_mult  = cfg.get("atr_sl_multiplier",  1.8)
        atr_tp_mult  = cfg.get("atr_tp_multiplier",  2.8)
        min_conf     = cfg.get("min_confidence",      0.40)

        if len(df) < 30:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Not enough bars ({len(df)} < 30).")

        if regime == "panic":
            return no_trade(self.name, asset, timeframe, self.horizon, "Panic — no momentum entries.")

        close   = df["close"]
        price   = float(close.iloc[-1])
        prev5   = float(close.iloc[-6]) if len(close) >= 6 else price
        ret5    = (price / prev5 - 1) if prev5 > 0 else 0.0
        ret1    = (price / float(close.iloc[-2]) - 1) if len(close) >= 2 else 0.0

        e20     = float(ema(close, 20).iloc[-1])
        e50     = float(ema(close, 50).iloc[-1])
        rsi_val = float(rsi(close, 14).iloc[-1])
        atr_val = float(atr(df, 14).iloc[-1])

        if any(pd.isna(v) for v in (e20, e50, rsi_val)) or atr_val <= 0:
            return no_trade(self.name, asset, timeframe, self.horizon, "Indicator NaN.")

        # ── BUY : momentum positif ───────────────────────────────────────────
        if price > e20 and ret5 >= min_ret5 and 35 < rsi_val < 78:
            confidence = 0.55
            # Boost for strong momentum
            if ret5 > 0.03:
                confidence += 0.10   # +3% sur 5j
            if ret5 > 0.06:
                confidence += 0.08   # +6% sur 5j — très fort
            if price > e20 > e50:
                confidence += 0.07   # EMA alignment
            if 45 < rsi_val < 68:
                confidence += 0.05   # RSI zone idéale
            if ret1 > 0.005:
                confidence += 0.04   # Momentum aujourd'hui aussi

            confidence = round(min(confidence, 0.95), 3)
            if confidence < min_conf:
                return no_trade(self.name, asset, timeframe, self.horizon,
                                f"Confidence {confidence:.2f} below {min_conf}.")

            sl = price - atr_sl_mult * atr_val
            tp = price + atr_tp_mult * atr_val
            rr = self._rr_ratio(price, sl, tp)
            return Signal(
                strategy_name=self.name, asset=asset, timeframe=timeframe,
                signal=SignalType.BUY, confidence=confidence,
                entry_price=round(price, 4),
                stop_loss=round(sl, 4), take_profit=round(tp, 4),
                risk_reward=rr, horizon=self.horizon,
                reason=f"Momentum +{ret5*100:.1f}% 5j | RSI={rsi_val:.0f} | prix>{e20:.0f}(EMA20)",
                metadata={"ret5d": round(ret5, 4), "ret1d": round(ret1, 4),
                           "rsi": round(rsi_val, 2), "ema20": round(e20, 4)},
            )

        # ── SELL : momentum négatif ──────────────────────────────────────────
        if price < e20 and ret5 <= -min_ret5 and 22 < rsi_val < 65:
            confidence = 0.55
            if ret5 < -0.03:
                confidence += 0.10
            if ret5 < -0.06:
                confidence += 0.08
            if price < e20 < e50:
                confidence += 0.07
            if 30 < rsi_val < 55:
                confidence += 0.05
            confidence = round(min(confidence, 0.92), 3)
            if confidence < min_conf:
                return no_trade(self.name, asset, timeframe, self.horizon,
                                f"Confidence {confidence:.2f} below {min_conf}.")
            sl = price + atr_sl_mult * atr_val
            tp = price - atr_tp_mult * atr_val
            rr = self._rr_ratio(price, sl, tp)
            return Signal(
                strategy_name=self.name, asset=asset, timeframe=timeframe,
                signal=SignalType.SELL, confidence=confidence,
                entry_price=round(price, 4),
                stop_loss=round(sl, 4), take_profit=round(tp, 4),
                risk_reward=rr, horizon=self.horizon,
                reason=f"Momentum {ret5*100:.1f}% 5j | RSI={rsi_val:.0f} | prix<{e20:.0f}(EMA20)",
                metadata={"ret5d": round(ret5, 4), "rsi": round(rsi_val, 2)},
            )

        return no_trade(
            self.name, asset, timeframe, self.horizon,
            f"Pas de momentum: ret5={ret5*100:.1f}% prix={'>' if price>e20 else '<'}EMA20 RSI={rsi_val:.0f}",
        )
