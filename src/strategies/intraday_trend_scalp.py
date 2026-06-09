"""
src/strategies/intraday_trend_scalp.py

Trend scalp intraday 5min — avec filtres qualité.

Conditions d'entrée LONG :
  1. EMA9 > EMA21 avec spread significatif (>= min_ema_sep_atr_ratio × ATR)
  2. Prix AU-DESSUS de EMA9 (ne pas chasser le prix en bas)
  3. RSI entre 40-68 (pas de surachat, tendance active)
  4. Dernière bougie fermée dans le sens du trade
  5. EMA9 > EMA50 (filtre tendance moyen-terme)

Conditions d'entrée SHORT : symétrique (inverse)

SL/TP :
  - SL = 1.5×ATR (place pour respirer)
  - TP = 3.0×ATR → R:R = 2.0 minimum
"""
from __future__ import annotations

import pandas as pd

from src.features.indicators import atr, ema, rsi
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)

_MIN_BARS = 55


class IntradayTrendScalpStrategy(BaseStrategy):
    name = "intraday_trend_scalp"
    horizon = Horizon.INTRADAY

    def generate_signal(self, df: pd.DataFrame, asset: str, regime: str) -> Signal:
        cfg = self.config
        timeframe            = cfg.get("timeframe",              "5m")
        ema_fast             = cfg.get("ema_fast",                9)
        ema_slow             = cfg.get("ema_slow",               21)
        ema_medium           = cfg.get("ema_medium",             50)
        atr_sl_mult          = cfg.get("atr_multiplier_sl",      1.5)
        atr_tp_mult          = cfg.get("atr_multiplier_tp",      3.0)
        min_conf             = cfg.get("min_confidence",         0.45)
        # EMA spread doit être >= ratio × ATR pour éviter le bruit
        min_sep_atr_ratio    = cfg.get("min_sep_atr_ratio",      0.30)
        # RSI window
        rsi_low              = cfg.get("rsi_low",                40)
        rsi_high             = cfg.get("rsi_high",               68)

        if len(df) < _MIN_BARS:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Not enough bars ({len(df)} < {_MIN_BARS})")

        if regime in ("panic", "extreme_bear"):
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Regime {regime} — skipping scalp")

        close   = df["close"]
        atr_val = float(atr(df, 14).iloc[-1])

        if atr_val <= 0:
            return no_trade(self.name, asset, timeframe, self.horizon, "ATR=0")

        fast   = ema(close, ema_fast)
        slow   = ema(close, ema_slow)
        medium = ema(close, ema_medium)
        rsi_v  = rsi(close, 14)

        curr_fast   = float(fast.iloc[-1])
        curr_slow   = float(slow.iloc[-1])
        curr_medium = float(medium.iloc[-1])
        curr_rsi    = float(rsi_v.iloc[-1])
        c           = float(close.iloc[-1])
        prev_c      = float(close.iloc[-2])

        if any(pd.isna(v) for v in [curr_fast, curr_slow, curr_medium, curr_rsi]):
            return no_trade(self.name, asset, timeframe, self.horizon, "NaN indicator")

        # ---- Direction ----
        bullish = curr_fast > curr_slow
        bearish = curr_fast < curr_slow

        if not bullish and not bearish:
            return no_trade(self.name, asset, timeframe, self.horizon, "EMA flat")

        # ---- Filtre 1 : spread EMA minimal (évite le bruit) ----
        sep = abs(curr_fast - curr_slow)
        min_sep = min_sep_atr_ratio * atr_val
        if sep < min_sep:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"EMA spread {sep:.6f} < min {min_sep:.6f} ({min_sep_atr_ratio}×ATR)")

        # ---- Filtre 2 : prix du bon côté de EMA9 ----
        # LONG : prix doit être ABOVE EMA9 (pas en train de chuter sous la moyenne)
        # SHORT : prix doit être BELOW EMA9
        if bullish and c < curr_fast:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"LONG bloqué: prix {c:.5f} < EMA9 {curr_fast:.5f} (momentum négatif)")
        if bearish and c > curr_fast:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"SHORT bloqué: prix {c:.5f} > EMA9 {curr_fast:.5f} (momentum positif)")

        # ---- Filtre 3 : filtre tendance moyen-terme (EMA50) ----
        if bullish and curr_fast < curr_medium:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"LONG bloqué: EMA9 {curr_fast:.5f} < EMA50 {curr_medium:.5f} (contre-tendance MT)")
        if bearish and curr_fast > curr_medium:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"SHORT bloqué: EMA9 {curr_fast:.5f} > EMA50 {curr_medium:.5f} (contre-tendance MT)")

        # ---- Filtre 4 : RSI ----
        if bullish and not (rsi_low <= curr_rsi <= rsi_high):
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"LONG bloqué: RSI {curr_rsi:.1f} hors fenêtre [{rsi_low},{rsi_high}]")
        if bearish and not (100 - rsi_high <= curr_rsi <= 100 - rsi_low):
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"SHORT bloqué: RSI {curr_rsi:.1f} hors fenêtre [{100-rsi_high},{100-rsi_low}]")

        # ---- Filtre 5 : dernière bougie dans le sens du trade ----
        last_bar_bullish = c >= prev_c
        if bullish and not last_bar_bullish:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"LONG bloqué: dernière bougie baissière ({prev_c:.5f} → {c:.5f})")
        if bearish and last_bar_bullish:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"SHORT bloqué: dernière bougie haussière ({prev_c:.5f} → {c:.5f})")

        # ---- SL/TP ----
        direction   = 1 if bullish else -1
        signal_type = SignalType.BUY if bullish else SignalType.SELL

        sl = self._atr_stop(c, atr_val, direction, atr_sl_mult)
        tp = self._atr_target(c, atr_val, direction, atr_tp_mult)
        rr = self._rr_ratio(c, sl, tp)

        if rr is None or rr < 1.8:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr:.2f} < 1.8")

        # ---- Score de confiance ----
        score = 0.45
        sep_ratio = sep / atr_val
        if sep_ratio > 0.50: score += 0.05
        if sep_ratio > 0.80: score += 0.05
        if sep_ratio > 1.20: score += 0.05

        # Bonus si RSI centré (45-60 pour long, 40-55 pour short)
        if bullish and 45 <= curr_rsi <= 60: score += 0.05
        if bearish and 40 <= curr_rsi <= 55: score += 0.05

        # Bonus régime aligné
        if regime == "bull_trend" and bullish:  score += 0.05
        if regime == "bear_trend" and bearish:  score += 0.05

        score = round(min(score, 1.0), 3)
        if score < min_conf:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Confidence {score:.2f} < {min_conf}")

        return Signal(
            strategy_name=self.name,
            asset=asset,
            timeframe=timeframe,
            signal=signal_type,
            confidence=score,
            entry_price=round(c, 6),
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            risk_reward=rr,
            horizon=self.horizon,
            reason=(
                f"TrendScalp {'BUY' if bullish else 'SELL'}: "
                f"EMA9={curr_fast:.5f} {'>' if bullish else '<'} "
                f"EMA21={curr_slow:.5f}; sep={sep:.5f}({sep_ratio:.2f}×ATR); "
                f"RSI={curr_rsi:.1f}; ATR={atr_val:.6f}"
            ),
            metadata={
                "strategy":   self.name,
                "ema_fast":   round(curr_fast, 6),
                "ema_slow":   round(curr_slow, 6),
                "ema_medium": round(curr_medium, 6),
                "sep":        round(sep, 6),
                "sep_ratio":  round(sep_ratio, 3),
                "rsi":        round(curr_rsi, 1),
                "atr":        round(atr_val, 6),
                "regime":     regime,
            },
        )
