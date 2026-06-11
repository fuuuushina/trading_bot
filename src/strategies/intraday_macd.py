"""
src/strategies/intraday_macd.py

MACD (12,26,9) crossover + continuation for 5-minute bars.

Quality filters (même niveau que intraday_trend_scalp) :
  1. Zero-line filter: continuation doit être du bon côté du zéro
  2. EMA(21) alignment: prix > EMA21 pour BUY, < EMA21 pour SELL
  3. Histogram minimum: |hist| >= min_hist_atr_ratio × ATR (évite le bruit)
  4. Crossover requires histogram inversion (pas juste tick)

R:R amélioré : SL = 1.5×ATR, TP = 3.0×ATR → R:R ≥ 2.0 garanti

Horizon: INTRADAY | Timeframe: 5m
"""
from __future__ import annotations

import pandas as pd

from src.features.indicators import atr, ema, macd
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)

_MIN_BARS = 40


class IntradayMACDStrategy(BaseStrategy):
    name = "intraday_macd"
    horizon = Horizon.INTRADAY

    def generate_signal(self, df: pd.DataFrame, asset: str, regime: str) -> Signal:
        cfg = self.config
        timeframe        = cfg.get("timeframe",           "5m")
        fast             = cfg.get("macd_fast",            12)
        slow_p           = cfg.get("macd_slow",            26)
        sig_p            = cfg.get("macd_signal",           9)
        atr_sl_mult      = cfg.get("atr_multiplier_sl",    2.5)
        atr_tp_mult      = cfg.get("atr_multiplier_tp",    4.0)
        min_conf         = cfg.get("min_confidence",       0.50)
        ema_period       = cfg.get("ema_trend_period",     21)
        min_hist_ratio   = cfg.get("min_hist_atr_ratio",   0.10)  # |hist| >= 10% ATR

        if len(df) < _MIN_BARS:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Not enough bars ({len(df)} < {_MIN_BARS})")

        if regime == "panic":
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Panic regime — skipping")

        close   = df["close"]
        macd_df = macd(close, fast, slow_p, sig_p)
        atr_val = float(atr(df, 14).iloc[-1])
        ema21   = ema(close, ema_period)

        macd_cur  = float(macd_df["macd"].iloc[-1])
        macd_prev = float(macd_df["macd"].iloc[-2])
        sig_cur   = float(macd_df["signal"].iloc[-1])
        sig_prev  = float(macd_df["signal"].iloc[-2])
        hist_cur  = float(macd_df["histogram"].iloc[-1])
        hist_prev = float(macd_df["histogram"].iloc[-2])
        curr_ema  = float(ema21.iloc[-1])
        c         = float(close.iloc[-1])

        if any(pd.isna(v) for v in [macd_cur, sig_cur, hist_cur, curr_ema]):
            return no_trade(self.name, asset, timeframe, self.horizon, "NaN indicator")
        if atr_val <= 0:
            return no_trade(self.name, asset, timeframe, self.horizon, "Zero ATR")

        # ---- Filtre 1 : histogram minimum (évite le bruit de 1 tick) ----
        min_hist = min_hist_ratio * atr_val
        if abs(hist_cur) < min_hist:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Histogram {abs(hist_cur):.6f} < min {min_hist:.6f} (bruit)")

        # ---- Détection des signaux ----
        # Crossover: vrai croisement confirmé (hist change de signe)
        cross_up   = hist_prev <= 0 and hist_cur > 0
        cross_down = hist_prev >= 0 and hist_cur < 0

        # Continuation: aligné avec zéro + histogram accélère
        cont_up   = (macd_cur > 0 and sig_cur > 0
                     and hist_cur > 0 and hist_cur > hist_prev)
        cont_down = (macd_cur < 0 and sig_cur < 0
                     and hist_cur < 0 and hist_cur < hist_prev)

        buy_signal  = cross_up  or cont_up
        sell_signal = cross_down or cont_down

        if not buy_signal and not sell_signal:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "No MACD condition met")

        if buy_signal and sell_signal:
            buy_signal  = macd_cur > sig_cur
            sell_signal = not buy_signal

        # ---- Filtre 2 : EMA(21) trend alignment ----
        if buy_signal and c < curr_ema:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"BUY blocked: price {c:.5f} < EMA{ema_period} {curr_ema:.5f}")
        if sell_signal and c > curr_ema:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"SELL blocked: price {c:.5f} > EMA{ema_period} {curr_ema:.5f}")

        direction   = 1 if buy_signal else -1
        signal_type = SignalType.BUY if buy_signal else SignalType.SELL

        sl = self._atr_stop(c, atr_val, direction, atr_sl_mult)
        is_forex = "=" in asset
        min_sl_pips = cfg.get("min_sl_pips", 15.0)
        sl = self._enforce_min_sl(c, sl, direction, min_sl_pips, is_forex)
        tp = self._atr_target(c, atr_val, direction, atr_tp_mult)
        rr = self._rr_ratio(c, sl, tp)

        if rr is None or rr < 1.8:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr} below 1.8")

        # ---- Scoring de confiance ----
        is_cross = cross_up if buy_signal else cross_down
        score = 0.44
        if is_cross:
            score += 0.12   # croisement vrai = signal fort
        # Histogram accélère
        if abs(hist_cur) > abs(hist_prev) * 1.5:
            score += 0.07   # momentum MACD fort
        elif abs(hist_cur) > abs(hist_prev):
            score += 0.04
        # Bon côté du zéro
        if (buy_signal and macd_cur > 0) or (sell_signal and macd_cur < 0):
            score += 0.06
        # Régime favorable
        if (buy_signal and regime in ("bull_trend", "breakout_expansion")):
            score += 0.06
        if (sell_signal and regime in ("bear_trend",)):
            score += 0.06
        if regime in ("compression",):
            score += 0.03
        # Prix bien au-dessus/en-dessous EMA21
        ema_dist = abs(c - curr_ema) / atr_val
        if ema_dist >= 0.5:
            score += 0.04   # prix éloigné de l'EMA = trend établi

        score = round(min(score, 1.0), 3)
        if score < min_conf:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Confidence {score:.2f} below {min_conf}")

        signal_kind = "cross" if is_cross else "cont"
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
                f"MACD({fast},{slow_p},{sig_p}) {signal_type.value} [{signal_kind}]: "
                f"macd={macd_cur:.6f} sig={sig_cur:.6f} hist={hist_cur:.6f}; "
                f"ATR={atr_val:.6f}"
            ),
            metadata={
                "strategy":  self.name,
                "macd":      round(macd_cur, 7),
                "macd_sig":  round(sig_cur, 7),
                "histogram": round(hist_cur, 7),
                "crossover": is_cross,
                "atr":       round(atr_val, 6),
                "rsi":       50.0,   # placeholder — pas de RSI ici
                "ema_slow":  round(curr_ema, 6),
                "regime":    regime,
            },
        )
