"""
src/strategies/intraday_macd.py

MACD (12,26,9) crossover + continuation for EUR/USD 5-minute bars.

BUY  conditions (any one):
  - MACD line crosses above signal line (crossover)
  - MACD line already above signal AND histogram growing (continuation)

SELL conditions (mirror):
  - MACD line crosses below signal line
  - MACD line already below signal AND histogram shrinking (continuation)

Additional filter: MACD must be on the correct side of zero for continuation
signals to reduce counter-trend noise.

Horizon: INTRADAY | Timeframe: 5m
"""
from __future__ import annotations

import pandas as pd

from src.features.indicators import atr, macd
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)

_MIN_BARS = 35


class IntradayMACDStrategy(BaseStrategy):
    name = "intraday_macd"
    horizon = Horizon.INTRADAY

    def generate_signal(self, df: pd.DataFrame, asset: str, regime: str) -> Signal:
        cfg = self.config
        timeframe    = cfg.get("timeframe",         "5m")
        fast         = cfg.get("macd_fast",          12)
        slow         = cfg.get("macd_slow",          26)
        sig          = cfg.get("macd_signal",         9)
        atr_sl_mult  = cfg.get("atr_multiplier_sl",  1.0)
        atr_tp_mult  = cfg.get("atr_multiplier_tp",  1.8)
        min_conf     = cfg.get("min_confidence",     0.35)
        use_zero_filter = cfg.get("zero_line_filter", False)

        if len(df) < _MIN_BARS:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Not enough bars ({len(df)} < {_MIN_BARS})")

        if regime == "panic":
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Panic regime — skipping")

        close   = df["close"]
        macd_df = macd(close, fast, slow, sig)
        atr_val = float(atr(df, 14).iloc[-1])

        macd_cur  = float(macd_df["macd"].iloc[-1])
        macd_prev = float(macd_df["macd"].iloc[-2])
        sig_cur   = float(macd_df["signal"].iloc[-1])
        sig_prev  = float(macd_df["signal"].iloc[-2])
        hist_cur  = float(macd_df["histogram"].iloc[-1])
        hist_prev = float(macd_df["histogram"].iloc[-2])

        if any(pd.isna(v) for v in [macd_cur, sig_cur, hist_cur]):
            return no_trade(self.name, asset, timeframe, self.horizon, "NaN indicator")
        if atr_val <= 0:
            return no_trade(self.name, asset, timeframe, self.horizon, "Zero ATR")

        c = float(close.iloc[-1])

        # Crossover detection
        cross_up   = macd_prev <= sig_prev and macd_cur > sig_cur
        cross_down = macd_prev >= sig_prev and macd_cur < sig_cur

        # Continuation: already on correct side + histogram growing
        cont_up   = macd_cur > sig_cur and hist_cur > hist_prev
        cont_down = macd_cur < sig_cur and hist_cur < hist_prev

        # Zero-line filter: continuation must be above/below zero
        if use_zero_filter:
            cont_up   = cont_up   and macd_cur > 0
            cont_down = cont_down and macd_cur < 0

        buy_signal  = cross_up  or cont_up
        sell_signal = cross_down or cont_down

        if not buy_signal and not sell_signal:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "No MACD condition met")

        # If both triggered (edge case), pick the stronger
        if buy_signal and sell_signal:
            if abs(macd_cur - sig_cur) > abs(macd_prev - sig_prev):
                sell_signal = False
            else:
                buy_signal = False

        direction   = 1 if buy_signal else -1
        signal_type = SignalType.BUY if buy_signal else SignalType.SELL

        sl = self._atr_stop(c, atr_val, direction, atr_sl_mult)
        tp = self._atr_target(c, atr_val, direction, atr_tp_mult)
        rr = self._rr_ratio(c, sl, tp)

        if rr is None or rr < 1.2:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr} below 1.2")

        # Confidence
        score = 0.38
        is_cross = cross_up if buy_signal else cross_down
        if is_cross:
            score += 0.08  # crossover bonus
        # Histogram acceleration
        if hist_cur != 0 and abs(hist_cur) > abs(hist_prev):
            score += 0.06
        # MACD on correct side of zero
        if (buy_signal and macd_cur > 0) or (sell_signal and macd_cur < 0):
            score += 0.06
        # Regime bonus
        if (buy_signal and regime in ("bull_trend",)) or \
           (sell_signal and regime in ("bear_trend",)):
            score += 0.06
        if regime in ("breakout_expansion",):
            score += 0.04

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
                f"MACD({fast},{slow},{sig}) {signal_type.value} [{signal_kind}]: "
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
                "regime":    regime,
            },
        )
