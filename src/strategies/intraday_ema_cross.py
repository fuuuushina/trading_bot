"""
src/strategies/intraday_ema_cross.py

EMA 9/21 crossover for EUR/USD 5-minute bars.

Entry rules:
  BUY:  EMA(9) crosses above EMA(21) in the last 2 bars,
        price > EMA(9), RSI between 40 and 70 (healthy momentum)
  SELL: EMA(9) crosses below EMA(21) in the last 2 bars,
        price < EMA(9), RSI between 30 and 60

Stops: ATR(14) × 1.5 (tight intraday)
Target: ATR(14) × 2.5 → min R:R 1.5
Horizon: INTRADAY | Timeframe: 5m
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

_MIN_BARS = 35   # ~3h of 5min bars for EMA(21) to stabilize
_CROSS_LOOKBACK = 5  # detect cross within last N bars (~25 min)


class IntradayEMACrossStrategy(BaseStrategy):
    name = "intraday_ema_cross"
    horizon = Horizon.INTRADAY

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        cfg = self.config
        timeframe   = cfg.get("timeframe",      "5m")
        ema_fast    = cfg.get("ema_fast",        9)
        ema_slow    = cfg.get("ema_slow",        21)
        atr_sl_mult = cfg.get("atr_multiplier_sl", 1.5)
        atr_tp_mult = cfg.get("atr_multiplier_tp", 2.5)
        min_conf    = cfg.get("min_confidence",  0.55)
        rsi_period  = cfg.get("rsi_period",      14)

        if len(df) < _MIN_BARS:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Not enough bars ({len(df)} < {_MIN_BARS})")

        close    = df["close"]
        fast     = ema(close, ema_fast)
        slow     = ema(close, ema_slow)
        rsi_vals = rsi(close, rsi_period)
        atr_val  = float(atr(df, 14).iloc[-1])

        curr_fast, curr_slow = float(fast.iloc[-1]), float(slow.iloc[-1])
        c = float(close.iloc[-1])

        if pd.isna(curr_fast) or pd.isna(curr_slow) or atr_val <= 0:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Indicator NaN or zero ATR")

        rsi_val = float(rsi_vals.iloc[-1])

        # Detect cross within last _CROSS_LOOKBACK bars
        n_look = min(_CROSS_LOOKBACK, len(fast) - 1)
        bullish_cross = False
        bearish_cross = False
        for i in range(1, n_look + 1):
            f_cur = float(fast.iloc[-i])
            f_prv = float(fast.iloc[-(i + 1)])
            s_cur = float(slow.iloc[-i])
            s_prv = float(slow.iloc[-(i + 1)])
            if (f_prv <= s_prv) and (f_cur > s_cur):
                bullish_cross = True
                break
            if (f_prv >= s_prv) and (f_cur < s_cur):
                bearish_cross = True
                break

        if not bullish_cross and not bearish_cross:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"No EMA cross in last {n_look} bars")

        # RSI filter
        if bullish_cross and not (38 <= rsi_val <= 72):
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"RSI={rsi_val:.1f} outside [38,72] for bullish cross")
        if bearish_cross and not (28 <= rsi_val <= 62):
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"RSI={rsi_val:.1f} outside [28,62] for bearish cross")

        # Price must be on correct side of EMA(fast)
        if bullish_cross and c < curr_fast * 0.9998:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Price below fast EMA despite bullish cross")
        if bearish_cross and c > curr_fast * 1.0002:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Price above fast EMA despite bearish cross")

        # Confidence scoring
        ema_separation = abs(curr_fast - curr_slow) / curr_slow if curr_slow > 0 else 0
        score = 0.60  # base for confirmed cross
        if ema_separation > 0.0005:
            score += 0.10   # clean separation
        if bullish_cross and 45 <= rsi_val <= 65:
            score += 0.15   # ideal RSI zone
        elif bearish_cross and 35 <= rsi_val <= 55:
            score += 0.15

        # Small penalty in high-volatility regimes (choppier for scalping)
        if regime in ("panic", "high_volatility"):
            score -= 0.10

        score = round(min(score, 1.0), 3)
        if score < min_conf:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Confidence {score:.2f} below {min_conf}")

        direction   = 1 if bullish_cross else -1
        signal_type = SignalType.BUY if bullish_cross else SignalType.SELL

        sl = self._atr_stop(c, atr_val, direction, atr_sl_mult)
        tp = self._atr_target(c, atr_val, direction, atr_tp_mult)
        rr = self._rr_ratio(c, sl, tp)

        if rr is None or rr < 1.5:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr} below 1.5")

        gap_pct = round(ema_separation * 100, 4)
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
                f"EMA({ema_fast}/{ema_slow}) {'bull' if bullish_cross else 'bear'} cross; "
                f"RSI={rsi_val:.1f}; gap={gap_pct}%; ATR={atr_val:.6f}"
            ),
            metadata={
                "strategy": self.name,
                "ema_fast": round(curr_fast, 6),
                "ema_slow": round(curr_slow, 6),
                "rsi":      round(rsi_val, 2),
                "atr":      round(atr_val, 6),
                "regime":   regime,
            },
        )
