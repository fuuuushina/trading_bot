"""
src/strategies/intraday_local_rebound.py

Intraday local rebound strategy for 5-minute bars.

This catches the pattern that pure trend/MACD strategies miss: price tags a
recent local low, then confirms a small rebound before the EMAs fully turn.

Entry rules for BUY:
  - A low in the recent touch window is close to the broader lookback low.
  - Current close has bounced by at least N ATR from that touched low.
  - Current candle/close confirms the rebound.
  - RSI is in a rebound zone and rising.
  - Price is not too far below EMA21/EMA50, avoiding falling knives.
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

_MIN_BARS = 60


class IntradayLocalReboundStrategy(BaseStrategy):
    name = "intraday_local_rebound"
    horizon = Horizon.INTRADAY

    def generate_signal(self, df: pd.DataFrame, asset: str, regime: str) -> Signal:
        cfg = self.config
        timeframe = cfg.get("timeframe", "5m")
        lookback_bars = int(cfg.get("lookback_bars", 36))
        touch_lookback_bars = int(cfg.get("touch_lookback_bars", 8))
        rsi_period = int(cfg.get("rsi_period", 14))
        rsi_min = float(cfg.get("rsi_min", 28.0))
        rsi_max = float(cfg.get("rsi_max", 58.0))
        low_tolerance_atr = float(cfg.get("low_tolerance_atr", 0.35))
        min_rebound_atr = float(cfg.get("min_rebound_atr", 0.18))
        max_rebound_atr = float(cfg.get("max_rebound_atr", 2.20))
        max_ema21_below_atr = float(cfg.get("max_ema21_below_atr", 0.85))
        max_ema50_below_atr = float(cfg.get("max_ema50_below_atr", 1.75))
        atr_sl_mult = float(cfg.get("atr_multiplier_sl", 2.0))
        atr_tp_mult = float(cfg.get("atr_multiplier_tp", 4.0))
        stop_buffer_atr = float(cfg.get("stop_buffer_atr", 0.35))
        min_conf = float(cfg.get("min_confidence", 0.52))

        min_bars = max(_MIN_BARS, lookback_bars + 3, touch_lookback_bars + 3)
        if len(df) < min_bars:
            return no_trade(
                self.name, asset, timeframe, self.horizon,
                f"Not enough bars ({len(df)} < {min_bars})",
            )

        if regime in ("panic", "bear_trend"):
            return no_trade(
                self.name, asset, timeframe, self.horizon,
                f"Regime {regime} blocks local rebound",
            )

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"] if "open" in df.columns else close.shift(1)

        atr_val = float(atr(df, 14).iloc[-1])
        if atr_val <= 0 or pd.isna(atr_val):
            return no_trade(self.name, asset, timeframe, self.horizon, "Zero ATR")

        ema21 = ema(close, 21)
        ema50 = ema(close, 50)
        rsi_vals = rsi(close, rsi_period)

        c = float(close.iloc[-1])
        prev_c = float(close.iloc[-2])
        o = float(open_.iloc[-1])
        curr_rsi = float(rsi_vals.iloc[-1])
        prev_rsi = float(rsi_vals.iloc[-2])
        curr_ema21 = float(ema21.iloc[-1])
        curr_ema50 = float(ema50.iloc[-1])

        if any(pd.isna(v) for v in [c, prev_c, o, curr_rsi, prev_rsi, curr_ema21, curr_ema50]):
            return no_trade(self.name, asset, timeframe, self.horizon, "NaN indicator")

        broad_window = df.iloc[-lookback_bars:]
        touch_window = df.iloc[-touch_lookback_bars:]
        broad_low = float(broad_window["low"].min())
        touch_low = float(touch_window["low"].min())
        touch_pos = int(touch_window["low"].reset_index(drop=True).idxmin())
        bars_since_touch = touch_lookback_bars - 1 - touch_pos

        if touch_low > broad_low + low_tolerance_atr * atr_val:
            return no_trade(
                self.name, asset, timeframe, self.horizon,
                f"No recent local-low touch (touch={touch_low:.5f}, low={broad_low:.5f})",
            )

        rebound_atr = (c - touch_low) / atr_val
        if rebound_atr < min_rebound_atr:
            return no_trade(
                self.name, asset, timeframe, self.horizon,
                f"Rebound {rebound_atr:.2f}xATR below {min_rebound_atr:.2f}",
            )
        if rebound_atr > max_rebound_atr:
            return no_trade(
                self.name, asset, timeframe, self.horizon,
                f"Rebound {rebound_atr:.2f}xATR already extended",
            )

        last_bar_green = c >= o
        close_rising = c >= prev_c
        if not (last_bar_green or close_rising):
            return no_trade(
                self.name, asset, timeframe, self.horizon,
                f"No rebound candle confirmation ({prev_c:.5f} -> {c:.5f})",
            )

        if not (rsi_min <= curr_rsi <= rsi_max):
            return no_trade(
                self.name, asset, timeframe, self.horizon,
                f"RSI {curr_rsi:.1f} outside rebound zone [{rsi_min:.0f},{rsi_max:.0f}]",
            )
        if curr_rsi < prev_rsi:
            return no_trade(
                self.name, asset, timeframe, self.horizon,
                f"RSI not rising ({prev_rsi:.1f} -> {curr_rsi:.1f})",
            )

        ema21_gap_atr = (curr_ema21 - c) / atr_val
        ema50_gap_atr = (curr_ema50 - c) / atr_val
        if ema21_gap_atr > max_ema21_below_atr:
            return no_trade(
                self.name, asset, timeframe, self.horizon,
                f"Price too far below EMA21 ({ema21_gap_atr:.2f}xATR)",
            )
        if ema50_gap_atr > max_ema50_below_atr:
            return no_trade(
                self.name, asset, timeframe, self.horizon,
                f"Price too far below EMA50 ({ema50_gap_atr:.2f}xATR)",
            )

        entry = c
        atr_stop = self._atr_stop(entry, atr_val, 1, atr_sl_mult)
        low_stop = touch_low - stop_buffer_atr * atr_val
        sl = min(atr_stop, low_stop)

        is_forex = "=" in asset
        min_sl_pips = float(cfg.get("min_sl_pips", 15.0))
        sl = self._enforce_min_sl(entry, sl, 1, min_sl_pips, is_forex)
        tp = self._tp_from_sl(entry, sl, 1, atr_tp_mult / atr_sl_mult)
        rr = self._rr_ratio(entry, sl, tp)

        if rr is None or rr < 1.8:
            return no_trade(self.name, asset, timeframe, self.horizon, f"R:R {rr} below 1.8")

        score = 0.46
        if bars_since_touch <= 3:
            score += 0.06
        if close_rising and last_bar_green:
            score += 0.07
        if curr_rsi > prev_rsi + 2:
            score += 0.06
        if 35 <= curr_rsi <= 52:
            score += 0.06
        if c >= curr_ema21:
            score += 0.06
        elif ema21_gap_atr <= 0.25:
            score += 0.03
        if regime in ("range", "compression", "low_volatility"):
            score += 0.06
        if regime == "bull_trend":
            score += 0.04

        score = round(min(score, 1.0), 3)
        if score < min_conf:
            return no_trade(
                self.name, asset, timeframe, self.horizon,
                f"Confidence {score:.2f} below {min_conf:.2f}",
            )

        return Signal(
            strategy_name=self.name,
            asset=asset,
            timeframe=timeframe,
            signal=SignalType.BUY,
            confidence=score,
            entry_price=round(entry, 6),
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            risk_reward=rr,
            horizon=self.horizon,
            reason=(
                f"Local rebound BUY: low={touch_low:.5f}, bounce={rebound_atr:.2f}xATR, "
                f"RSI={prev_rsi:.1f}->{curr_rsi:.1f}, EMA21_gap={ema21_gap_atr:.2f}xATR"
            ),
            metadata={
                "strategy": self.name,
                "trigger": "local_rebound",
                "touch_low": round(touch_low, 6),
                "broad_low": round(broad_low, 6),
                "bars_since_touch": bars_since_touch,
                "rebound_atr": round(rebound_atr, 3),
                "rsi": round(curr_rsi, 2),
                "atr": round(atr_val, 6),
                "ema21": round(curr_ema21, 6),
                "ema50": round(curr_ema50, 6),
                "regime": regime,
                "risk_multiplier": cfg.get("risk_multiplier", 0.75),
            },
        )
