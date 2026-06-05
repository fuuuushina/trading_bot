"""
src/strategies/rsi_dip_buyer.py

RSI(2) Dip Buyer — Connors mean-reversion strategy for ETFs.

Logic:
  - Only buy when price > SMA(200)  (avoid bear markets)
  - Entry when RSI(2) < 10          (extreme short-term oversold)
  - Strong entry when RSI(2) < 5    (panic-level oversold)
  - Stop : 1.5 × ATR below entry
  - Target: 2.5 × ATR above entry

Historical edge on SPY/QQQ: ~75-80% win rate, avg hold 3-7 bars.
Perfect for small capital: low friction (~10-15 signals/year), high win rate.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.indicators import atr, rsi
from src.strategies.base import BaseStrategy, Horizon, Signal, SignalType, no_trade


class RSIDipBuyerStrategy(BaseStrategy):
    name = "rsi_dip_buyer"
    horizon = Horizon.SWING

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        if len(df) < 210:
            return no_trade(self.name, asset, "1d", self.horizon,
                            "Insufficient bars for RSI dip buyer.")

        cfg = self.config
        entry_thresh  = cfg.get("rsi2_entry_threshold",  15.0)
        strong_thresh = cfg.get("rsi2_strong_threshold",  8.0)
        atr_sl_mult   = cfg.get("atr_multiplier_sl",      1.5)
        atr_tp_mult   = cfg.get("atr_multiplier_tp",      2.5)
        min_conf      = cfg.get("min_confidence",          0.65)
        position_size = cfg.get("position_size_pct",       0.25)

        close = df["close"]
        price = float(close.iloc[-1])

        # ── Primary trend filter: only long above SMA(200) ──────────────────
        sma200 = float(close.rolling(200).mean().iloc[-1])
        if pd.isna(sma200) or price < sma200:
            return no_trade(self.name, asset, "1d", self.horizon,
                            f"Price {price:.2f} below SMA200 {sma200:.2f}. No longs in downtrend.")

        # ── Skip hard bear regimes ───────────────────────────────────────────
        if regime in ("bear_trend", "panic"):
            return no_trade(self.name, asset, "1d", self.horizon,
                            f"Regime={regime}: avoid mean-reversion counter-trend trades.")

        # ── RSI(2) — extreme oversold ────────────────────────────────────────
        rsi2 = float(rsi(close, 2).iloc[-1])
        if pd.isna(rsi2) or rsi2 > entry_thresh:
            return no_trade(self.name, asset, "1d", self.horizon,
                            f"RSI(2)={rsi2:.1f} not oversold (threshold={entry_thresh}).")

        # ── ATR ──────────────────────────────────────────────────────────────
        atr_val = float(atr(df, 14).iloc[-1])
        if atr_val <= 0 or pd.isna(atr_val):
            return no_trade(self.name, asset, "1d", self.horizon, "Invalid ATR.")

        # ── Confidence based on oversold severity ────────────────────────────
        if rsi2 <= strong_thresh:
            confidence = 0.84
            trigger = "extreme_dip"
        else:
            confidence = 0.70
            trigger = "dip"

        # Regime fine-tuning
        if regime in ("range", "low_volatility", "compression"):
            confidence = min(0.92, confidence + 0.05)
        elif regime in ("high_volatility",):
            confidence -= 0.05

        if confidence < min_conf:
            return no_trade(self.name, asset, "1d", self.horizon,
                            f"Confidence {confidence:.2f} too low after regime adj.")

        # ── Build signal ──────────────────────────────────────────────────────
        entry = price
        sl    = self._atr_stop(entry, atr_val, 1, atr_sl_mult)
        tp    = self._atr_target(entry, atr_val, 1, atr_tp_mult)
        rr    = self._rr_ratio(entry, sl, tp)

        if rr is None or rr < 1.4:
            return no_trade(self.name, asset, "1d", self.horizon,
                            f"R:R {rr} below minimum 1.4.")

        return Signal(
            strategy_name=self.name,
            asset=asset,
            timeframe="1d",
            signal=SignalType.BUY,
            confidence=round(confidence, 3),
            entry_price=round(entry, 4),
            stop_loss=round(sl, 4),
            take_profit=round(tp, 4),
            risk_reward=round(rr, 2),
            horizon=self.horizon,
            reason=(
                f"RSI(2)={rsi2:.1f} [{trigger}] while price > SMA200={sma200:.0f}. "
                f"Regime={regime}."
            ),
            metadata={
                "strategy":          self.name,
                "strategy_type":     "mean_reversion",
                "sizing_mode":       "fixed_allocation",
                "requested_size_pct": position_size,
                "min_cash_reserve_pct": 0.0,
                "trigger":           trigger,
                "rsi2":          round(rsi2, 2),
                "sma200":        round(sma200, 2),
                "atr":           round(atr_val, 4),
                "regime":        regime,
            },
        )
