"""
src/strategies/intraday_bollinger_rsi.py

Bollinger Band + RSI mean reversion for EUR/USD 5-minute bars.

Entry rules:
  BUY:  close <= lower band  AND  RSI(14) < 35  AND  RSI rising (last 2 bars)
  SELL: close >= upper band  AND  RSI(14) > 65  AND  RSI falling (last 2 bars)

TP:  Middle band (basis SMA)
SL:  ATR(14) × 1.5 beyond the band touch

Works best in ranging / low_volatility regime. Penalised in trending regimes.
Horizon: INTRADAY | Timeframe: 5m
"""
from __future__ import annotations

import pandas as pd

from src.features.indicators import atr, ema, rsi, sma
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)

_MIN_BARS = 30


def _bollinger(close: pd.Series, period: int = 20, mult: float = 2.0):
    """Returns (upper, basis, lower) as pd.Series."""
    basis = sma(close, period)
    std   = close.rolling(period).std(ddof=0)
    return basis + mult * std, basis, basis - mult * std


class IntradayBollingerRSIStrategy(BaseStrategy):
    name = "intraday_bollinger_rsi"
    horizon = Horizon.INTRADAY

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        cfg = self.config
        timeframe    = cfg.get("timeframe",       "5m")
        bb_period    = cfg.get("bb_period",        20)
        bb_mult      = cfg.get("bb_mult",          2.0)
        rsi_period   = cfg.get("rsi_period",       14)
        rsi_ob       = cfg.get("rsi_overbought",   62)
        rsi_os       = cfg.get("rsi_oversold",     38)
        atr_sl_mult  = cfg.get("atr_multiplier_sl", 1.5)
        atr_tp_mult  = cfg.get("atr_multiplier_tp", 4.5)
        min_conf     = cfg.get("min_confidence",   0.55)

        if len(df) < _MIN_BARS:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Not enough bars ({len(df)} < {_MIN_BARS})")

        # Penalise strong trends — mean reversion underperforms
        if regime in ("bull_trend", "bear_trend", "breakout_expansion"):
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Regime {regime} unfavourable for mean reversion")

        close = df["close"]
        upper, basis, lower = _bollinger(close, bb_period, bb_mult)
        rsi_vals = rsi(close, rsi_period)
        atr_val  = float(atr(df, 14).iloc[-1])

        c        = float(close.iloc[-1])
        up_val   = float(upper.iloc[-1])
        bas_val  = float(basis.iloc[-1])
        low_val  = float(lower.iloc[-1])
        rsi_cur  = float(rsi_vals.iloc[-1])
        rsi_prev = float(rsi_vals.iloc[-2])

        if any(pd.isna(v) for v in [up_val, bas_val, low_val, rsi_cur, rsi_prev]):
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Indicator NaN")
        if atr_val <= 0 or bas_val <= 0:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Zero ATR or basis")

        # Band width as % of price — skip very narrow bands (low volatility, risky)
        band_width_pct = (up_val - low_val) / bas_val
        if band_width_pct < 0.0004:   # < 4 pips for EURUSD
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Band too narrow ({band_width_pct:.4%})")

        buy_signal  = c <= low_val and rsi_cur < rsi_os and rsi_cur > rsi_prev
        sell_signal = c >= up_val  and rsi_cur > rsi_ob and rsi_cur < rsi_prev

        if not buy_signal and not sell_signal:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "No BB+RSI condition met")

        direction   = 1 if buy_signal else -1
        signal_type = SignalType.BUY if buy_signal else SignalType.SELL

        entry = c
        # TP = ATR × multiplicateur (R:R explicite, plus fiable que la bande médiane)
        tp = self._atr_target(entry, atr_val, direction, atr_tp_mult)
        # SL = beyond the band by ATR×sl_mult
        sl = entry - direction * atr_sl_mult * atr_val
        rr = self._rr_ratio(entry, sl, tp)

        if rr is None or rr < 1.5:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr} below 1.2")

        # Confidence
        score = 0.50
        # Deeper into the band = stronger signal
        deviation = abs(c - bas_val) / (up_val - bas_val) if (up_val - bas_val) > 0 else 0
        if deviation > 1.0:    # price beyond 1 std from basis
            score += 0.10
        # RSI extreme
        if (buy_signal and rsi_cur < 25) or (sell_signal and rsi_cur > 75):
            score += 0.15
        # Regime bonus for range
        if regime in ("range", "low_volatility", "compression"):
            score += 0.10

        score = round(min(score, 1.0), 3)
        if score < min_conf:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Confidence {score:.2f} below {min_conf}")

        return Signal(
            strategy_name=self.name,
            asset=asset,
            timeframe=timeframe,
            signal=signal_type,
            confidence=score,
            entry_price=round(entry, 6),
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            risk_reward=rr,
            horizon=self.horizon,
            reason=(
                f"BB({'BUY: price<=lower' if buy_signal else 'SELL: price>=upper'}); "
                f"RSI={rsi_cur:.1f}; BB_width={band_width_pct:.3%}; ATR={atr_val:.6f}"
            ),
            metadata={
                "strategy":   self.name,
                "bb_upper":   round(up_val, 6),
                "bb_basis":   round(bas_val, 6),
                "bb_lower":   round(low_val, 6),
                "rsi":        round(rsi_cur, 2),
                "band_width": round(band_width_pct, 5),
                "atr":        round(atr_val, 6),
                "regime":     regime,
            },
        )
