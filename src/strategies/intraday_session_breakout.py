"""
src/strategies/intraday_session_breakout.py

Session breakout for EUR/USD 5-minute bars.

EUR/USD liquidity peaks at London (07:00-09:30 UTC) and NY (13:30-16:00 UTC) opens.
The strategy computes a pre-session consolidation range (last `consolidation_bars`
before the session window) and trades the initial breakout.

Active windows (UTC):
  London open:  07:00 – 09:30
  NY open:      13:30 – 16:00

Logic:
  - Take the high/low over `consolidation_bars` (default 24 = 2h) immediately
    before the session window.
  - BUY  if current close > consolidation high + ATR × buffer_mult
  - SELL if current close < consolidation low  - ATR × buffer_mult
  - Only fires when the current bar falls within an active session window.

Horizon: INTRADAY | Timeframe: 5m
"""
from __future__ import annotations

from datetime import timezone

import pandas as pd

from src.features.indicators import atr, ema
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)

_MIN_BARS = 40
# UTC hours for session windows
_SESSIONS = {
    "asian":  (0, 0, 3, 0),     # Tokyo/Asian open
    "london": (7, 0, 9, 30),    # London open
    "ny":     (13, 30, 16, 0),  # New York open
}


def _in_session(ts: pd.Timestamp) -> bool:
    """Return True if ts falls within London or NY open window (UTC)."""
    try:
        utc_ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts
    except Exception:
        utc_ts = ts
    h, m = utc_ts.hour, utc_ts.minute
    total_min = h * 60 + m
    for (oh, om, ch, cm) in _SESSIONS.values():
        if oh * 60 + om <= total_min < ch * 60 + cm:
            return True
    return False


class IntradaySessionBreakoutStrategy(BaseStrategy):
    name = "intraday_session_breakout"
    horizon = Horizon.INTRADAY

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        cfg = self.config
        timeframe           = cfg.get("timeframe",           "5m")
        consolidation_bars  = cfg.get("consolidation_bars",  24)   # 2h
        buffer_mult         = cfg.get("atr_buffer_mult",     0.3)   # breakout buffer
        atr_sl_mult         = cfg.get("atr_multiplier_sl",   1.8)
        atr_tp_mult         = cfg.get("atr_multiplier_tp",   3.0)
        min_conf            = cfg.get("min_confidence",      0.55)

        if len(df) < _MIN_BARS:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Not enough bars ({len(df)} < {_MIN_BARS})")

        # Check session window
        last_idx = df.index[-1]
        if not isinstance(last_idx, pd.Timestamp):
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Index is not DatetimeIndex")

        if not _in_session(last_idx):
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Outside London/NY session windows")

        # ATR on full history
        atr_val = float(atr(df, 14).iloc[-1])
        if atr_val <= 0 or pd.isna(atr_val):
            return no_trade(self.name, asset, timeframe, self.horizon, "Zero ATR")

        # Consolidation range = last `consolidation_bars` BEFORE current bar
        consol_window = df.iloc[-(consolidation_bars + 1):-1]
        if len(consol_window) < consolidation_bars // 2:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Not enough consolidation data")

        consol_high = float(consol_window["high"].max())
        consol_low  = float(consol_window["low"].min())
        consol_range = consol_high - consol_low

        # Range sanity: skip if already very wide (already broken out)
        range_pct = consol_range / consol_high if consol_high > 0 else 0
        if range_pct > 0.006:   # > 60 pips for EURUSD — too chaotic
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Consolidation range too wide ({range_pct:.3%})")
        if range_pct < 0.0003:  # < 3 pips — no range to break
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Consolidation range too narrow ({range_pct:.3%})")

        c = float(df["close"].iloc[-1])
        breakout_up   = c > consol_high + buffer_mult * atr_val
        breakout_down = c < consol_low  - buffer_mult * atr_val

        if not breakout_up and not breakout_down:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"No breakout (H={consol_high:.5f} L={consol_low:.5f})")

        # Regime filter — avoid in panic (whipsaw)
        if regime == "panic":
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Panic regime — skipping breakout")

        direction   = 1 if breakout_up else -1
        signal_type = SignalType.BUY if breakout_up else SignalType.SELL

        entry = c
        sl    = self._atr_stop(entry, atr_val, direction, atr_sl_mult)
        tp    = self._atr_target(entry, atr_val, direction, atr_tp_mult)
        rr    = self._rr_ratio(entry, sl, tp)

        if rr is None or rr < 1.5:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr} below 1.5")

        # Confidence
        score = 0.55
        # Breakout magnitude
        if breakout_up:
            excess = (c - consol_high) / atr_val
        else:
            excess = (consol_low - c) / atr_val
        if excess > 0.5:
            score += 0.10
        if excess > 1.0:
            score += 0.10
        # Bonus for trend-aligned regimes
        if (breakout_up and regime == "bull_trend") or (breakout_down and regime == "bear_trend"):
            score += 0.10
        if regime in ("breakout_expansion",):
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
                f"Session breakout {'UP' if breakout_up else 'DOWN'}; "
                f"range=[{consol_low:.5f},{consol_high:.5f}] "
                f"({range_pct:.3%}); excess={excess:.2f}×ATR"
            ),
            metadata={
                "strategy":     self.name,
                "consol_high":  round(consol_high, 6),
                "consol_low":   round(consol_low, 6),
                "range_pct":    round(range_pct, 5),
                "atr":          round(atr_val, 6),
                "regime":       regime,
            },
        )
