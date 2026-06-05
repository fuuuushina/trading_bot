"""
src/strategies/breakout.py

N-day high/low breakout with volume and consolidation confirmation.
"""
from __future__ import annotations

import pandas as pd

from src.features.indicators import atr, bollinger_bands, volume_ratio
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)


class BreakoutStrategy(BaseStrategy):
    name = "breakout"
    horizon = Horizon.SWING

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        cfg = self.config
        timeframe = cfg.get("timeframe", "1d")
        lookback = cfg.get("lookback_period", 20)
        vol_confirm = cfg.get("volume_confirm_ratio", 1.5)
        min_atr_pct = cfg.get("min_atr_pct", 0.005)
        consol_days = cfg.get("consolidation_days", 5)
        min_conf = cfg.get("min_confidence", 0.65)
        atr_sl_mult = cfg.get("atr_multiplier_sl", 1.0)
        atr_tp_mult = cfg.get("atr_multiplier_tp", 3.0)

        if len(df) < lookback + consol_days + 5:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Insufficient data for breakout.")

        close = df["close"]
        high = df["high"]
        low = df["low"]
        c = float(close.iloc[-1])
        h = float(high.iloc[-1])
        lo = float(low.iloc[-1])

        # N-day high/low (excluding current bar)
        window_high = float(high.iloc[-(lookback + 1):-1].max())
        window_low = float(low.iloc[-(lookback + 1):-1].min())

        atr_val = float(atr(df, 14).iloc[-1])
        atr_pct = atr_val / c

        if atr_pct < min_atr_pct:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"ATR {atr_pct:.4f} too small — low volatility trap.")

        # ---- Breakout detection ----
        breakout_up = c > window_high
        breakout_down = c < window_low

        if not breakout_up and not breakout_down:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"No breakout. Price={c:.2f}, Range=[{window_low:.2f}, {window_high:.2f}]")

        # ---- Consolidation check ----
        # The N days before breakout should have tightening range
        pre_break = df.iloc[-(consol_days + 2):-1]
        pre_high = float((pre_break["high"] - pre_break["low"]).mean())
        pre_atr = float(atr(df.iloc[:-(1)], 14).iloc[-consol_days:].mean())
        consolidating = pre_high < pre_atr * 1.5

        # ---- Volume confirmation ----
        vol_r = float(volume_ratio(df, 20).iloc[-1])
        volume_confirmed = vol_r >= vol_confirm

        # ---- BB width (prefer breakout from squeeze) ----
        bb = bollinger_bands(close, 20, 2.0)
        bb_prev_width = float(bb["bb_bandwidth"].iloc[-6])
        bb_curr_width = float(bb["bb_bandwidth"].iloc[-1])
        expanding = bb_curr_width > bb_prev_width

        # ---- Score ----
        score = 0.30
        reasons = []

        if breakout_up or breakout_down:
            score += 0.25
            lvl = window_high if breakout_up else window_low
            reasons.append(f"Price broke {lookback}d {'high' if breakout_up else 'low'} @ {lvl:.2f}")

        if consolidating:
            score += 0.20
            reasons.append("Pre-breakout consolidation confirmed")

        if volume_confirmed:
            score += 0.20
            reasons.append(f"Volume ratio={vol_r:.2f}")

        if expanding:
            score += 0.05
            reasons.append("BB expanding")

        score = round(min(score, 1.0), 3)

        if score < min_conf:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Confidence {score:.2f} < {min_conf}.")

        # False breakout filter: require close near the extreme
        if breakout_up and c < window_high * 1.001:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Marginal breakout — potential false signal.")

        direction = 1 if breakout_up else -1
        signal_type = SignalType.BUY if breakout_up else SignalType.SELL

        entry = c
        sl = self._atr_stop(entry, atr_val, direction, atr_sl_mult)
        tp = self._atr_target(entry, atr_val, direction, atr_tp_mult)
        rr = self._rr_ratio(entry, sl, tp)

        if rr is None or rr < 1.5:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr} below 1.5.")

        return Signal(
            strategy_name=self.name,
            asset=asset,
            timeframe=timeframe,
            signal=signal_type,
            confidence=score,
            entry_price=round(entry, 4),
            stop_loss=round(sl, 4),
            take_profit=round(tp, 4),
            risk_reward=rr,
            horizon=self.horizon,
            reason="; ".join(reasons),
            metadata={
                "window_high": round(window_high, 4),
                "window_low": round(window_low, 4),
                "volume_ratio": round(vol_r, 3),
                "atr": round(atr_val, 4),
                "consolidating": consolidating,
                "bb_expanding": expanding,
                "regime": regime,
            },
        )
