"""
src/strategies/volatility_compression.py

Volatility compression / Bollinger Band squeeze breakout.
Detects periods of low volatility (compression) and positions for the
impending expansion move.
"""
from __future__ import annotations

import pandas as pd

from src.features.indicators import (
    atr,
    bollinger_bands,
    keltner_channels,
    volume_ratio,
    ema,
)
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)


class VolatilityCompressionStrategy(BaseStrategy):
    name = "volatility_compression"
    horizon = Horizon.SWING

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        cfg = self.config
        timeframe = cfg.get("timeframe", "1d")
        squeeze_threshold = cfg.get("bb_squeeze_threshold", 0.02)
        compression_days = cfg.get("compression_days", 10)
        min_conf = cfg.get("min_confidence", 0.65)
        atr_sl_mult = cfg.get("atr_multiplier_sl", 1.5)
        atr_tp_mult = cfg.get("atr_multiplier_tp", 3.5)

        if len(df) < max(compression_days + 20, 50):
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Insufficient data.")

        close = df["close"]
        c = float(close.iloc[-1])

        # ---- Bollinger Band squeeze detection ----
        bb = bollinger_bands(close, 20, 2.0)
        bb_width = float(bb["bb_bandwidth"].iloc[-1])
        bb_width_series = bb["bb_bandwidth"].iloc[-compression_days:]
        is_squeezed = (bb_width_series < squeeze_threshold).all()

        if not is_squeezed:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"No BB squeeze. Width={bb_width:.4f} (thresh={squeeze_threshold})")

        # ---- Keltner Channel confirmation (BB inside KC = TTM Squeeze) ----
        kc = keltner_channels(df, 20, 2.0)
        bb_inside_kc = (
            float(bb["bb_upper"].iloc[-1]) < float(kc["kc_upper"].iloc[-1])
            and float(bb["bb_lower"].iloc[-1]) > float(kc["kc_lower"].iloc[-1])
        )

        # ---- Expansion detection — has the move started? ----
        prev_width = float(bb["bb_bandwidth"].iloc[-compression_days - 1])
        expanding_now = bb_width > prev_width * 1.1

        # ---- Direction bias via EMA slope ----
        e20_now = float(ema(close, 20).iloc[-1])
        e20_prev = float(ema(close, 20).iloc[-5])
        ema_slope_up = e20_now > e20_prev

        # ---- Volume check ----
        vol_r = float(volume_ratio(df, 20).iloc[-1])
        vol_expanding = vol_r > 1.2

        # ---- ATR for sizing ----
        atr_val = float(atr(df, 14).iloc[-1])

        # ---- Score ----
        score = 0.30
        reasons = []

        if is_squeezed:
            score += 0.25
            reasons.append(f"BB squeeze confirmed ({compression_days}d)")
        if bb_inside_kc:
            score += 0.15
            reasons.append("BB inside KC (TTM squeeze)")
        if expanding_now:
            score += 0.15
            reasons.append("BB starting to expand")
        if vol_expanding:
            score += 0.10
            reasons.append(f"Volume ratio={vol_r:.2f}")

        score = round(min(score, 1.0), 3)

        if score < min_conf:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Confidence {score:.2f} < {min_conf}.")

        # Direction from EMA slope
        direction = 1 if ema_slope_up else -1
        signal_type = SignalType.BUY if direction == 1 else SignalType.SELL

        entry = c
        sl = self._atr_stop(entry, atr_val, direction, atr_sl_mult)
        tp = self._atr_target(entry, atr_val, direction, atr_tp_mult)
        rr = self._rr_ratio(entry, sl, tp)

        if rr is None or rr < 1.5:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr} below minimum.")

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
                "bb_width": round(bb_width, 5),
                "bb_inside_kc": bb_inside_kc,
                "expanding": expanding_now,
                "volume_ratio": round(vol_r, 3),
                "ema_slope_up": ema_slope_up,
                "atr": round(atr_val, 4),
                "regime": regime,
            },
        )
