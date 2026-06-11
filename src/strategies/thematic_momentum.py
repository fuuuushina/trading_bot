"""
src/strategies/thematic_momentum.py

Thematic momentum strategy.

Buys stocks from sectors with a strong positive theme score (from LLM/news analysis).
Uses EMA trend + RSI as technical confirmation before entry.
Generates SELL when a held sector's theme score turns negative.

Horizon: SWING (daily bars, hold 3-15 days)
Min bars: 60 (≈3 months daily data for EMA200 to stabilize)
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from src.analysis.sector_universe import SECTOR_UNIVERSE
from src.features.indicators import atr, ema, rsi
from src.strategies.base import (
    BaseStrategy,
    Horizon,
    Signal,
    SignalType,
    no_trade,
)

logger = logging.getLogger(__name__)

_MIN_BARS = 60


class ThematicMomentumStrategy(BaseStrategy):
    """
    Swing strategy driven by sector theme scores.

    Call update_themes() from the market_watcher before each decision cycle
    to provide the latest LLM sector analysis.
    """

    name = "thematic_momentum"
    horizon = Horizon.SWING

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._theme_scores: dict[str, float] = {}   # sector_key → score
        self._theme_picks:  dict[str, list[str]] = {}  # sector_key → top tickers

    def update_themes(self, themes: dict) -> None:
        """Called by market_watcher before the decision cycle."""
        for k, v in themes.items():
            if hasattr(v, "score"):
                self._theme_scores[k] = float(v.score)
                self._theme_picks[k]  = list(getattr(v, "top_picks", []))
            else:
                self._theme_scores[k] = float(v)

    def _get_asset_theme(self, asset: str) -> tuple[Optional[str], float]:
        """Return (sector_key, score) for an asset, or (None, 0.0) if not tracked."""
        for sector_key, info in SECTOR_UNIVERSE.items():
            if asset in info["tickers"]:
                return sector_key, self._theme_scores.get(sector_key, 0.0)
        return None, 0.0

    def _is_top_pick(self, asset: str, sector: str) -> bool:
        """True if asset is in the sector's top_picks list."""
        picks = self._theme_picks.get(sector, [])
        return (not picks) or (asset in picks)

    # ------------------------------------------------------------------ #

    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        cfg = self.config
        timeframe       = cfg.get("timeframe",        "1d")
        min_theme       = cfg.get("min_theme_score",   0.35)
        min_conf        = cfg.get("min_confidence",    0.50)
        atr_sl_mult     = cfg.get("atr_sl_multiplier", 2.0)
        atr_tp_mult     = cfg.get("atr_tp_multiplier", 3.5)
        rsi_max_buy     = cfg.get("rsi_max_buy",       72)
        rsi_min_sell    = cfg.get("rsi_min_sell",       30)

        if len(df) < _MIN_BARS:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Not enough bars ({len(df)} < {_MIN_BARS})")

        sector, theme_score = self._get_asset_theme(asset)
        if sector is None:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            "Asset not in any tracked sector")

        # Fallback: if scores absent OR all neutral (Groq hasn't fired yet / no news),
        # use a mild default so technical filters can still fire.
        if not self._theme_scores or all(abs(v) < 0.05 for v in self._theme_scores.values()):
            theme_score = 0.30

        if abs(theme_score) < min_theme:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Theme score {theme_score:+.2f} below ±{min_theme}")

        close = df["close"]
        ema20_s  = ema(close, 20)
        ema50_s  = ema(close, 50)
        rsi_s    = rsi(close, 14)
        atr_val  = float(atr(df, 14).iloc[-1])

        price    = float(close.iloc[-1])
        curr_e20 = float(ema20_s.iloc[-1])
        curr_e50 = float(ema50_s.iloc[-1])
        rsi_val  = float(rsi_s.iloc[-1])

        if any(pd.isna(v) for v in (curr_e20, curr_e50, rsi_val)) or atr_val <= 0:
            return no_trade(self.name, asset, timeframe, self.horizon, "Indicator NaN")

        if theme_score > 0:
            # BUY: price within 4% of EMA20 (allows entries slightly below EMA in corrections)
            if price < curr_e20 * 0.96:
                return no_trade(self.name, asset, timeframe, self.horizon,
                                f"Price {price:.2f} more than 4% below EMA20 {curr_e20:.2f}")
            if curr_e20 < curr_e50 and regime not in ("range", "compression", "unknown"):
                return no_trade(self.name, asset, timeframe, self.horizon,
                                "EMA20 below EMA50 — uptrend not confirmed")
            if rsi_val > rsi_max_buy:
                return no_trade(self.name, asset, timeframe, self.horizon,
                                f"RSI={rsi_val:.1f} overbought (>{rsi_max_buy})")
            signal_type = SignalType.BUY
            direction = 1

        else:
            # SELL: price below EMA20, RSI not oversold
            if price > curr_e20 * 1.002:
                return no_trade(self.name, asset, timeframe, self.horizon,
                                f"Price {price:.2f} above EMA20 {curr_e20:.2f}")
            if curr_e20 > curr_e50:
                return no_trade(self.name, asset, timeframe, self.horizon,
                                "EMA20 above EMA50 — downtrend not confirmed")
            if rsi_val < rsi_min_sell:
                return no_trade(self.name, asset, timeframe, self.horizon,
                                f"RSI={rsi_val:.1f} oversold (<{rsi_min_sell})")
            signal_type = SignalType.SELL
            direction = -1

        # Confidence scoring
        conf = 0.50
        conf += min(abs(theme_score) * 0.30, 0.25)   # theme strength bonus (max 0.25)
        if direction > 0:
            if price > curr_e20 > curr_e50:
                conf += 0.10   # full EMA alignment
            if 45 <= rsi_val <= 65:
                conf += 0.05   # ideal RSI zone
        else:
            if price < curr_e20 < curr_e50:
                conf += 0.10
            if 35 <= rsi_val <= 55:
                conf += 0.05

        # Prefer top-picks from the LLM analysis
        if self._is_top_pick(asset, sector):
            conf += 0.05

        # Small penalty in panic/high_volatility (swing strategies are riskier)
        if regime in ("panic", "high_volatility"):
            conf -= 0.10

        confidence = round(min(max(conf, 0.0), 0.95), 3)
        if confidence < min_conf:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"Confidence {confidence:.2f} below {min_conf}")

        sl = self._atr_stop(price, atr_val, direction, atr_sl_mult)
        tp = self._atr_target(price, atr_val, direction, atr_tp_mult)
        rr = self._rr_ratio(price, sl, tp)

        if rr is None or rr < 1.5:
            return no_trade(self.name, asset, timeframe, self.horizon,
                            f"R:R {rr} below 1.5")

        return Signal(
            strategy_name=self.name,
            asset=asset,
            timeframe=timeframe,
            signal=signal_type,
            confidence=confidence,
            entry_price=round(price, 4),
            stop_loss=round(sl, 4),
            take_profit=round(tp, 4),
            risk_reward=rr,
            horizon=self.horizon,
            reason=(
                f"Thème [{sector} score={theme_score:+.2f}]; "
                f"RSI={rsi_val:.1f}; EMA20={curr_e20:.2f}; ATR={atr_val:.4f}"
            ),
            metadata={
                "strategy":    self.name,
                "sector":      sector,
                "theme_score": theme_score,
                "rsi":         round(rsi_val, 2),
                "ema20":       round(curr_e20, 4),
                "ema50":       round(curr_e50, 4),
                "regime":      regime,
            },
        )
