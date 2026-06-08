"""
src/features/regime_detector.py

Classifies the current market into one of 9 regimes using deterministic
rules based on technical indicators. All rules are explicit and auditable.

Regimes:
  bull_trend | bear_trend | range | high_volatility | low_volatility |
  panic | euphoric | compression | breakout_expansion
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from src.features.indicators import (
    atr,
    bollinger_bands,
    drawdown,
    ema,
    historical_volatility,
    rsi,
    sma,
    volume_ratio,
)


class MarketRegime(str, Enum):
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    RANGE = "range"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    PANIC = "panic"
    EUPHORIC = "euphoric"
    COMPRESSION = "compression"
    BREAKOUT_EXPANSION = "breakout_expansion"
    UNKNOWN = "unknown"


@dataclass
class RegimeResult:
    regime: MarketRegime
    confidence: float                   # 0–1
    score: int = 0
    market_label: str = "unknown"
    sub_signals: dict[str, str | float] = field(default_factory=dict)
    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 3),
            "score": self.score,
            "market_label": self.market_label,
            "sub_signals": self.sub_signals,
            "explanation": self.explanation,
        }


class MarketRegimeDetector:
    """
    Stateless detector — call detect() with a feature-enriched DataFrame.
    The DataFrame must have at minimum: close, high, low, volume.
    """

    # Thresholds — could be moved to config
    BULL_EMA_CONDITION = True   # close > ema20 > ema50 > ema200
    HIST_VOL_HIGH = 0.25        # Annualised vol > 25%
    HIST_VOL_LOW = 0.10         # Annualised vol < 10%
    ADX_TREND_MIN = 25
    RSI_PANIC = 25
    RSI_EUPHORIC = 78
    BB_SQUEEZE_WIDTH = 0.025    # BB bandwidth / price < 2.5%
    DRAWDOWN_PANIC = -0.10      # -10% from peak triggers panic flag
    VOL_SPIKE_RATIO = 2.0       # Volume 2x average

    def detect(
        self,
        df: pd.DataFrame,
        vix: Optional[pd.Series] = None,
    ) -> RegimeResult:
        """
        Main detection method.
        df: daily OHLCV DataFrame (last row = current state).
        vix: optional VIX series aligned with df index.
        """
        if len(df) < 200:
            return RegimeResult(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                explanation="Insufficient data (need 200+ bars).",
            )

        signals: dict[str, str | float] = {}
        votes: dict[MarketRegime, float] = {r: 0.0 for r in MarketRegime}

        close = df["close"]
        last = df.iloc[-1]

        # ---- EMA trend signals ----
        e20 = ema(close, 20).iloc[-1]
        e50 = ema(close, 50).iloc[-1]
        e200 = ema(close, 200).iloc[-1]
        c = close.iloc[-1]

        above_ema20 = c > e20
        above_ema50 = c > e50
        above_ema200 = c > e200
        ema_stack_bull = e20 > e50 > e200
        ema_stack_bear = e20 < e50 < e200

        signals["above_ema20"] = above_ema20
        signals["above_ema200"] = above_ema200
        signals["ema_stack_bull"] = ema_stack_bull
        signals["ema_stack_bear"] = ema_stack_bear

        if ema_stack_bull and above_ema20 and above_ema200:
            votes[MarketRegime.BULL_TREND] += 3
        if ema_stack_bear and not above_ema20 and not above_ema200:
            votes[MarketRegime.BEAR_TREND] += 3
        if abs(c - e20) / e20 < 0.03 and not ema_stack_bull and not ema_stack_bear:
            votes[MarketRegime.RANGE] += 2

        # ---- RSI ----
        rsi_val = rsi(close, 14).iloc[-1]
        signals["rsi_14"] = round(rsi_val, 2)

        if rsi_val < self.RSI_PANIC:
            votes[MarketRegime.PANIC] += 3
        elif rsi_val > self.RSI_EUPHORIC:
            votes[MarketRegime.EUPHORIC] += 2
        elif 45 <= rsi_val <= 55:
            votes[MarketRegime.RANGE] += 1

        # ---- Historical volatility ----
        hist_vol = historical_volatility(close, 20).iloc[-1]
        signals["hist_vol_20"] = round(float(hist_vol), 4)

        if hist_vol > self.HIST_VOL_HIGH:
            votes[MarketRegime.HIGH_VOLATILITY] += 2
        elif hist_vol < self.HIST_VOL_LOW:
            votes[MarketRegime.LOW_VOLATILITY] += 2

        # ---- VIX (if available) ----
        if vix is not None and len(vix) > 0:
            last_vix = vix.iloc[-1]
            signals["vix"] = round(float(last_vix), 2)
            if last_vix > 35:
                votes[MarketRegime.PANIC] += 3
            elif last_vix > 25:
                votes[MarketRegime.HIGH_VOLATILITY] += 2
            elif last_vix < 14:
                votes[MarketRegime.LOW_VOLATILITY] += 1

        # ---- ATR trend ----
        atr_vals = atr(df, 14)
        atr_sma = atr_vals.rolling(20).mean().iloc[-1]
        atr_now = atr_vals.iloc[-1]
        atr_ratio = atr_now / (atr_sma + 1e-10)
        signals["atr_ratio"] = round(float(atr_ratio), 3)

        if atr_ratio > 1.5:
            votes[MarketRegime.HIGH_VOLATILITY] += 1
        elif atr_ratio < 0.7:
            votes[MarketRegime.COMPRESSION] += 2

        # ---- Bollinger Bands squeeze ----
        bb = bollinger_bands(close, 20, 2.0)
        bb_width = bb["bb_bandwidth"].iloc[-1]
        signals["bb_bandwidth"] = round(float(bb_width), 4)

        if bb_width < self.BB_SQUEEZE_WIDTH:
            votes[MarketRegime.COMPRESSION] += 3

        # Previous bandwidth (expansion detection)
        prev_bb_width = bb["bb_bandwidth"].iloc[-6:-1].mean() if len(bb) > 6 else bb_width
        if bb_width > prev_bb_width * 1.4:
            votes[MarketRegime.BREAKOUT_EXPANSION] += 2

        # ---- Volume ----
        if "volume" in df.columns:
            vol_r = volume_ratio(df, 20).iloc[-1]
            signals["volume_ratio"] = round(float(vol_r), 3)
            if vol_r > self.VOL_SPIKE_RATIO:
                if votes[MarketRegime.PANIC] > 0:
                    votes[MarketRegime.PANIC] += 2
                elif votes[MarketRegime.BREAKOUT_EXPANSION] > 0:
                    votes[MarketRegime.BREAKOUT_EXPANSION] += 2

        # ---- Drawdown ----
        dd = drawdown(close).iloc[-1]
        signals["drawdown"] = round(float(dd), 4)

        if dd < self.DRAWDOWN_PANIC:
            votes[MarketRegime.PANIC] += 2
        if dd < -0.20:
            votes[MarketRegime.BEAR_TREND] += 2

        # ---- Momentum check ----
        ret_20 = (c - close.iloc[-21]) / close.iloc[-21] if len(close) > 21 else 0.0
        signals["ret_20d"] = round(float(ret_20), 4)

        if ret_20 > 0.08:
            votes[MarketRegime.EUPHORIC] += 1
        elif ret_20 < -0.08:
            votes[MarketRegime.PANIC] += 1

        # ---- Calculate market regime score ----
        score, label = self._compute_market_score(close, e20, e50, e200, vix)
        signals["market_score"] = score
        signals["market_label"] = label

        # ---- Determine winner ----
        votes.pop(MarketRegime.UNKNOWN, None)
        best_regime = max(votes, key=lambda r: votes[r])
        best_score = votes[best_regime]
        total_score = sum(votes.values()) + 1e-10
        confidence = min(best_score / total_score * 2, 1.0)  # normalise

        # Safety: zero votes → RANGE as neutral fallback
        if best_score == 0:
            return RegimeResult(
                regime=MarketRegime.RANGE,
                confidence=0.2,
                score=score,
                market_label=label,
                sub_signals=signals,
                explanation="No clear regime signal — defaulting to range.",
            )

        explanation = self._build_explanation(best_regime, signals)

        return RegimeResult(
            regime=best_regime,
            confidence=round(confidence, 3),
            score=score,
            market_label=label,
            sub_signals=signals,
            explanation=explanation,
        )

    @staticmethod
    def _build_explanation(regime: MarketRegime, signals: dict) -> str:
        parts = []
        regime_descriptions = {
            MarketRegime.BULL_TREND: "EMA stack bullish, price above key averages.",
            MarketRegime.BEAR_TREND: "EMA stack bearish, price below key averages.",
            MarketRegime.RANGE: "Price near EMA20, RSI neutral, no clear trend.",
            MarketRegime.HIGH_VOLATILITY: "Elevated ATR and historical volatility.",
            MarketRegime.LOW_VOLATILITY: "Compressed ATR, low historical volatility.",
            MarketRegime.PANIC: "RSI oversold, large drawdown, possible volume spike.",
            MarketRegime.EUPHORIC: "RSI overbought, strong recent gains.",
            MarketRegime.COMPRESSION: "BB squeeze detected, ATR contracted.",
            MarketRegime.BREAKOUT_EXPANSION: "BB bandwidth expanding, volume confirming.",
        }
        base = regime_descriptions.get(regime, "")
        parts.append(base)
        parts.append(
            f"RSI={signals.get('rsi_14', '?')}, "
            f"vol={signals.get('hist_vol_20', '?'):.1%}, "
            f"dd={signals.get('drawdown', 0):.1%}"
            if isinstance(signals.get("hist_vol_20"), float)
            else ""
        )
        return " ".join(p for p in parts if p)

    def _compute_market_score(
        self,
        close: pd.Series,
        e20: float,
        e50: float,
        e200: float,
        vix: Optional[pd.Series],
    ) -> tuple[int, str]:
        score = 0

        if close.iloc[-1] > e200:
            score += 1
        if close.iloc[-1] > e50:
            score += 1
        if close.iloc[-1] > e20:
            score += 1

        if len(close) > 63:
            ret_3m = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]
            if ret_3m > 0:
                score += 1
        if len(close) > 126:
            ret_6m = (close.iloc[-1] - close.iloc[-126]) / close.iloc[-126]
            if ret_6m > 0:
                score += 1

        if vix is not None and len(vix) >= 200:
            vix_200 = vix.rolling(200).mean().iloc[-1]
            if not np.isnan(vix_200) and vix.iloc[-1] < vix_200:
                # Add a small confidence boost when volatility is below its long-term average.
                score = min(score + 1, 5)

        if score <= 1:
            label = "bear"
        elif score <= 3:
            label = "neutral"
        else:
            label = "bull"

        return score, label
