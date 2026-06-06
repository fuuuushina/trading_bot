"""
src/features/feature_engine.py

Feature Engine — calcule un vecteur de features standardisé par asset.

À chaque cycle le Market Watcher appelle compute() pour chaque asset.
Les features produites alimentent :
  - le ML Regime Model
  - le LLM Market Analyst (contexte enrichi)
  - les stratégies existantes

Features calculées :
  Prix & variations   : close, change_1h, change_1d, change_5d, change_20d
  Tendance            : ma50, ma200, ema20, price_vs_ma200 (%), slope_ma50
  Momentum            : rsi14, macd_hist, adx14
  Volatilité          : atr14_pct, hv20, bb_width
  Volume              : volume_ratio_20 (vs moyenne 20j)
  Risque              : drawdown_pct (depuis le plus haut 52 semaines)
  Macro (si VIX dispo): vix_level, vix_trend
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.features.indicators import (
    adx,
    atr,
    bollinger_bands,
    drawdown,
    ema,
    historical_volatility,
    macd,
    rsi,
    sma,
    volume_ratio,
)

logger = logging.getLogger(__name__)


@dataclass
class AssetFeatures:
    """Snapshot des features pour un asset à un instant donné."""
    ticker: str
    timestamp: float                        # UNIX timestamp de la dernière barre

    # Prix
    close: float = 0.0
    change_1d: float = 0.0                 # variation % sur 1 jour
    change_5d: float = 0.0                 # variation % sur 5 jours
    change_20d: float = 0.0               # variation % sur 20 jours

    # Tendance
    ma50: float = 0.0
    ma200: float = 0.0
    ema20: float = 0.0
    price_vs_ma200_pct: float = 0.0        # (close - ma200) / ma200
    slope_ma50: float = 0.0               # pente normalisée sur 10 barres

    # Momentum
    rsi14: float = 50.0
    macd_hist: float = 0.0
    adx14: float = 0.0

    # Volatilité
    atr14_pct: float = 0.0                 # ATR / close
    hv20: float = 0.0                      # volatilité historique 20j
    bb_width: float = 0.0                  # largeur Bollinger / prix

    # Volume
    vol_ratio_20: float = 1.0              # volume courant / moyenne 20j

    # Risque
    drawdown_pct: float = 0.0             # depuis le plus haut 52 sem.

    # Macro (optionnel)
    vix_level: Optional[float] = None
    vix_trend: Optional[str] = None       # rising | falling | stable

    # Méta
    data_quality: float = 1.0             # 0 = mauvaise, 1 = parfaite
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "close": round(self.close, 4),
            "change_1d": round(self.change_1d, 4),
            "change_5d": round(self.change_5d, 4),
            "change_20d": round(self.change_20d, 4),
            "price_vs_ma200_pct": round(self.price_vs_ma200_pct, 4),
            "slope_ma50": round(self.slope_ma50, 4),
            "rsi14": round(self.rsi14, 2),
            "macd_hist": round(self.macd_hist, 4),
            "adx14": round(self.adx14, 2),
            "atr14_pct": round(self.atr14_pct, 4),
            "hv20": round(self.hv20, 4),
            "bb_width": round(self.bb_width, 4),
            "vol_ratio_20": round(self.vol_ratio_20, 3),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "vix_level": self.vix_level,
            "vix_trend": self.vix_trend,
        }

    def to_ml_vector(self) -> list[float]:
        """Vecteur numérique pour le ML Regime Model."""
        return [
            self.change_1d,
            self.change_5d,
            self.change_20d,
            self.price_vs_ma200_pct,
            self.slope_ma50,
            self.rsi14 / 100.0,
            self.macd_hist,
            self.adx14 / 100.0,
            self.atr14_pct,
            self.hv20,
            self.bb_width,
            self.vol_ratio_20,
            self.drawdown_pct,
            self.vix_level / 100.0 if self.vix_level else 0.20,
        ]

    @property
    def ml_feature_names(self) -> list[str]:
        return [
            "change_1d", "change_5d", "change_20d",
            "price_vs_ma200_pct", "slope_ma50",
            "rsi14_norm", "macd_hist", "adx14_norm",
            "atr14_pct", "hv20", "bb_width",
            "vol_ratio_20", "drawdown_pct", "vix_norm",
        ]


@dataclass
class MarketSnapshot:
    """Features agrégées pour tous les actifs surveillés."""
    computed_at: float
    assets: dict[str, AssetFeatures] = field(default_factory=dict)
    vix_level: Optional[float] = None

    def get(self, ticker: str) -> Optional[AssetFeatures]:
        return self.assets.get(ticker)

    def benchmark_features(self) -> Optional[AssetFeatures]:
        """Retourne les features de SPY (benchmark principal)."""
        return self.assets.get("SPY") or next(iter(self.assets.values()), None)


class FeatureEngine:
    """
    Calcule les AssetFeatures pour un ensemble d'assets.

    Usage :
        engine = FeatureEngine()
        snapshot = engine.compute(data_map, vix_series=vix_df["close"])
    """

    def compute(
        self,
        data_map: dict[str, pd.DataFrame],
        vix_series: Optional[pd.Series] = None,
    ) -> MarketSnapshot:
        """
        Calcule les features pour tous les DataFrames fournis.

        Parameters
        ----------
        data_map   : {ticker: OHLCV DataFrame enrichi (colonnes lowercase)}
        vix_series : Série de prix VIX optionnelle

        Returns
        -------
        MarketSnapshot
        """
        import time

        snapshot = MarketSnapshot(computed_at=time.time())
        snapshot.vix_level = _latest(vix_series) if vix_series is not None else None

        for ticker, df in data_map.items():
            if df is None or len(df) < 10:
                logger.debug("Skipping %s — not enough data (%d rows)", ticker, len(df) if df is not None else 0)
                continue
            try:
                feat = self._compute_asset(ticker, df, snapshot.vix_level, vix_series)
                snapshot.assets[ticker] = feat
            except Exception as exc:
                logger.warning("FeatureEngine failed for %s: %s", ticker, exc)

        return snapshot

    def _compute_asset(
        self,
        ticker: str,
        df: pd.DataFrame,
        vix_level: Optional[float],
        vix_series: Optional[pd.Series],
    ) -> AssetFeatures:
        import time

        close = df["close"]
        last = close.iloc[-1]
        n = len(df)

        # --- Variations de prix ---
        change_1d  = _pct_change(close, 1)
        change_5d  = _pct_change(close, min(5, n - 1))
        change_20d = _pct_change(close, min(20, n - 1))

        # --- Tendance ---
        ma50_s  = sma(close, min(50, n))
        ma200_s = sma(close, min(200, n))
        ema20_s = ema(close, min(20, n))

        ma50_val  = float(ma50_s.iloc[-1])  if not ma50_s.empty  else last
        ma200_val = float(ma200_s.iloc[-1]) if not ma200_s.empty else last
        ema20_val = float(ema20_s.iloc[-1]) if not ema20_s.empty else last

        price_vs_ma200 = (last - ma200_val) / ma200_val if ma200_val > 0 else 0.0

        # Pente MA50 : variation sur 10 barres, normalisée par price
        window = 10
        if len(ma50_s.dropna()) >= window:
            slope_raw = (ma50_s.iloc[-1] - ma50_s.iloc[-window]) / ma50_s.iloc[-window]
        else:
            slope_raw = 0.0

        # --- Momentum ---
        rsi14_s = rsi(close, min(14, n - 1))
        rsi14_val = float(rsi14_s.iloc[-1]) if not rsi14_s.dropna().empty else 50.0

        macd_df = macd(close)
        macd_hist_val = float(macd_df["histogram"].iloc[-1]) if not macd_df.empty else 0.0

        adx14_val = 0.0
        if all(c in df.columns for c in ("high", "low", "close")) and n >= 15:
            adx_s = adx(df, min(14, n - 2))
            if not adx_s.dropna().empty:
                adx14_val = float(adx_s.iloc[-1])

        # --- Volatilité ---
        atr14_pct = 0.0
        if all(c in df.columns for c in ("high", "low", "close")) and n >= 15:
            atr14_s = atr(df, min(14, n - 2))
            if not atr14_s.dropna().empty:
                atr14_pct = float(atr14_s.iloc[-1]) / last if last > 0 else 0.0

        hv20_val = 0.0
        if n >= 21:
            hv_s = historical_volatility(close, min(20, n - 1))
            if not hv_s.dropna().empty:
                hv20_val = float(hv_s.iloc[-1])

        bb_width_val = 0.0
        if n >= 20:
            bb = bollinger_bands(close, min(20, n))
            if "upper" in bb.columns and "lower" in bb.columns:
                width = bb["upper"].iloc[-1] - bb["lower"].iloc[-1]
                bb_width_val = width / last if last > 0 else 0.0

        # --- Volume ---
        vol_ratio_val = 1.0
        if "volume" in df.columns and n >= 20:
            vr = volume_ratio(df["volume"], min(20, n))
            if not vr.dropna().empty:
                vol_ratio_val = float(vr.iloc[-1])

        # --- Drawdown 52 semaines ---
        dd_val = 0.0
        lookback = min(252, n)
        if lookback > 1:
            dd_s = drawdown(close.iloc[-lookback:])
            dd_val = float(dd_s.iloc[-1])

        # --- VIX trend ---
        vix_trend = None
        if vix_series is not None and len(vix_series) >= 5:
            vix_recent = vix_series.iloc[-5:]
            vix_trend = "rising" if vix_recent.iloc[-1] > vix_recent.iloc[0] * 1.05 else (
                "falling" if vix_recent.iloc[-1] < vix_recent.iloc[0] * 0.95 else "stable"
            )

        # --- Qualité des données ---
        expected = 200
        data_quality = min(n / expected, 1.0)

        return AssetFeatures(
            ticker=ticker,
            timestamp=time.time(),
            close=round(last, 4),
            change_1d=round(change_1d, 4),
            change_5d=round(change_5d, 4),
            change_20d=round(change_20d, 4),
            ma50=round(ma50_val, 4),
            ma200=round(ma200_val, 4),
            ema20=round(ema20_val, 4),
            price_vs_ma200_pct=round(price_vs_ma200, 4),
            slope_ma50=round(slope_raw, 4),
            rsi14=round(rsi14_val, 2),
            macd_hist=round(macd_hist_val, 6),
            adx14=round(adx14_val, 2),
            atr14_pct=round(atr14_pct, 4),
            hv20=round(hv20_val, 4),
            bb_width=round(bb_width_val, 4),
            vol_ratio_20=round(vol_ratio_val, 3),
            drawdown_pct=round(dd_val, 4),
            vix_level=vix_level,
            vix_trend=vix_trend,
            data_quality=round(data_quality, 2),
        )


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _pct_change(series: pd.Series, periods: int) -> float:
    if len(series) <= periods or periods <= 0:
        return 0.0
    old = series.iloc[-(periods + 1)]
    new = series.iloc[-1]
    return (new - old) / old if old != 0 else 0.0


def _latest(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    val = series.dropna()
    return float(val.iloc[-1]) if not val.empty else None
