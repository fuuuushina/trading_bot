"""
src/risk/kelly_sizer.py

Sizing dynamique basé sur le critère de Kelly fractionné.

Formule Kelly :
    f* = (p × b - q) / b
    où :
        p = win_rate (probabilité de gain)
        q = 1 - p
        b = avg_win / avg_loss (ratio gain/perte moyen)

On utilise Quarter-Kelly (f* × 0.25) pour la prudence.

Ajustements :
  - Confidence de la stratégie : ±20%
  - Score ML du filtre signal : ±30%
  - Régime de marché : facteur 0.5–1.2
  - Streak de pertes : réduction progressive

Sortie : fraction du capital à déployer en MARGE (avant levier).
"""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Paramètres par défaut (utilisés avant accumulation de données)
_DEFAULT_WIN_RATE  = 0.48
_DEFAULT_AVG_WIN   = 1.2   # $ moyens gagnés
_DEFAULT_AVG_LOSS  = 1.0   # $ moyens perdus
_QUARTER_KELLY     = 0.25
_MIN_FRACTION      = 0.02  # 2% capital minimum
_MAX_FRACTION      = 0.15  # 15% capital maximum (hard cap prudent)


class KellySizer:
    """
    Calcule la taille de position optimale via Kelly fractionné.

    Usage :
        sizer = KellySizer()
        fraction = sizer.compute(
            capital=2000,
            win_rate=0.52,
            avg_win=1.5,
            avg_loss=1.0,
            confidence=0.65,
            ml_score=0.60,
            regime="bull_trend",
            loss_streak=0,
        )
        margin_usd = capital * fraction
    """

    def __init__(
        self,
        kelly_fraction: float = _QUARTER_KELLY,
        min_fraction:   float = _MIN_FRACTION,
        max_fraction:   float = _MAX_FRACTION,
    ) -> None:
        self.kelly_fraction = kelly_fraction
        self.min_fraction   = min_fraction
        self.max_fraction   = max_fraction

    def compute(
        self,
        capital: float,
        win_rate:   float = _DEFAULT_WIN_RATE,
        avg_win:    float = _DEFAULT_AVG_WIN,
        avg_loss:   float = _DEFAULT_AVG_LOSS,
        confidence: float = 0.5,
        ml_score:   float = 0.5,
        regime:     str   = "range",
        loss_streak: int  = 0,
        asset_type: str   = "forex",
    ) -> float:
        """
        Retourne la fraction de capital à déployer [0, max_fraction].

        Parameters
        ----------
        capital     : Capital total en USD
        win_rate    : Taux de succès historique [0, 1]
        avg_win     : Gain moyen par trade gagnant
        avg_loss    : Perte moyenne par trade perdant (valeur absolue)
        confidence  : Score de confiance de la stratégie [0, 1]
        ml_score    : Score du filtre ML [0, 1]
        regime      : Régime de marché actuel
        loss_streak : Nombre de pertes consécutives
        asset_type  : "forex", "crypto", "commodity", "equity"
        """
        if capital <= 0 or avg_loss <= 0:
            return self.min_fraction

        # 1. Kelly brut
        b = avg_win / avg_loss  # odds ratio
        p = max(0.01, min(0.99, win_rate))
        q = 1 - p
        kelly_raw = (p * b - q) / b  # Kelly formula

        if kelly_raw <= 0:
            # Kelly négatif = edge négatif, on trade au minimum
            logger.debug("Kelly négatif (wr=%.2f, b=%.2f) → fraction minimale", p, b)
            return self.min_fraction

        # 2. Quarter-Kelly de base
        fraction = kelly_raw * self.kelly_fraction

        # 3. Ajustement confiance stratégie (±20%)
        conf_adj = 0.8 + (confidence - 0.5) * 0.8  # 0.4 → 1.2 range
        fraction *= conf_adj

        # 4. Ajustement score ML (±30%)
        ml_adj = 0.7 + (ml_score - 0.5) * 1.2  # 0.1 → 1.3 range
        fraction *= max(0.1, ml_adj)

        # 5. Ajustement régime
        regime_multiplier = _regime_multiplier(regime)
        fraction *= regime_multiplier

        # 6. Réduction sur streak de pertes
        if loss_streak >= 2:
            streak_reduction = max(0.3, 1.0 - (loss_streak - 1) * 0.15)
            fraction *= streak_reduction
            if loss_streak >= 3:
                logger.debug("Streak %d → réduction sizing ×%.2f", loss_streak, streak_reduction)

        # 7. Asset-type cap
        asset_cap = _asset_type_cap(asset_type)
        fraction = min(fraction, asset_cap)

        # 8. Hard limits
        fraction = max(self.min_fraction, min(self.max_fraction, fraction))

        return round(fraction, 4)

    def margin_usd(
        self,
        capital: float,
        **kwargs,
    ) -> float:
        """Retourne directement le montant de marge en USD."""
        return capital * self.compute(capital, **kwargs)

    def log_sizing(
        self,
        asset: str,
        capital: float,
        fraction: float,
        leverage: float = 1.0,
    ) -> None:
        margin = capital * fraction
        notional = margin * leverage
        logger.info(
            "KellySizer [%s] fraction=%.1f%% → margin=$%.0f notional=$%.0f (×%dx)",
            asset, fraction * 100, margin, notional, int(leverage),
        )


def _regime_multiplier(regime: str) -> float:
    """Facteur de sizing selon le régime de marché."""
    multipliers = {
        "bull_trend":         1.20,
        "bear_trend":         0.80,
        "range":              0.90,
        "high_volatility":    0.70,
        "low_volatility":     1.10,
        "panic":              0.40,
        "euphoric":           0.75,  # prudent en euphorie (retournement probable)
        "compression":        0.85,
        "breakout_expansion": 1.10,
        "unknown":            0.85,
    }
    return multipliers.get(regime, 0.85)


def _asset_type_cap(asset_type: str) -> float:
    """Cap de fraction par type d'asset (avant levier)."""
    caps = {
        "forex":     0.12,   # 12% capital max → $240 @ $2000 → $12k notionnel @50x
        "crypto":    0.08,   # crypto plus volatile
        "commodity": 0.10,
        "equity":    0.15,
        "index":     0.15,
    }
    return caps.get(asset_type, 0.10)
