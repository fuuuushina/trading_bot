"""
src/profile/client_profile.py

Profil client — définit pour qui le bot trade.

Le profil pilote l'IA stratégique qui produira une allocation dynamique
adaptée aux objectifs, à l'horizon et à la tolérance au drawdown.

Les préférences (etf / stocks / intraday / crypto) activent ou désactivent
les catégories de bots dans le Decision Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskTolerance(str, Enum):
    CONSERVATIVE  = "conservative"   # Max 10% drawdown
    MODERATE      = "moderate"       # Max 20% drawdown
    AGGRESSIVE    = "aggressive"     # Max 35% drawdown
    SPECULATIVE   = "speculative"    # Max 50% drawdown


class Objective(str, Enum):
    INCOME             = "income"            # Revenus réguliers, dividendes
    GROWTH             = "growth"            # Croissance du capital
    WEALTH_PRESERVATION = "wealth_preservation"  # Préserver le capital
    BALANCED           = "balanced"          # Équilibre croissance/sécurité


@dataclass
class AssetPreferences:
    etf: bool      = True
    stocks: bool   = True
    intraday: bool = False
    crypto: bool   = False

    def to_dict(self) -> dict:
        return {
            "etf": self.etf,
            "stocks": self.stocks,
            "intraday": self.intraday,
            "crypto": self.crypto,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AssetPreferences":
        return cls(
            etf=d.get("etf", True),
            stocks=d.get("stocks", True),
            intraday=d.get("intraday", False),
            crypto=d.get("crypto", False),
        )


@dataclass
class ClientProfile:
    """
    Profil complet d'un client.

    Tous les paramètres ont des valeurs par défaut raisonnables (profil modéré).
    """
    name: str                              = "default"
    capital: float                         = 10_000.0
    risk_tolerance: RiskTolerance          = RiskTolerance.MODERATE
    objective: Objective                   = Objective.GROWTH
    horizon_years: int                     = 5
    max_drawdown_tolerance: float          = 0.20     # 20% max acceptable
    target_annual_return: Optional[float]  = None     # None = calculé automatiquement
    age: Optional[int]                     = None
    preferences: AssetPreferences         = field(default_factory=AssetPreferences)

    def __post_init__(self) -> None:
        # Calcul automatique du rendement cible si non fourni
        if self.target_annual_return is None:
            self.target_annual_return = _default_target_return(
                self.risk_tolerance, self.objective
            )

    @property
    def risk_profile_label(self) -> str:
        """Label lisible combinant tolérance et objectif."""
        return f"{self.risk_tolerance.value}_{self.objective.value}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "capital": self.capital,
            "risk_tolerance": self.risk_tolerance.value,
            "objective": self.objective.value,
            "horizon_years": self.horizon_years,
            "max_drawdown_tolerance": self.max_drawdown_tolerance,
            "target_annual_return": self.target_annual_return,
            "age": self.age,
            "preferences": self.preferences.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClientProfile":
        prefs_raw = d.get("preferences", {})
        return cls(
            name=d.get("name", "default"),
            capital=float(d.get("capital", 10_000.0)),
            risk_tolerance=RiskTolerance(d.get("risk_tolerance", "moderate")),
            objective=Objective(d.get("objective", "growth")),
            horizon_years=int(d.get("horizon_years", 5)),
            max_drawdown_tolerance=float(d.get("max_drawdown_tolerance", 0.20)),
            target_annual_return=d.get("target_annual_return"),
            age=d.get("age"),
            preferences=AssetPreferences.from_dict(prefs_raw),
        )


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

_RETURN_TABLE: dict[tuple[str, str], float] = {
    ("conservative",  "income"):              0.06,
    ("conservative",  "growth"):              0.08,
    ("conservative",  "wealth_preservation"): 0.05,
    ("conservative",  "balanced"):            0.07,
    ("moderate",      "income"):              0.10,
    ("moderate",      "growth"):              0.15,
    ("moderate",      "wealth_preservation"): 0.08,
    ("moderate",      "balanced"):            0.12,
    ("aggressive",    "income"):              0.15,
    ("aggressive",    "growth"):              0.22,
    ("aggressive",    "wealth_preservation"): 0.10,
    ("aggressive",    "balanced"):            0.18,
    ("speculative",   "income"):              0.20,
    ("speculative",   "growth"):              0.35,
    ("speculative",   "wealth_preservation"): 0.12,
    ("speculative",   "balanced"):            0.25,
}


def _default_target_return(risk: RiskTolerance, obj: Objective) -> float:
    return _RETURN_TABLE.get((risk.value, obj.value), 0.12)
