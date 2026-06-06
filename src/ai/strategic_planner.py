"""
src/ai/strategic_planner.py

Strategic AI Planner — Niveau 1 de l'architecture hiérarchique.

Il reçoit le profil client et produit une StrategyPlan :
  - allocation par horizon (long_term / swing / intraday / cash)
  - rendement cible
  - drawdown maximum
  - horizons activés
  - explication de la stratégie

Deux modes :
  1. Rule-based (rapide, sans API) : mapping profil → allocation pré-définie
  2. AI-driven (Claude) : allocation personnalisée via LLM (optionnel)

L'IA stratégique ne passe JAMAIS d'ordres.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.profile.client_profile import ClientProfile, Objective, RiskTolerance

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# StrategyPlan — sortie du planificateur
# ------------------------------------------------------------------ #

@dataclass
class StrategyPlan:
    """
    Plan stratégique produit par le Strategic Planner.
    Transmis à l'AllocationEngine pour piloter les budgets par horizon.
    """
    profile_label: str            # ex: "moderate_growth"
    target_annual_return: float   # ex: 0.15 → 15% par an
    max_drawdown: float           # ex: 0.20 → 20% max acceptable
    allocation: dict[str, float]  # {"long_term": 0.60, "swing": 0.25, "intraday": 0.10, "cash": 0.05}
    enabled_horizons: list[str]   # horizons autorisés selon préférences
    reasoning: str                # Explication lisible de la stratégie
    ai_generated: bool = False    # True si produit par LLM

    def __post_init__(self) -> None:
        total = sum(self.allocation.values())
        assert abs(total - 1.0) < 0.02, f"Allocations must sum to 1.0, got {total:.3f}"

    def to_dict(self) -> dict:
        return {
            "profile_label": self.profile_label,
            "target_annual_return": round(self.target_annual_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "allocation": {k: round(v, 4) for k, v in self.allocation.items()},
            "enabled_horizons": self.enabled_horizons,
            "reasoning": self.reasoning,
            "ai_generated": self.ai_generated,
        }


# ------------------------------------------------------------------ #
# Tables d'allocation pré-définies par profil (rule-based)
# ------------------------------------------------------------------ #

# Format: (risk_tolerance, objective) → AllocationTarget dict
_ALLOCATION_TABLE: dict[tuple[str, str], dict[str, float]] = {
    # Conservative
    ("conservative", "income"):              {"long_term": 0.75, "swing": 0.10, "intraday": 0.00, "cash": 0.15},
    ("conservative", "growth"):              {"long_term": 0.70, "swing": 0.15, "intraday": 0.00, "cash": 0.15},
    ("conservative", "wealth_preservation"): {"long_term": 0.65, "swing": 0.05, "intraday": 0.00, "cash": 0.30},
    ("conservative", "balanced"):            {"long_term": 0.70, "swing": 0.10, "intraday": 0.00, "cash": 0.20},

    # Moderate
    ("moderate", "income"):              {"long_term": 0.65, "swing": 0.20, "intraday": 0.00, "cash": 0.15},
    ("moderate", "growth"):              {"long_term": 0.60, "swing": 0.25, "intraday": 0.05, "cash": 0.10},
    ("moderate", "wealth_preservation"): {"long_term": 0.60, "swing": 0.15, "intraday": 0.00, "cash": 0.25},
    ("moderate", "balanced"):            {"long_term": 0.62, "swing": 0.20, "intraday": 0.03, "cash": 0.15},

    # Aggressive
    ("aggressive", "income"):              {"long_term": 0.55, "swing": 0.30, "intraday": 0.05, "cash": 0.10},
    ("aggressive", "growth"):              {"long_term": 0.50, "swing": 0.30, "intraday": 0.10, "cash": 0.10},
    ("aggressive", "wealth_preservation"): {"long_term": 0.55, "swing": 0.25, "intraday": 0.00, "cash": 0.20},
    ("aggressive", "balanced"):            {"long_term": 0.52, "swing": 0.28, "intraday": 0.08, "cash": 0.12},

    # Speculative
    ("speculative", "income"):              {"long_term": 0.45, "swing": 0.35, "intraday": 0.10, "cash": 0.10},
    ("speculative", "growth"):              {"long_term": 0.40, "swing": 0.35, "intraday": 0.15, "cash": 0.10},
    ("speculative", "wealth_preservation"): {"long_term": 0.50, "swing": 0.30, "intraday": 0.05, "cash": 0.15},
    ("speculative", "balanced"):            {"long_term": 0.45, "swing": 0.32, "intraday": 0.13, "cash": 0.10},
}

_REASONING_TEMPLATES: dict[str, str] = {
    "conservative": (
        "Profil conservateur : priorité à la préservation du capital. "
        "Forte pondération long terme (ETF diversifiés), cash élevé comme amortisseur. "
        "Trading swing limité, intraday désactivé."
    ),
    "moderate": (
        "Profil modéré : équilibre croissance et sécurité. "
        "Mix long terme + swing sur actions de qualité. "
        "Exposition intraday très limitée selon objectif."
    ),
    "aggressive": (
        "Profil agressif : croissance du capital en priorité. "
        "Swing trading actif sur momentum et breakouts. "
        "Intraday autorisé avec risk management strict."
    ),
    "speculative": (
        "Profil spéculatif : rendement maximal, risque élevé assumé. "
        "Swing et intraday actifs. Capital long terme réduit au minimum. "
        "Kill switch et defensive mode critiques à ce niveau de risque."
    ),
}


# ------------------------------------------------------------------ #
# Strategic Planner (rule-based + AI optionnel)
# ------------------------------------------------------------------ #

class StrategicPlanner:
    """
    Produit un StrategyPlan à partir d'un ClientProfile.

    Usage :
        planner = StrategicPlanner(ai_cfg)
        plan = planner.build_plan(profile)
    """

    def __init__(self, ai_cfg: dict | None = None) -> None:
        self.ai_cfg = ai_cfg or {}
        self._use_ai = (
            self.ai_cfg.get("enabled", False)
            and self.ai_cfg.get("strategic_model")
        )

    def build_plan(self, profile: ClientProfile) -> StrategyPlan:
        """
        Construit le plan stratégique.
        Tente l'IA si configurée, sinon utilise le mode rule-based.
        """
        if self._use_ai:
            try:
                plan = self._ai_plan(profile)
                if plan:
                    logger.info(
                        "StrategyPlan (AI): %s → alloc=%s",
                        profile.risk_profile_label, plan.allocation
                    )
                    return plan
            except Exception as exc:
                logger.warning("AI planning failed, falling back to rule-based: %s", exc)

        plan = self._rule_based_plan(profile)
        logger.info(
            "StrategyPlan (rules): %s → alloc=%s target=%.0f%%",
            profile.risk_profile_label,
            plan.allocation,
            plan.target_annual_return * 100,
        )
        return plan

    # ------------------------------------------------------------------ #
    # Rule-based planner
    # ------------------------------------------------------------------ #

    def _rule_based_plan(self, profile: ClientProfile) -> StrategyPlan:
        key = (profile.risk_tolerance.value, profile.objective.value)
        base_alloc = dict(_ALLOCATION_TABLE.get(key, _ALLOCATION_TABLE[("moderate", "growth")]))

        # Désactiver intraday si non préféré
        if not profile.preferences.intraday:
            extra = base_alloc.pop("intraday", 0.0)
            base_alloc["cash"] = base_alloc.get("cash", 0.0) + extra

        # Ajustement horizon : horizon court → plus de cash
        if profile.horizon_years <= 2:
            shift = 0.10
            base_alloc["long_term"] = max(base_alloc["long_term"] - shift, 0.20)
            base_alloc["cash"] = base_alloc.get("cash", 0.0) + shift
        elif profile.horizon_years >= 10:
            shift = 0.05
            base_alloc["long_term"] = min(base_alloc["long_term"] + shift, 0.90)
            base_alloc["cash"] = max(base_alloc.get("cash", 0.0) - shift, 0.05)

        # Normaliser à 1.0
        base_alloc = _normalize(base_alloc)

        enabled_horizons = _compute_enabled_horizons(profile, base_alloc)
        reasoning = _build_reasoning(profile, base_alloc)

        return StrategyPlan(
            profile_label=profile.risk_profile_label,
            target_annual_return=profile.target_annual_return or 0.12,
            max_drawdown=profile.max_drawdown_tolerance,
            allocation=base_alloc,
            enabled_horizons=enabled_horizons,
            reasoning=reasoning,
            ai_generated=False,
        )

    # ------------------------------------------------------------------ #
    # AI-driven planner (Claude)
    # ------------------------------------------------------------------ #

    def _ai_plan(self, profile: ClientProfile) -> Optional[StrategyPlan]:
        try:
            import anthropic  # type: ignore
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return None

            system = (
                "Tu es un conseiller financier algorithmique. "
                "Tu reçois un profil client et tu produis une allocation de portefeuille optimale. "
                "Tu réponds UNIQUEMENT en JSON valide, sans texte autour. "
                "Schéma attendu :\n"
                + json.dumps({
                    "allocation": {
                        "long_term": "float 0-1",
                        "swing": "float 0-1",
                        "intraday": "float 0-1",
                        "cash": "float 0-1",
                    },
                    "target_annual_return": "float ex: 0.15",
                    "max_drawdown": "float ex: 0.20",
                    "enabled_horizons": ["long_term", "swing"],
                    "reasoning": "string",
                }, indent=2, ensure_ascii=False)
                + "\n\nLes allocations DOIVENT sommer à 1.0."
            )

            user = (
                f"Profil client :\n{json.dumps(profile.to_dict(), indent=2, ensure_ascii=False)}\n\n"
                "Produis le plan d'allocation optimal pour ce profil."
            )

            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=self.ai_cfg.get("strategic_model", "claude-sonnet-4-6"),
                max_tokens=512,
                temperature=self.ai_cfg.get("strategic_temperature", 0.2),
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            raw = msg.content[0].text.strip()

            # Nettoyer éventuels blocs markdown
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            data = json.loads(raw)

            alloc = {k: float(v) for k, v in data["allocation"].items()}
            alloc = _normalize(alloc)

            # Valider les horizons fournis par l'IA
            enabled = [
                h for h in data.get("enabled_horizons", ["long_term", "swing"])
                if h in ("long_term", "swing", "intraday")
            ]
            if not enabled:
                enabled = ["long_term", "swing"]

            return StrategyPlan(
                profile_label=profile.risk_profile_label,
                target_annual_return=float(data.get("target_annual_return", 0.12)),
                max_drawdown=float(data.get("max_drawdown", profile.max_drawdown_tolerance)),
                allocation=alloc,
                enabled_horizons=enabled,
                reasoning=str(data.get("reasoning", "")),
                ai_generated=True,
            )
        except Exception as exc:
            logger.warning("AI strategic plan failed: %s", exc)
            return None


# ------------------------------------------------------------------ #
# Helpers internes
# ------------------------------------------------------------------ #

def _normalize(alloc: dict[str, float]) -> dict[str, float]:
    """Normalise les allocations pour qu'elles somment à 1.0."""
    total = sum(alloc.values())
    if total <= 0:
        return {"long_term": 0.60, "swing": 0.20, "intraday": 0.00, "cash": 0.20}
    return {k: round(v / total, 4) for k, v in alloc.items()}


def _compute_enabled_horizons(profile: ClientProfile, alloc: dict[str, float]) -> list[str]:
    horizons = []
    if alloc.get("long_term", 0) > 0:
        horizons.append("long_term")
    if alloc.get("swing", 0) > 0 and profile.preferences.stocks or profile.preferences.etf:
        horizons.append("swing")
    if alloc.get("intraday", 0) > 0 and profile.preferences.intraday:
        horizons.append("intraday")
    return horizons or ["long_term"]


def _build_reasoning(profile: ClientProfile, alloc: dict[str, float]) -> str:
    base = _REASONING_TEMPLATES.get(profile.risk_tolerance.value, "")
    horizon_note = f"Horizon : {profile.horizon_years} ans."
    alloc_note = (
        f"Allocation : {alloc.get('long_term', 0):.0%} long terme, "
        f"{alloc.get('swing', 0):.0%} swing, "
        f"{alloc.get('intraday', 0):.0%} intraday, "
        f"{alloc.get('cash', 0):.0%} cash."
    )
    target_note = f"Rendement cible : {(profile.target_annual_return or 0.12):.0%}/an."
    return f"{base} {horizon_note} {alloc_note} {target_note}"
