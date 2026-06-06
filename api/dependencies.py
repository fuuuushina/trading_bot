"""
api/dependencies.py

Dépendances FastAPI (injection).

Le BotState est un dictionnaire partagé entre le loop de trading
et l'API. Il est initialisé dans api/main.py au démarrage.
"""
from __future__ import annotations

from typing import Any, Optional

# État partagé entre le bot et l'API (initialisé au démarrage)
_bot_state: dict[str, Any] = {
    "mode": "paper",
    "kill_switch_active": False,
    "defensive_mode": False,
    "regime": "unknown",
    "regime_confidence": 0.0,
    "cycle_count": 0,
    "last_cycle_at": None,
    "last_decisions": [],
    "client_profile": {},
    "strategy_plan": None,
    "daily_pnl_pct": 0.0,
    "weekly_pnl_pct": 0.0,
    "consecutive_blocked": 0,
}

_broker_instance = None
_planner_instance = None
_allocation_engine_instance = None
_news_manager_instance = None


def init_state(
    broker,
    planner,
    allocation_engine,
    profile,
    plan,
    news_manager=None,
) -> None:
    """Appelé une fois au démarrage du serveur pour injecter les instances."""
    global _broker_instance, _planner_instance, _allocation_engine_instance, _news_manager_instance
    _broker_instance = broker
    _planner_instance = planner
    _allocation_engine_instance = allocation_engine
    _news_manager_instance = news_manager
    _bot_state["client_profile"] = profile.to_dict() if profile else {}
    _bot_state["strategy_plan"] = plan.to_dict() if plan else None


def update_cycle_state(
    decisions: list[dict],
    regime: str,
    regime_confidence: float,
    kill_switch: bool,
    defensive: bool,
    cycle_count: int,
    last_cycle_at: str,
) -> None:
    """Mise à jour à chaque cycle de trading."""
    _bot_state["last_decisions"] = decisions
    _bot_state["regime"] = regime
    _bot_state["regime_confidence"] = regime_confidence
    _bot_state["kill_switch_active"] = kill_switch
    _bot_state["defensive_mode"] = defensive
    _bot_state["cycle_count"] = cycle_count
    _bot_state["last_cycle_at"] = last_cycle_at


def get_bot_state() -> dict[str, Any]:
    return _bot_state


def get_broker():
    if _broker_instance is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Broker non initialisé.")
    return _broker_instance


def get_planner():
    if _planner_instance is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Planner non initialisé.")
    return _planner_instance


def get_allocation_engine():
    if _allocation_engine_instance is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="AllocationEngine non initialisé.")
    return _allocation_engine_instance


def get_news_manager() -> Optional[Any]:
    return _news_manager_instance
