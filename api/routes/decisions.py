"""
api/routes/decisions.py

Routes décisions : dernières décisions du moteur, signaux agrégés,
état du risk manager.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_bot_state
from api.models import DecisionOut, SystemStatusOut

router = APIRouter(prefix="/decisions", tags=["decisions"])


@router.get("/latest", response_model=list[DecisionOut])
async def get_latest_decisions(limit: int = 20, state=Depends(get_bot_state)):
    """Retourne les dernières décisions du moteur."""
    decisions = state.get("last_decisions", [])[-limit:]
    return [_map_decision(d) for d in decisions]


@router.get("/system", response_model=SystemStatusOut)
async def get_system_status(state=Depends(get_bot_state)):
    """Retourne l'état global du système."""
    plan_data = state.get("strategy_plan")

    from api.models import StrategyPlanOut
    plan_out = None
    if plan_data:
        plan_out = StrategyPlanOut(**plan_data)

    return SystemStatusOut(
        mode=state.get("mode", "paper"),
        kill_switch_active=state.get("kill_switch_active", False),
        defensive_mode=state.get("defensive_mode", False),
        regime=state.get("regime", "unknown"),
        regime_confidence=state.get("regime_confidence", 0.0),
        cycle_count=state.get("cycle_count", 0),
        last_cycle_at=state.get("last_cycle_at"),
        strategy_plan=plan_out,
    )


@router.get("/risk")
async def get_risk_status(state=Depends(get_bot_state)):
    """Retourne les métriques risk manager."""
    return {
        "kill_switch_active": state.get("kill_switch_active", False),
        "defensive_mode": state.get("defensive_mode", False),
        "daily_pnl_pct": state.get("daily_pnl_pct", 0.0),
        "weekly_pnl_pct": state.get("weekly_pnl_pct", 0.0),
        "consecutive_blocked": state.get("consecutive_blocked", 0),
    }


def _map_decision(d: dict) -> DecisionOut:
    signal = d.get("signal", {})
    meta = signal.get("metadata", {})

    return DecisionOut(
        asset=d.get("asset", ""),
        action=signal.get("signal", ""),
        confidence=signal.get("confidence", 0.0),
        horizon=signal.get("horizon", ""),
        entry_price=signal.get("entry_price"),
        stop_loss=signal.get("stop_loss"),
        take_profit=signal.get("take_profit"),
        risk_reward=signal.get("risk_reward"),
        n_agreeing=meta.get("n_agreeing", 1),
        n_dissenting=meta.get("n_dissenting", 0),
        contributors=[
            {"strategy": c.get("strategy", ""), "signal": c.get("signal", ""), "confidence": c.get("confidence", 0.0)}
            for c in meta.get("contributors", [])
        ],
        final_action=d.get("final_action", "BLOCK"),
        risk_decision=d.get("risk_verdict", {}).get("decision", ""),
        approved_size_usd=d.get("risk_verdict", {}).get("approved_size_usd", 0.0),
        reason=signal.get("reason", ""),
        explanation=d.get("explanation", ""),
        regime=d.get("regime", {}).get("regime", "unknown") if isinstance(d.get("regime"), dict) else "unknown",
    )
