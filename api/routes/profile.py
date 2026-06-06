"""
api/routes/profile.py

Routes profil client : lecture, mise à jour, re-calcul du plan stratégique.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_bot_state, get_planner, get_allocation_engine
from api.models import ProfileIn, StrategyPlanOut

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("/", response_model=dict)
async def get_profile(state=Depends(get_bot_state)):
    """Retourne le profil client actuel."""
    return state.get("client_profile", {})


@router.put("/", response_model=StrategyPlanOut)
async def update_profile(
    profile_in: ProfileIn,
    state=Depends(get_bot_state),
    planner=Depends(get_planner),
    alloc_engine=Depends(get_allocation_engine),
):
    """
    Met à jour le profil client et recalcule le plan stratégique.
    Le nouveau plan prend effet au prochain cycle de trading.
    """
    from src.profile.client_profile import ClientProfile, AssetPreferences, RiskTolerance, Objective

    try:
        profile = ClientProfile(
            name=profile_in.name,
            capital=profile_in.capital,
            risk_tolerance=RiskTolerance(profile_in.risk_tolerance),
            objective=Objective(profile_in.objective),
            horizon_years=profile_in.horizon_years,
            max_drawdown_tolerance=profile_in.max_drawdown_tolerance,
            target_annual_return=profile_in.target_annual_return,
            age=profile_in.age,
            preferences=AssetPreferences(
                etf=profile_in.preferences.etf,
                stocks=profile_in.preferences.stocks,
                intraday=profile_in.preferences.intraday,
                crypto=profile_in.preferences.crypto,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    plan = planner.build_plan(profile)
    alloc_engine.update_plan(plan)

    # Persister dans l'état partagé
    state["client_profile"] = profile.to_dict()
    state["strategy_plan"] = plan.to_dict()

    return StrategyPlanOut(**plan.to_dict())


@router.get("/plan", response_model=StrategyPlanOut)
async def get_strategy_plan(state=Depends(get_bot_state)):
    """Retourne le plan stratégique actuel."""
    plan_data = state.get("strategy_plan")
    if not plan_data:
        raise HTTPException(status_code=404, detail="Aucun plan stratégique calculé.")
    return StrategyPlanOut(**plan_data)
