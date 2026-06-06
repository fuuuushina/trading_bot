"""
api/routes/portfolio.py

Routes portfolio : positions, état du portefeuille, historique des trades.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_broker
from api.models import PortfolioStateOut, PositionOut

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("/state", response_model=PortfolioStateOut)
async def get_portfolio_state(broker=Depends(get_broker)):
    """Retourne l'état courant du portefeuille."""
    state = broker.get_portfolio_state()
    positions_raw = broker.get_open_positions()

    positions = [
        PositionOut(
            asset=p.get("asset", ""),
            quantity=p.get("quantity", 0.0),
            avg_cost=p.get("avg_cost", 0.0),
            current_price=p.get("current_price"),
            market_value=p.get("market_value"),
            unrealized_pnl=p.get("unrealized_pnl"),
            unrealized_pnl_pct=p.get("unrealized_pnl_pct"),
            strategy_name=p.get("strategy_name", ""),
            horizon=p.get("horizon", ""),
        )
        for p in positions_raw
    ]

    return PortfolioStateOut(
        total_capital=state.get("total_capital", 0.0),
        cash=state.get("cash", 0.0),
        total_exposure=state.get("total_exposure", 0.0),
        total_exposure_pct=state.get("total_exposure", 0.0) / max(state.get("total_capital", 1.0), 1.0),
        drawdown_pct=state.get("drawdown_pct", 0.0),
        open_positions=len(positions),
        daily_pnl=state.get("daily_pnl"),
        positions=positions,
    )


@router.get("/trades")
async def get_trade_history(limit: int = 50, broker=Depends(get_broker)):
    """Retourne l'historique des trades (les N derniers)."""
    history = broker.get_trade_history()
    return {"trades": history[-limit:], "total": len(history)}


@router.get("/positions")
async def get_positions(broker=Depends(get_broker)):
    """Retourne les positions ouvertes."""
    return {"positions": broker.get_open_positions()}
