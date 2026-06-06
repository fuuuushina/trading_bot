"""
api/routes/news.py

Routes News Layer : impacts par asset, scores de sentiment, dernières news.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_news_manager
from api.models import NewsImpactOut

router = APIRouter(prefix="/news", tags=["news"])


@router.get("/impacts", response_model=list[NewsImpactOut])
async def get_news_impacts(news_manager=Depends(get_news_manager)):
    """Retourne les impacts news calculés sur les assets de l'univers."""
    if news_manager is None:
        return []
    impacts = news_manager.get_latest_impacts()
    return [NewsImpactOut(**i) for i in impacts]


@router.get("/risk-scores")
async def get_risk_scores(news_manager=Depends(get_news_manager)):
    """Retourne les scores de risque injectés dans le Signal Aggregator."""
    if news_manager is None:
        return {"risk_scores": {}, "enabled": False}
    return {"risk_scores": news_manager.get_risk_scores(), "enabled": True}


@router.get("/headlines/{asset}")
async def get_asset_headlines(asset: str, limit: int = 10, news_manager=Depends(get_news_manager)):
    """Retourne les dernières headlines pour un asset."""
    if news_manager is None:
        return {"asset": asset, "headlines": [], "enabled": False}
    return {
        "asset": asset,
        "headlines": news_manager.get_headlines(asset, limit=limit),
        "enabled": True,
    }
