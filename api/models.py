"""
api/models.py

Modèles Pydantic pour l'API FastAPI.
Définit les schémas de requête/réponse.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---- Portfolio ----

class PositionOut(BaseModel):
    asset: str
    quantity: float
    avg_cost: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    strategy_name: str
    horizon: str


class PortfolioStateOut(BaseModel):
    total_capital: float
    cash: float
    total_exposure: float
    total_exposure_pct: float
    drawdown_pct: float
    open_positions: int
    daily_pnl: Optional[float] = None
    positions: list[PositionOut] = []


# ---- Decisions ----

class ContributorOut(BaseModel):
    strategy: str
    signal: str
    confidence: float


class DecisionOut(BaseModel):
    asset: str
    action: str
    confidence: float
    horizon: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None
    n_agreeing: int = 1
    n_dissenting: int = 0
    contributors: list[ContributorOut] = []
    final_action: str       # EXECUTE | BLOCK | NO_TRADE
    risk_decision: str
    approved_size_usd: float
    reason: str
    explanation: str
    regime: str


# ---- Profile ----

class AssetPrefsIn(BaseModel):
    etf: bool = True
    stocks: bool = True
    intraday: bool = False
    crypto: bool = False


class ProfileIn(BaseModel):
    name: str = "default"
    capital: float = Field(gt=0)
    risk_tolerance: str = "moderate"
    objective: str = "growth"
    horizon_years: int = Field(ge=1, le=50, default=5)
    max_drawdown_tolerance: float = Field(ge=0.01, le=0.80, default=0.20)
    target_annual_return: Optional[float] = None
    age: Optional[int] = Field(ge=18, le=120, default=None)
    preferences: AssetPrefsIn = Field(default_factory=AssetPrefsIn)


class StrategyPlanOut(BaseModel):
    profile_label: str
    target_annual_return: float
    max_drawdown: float
    allocation: dict[str, float]
    enabled_horizons: list[str]
    reasoning: str
    ai_generated: bool


# ---- News ----

class NewsImpactOut(BaseModel):
    asset: str
    sentiment: float      # -1.0 (très négatif) à +1.0 (très positif)
    risk_score: float     # 0.0 à 1.0
    topics: list[str]
    impact: str           # positive_high | positive_medium | neutral | negative_medium | negative_high
    headline_count: int
    source: str


# ---- System ----

class SystemStatusOut(BaseModel):
    mode: str
    kill_switch_active: bool
    defensive_mode: bool
    regime: str
    regime_confidence: float
    cycle_count: int
    last_cycle_at: Optional[str] = None
    strategy_plan: Optional[StrategyPlanOut] = None
