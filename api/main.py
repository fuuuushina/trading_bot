"""
api/main.py

Serveur FastAPI — expose l'état du bot via une API REST.

Démarrage :
    uvicorn api.main:app --reload --port 8000

Endpoints :
    GET  /portfolio/state        → état portefeuille
    GET  /portfolio/trades       → historique trades
    GET  /portfolio/positions    → positions ouvertes
    GET  /decisions/latest       → dernières décisions
    GET  /decisions/system       → état système
    GET  /decisions/risk         → métriques risk manager
    GET  /profile/               → profil client
    PUT  /profile/               → mise à jour profil + recalcul plan
    GET  /profile/plan           → plan stratégique actuel
    GET  /news/impacts           → impacts news par asset
    GET  /news/risk-scores       → scores de risque news
    GET  /news/headlines/{asset} → headlines par asset
    GET  /health                 → healthcheck
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes import portfolio, decisions, profile, news
from api.dependencies import init_state

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Initialisation au démarrage : charge le profil, crée le plan,
    initialise le broker paper en mode lecture seule pour l'API.
    """
    try:
        from config.loader import get_profile_config, get_risk_config, get_settings
        from src.ai.strategic_planner import StrategicPlanner
        from src.execution.paper_broker import PaperBroker
        from src.portfolio.allocation_engine import AllocationEngine
        from src.profile.client_profile import ClientProfile

        settings = get_settings()
        risk_cfg = get_risk_config()
        profile_cfg = get_profile_config()

        broker_cfg = settings.get("broker", {}).get("paper", {})
        ai_cfg = settings.get("ai", {})

        profile_obj = ClientProfile.from_dict(profile_cfg.get("client", {}))
        planner = StrategicPlanner(ai_cfg)
        plan = planner.build_plan(profile_obj)
        alloc_engine = AllocationEngine(risk_cfg, strategy_plan=plan)

        broker = PaperBroker(
            initial_capital=broker_cfg.get("initial_capital", 10_000.0),
            commission_flat=broker_cfg.get("commission_per_trade", 0.0),
            slippage_pct=broker_cfg.get("slippage_pct", 0.001),
        )

        init_state(
            broker=broker,
            planner=planner,
            allocation_engine=alloc_engine,
            profile=profile_obj,
            plan=plan,
        )

        logger.info("API initialized: profile=%s plan=%s", profile_obj.risk_profile_label, plan.allocation)
    except Exception as exc:
        logger.error("API initialization error: %s", exc)

    yield


app = FastAPI(
    title="Trading Bot API",
    description="API REST pour le bot de trading multi-agents",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portfolio.router)
app.include_router(decisions.router)
app.include_router(profile.router)
app.include_router(news.router)


@app.get("/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/", tags=["system"])
async def root() -> dict:
    return {
        "message": "Trading Bot API",
        "docs": "/docs",
        "version": app.version,
    }
