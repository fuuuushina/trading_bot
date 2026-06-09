"""
src/engine/trading_orchestrator.py

TradingOrchestrator — registre central des assets tradables.

Responsabilités :
  - Charger et exposer la configuration de chaque asset (assets.yaml)
  - Fournir leverage, stratégies, paramètres de données par asset
  - Calculer les budgets de marge disponibles par asset
  - Générer la map asset → stratégies pour le DecisionEngine
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# AssetConfig — structure de configuration d'un asset
# ------------------------------------------------------------------ #

@dataclass
class AssetConfig:
    asset_id: str
    name: str
    asset_type: str          # forex | crypto | commodity | equity
    enabled: bool
    leverage: float
    strategies: list[str]
    max_margin_pct: float
    data_interval: str
    data_period: str
    min_confidence: float = 0.40


# ------------------------------------------------------------------ #
# TradingOrchestrator
# ------------------------------------------------------------------ #

class TradingOrchestrator:
    """
    Orchestrateur central des assets tradables.

    Paramètres
    ----------
    assets_cfg   : dict chargé depuis config/assets.yaml
    risk_cfg     : dict chargé depuis config/risk.yaml  (optionnel)
    settings_cfg : dict chargé depuis config/settings.yaml (optionnel)

    Exemple d'utilisation
    ---------------------
        from src.engine.trading_orchestrator import TradingOrchestrator
        orch = TradingOrchestrator(assets_cfg)
        for asset in orch.tradable_assets:
            lev = orch.get_leverage(asset)
    """

    def __init__(
        self,
        assets_cfg: dict,
        risk_cfg: Optional[dict] = None,
        settings_cfg: Optional[dict] = None,
    ) -> None:
        self._risk_cfg     = risk_cfg     or {}
        self._settings_cfg = settings_cfg or {}
        self._assets: dict[str, AssetConfig] = {}
        self._load(assets_cfg)
        logger.info(
            "TradingOrchestrator initialized — %d assets total, %d enabled",
            len(self._assets),
            len(self.tradable_assets),
        )

    # ------------------------------------------------------------------ #
    # Chargement
    # ------------------------------------------------------------------ #

    def _load(self, raw: dict) -> None:
        """Parse assets_cfg dict → AssetConfig objects."""
        assets_section = raw.get("assets", raw)  # support both {assets: {...}} and flat dict
        if not isinstance(assets_section, dict):
            logger.warning("TradingOrchestrator: assets config is empty or malformed")
            return

        for asset_id, cfg in assets_section.items():
            if not isinstance(cfg, dict):
                continue
            try:
                obj = AssetConfig(
                    asset_id=asset_id,
                    name=cfg.get("name", asset_id),
                    asset_type=cfg.get("type", "unknown"),
                    enabled=bool(cfg.get("enabled", False)),
                    leverage=float(cfg.get("leverage", 1.0)),
                    strategies=list(cfg.get("strategies", [])),
                    max_margin_pct=float(cfg.get("max_margin_pct", 0.20)),
                    data_interval=str(cfg.get("data_interval", "5m")),
                    data_period=str(cfg.get("data_period", "5d")),
                    min_confidence=float(cfg.get("min_confidence", 0.40)),
                )
                self._assets[asset_id] = obj
            except Exception as exc:
                logger.warning("Failed to parse asset config for %s: %s", asset_id, exc)

    # ------------------------------------------------------------------ #
    # Properties / accesseurs principaux
    # ------------------------------------------------------------------ #

    @property
    def tradable_assets(self) -> list[str]:
        """Liste des asset_id ayant enabled=true."""
        return [aid for aid, cfg in self._assets.items() if cfg.enabled]

    def is_managed_asset(self, asset_id: str) -> bool:
        """Retourne True si l'asset est dans le registre (enabled ou non)."""
        return asset_id in self._assets

    def get_leverage(self, asset_id: str) -> float:
        """Retourne le levier configuré pour l'asset (1.0 si inconnu)."""
        cfg = self._assets.get(asset_id)
        return cfg.leverage if cfg else 1.0

    def get_strategy_names(self, asset_id: str) -> list[str]:
        """Retourne la liste des stratégies configurées pour l'asset."""
        cfg = self._assets.get(asset_id)
        return list(cfg.strategies) if cfg else []

    def get_data_params(self, asset_id: str) -> tuple[str, str]:
        """Retourne (interval, period) pour yfinance."""
        cfg = self._assets.get(asset_id)
        if cfg:
            return cfg.data_interval, cfg.data_period
        return "5m", "5d"

    def get_max_margin_usd(self, asset_id: str, total_capital: float) -> float:
        """Budget maximum de marge en USD pour cet asset."""
        cfg = self._assets.get(asset_id)
        if cfg is None or total_capital <= 0:
            return 0.0
        return total_capital * cfg.max_margin_pct

    def get_available_margin(
        self,
        asset_id: str,
        total_capital: float,
        deployed_margin: float,
    ) -> float:
        """
        Marge disponible = max_margin - marge déjà déployée sur cet asset.
        """
        max_m = self.get_max_margin_usd(asset_id, total_capital)
        return max(0.0, max_m - deployed_margin)

    def get_asset_strategy_map(self) -> dict[str, list[str]]:
        """
        Retourne un dict {asset_id: [strategy_names]} pour tous les assets enabled.

        Utilisé par DecisionEngine.run_cycle() comme asset_strategy_map.
        """
        return {
            aid: list(cfg.strategies)
            for aid, cfg in self._assets.items()
            if cfg.enabled and cfg.strategies
        }

    # ------------------------------------------------------------------ #
    # Calcul de la marge déployée depuis les positions ouvertes
    # ------------------------------------------------------------------ #

    def get_deployed_margin(
        self,
        asset_id: str,
        open_positions: list[dict],
    ) -> float:
        """
        Somme de la marge déployée sur un asset dans les positions ouvertes.

        Utilise position["cost_basis"] si disponible, sinon
        position["quantity"] * position["entry_price"] / leverage.
        """
        cfg = self._assets.get(asset_id)
        leverage = cfg.leverage if cfg else 1.0
        total = 0.0
        for pos in open_positions:
            if pos.get("asset") != asset_id:
                continue
            cost = pos.get("cost_basis") or pos.get("margin")
            if cost is not None:
                total += float(cost)
            else:
                qty   = float(pos.get("quantity", 0.0) or 0.0)
                price = float(pos.get("entry_price", 0.0) or pos.get("avg_price", 0.0) or 0.0)
                total += (qty * price) / max(leverage, 1.0)
        return total

    # ------------------------------------------------------------------ #
    # Résumé du portefeuille
    # ------------------------------------------------------------------ #

    def portfolio_summary(self, open_positions: list[dict]) -> dict:
        """
        Retourne un dict résumant l'exposition par asset géré.

        {
          asset_id: {
            "name": ...,
            "type": ...,
            "leverage": ...,
            "deployed_margin": ...,
            "max_margin_pct": ...,
          },
          ...
        }
        """
        summary: dict = {}
        for aid, cfg in self._assets.items():
            if not cfg.enabled:
                continue
            deployed = self.get_deployed_margin(aid, open_positions)
            summary[aid] = {
                "name":           cfg.name,
                "type":           cfg.asset_type,
                "leverage":       cfg.leverage,
                "deployed_margin": round(deployed, 2),
                "max_margin_pct": cfg.max_margin_pct,
            }
        return summary
