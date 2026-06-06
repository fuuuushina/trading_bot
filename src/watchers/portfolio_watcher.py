"""
src/watchers/portfolio_watcher.py

Portfolio Watcher — surveille en continu les positions ouvertes.

Responsabilités :
  - Calculer le P&L courant (unrealized + realized)
  - Surveiller les drawdowns par position et globaux
  - Déclencher des alertes sur les seuils de stop / drawdown
  - Produire un PortfolioState enrichi à chaque cycle

Fréquence recommandée : HOURLY (ou REALTIME pour les stops)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.alerts.alert_manager import AlertLevel, AlertManager, AlertType

logger = logging.getLogger(__name__)


@dataclass
class PositionStatus:
    """État courant d'une position."""
    asset: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    strategy_name: str
    horizon: str
    is_at_risk: bool = False   # True si proche du stop ou drawdown élevé
    days_held: int = 0


@dataclass
class WatchedPortfolioState:
    """État enrichi du portefeuille produit par le Portfolio Watcher."""
    total_capital: float
    cash: float
    total_exposure: float
    total_exposure_pct: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    daily_pnl: float
    weekly_pnl: float
    drawdown_pct: float
    max_drawdown_ever: float
    open_positions: int
    positions: list[PositionStatus] = field(default_factory=list)
    at_risk_count: int = 0        # positions proches du stop
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "total_capital": round(self.total_capital, 2),
            "cash": round(self.cash, 2),
            "total_exposure": round(self.total_exposure, 2),
            "total_exposure_pct": round(self.total_exposure_pct, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 4),
            "daily_pnl": round(self.daily_pnl, 2),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "max_drawdown_ever": round(self.max_drawdown_ever, 4),
            "open_positions": self.open_positions,
            "at_risk_positions": self.at_risk_count,
        }


class PortfolioWatcher:
    """
    Surveille l'état du portefeuille et déclenche des alertes.

    Usage :
        watcher = PortfolioWatcher(risk_cfg, alert_manager)
        state = watcher.watch(broker, current_prices)
    """

    def __init__(
        self,
        risk_cfg: dict,
        alert_manager: AlertManager,
    ) -> None:
        self.r = risk_cfg.get("risk", {})
        self.alerts = alert_manager
        self._peak_capital: float = 0.0
        self._max_drawdown: float = 0.0
        self._last_regime: str = "unknown"

    def watch(
        self,
        broker,                              # PaperBroker / AlpacaPaperTrader
        current_prices: dict[str, float],
    ) -> WatchedPortfolioState:
        """
        Calcule l'état enrichi du portefeuille et émet les alertes nécessaires.
        """
        raw_state = broker.get_portfolio_state()
        raw_positions = broker.get_open_positions()

        total_capital = raw_state.get("total_capital", 0.0)
        cash = raw_state.get("cash", total_capital)
        total_exposure = raw_state.get("total_exposure", 0.0)

        # Mise à jour du peak capital (pour drawdown exact)
        if total_capital > self._peak_capital:
            self._peak_capital = total_capital

        # Drawdown courant depuis le pic
        if self._peak_capital > 0:
            drawdown_pct = (total_capital - self._peak_capital) / self._peak_capital
        else:
            drawdown_pct = raw_state.get("drawdown_pct", 0.0)

        if drawdown_pct < self._max_drawdown:
            self._max_drawdown = drawdown_pct

        # Évaluer chaque position
        positions: list[PositionStatus] = []
        total_unrealized = 0.0

        for pos in raw_positions:
            asset = pos.get("asset", "")
            qty = float(pos.get("quantity", 0.0))
            avg_cost = float(pos.get("avg_cost", 0.0))
            cur_price = current_prices.get(asset, avg_cost)
            market_value = qty * cur_price
            unrealized = market_value - qty * avg_cost
            unrealized_pct = (cur_price - avg_cost) / avg_cost if avg_cost > 0 else 0.0
            total_unrealized += unrealized

            # Position "at risk" si perte > 5%
            at_risk = unrealized_pct < -0.05

            positions.append(PositionStatus(
                asset=asset,
                quantity=qty,
                avg_cost=avg_cost,
                current_price=cur_price,
                market_value=market_value,
                unrealized_pnl=round(unrealized, 2),
                unrealized_pnl_pct=round(unrealized_pct, 4),
                strategy_name=pos.get("strategy_name", ""),
                horizon=pos.get("horizon", ""),
                is_at_risk=at_risk,
            ))

        at_risk_count = sum(1 for p in positions if p.is_at_risk)
        unrealized_pnl_pct = total_unrealized / total_capital if total_capital > 0 else 0.0

        state = WatchedPortfolioState(
            total_capital=total_capital,
            cash=cash,
            total_exposure=total_exposure,
            total_exposure_pct=total_exposure / total_capital if total_capital > 0 else 0.0,
            unrealized_pnl=round(total_unrealized, 2),
            unrealized_pnl_pct=round(unrealized_pnl_pct, 4),
            daily_pnl=raw_state.get("daily_pnl", 0.0),
            weekly_pnl=raw_state.get("weekly_pnl", 0.0),
            drawdown_pct=round(drawdown_pct, 4),
            max_drawdown_ever=round(self._max_drawdown, 4),
            open_positions=len(positions),
            positions=positions,
            at_risk_count=at_risk_count,
        )

        self._check_alerts(state)
        return state

    def _check_alerts(self, state: WatchedPortfolioState) -> None:
        """Émet des alertes si des seuils sont dépassés."""
        dd_threshold = self.r.get("max_total_drawdown_pct", 0.15)
        dd_warning = dd_threshold * 0.70    # Alerte à 70% du seuil max

        # Drawdown warning
        if state.drawdown_pct < -dd_warning:
            self.alerts.drawdown_alert(state.drawdown_pct, dd_warning)

        # Positions at risk
        if state.at_risk_count >= 2:
            self.alerts.send(
                AlertLevel.WARNING, AlertType.RISK_LIMIT,
                f"{state.at_risk_count} positions en risque (perte > 5%)",
                {"at_risk_count": state.at_risk_count},
            )

        # Exposition maximale approchée
        max_exp = self.r.get("max_total_exposure_pct", 0.80)
        if state.total_exposure_pct > max_exp * 0.90:
            self.alerts.send(
                AlertLevel.WARNING, AlertType.RISK_LIMIT,
                f"Exposition {state.total_exposure_pct:.0%} approche du max {max_exp:.0%}",
            )
