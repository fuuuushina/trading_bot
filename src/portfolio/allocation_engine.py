"""
src/portfolio/allocation_engine.py

Allocation Engine.
Determines target allocation across long_term / swing / intraday / cash
based on the current market regime and portfolio drawdown.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AllocationTarget:
    long_term: float    # % of total capital
    swing: float
    intraday: float
    cash: float

    def __post_init__(self):
        total = self.long_term + self.swing + self.intraday + self.cash
        assert abs(total - 1.0) < 0.01, f"Allocations must sum to 1.0, got {total:.3f}"

    def to_dict(self) -> dict:
        return {
            "long_term": round(self.long_term, 4),
            "swing": round(self.swing, 4),
            "intraday": round(self.intraday, 4),
            "cash": round(self.cash, 4),
        }


# Pre-defined allocation tables per regime
REGIME_ALLOCATIONS: dict[str, AllocationTarget] = {
    "bull_trend": AllocationTarget(
        long_term=0.70, swing=0.20, intraday=0.05, cash=0.05
    ),
    "breakout_expansion": AllocationTarget(
        long_term=0.65, swing=0.25, intraday=0.05, cash=0.05
    ),
    "low_volatility": AllocationTarget(
        long_term=0.70, swing=0.20, intraday=0.05, cash=0.05
    ),
    "range": AllocationTarget(
        long_term=0.60, swing=0.20, intraday=0.05, cash=0.15
    ),
    "compression": AllocationTarget(
        long_term=0.60, swing=0.15, intraday=0.00, cash=0.25
    ),
    "high_volatility": AllocationTarget(
        long_term=0.50, swing=0.10, intraday=0.00, cash=0.40
    ),
    "euphoric": AllocationTarget(
        long_term=0.55, swing=0.10, intraday=0.00, cash=0.35  # Trim on euphoria
    ),
    "bear_trend": AllocationTarget(
        long_term=0.35, swing=0.05, intraday=0.00, cash=0.60
    ),
    "panic": AllocationTarget(
        long_term=0.20, swing=0.00, intraday=0.00, cash=0.80
    ),
    "unknown": AllocationTarget(
        long_term=0.50, swing=0.10, intraday=0.00, cash=0.40
    ),
}


class AllocationEngine:
    """
    Computes target allocation and checks if a proposed trade
    is within its horizon budget.
    """

    def __init__(self, risk_cfg: dict) -> None:
        self.risk = risk_cfg.get("risk", {})

    def get_target(
        self,
        regime: str,
        drawdown_pct: float,
    ) -> AllocationTarget:
        """
        Get allocation target for given regime and drawdown.
        Drawdown acts as an additional defensive multiplier.
        """
        base = REGIME_ALLOCATIONS.get(regime, REGIME_ALLOCATIONS["unknown"])

        # Progressive defensive shift based on drawdown
        if drawdown_pct < -0.10:
            # Deep drawdown → go very defensive
            adjusted = AllocationTarget(
                long_term=base.long_term * 0.50,
                swing=0.0,
                intraday=0.0,
                cash=1.0 - base.long_term * 0.50,
            )
        elif drawdown_pct < -0.05:
            # Moderate drawdown → tighten
            factor = 0.70
            lt = base.long_term * factor
            sw = base.swing * factor
            intra = 0.0
            cash = 1.0 - lt - sw
            adjusted = AllocationTarget(lt, sw, intra, max(cash, 0.30))
        else:
            adjusted = base

        logger.debug(
            "Allocation target: regime=%s dd=%.1f%% → %s",
            regime, drawdown_pct * 100, adjusted.to_dict()
        )
        return adjusted

    def get_available_budget(
        self,
        horizon: str,
        current_target: AllocationTarget,
        portfolio_state: dict,
    ) -> float:
        """
        Returns available USD budget for a given horizon.
        """
        total_capital = portfolio_state.get("total_capital", 0.0)
        horizon_exposure = portfolio_state.get("horizon_exposure", {})

        target_pct = getattr(current_target, horizon, 0.0)
        target_usd = total_capital * target_pct
        current_usd = horizon_exposure.get(horizon, 0.0)
        available = max(target_usd - current_usd, 0.0)

        logger.debug(
            "Budget for %s: target=$%.0f current=$%.0f available=$%.0f",
            horizon, target_usd, current_usd, available
        )
        return available

    def is_within_budget(
        self,
        horizon: str,
        proposed_size_usd: float,
        current_target: AllocationTarget,
        portfolio_state: dict,
    ) -> bool:
        available = self.get_available_budget(horizon, current_target, portfolio_state)
        return proposed_size_usd <= available + 0.01
