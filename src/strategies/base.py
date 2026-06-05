"""
src/strategies/base.py

Defines the canonical Signal object returned by ALL strategies,
and the abstract base class every strategy must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"
    NO_TRADE = "NO_TRADE"


class Horizon(str, Enum):
    LONG_TERM = "long_term"
    SWING = "swing"
    INTRADAY = "intraday"


@dataclass
class Signal:
    """
    Standardised output of every strategy.
    Immutable after creation — strategies must not modify signals in-flight.
    """
    strategy_name: str
    asset: str
    timeframe: str
    signal: SignalType
    confidence: float               # 0.0 – 1.0
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward: Optional[float]
    horizon: Horizon
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Filled in by RiskManager / execution layer
    approved: bool = False
    approved_size_usd: float = 0.0
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "signal": self.signal.value,
            "confidence": round(self.confidence, 4),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward": self.risk_reward,
            "horizon": self.horizon.value,
            "reason": self.reason,
            "approved": self.approved,
            "approved_size_usd": self.approved_size_usd,
            "rejection_reason": self.rejection_reason,
            "metadata": self.metadata,
        }

    @property
    def is_actionable(self) -> bool:
        return self.signal in (SignalType.BUY, SignalType.SELL) and self.approved

    def __repr__(self) -> str:
        return (
            f"Signal({self.strategy_name} | {self.asset} | "
            f"{self.signal.value} @ {self.entry_price} | "
            f"conf={self.confidence:.2f})"
        )


def no_trade(
    strategy_name: str,
    asset: str,
    timeframe: str,
    horizon: Horizon,
    reason: str,
    metadata: dict | None = None,
) -> Signal:
    """Convenience constructor for a NO_TRADE signal."""
    return Signal(
        strategy_name=strategy_name,
        asset=asset,
        timeframe=timeframe,
        signal=SignalType.NO_TRADE,
        confidence=0.0,
        entry_price=None,
        stop_loss=None,
        take_profit=None,
        risk_reward=None,
        horizon=horizon,
        reason=reason,
        metadata=metadata or {},
    )


class BaseStrategy(ABC):
    """
    Every strategy inherits from BaseStrategy.

    Subclasses implement generate_signal() only.
    They must NOT submit orders — they only return Signal objects.
    """

    name: str = "unnamed"
    horizon: Horizon = Horizon.SWING

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        asset: str,
        regime: str,
    ) -> Signal:
        """
        Generate a trading signal.

        Parameters
        ----------
        df      : Feature-enriched OHLCV DataFrame (last row = current bar).
        asset   : Ticker symbol.
        regime  : Current MarketRegime value string.

        Returns
        -------
        Signal object — never raises, always returns.
        """
        ...

    def _rr_ratio(
        self, entry: float, stop: float, target: float
    ) -> Optional[float]:
        risk = abs(entry - stop)
        reward = abs(target - entry)
        if risk == 0:
            return None
        return round(reward / risk, 2)

    def _atr_stop(
        self, entry: float, atr_val: float, direction: int, mult: float
    ) -> float:
        """direction: +1 for long, -1 for short."""
        return entry - direction * mult * atr_val

    def _atr_target(
        self, entry: float, atr_val: float, direction: int, mult: float
    ) -> float:
        return entry + direction * mult * atr_val
