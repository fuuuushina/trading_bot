"""
src/execution/paper_broker.py

Paper trading broker. Simulates realistic order execution with:
  - Commission (flat + percentage)
  - Slippage
  - Fill at next-bar open (when using daily bars)
  - Order state management
  - Full audit trail

This is the ONLY execution path in paper mode.
Live broker is a separate module requiring explicit unlock.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    order_id: str
    asset: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float]
    stop_price: Optional[float]
    strategy_name: str
    horizon: str
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "asset": self.asset,
            "side": self.side.value,
            "type": self.order_type.value,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "strategy": self.strategy_name,
            "horizon": self.horizon,
            "status": self.status.value,
            "filled_qty": self.filled_quantity,
            "filled_price": self.filled_price,
            "commission": self.commission,
            "slippage": self.slippage,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "metadata": self.metadata,
        }

    @property
    def gross_value(self) -> float:
        return self.filled_quantity * self.filled_price

    @property
    def net_value(self) -> float:
        return self.gross_value + self.commission


@dataclass
class Position:
    asset: str
    side: str               # long | short
    quantity: float
    avg_entry_price: float
    current_price: float
    strategy_name: str
    horizon: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    @property
    def unrealized_pnl(self) -> float:
        direction = 1 if self.side == "long" else -1
        return direction * (self.current_price - self.avg_entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        cost = self.avg_entry_price * self.quantity
        return self.unrealized_pnl / cost if cost != 0 else 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "side": self.side,
            "quantity": self.quantity,
            "avg_entry": self.avg_entry_price,
            "current_price": self.current_price,
            "market_value": round(self.market_value, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct, 4),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy": self.strategy_name,
            "horizon": self.horizon,
            "opened_at": self.opened_at.isoformat(),
        }


class PaperBroker:
    """
    Simulated broker for paper trading.

    Maintains:
      - Cash balance
      - Open positions
      - Order history
      - Trade history (closed positions)

    Usage:
        broker = PaperBroker(initial_capital=100_000, commission=1.0, slippage_pct=0.001)
        order = broker.submit_market_order("SPY", OrderSide.BUY, 10, "trend_following", "swing")
        broker.fill_pending_orders({"SPY": 450.0})   # call with current prices
    """

    def __init__(
        self,
        initial_capital: float,
        commission_flat: float = 1.0,
        commission_pct: float = 0.0,
        slippage_pct: float = 0.001,
    ) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_flat = commission_flat
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        self._positions: dict[str, Position] = {}   # asset → Position
        self._pending_orders: list[Order] = []
        self._order_history: list[Order] = []
        self._trade_history: list[dict] = []

    # ------------------------------------------------------------------ #
    # Order submission
    # ------------------------------------------------------------------ #

    def submit_market_order(
        self,
        asset: str,
        side: OrderSide,
        quantity: float,
        strategy_name: str,
        horizon: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: dict | None = None,
    ) -> Order:
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            asset=asset,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            limit_price=None,
            stop_price=None,
            strategy_name=strategy_name,
            horizon=horizon,
            metadata={
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                **(metadata or {}),
            },
        )
        self._pending_orders.append(order)
        logger.info("Order submitted: %s %s %s x%.4f", order.order_id, side.value, asset, quantity)
        return order

    def submit_limit_order(
        self,
        asset: str,
        side: OrderSide,
        quantity: float,
        limit_price: float,
        strategy_name: str,
        horizon: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Order:
        order = Order(
            order_id=str(uuid.uuid4())[:8],
            asset=asset,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=None,
            strategy_name=strategy_name,
            horizon=horizon,
            metadata={"stop_loss": stop_loss, "take_profit": take_profit},
        )
        self._pending_orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        for order in self._pending_orders:
            if order.order_id == order_id:
                order.status = OrderStatus.CANCELLED
                self._pending_orders.remove(order)
                self._order_history.append(order)
                logger.info("Order cancelled: %s", order_id)
                return True
        return False

    # ------------------------------------------------------------------ #
    # Order processing (call once per bar)
    # ------------------------------------------------------------------ #

    def fill_pending_orders(self, current_prices: dict[str, float]) -> list[Order]:
        """
        Attempt to fill all pending orders against current bar prices.
        Returns list of filled orders.
        """
        filled: list[Order] = []
        still_pending: list[Order] = []

        for order in self._pending_orders:
            price = current_prices.get(order.asset)
            if price is None:
                still_pending.append(order)
                continue

            if order.order_type == OrderType.MARKET:
                self._fill_order(order, price)
                filled.append(order)
            elif order.order_type == OrderType.LIMIT:
                if (order.side == OrderSide.BUY and price <= order.limit_price) or \
                   (order.side == OrderSide.SELL and price >= order.limit_price):
                    self._fill_order(order, order.limit_price)
                    filled.append(order)
                else:
                    still_pending.append(order)
            else:
                still_pending.append(order)

        self._pending_orders = still_pending
        return filled

    def check_stops_and_targets(
        self, current_prices: dict[str, float]
    ) -> list[dict]:
        """
        Check open positions against stop-loss and take-profit levels.
        Automatically closes positions that have hit their levels.
        Returns list of auto-close events.
        """
        events = []
        to_close: list[tuple[str, str, float]] = []  # (asset, reason, price)

        for asset, pos in self._positions.items():
            price = current_prices.get(asset)
            if price is None:
                continue

            pos.current_price = price

            if pos.side == "long":
                if pos.stop_loss and price <= pos.stop_loss:
                    to_close.append((asset, "stop_loss", price))
                elif pos.take_profit and price >= pos.take_profit:
                    to_close.append((asset, "take_profit", price))
            else:  # short
                if pos.stop_loss and price >= pos.stop_loss:
                    to_close.append((asset, "stop_loss", price))
                elif pos.take_profit and price <= pos.take_profit:
                    to_close.append((asset, "take_profit", price))

        for asset, reason, price in to_close:
            event = self._close_position(asset, price, reason)
            if event:
                events.append(event)

        return events

    def update_prices(self, current_prices: dict[str, float]) -> None:
        """Update mark-to-market prices without closing positions."""
        for asset, pos in self._positions.items():
            if asset in current_prices:
                pos.current_price = current_prices[asset]

    # ------------------------------------------------------------------ #
    # Portfolio state
    # ------------------------------------------------------------------ #

    def get_portfolio_state(self) -> dict:
        total_market_value = sum(p.market_value for p in self._positions.values())
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        total_equity = self.cash + total_market_value
        peak_equity = self.initial_capital  # simplified — track running peak separately
        drawdown_pct = (total_equity - peak_equity) / peak_equity

        asset_exposure = {a: p.market_value for a, p in self._positions.items()}
        horizon_exposure: dict[str, float] = {}
        for pos in self._positions.values():
            horizon_exposure[pos.horizon] = (
                horizon_exposure.get(pos.horizon, 0.0) + pos.market_value
            )

        realized_pnl = sum(t.get("pnl", 0.0) for t in self._trade_history)

        return {
            "total_capital": total_equity,
            "cash": round(self.cash, 2),
            "total_market_value": round(total_market_value, 2),
            "total_exposure": round(total_market_value, 2),
            "total_exposure_pct": round(total_market_value / (total_equity + 1e-10), 4),
            "total_equity": round(total_equity, 2),
            "unrealized_pnl": round(total_unrealized, 2),
            "realized_pnl": round(realized_pnl, 2),
            "drawdown_pct": round(drawdown_pct, 4),
            "open_positions": len(self._positions),
            "asset_exposure": {a: round(v, 2) for a, v in asset_exposure.items()},
            "horizon_exposure": {h: round(v, 2) for h, v in horizon_exposure.items()},
        }

    def get_open_positions(self) -> list[dict]:
        return [p.to_dict() for p in self._positions.values()]

    def get_trade_history(self) -> list[dict]:
        return self._trade_history.copy()

    def get_order_history(self) -> list[dict]:
        return [o.to_dict() for o in self._order_history]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _fill_order(self, order: Order, base_price: float) -> None:
        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + self.slippage_pct)
        else:
            fill_price = base_price * (1 - self.slippage_pct)

        commission = self.commission_flat + fill_price * order.quantity * self.commission_pct
        cost = fill_price * order.quantity + (commission if order.side == OrderSide.BUY else -commission)

        if order.side == OrderSide.BUY and cost > self.cash:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = f"Insufficient cash: need ${cost:.2f}, have ${self.cash:.2f}"
            self._order_history.append(order)
            logger.warning("Order rejected (cash): %s %s", order.order_id, order.asset)
            return

        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = round(fill_price, 4)
        order.commission = round(commission, 4)
        order.slippage = round(abs(fill_price - base_price) * order.quantity, 4)
        order.filled_at = datetime.utcnow()

        self._order_history.append(order)

        if order.side == OrderSide.BUY:
            self.cash -= cost
            self._open_or_add_position(order, fill_price)
        else:
            self.cash += fill_price * order.quantity - commission
            self._reduce_or_close_position(order, fill_price)

        logger.info(
            "FILL: %s %s %s x%.2f @ %.4f commission=%.2f",
            order.order_id, order.side.value, order.asset,
            order.quantity, fill_price, commission
        )

    def _open_or_add_position(self, order: Order, fill_price: float) -> None:
        meta = order.metadata
        if order.asset in self._positions:
            pos = self._positions[order.asset]
            total_qty = pos.quantity + order.quantity
            pos.avg_entry_price = (
                pos.avg_entry_price * pos.quantity + fill_price * order.quantity
            ) / total_qty
            pos.quantity = total_qty
        else:
            self._positions[order.asset] = Position(
                asset=order.asset,
                side="long" if order.side == OrderSide.BUY else "short",
                quantity=order.quantity,
                avg_entry_price=fill_price,
                current_price=fill_price,
                strategy_name=order.strategy_name,
                horizon=order.horizon,
                stop_loss=meta.get("stop_loss"),
                take_profit=meta.get("take_profit"),
                metadata=meta,
            )

    def _reduce_or_close_position(self, order: Order, fill_price: float) -> None:
        if order.asset not in self._positions:
            return
        pos = self._positions[order.asset]
        direction = 1 if pos.side == "long" else -1
        realized_pnl = direction * (fill_price - pos.avg_entry_price) * order.quantity - order.commission

        self._trade_history.append({
            "asset": order.asset,
            "strategy": order.strategy_name,
            "horizon": order.horizon,
            "side": pos.side,
            "quantity": order.quantity,
            "entry_price": pos.avg_entry_price,
            "exit_price": fill_price,
            "pnl": round(realized_pnl, 4),
            "pnl_pct": round(realized_pnl / (pos.avg_entry_price * order.quantity), 4),
            "commission": order.commission,
            "closed_at": datetime.utcnow().isoformat(),
        })

        if order.quantity >= pos.quantity:
            del self._positions[order.asset]
        else:
            pos.quantity -= order.quantity

    def _close_position(
        self, asset: str, price: float, reason: str
    ) -> Optional[dict]:
        if asset not in self._positions:
            return None
        pos = self._positions[asset]
        side = OrderSide.SELL if pos.side == "long" else OrderSide.BUY
        close_order = self.submit_market_order(
            asset=asset,
            side=side,
            quantity=pos.quantity,
            strategy_name=pos.strategy_name,
            horizon=pos.horizon,
            metadata={"auto_close_reason": reason},
        )
        self.fill_pending_orders({asset: price})
        logger.info("Auto-close %s: %s @ %.4f (reason=%s)", asset, side.value, price, reason)
        return {"asset": asset, "reason": reason, "price": price}
