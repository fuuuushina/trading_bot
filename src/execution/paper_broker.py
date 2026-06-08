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
            # Always update unrealized PnL even if price wasn't fetched this cycle
            # This forces recalculation based on current stored price

    # ------------------------------------------------------------------ #
    # Portfolio state
    # ------------------------------------------------------------------ #

    def get_portfolio_state(self) -> dict:
        long_market_value = sum(p.market_value for p in self._positions.values() if p.side == "long")
        short_market_value = sum(p.market_value for p in self._positions.values() if p.side == "short")
        total_market_value = long_market_value + short_market_value
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        total_equity = self.cash + long_market_value - short_market_value
        available_cash = max(0.0, self.cash - short_market_value)
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
            "available_cash": round(available_cash, 2),
            "restricted_short_proceeds": round(short_market_value, 2),
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
    # Persistence — save / load state to survive restarts
    # ------------------------------------------------------------------ #

    def save_state(self, path: str = "data/paper_trading/broker_state.json") -> None:
        """Persist cash, positions and trade history to JSON."""
        import json
        from pathlib import Path
        try:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "initial_capital": self.initial_capital,
                "cash":            round(self.cash, 4),
                "positions": [
                    {
                        "asset":         p.asset,
                        "side":          p.side,
                        "quantity":      p.quantity,
                        "avg_entry":     p.avg_entry_price,
                        "current_price": p.current_price,
                        "stop_loss":     p.stop_loss,
                        "take_profit":   p.take_profit,
                        "strategy":      p.strategy_name,
                        "horizon":       p.horizon,
                        "opened_at":     p.opened_at.isoformat(),
                    }
                    for p in self._positions.values()
                ],
                "trade_history": self._trade_history,
            }
            tmp = out.with_suffix(out.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2, default=str)
            tmp.replace(out)
        except Exception as exc:
            logger.warning("save_state failed: %s", exc)

    def load_state(self, path: str = "data/paper_trading/broker_state.json") -> bool:
        """Restore positions and trade history from a previous session. Returns True if loaded."""
        import json
        from pathlib import Path
        try:
            p = Path(path)
            if not p.exists():
                return False
            data = json.loads(p.read_text(encoding="utf-8"))
            self.cash = float(data.get("cash", self.initial_capital))
            self._trade_history = data.get("trade_history", [])
            self._positions = {}
            for pos in data.get("positions", []):
                opened = datetime.fromisoformat(pos["opened_at"]) if pos.get("opened_at") else datetime.utcnow()
                self._positions[pos["asset"]] = Position(
                    asset=pos["asset"],
                    side=pos["side"],
                    quantity=float(pos["quantity"]),
                    avg_entry_price=float(pos["avg_entry"]),
                    current_price=float(pos["current_price"]),
                    strategy_name=pos.get("strategy", "unknown"),
                    horizon=pos.get("horizon", "swing"),
                    stop_loss=pos.get("stop_loss"),
                    take_profit=pos.get("take_profit"),
                    opened_at=opened,
                )
            logger.info(
                "Broker state loaded: cash=$%.2f, %d positions, %d trades",
                self.cash, len(self._positions), len(self._trade_history),
            )
            return True
        except Exception as exc:
            logger.warning("load_state failed: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _fill_order(self, order: Order, base_price: float) -> None:
        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + self.slippage_pct)
        else:
            fill_price = base_price * (1 - self.slippage_pct)

        existing = self._positions.get(order.asset)
        if existing is not None:
            closes_existing = (
                (existing.side == "long" and order.side == OrderSide.SELL)
                or (existing.side == "short" and order.side == OrderSide.BUY)
            )
            if closes_existing and order.quantity > existing.quantity:
                order.metadata["quantity_capped_to_position"] = existing.quantity
                order.quantity = existing.quantity

        commission = self.commission_flat + fill_price * order.quantity * self.commission_pct
        gross_value = fill_price * order.quantity

        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = "Order quantity is zero after position cap."
            self._order_history.append(order)
            logger.warning("Order rejected (quantity): %s %s", order.order_id, order.asset)
            return

        if order.side == OrderSide.BUY and gross_value + commission > self.cash:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = (
                f"Insufficient cash: need ${gross_value + commission:.2f}, "
                f"have ${self.cash:.2f}"
            )
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

        existing = self._positions.get(order.asset)
        if order.side == OrderSide.BUY:
            self.cash -= gross_value + commission
            if existing is not None and existing.side == "short":
                self._reduce_or_close_position(order, fill_price)
            else:
                self._open_or_add_position(order, fill_price)
        else:
            self.cash += gross_value - commission
            if existing is not None and existing.side == "long":
                self._reduce_or_close_position(order, fill_price)
            else:
                self._open_or_add_position(order, fill_price)

        logger.info(
            "FILL: %s %s %s x%.2f @ %.4f commission=%.2f",
            order.order_id, order.side.value, order.asset,
            order.quantity, fill_price, commission
        )

    def _open_or_add_position(self, order: Order, fill_price: float) -> None:
        meta = order.metadata
        side = "long" if order.side == OrderSide.BUY else "short"
        if order.asset in self._positions:
            pos = self._positions[order.asset]
            if pos.side != side:
                logger.warning(
                    "Refusing to add %s order to existing %s position in %s",
                    side, pos.side, order.asset,
                )
                return
            total_qty = pos.quantity + order.quantity
            pos.avg_entry_price = (
                pos.avg_entry_price * pos.quantity + fill_price * order.quantity
            ) / total_qty
            pos.quantity = total_qty
            pos.current_price = fill_price
        else:
            self._positions[order.asset] = Position(
                asset=order.asset,
                side=side,
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
        closed_qty = min(order.quantity, pos.quantity)
        if closed_qty <= 0:
            return
        direction = 1 if pos.side == "long" else -1
        realized_pnl = direction * (fill_price - pos.avg_entry_price) * closed_qty - order.commission

        self._trade_history.append({
            "asset": order.asset,
            "strategy": order.strategy_name,
            "horizon": order.horizon,
            "side": pos.side,
            "quantity": closed_qty,
            "entry_price": pos.avg_entry_price,
            "exit_price": fill_price,
            "pnl": round(realized_pnl, 4),
            "pnl_pct": round(realized_pnl / (pos.avg_entry_price * closed_qty), 4),
            "commission": order.commission,
            "closed_at": datetime.utcnow().isoformat(),
        })

        if closed_qty >= pos.quantity:
            del self._positions[order.asset]
        else:
            pos.quantity -= closed_qty
            pos.current_price = fill_price

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
