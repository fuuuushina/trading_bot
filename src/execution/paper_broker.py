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
import math
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
    leverage: float = 1.0   # >1 for forex margin positions
    initial_stop_loss: Optional[float] = None  # SL original pour calcul trailing

    @property
    def unrealized_pnl(self) -> float:
        direction = 1 if self.side == "long" else -1
        return direction * (self.current_price - self.avg_entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        # For leveraged positions: pnl relative to margin (not notional)
        notional = self.avg_entry_price * self.quantity
        cost = notional / self.leverage if self.leverage > 1.0 else notional
        return self.unrealized_pnl / cost if cost != 0 else 0.0

    @property
    def margin(self) -> float:
        """Cash posted as margin (= notional for unleveraged positions)."""
        return self.avg_entry_price * self.quantity / self.leverage

    @property
    def market_value(self) -> float:
        """
        Value used for portfolio equity calculation.
        For leveraged: margin + unrealized PnL (what you'd get back on close).
        For non-leveraged: standard quantity × current_price.
        """
        if self.leverage > 1.0:
            return self.margin + self.unrealized_pnl
        return self.quantity * self.current_price

    @property
    def notional_value(self) -> float:
        return self.quantity * self.current_price

    def to_dict(self) -> dict:
        d = {
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
            "initial_stop_loss": self.initial_stop_loss,
            "strategy": self.strategy_name,
            "horizon": self.horizon,
            "opened_at": self.opened_at.isoformat(),
        }
        if self.leverage > 1.0:
            d["leverage"] = self.leverage
            d["notional"] = round(self.notional_value, 2)
            d["margin"] = round(self.margin, 2)
        return d


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
        self._peak_equity: float = initial_capital   # High-water mark pour drawdown réel

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
        self,
        current_prices: dict[str, float],
        bar_highs: dict[str, float] | None = None,
        bar_lows:  dict[str, float] | None = None,
        max_hold_hours_intraday: float = 6.0,
        trail_be_r: float = 0.5,
    ) -> list[dict]:
        """
        Check open positions against stop-loss and take-profit levels.
        Uses bar HIGH/LOW (when provided) so intra-bar SL/TP are properly detected.
        Priority: SL > TP (worst-case). Intraday positions auto-close after max_hold_hours_intraday.

        Trailing stop (trail_be_r):
          - At profit >= trail_be_r × initial_risk → SL moved to entry (breakeven)
          - Every additional trail_be_r × initial_risk of profit → SL trails by trail_be_r × initial_risk

        Returns list of auto-close events.
        """
        events = []
        to_close: list[tuple[str, str, float]] = []

        import math as _m
        now = datetime.utcnow()

        for asset, pos in list(self._positions.items()):
            price = current_prices.get(asset)
            if price is None:
                continue
            try:
                price = float(price)
            except Exception:
                continue
            if not _m.isfinite(price) or price <= 0:
                continue

            pos.current_price = price

            # Use bar high/low when available — catch intra-bar SL/TP touches
            bar_high = bar_highs.get(asset, price) if bar_highs else price
            bar_low  = bar_lows.get(asset, price)  if bar_lows  else price

            if pos.side == "long":
                sl_hit = pos.stop_loss   and bar_low  <= pos.stop_loss
                tp_hit = pos.take_profit and bar_high >= pos.take_profit
                if sl_hit:
                    to_close.append((asset, "stop_loss",   pos.stop_loss))
                elif tp_hit:
                    to_close.append((asset, "take_profit", pos.take_profit))
            else:  # short
                sl_hit = pos.stop_loss   and bar_high >= pos.stop_loss
                tp_hit = pos.take_profit and bar_low  <= pos.take_profit
                if sl_hit:
                    to_close.append((asset, "stop_loss",   pos.stop_loss))
                elif tp_hit:
                    to_close.append((asset, "take_profit", pos.take_profit))

            # Trailing stop continu (high-water mark)
            # trail_be_r = distance de trailing exprimée en fraction du risque initial
            # Ex: trail_be_r=0.25 → distance = 25% du risque initial
            # Activation : dès que le profit >= trail_be_r × initial_risk
            already_closing = asset in [a for a, _, _ in to_close]
            if not already_closing and pos.stop_loss is not None and trail_be_r > 0:
                initial_sl = pos.initial_stop_loss or pos.stop_loss
                initial_risk = abs(pos.avg_entry_price - initial_sl)
                if initial_risk > 0:
                    entry = pos.avg_entry_price
                    trail_dist = trail_be_r * initial_risk

                    if pos.side == "long":
                        # Meilleur prix vu = maximum (utilise bar_high quand dispo)
                        prev_best = pos.metadata.get("trail_best_price", entry)
                        new_best  = max(prev_best, bar_high)
                        pos.metadata["trail_best_price"] = new_best
                        profit = new_best - entry

                        if profit >= trail_dist:
                            target_sl = new_best - trail_dist
                            if pos.stop_loss < target_sl:
                                logger.info(
                                    "Trailing SL %s long: %.5f → %.5f "
                                    "(best=%.5f trail=%.1f pips)",
                                    asset, pos.stop_loss, target_sl,
                                    new_best, trail_dist * 10000,
                                )
                                pos.stop_loss = round(target_sl, 6)
                    else:  # short
                        # Meilleur prix vu = minimum (utilise bar_low quand dispo)
                        prev_best = pos.metadata.get("trail_best_price", entry)
                        new_best  = min(prev_best, bar_low)
                        pos.metadata["trail_best_price"] = new_best
                        profit = entry - new_best

                        if profit >= trail_dist:
                            target_sl = new_best + trail_dist
                            if pos.stop_loss > target_sl:
                                logger.info(
                                    "Trailing SL %s short: %.5f → %.5f "
                                    "(best=%.5f trail=%.1f pips)",
                                    asset, pos.stop_loss, target_sl,
                                    new_best, trail_dist * 10000,
                                )
                                pos.stop_loss = round(target_sl, 6)

            # Sortie temporelle : ferme les positions intraday après N heures
            if asset not in [a for a, _, _ in to_close]:
                if pos.horizon == "intraday" and max_hold_hours_intraday > 0:
                    hold_hours = (now - pos.opened_at).total_seconds() / 3600
                    if hold_hours >= max_hold_hours_intraday:
                        logger.info(
                            "Time-exit: %s held %.1fh >= %.1fh — closing at market",
                            asset, hold_hours, max_hold_hours_intraday,
                        )
                        to_close.append((asset, "time_exit", price))

        for asset, reason, close_price in to_close:
            event = self._close_position(asset, close_price, reason)
            if event:
                events.append(event)

        return events

    def update_prices(self, current_prices: dict[str, float]) -> None:
        """Update mark-to-market prices without closing positions."""
        import math
        for asset, pos in self._positions.items():
            if asset in current_prices:
                price = current_prices[asset]
                # Skip NaN/inf prices (yfinance hors-session) — conserver le dernier prix valide
                if price is not None and math.isfinite(float(price)) and float(price) > 0:
                    pos.current_price = float(price)

    # ------------------------------------------------------------------ #
    # Portfolio state
    # ------------------------------------------------------------------ #

    def get_portfolio_state(self) -> dict:
        # Separate leveraged vs non-leveraged positions
        # For leveraged: market_value = margin + pnl → always ADD to equity (long or short)
        # For non-leveraged: standard long +, short -
        non_lev_long = sum(p.market_value for p in self._positions.values()
                           if p.side == "long" and p.leverage <= 1.0)
        non_lev_short = sum(p.market_value for p in self._positions.values()
                            if p.side == "short" and p.leverage <= 1.0)
        lev_value = sum(p.market_value for p in self._positions.values()
                        if p.leverage > 1.0)

        total_market_value = non_lev_long + non_lev_short + sum(
            p.notional_value for p in self._positions.values() if p.leverage > 1.0
        )
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        total_equity = self.cash + non_lev_long - non_lev_short + lev_value
        available_cash = max(0.0, self.cash)
        # High-water mark : met à jour le pic seulement si equity monte
        if total_equity > self._peak_equity:
            self._peak_equity = total_equity
        drawdown_pct = (total_equity - self._peak_equity) / max(self._peak_equity, 1e-10)

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
            "restricted_short_proceeds": round(non_lev_short, 2),
            "total_market_value": round(total_market_value, 2),
            "total_exposure": round(non_lev_long + non_lev_short + sum(
                p.margin for p in self._positions.values() if p.leverage > 1.0
            ), 2),
            "total_exposure_pct": round((non_lev_long + non_lev_short + sum(
                p.margin for p in self._positions.values() if p.leverage > 1.0
            )) / (total_equity + 1e-10), 4),
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
                "peak_equity":     round(self._peak_equity, 4),
                "cash":            round(self.cash, 4),
                "positions": [
                    {
                        "asset":              p.asset,
                        "side":               p.side,
                        "quantity":           p.quantity,
                        "avg_entry":          p.avg_entry_price,
                        "current_price":      (
                            p.current_price if math.isfinite(p.current_price)
                            else p.avg_entry_price
                        ),
                        "stop_loss":          p.stop_loss,
                        "take_profit":        p.take_profit,
                        "initial_stop_loss":  p.initial_stop_loss,
                        "strategy":           p.strategy_name,
                        "horizon":            p.horizon,
                        "opened_at":          p.opened_at.isoformat(),
                        "leverage":           p.leverage,
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
            self._peak_equity = float(data.get("peak_equity", self.initial_capital))
            self._trade_history = data.get("trade_history", [])
            self._positions = {}
            for pos in data.get("positions", []):
                opened = datetime.fromisoformat(pos["opened_at"]) if pos.get("opened_at") else datetime.utcnow()
                avg_entry = float(pos["avg_entry"])
                cur_price = pos.get("current_price")
                try:
                    _cp = float(cur_price)
                    _cp = _cp if math.isfinite(_cp) and _cp > 0 else avg_entry
                except (TypeError, ValueError):
                    _cp = avg_entry
                isl_raw = pos.get("initial_stop_loss") or pos.get("stop_loss")
                self._positions[pos["asset"]] = Position(
                    asset=pos["asset"],
                    side=pos["side"],
                    quantity=float(pos["quantity"]),
                    avg_entry_price=avg_entry,
                    current_price=_cp,
                    strategy_name=pos.get("strategy", "unknown"),
                    horizon=pos.get("horizon", "swing"),
                    stop_loss=pos.get("stop_loss"),
                    take_profit=pos.get("take_profit"),
                    initial_stop_loss=float(isl_raw) if isl_raw is not None else None,
                    leverage=float(pos.get("leverage", 1.0)),
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
        is_closing = existing is not None and (
            (existing.side == "long" and order.side == OrderSide.SELL)
            or (existing.side == "short" and order.side == OrderSide.BUY)
        )
        if is_closing and order.quantity > existing.quantity:
            order.metadata["quantity_capped_to_position"] = existing.quantity
            order.quantity = existing.quantity

        # Inherit leverage from existing position when closing (auto-close via SL/TP)
        leverage = float(order.metadata.get("leverage", 1.0))
        if leverage <= 1.0 and existing is not None and is_closing:
            leverage = getattr(existing, "leverage", 1.0)

        commission = self.commission_flat + fill_price * order.quantity * self.commission_pct
        gross_value = fill_price * order.quantity

        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.metadata["rejection_reason"] = "Order quantity is zero after position cap."
            self._order_history.append(order)
            logger.warning("Order rejected (quantity): %s %s", order.order_id, order.asset)
            return

        # ---- Cash accounting (leverage-aware) ----
        if leverage > 1.0:
            if is_closing:
                # Return margin + PnL when closing a leveraged position
                closed_qty = min(order.quantity, existing.quantity)
                direction = 1 if existing.side == "long" else -1
                pnl = direction * (fill_price - existing.avg_entry_price) * closed_qty
                margin_back = existing.avg_entry_price * closed_qty / leverage
                self.cash += margin_back + pnl - commission
            else:
                # Opening: deduct only margin (not full notional)
                margin_needed = gross_value / leverage + commission
                if margin_needed > self.cash:
                    order.status = OrderStatus.REJECTED
                    order.metadata["rejection_reason"] = (
                        f"Insufficient margin: need ${margin_needed:.2f}, have ${self.cash:.2f}"
                    )
                    self._order_history.append(order)
                    logger.warning("Order rejected (margin): %s %s", order.order_id, order.asset)
                    return
                self.cash -= margin_needed
        else:
            # Standard non-leveraged accounting
            if order.side == OrderSide.BUY and gross_value + commission > self.cash:
                order.status = OrderStatus.REJECTED
                order.metadata["rejection_reason"] = (
                    f"Insufficient cash: need ${gross_value + commission:.2f}, "
                    f"have ${self.cash:.2f}"
                )
                self._order_history.append(order)
                logger.warning("Order rejected (cash): %s %s", order.order_id, order.asset)
                return
            if order.side == OrderSide.BUY:
                self.cash -= gross_value + commission
            else:
                self.cash += gross_value - commission

        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = round(fill_price, 6)
        order.commission = round(commission, 6)
        order.slippage = round(abs(fill_price - base_price) * order.quantity, 6)
        order.filled_at = datetime.utcnow()
        self._order_history.append(order)

        if order.side == OrderSide.BUY:
            if existing is not None and existing.side == "short":
                self._reduce_or_close_position(order, fill_price)
            else:
                self._open_or_add_position(order, fill_price, leverage=leverage)
        else:
            if existing is not None and existing.side == "long":
                self._reduce_or_close_position(order, fill_price)
            else:
                self._open_or_add_position(order, fill_price, leverage=leverage)

        logger.info(
            "FILL: %s %s %s x%.2f @ %.6f lev=%.0fx commission=%.4f",
            order.order_id, order.side.value, order.asset,
            order.quantity, fill_price, leverage, commission
        )

    def _open_or_add_position(
        self, order: Order, fill_price: float, leverage: float = 1.0
    ) -> None:
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
                initial_stop_loss=meta.get("stop_loss"),
                leverage=leverage,
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
            "asset":        order.asset,
            "strategy":     order.strategy_name,
            "horizon":      order.horizon,
            "side":         pos.side,
            "quantity":     closed_qty,
            "entry_price":  pos.avg_entry_price,
            "exit_price":   fill_price,
            "stop_loss":    pos.stop_loss,
            "take_profit":  pos.take_profit,
            "close_reason": order.metadata.get("auto_close_reason", "signal"),
            "pnl":          round(realized_pnl, 4),
            "pnl_pct":      round(realized_pnl / (pos.avg_entry_price * closed_qty), 4),
            "commission":   order.commission,
            "closed_at":    datetime.utcnow().isoformat(),
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
        # Propagate leverage so _fill_order uses correct margin accounting
        close_order = self.submit_market_order(
            asset=asset,
            side=side,
            quantity=pos.quantity,
            strategy_name=pos.strategy_name,
            horizon=pos.horizon,
            metadata={"auto_close_reason": reason, "leverage": pos.leverage},
        )
        self.fill_pending_orders({asset: price})
        logger.info("Auto-close %s: %s @ %.6f (reason=%s)", asset, side.value, price, reason)
        return {"asset": asset, "reason": reason, "price": price}
