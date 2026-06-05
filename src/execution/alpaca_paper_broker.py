"""
Alpaca paper broker adapter.

This module is for PAPER trading only unless explicitly reworked for live.
It reads credentials from environment variables loaded by config.loader.

Useful checks:
    python -m src.execution.alpaca_paper_broker --check
    python -m src.execution.alpaca_paper_broker --positions
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

from config.loader import get_env, get_settings


@dataclass
class AlpacaAccountSnapshot:
    account_id: str
    status: str
    currency: str
    cash: float
    buying_power: float
    portfolio_value: float
    paper: bool


class AlpacaPaperBroker:
    """Thin wrapper around alpaca-py for broker-paper validation."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://paper-api.alpaca.markets/v2",
    ) -> None:
        from alpaca.trading.client import TradingClient

        base_url = normalize_alpaca_base_url(base_url)

        if base_url != "https://paper-api.alpaca.markets":
            raise ValueError(
                "Refusing non-paper Alpaca URL. Expected https://paper-api.alpaca.markets/v2"
            )

        self.base_url = base_url
        self.client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=True,
            url_override=base_url,
        )

    @classmethod
    def from_env(cls) -> "AlpacaPaperBroker":
        settings = get_settings()
        live_cfg = settings.get("broker", {}).get("live", {})
        api_key = get_env(live_cfg.get("api_key_env", "BROKER_API_KEY"))
        api_secret = get_env(live_cfg.get("api_secret_env", "BROKER_API_SECRET"))
        base_url = get_env(
            live_cfg.get("base_url_env", "BROKER_BASE_URL"),
            "https://paper-api.alpaca.markets/v2",
        )
        return cls(api_key=api_key, api_secret=api_secret, base_url=base_url)

    def get_account_snapshot(self) -> AlpacaAccountSnapshot:
        account = self.client.get_account()
        return AlpacaAccountSnapshot(
            account_id=str(account.id),
            status=str(account.status),
            currency=str(account.currency),
            cash=_to_float(account.cash),
            buying_power=_to_float(account.buying_power),
            portfolio_value=_to_float(account.portfolio_value),
            paper=True,
        )

    def get_positions(self) -> list[dict[str, Any]]:
        positions = self.client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": _to_float(p.qty),
                "side": _enum_value(p.side),
                "market_value": _to_float(p.market_value),
                "avg_entry_price": _to_float(p.avg_entry_price),
                "unrealized_pl": _to_float(p.unrealized_pl),
                "unrealized_plpc": _to_float(p.unrealized_plpc),
            }
            for p in positions
        ]

    def get_orders(self, limit: int = 20) -> list[dict[str, Any]]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self.client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
        )
        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": _enum_value(o.side),
                "type": _enum_value(o.type),
                "qty": _to_float(o.qty),
                "filled_qty": _to_float(o.filled_qty),
                "status": _enum_value(o.status),
                "submitted_at": str(o.submitted_at),
                "filled_at": str(o.filled_at) if o.filled_at else None,
            }
            for o in orders
        ]

    def submit_market_order(
        self,
        symbol: str,
        qty: float | None = None,
        notional: float | None = None,
        side: str = "buy",
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        if qty is None and notional is None:
            raise ValueError("Either qty or notional is required.")
        if qty is not None and notional is not None:
            raise ValueError("Use qty or notional, not both.")

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.DAY if time_in_force.lower() == "day" else TimeInForce.GTC
        request = MarketOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            notional=notional,
            side=order_side,
            time_in_force=tif,
        )
        order = self.client.submit_order(order_data=request)
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": _enum_value(order.side),
            "qty": _to_float(order.qty),
            "notional": _to_float(getattr(order, "notional", None)),
            "status": str(order.status),
            "submitted_at": str(order.submitted_at),
        }

    def close_position(self, symbol: str, qty: float | None = None) -> dict[str, Any]:
        from alpaca.trading.requests import ClosePositionRequest

        close_options = ClosePositionRequest(qty=str(qty)) if qty is not None else None
        order = self.client.close_position(symbol.upper(), close_options=close_options)
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": _enum_value(order.side),
            "qty": _to_float(order.qty),
            "status": str(order.status),
            "submitted_at": str(order.submitted_at),
        }


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def normalize_alpaca_base_url(base_url: str) -> str:
    """
    alpaca-py appends /v2 internally, so accept either user-facing endpoint:
    https://paper-api.alpaca.markets/v2 or https://paper-api.alpaca.markets.
    """
    clean = base_url.strip().rstrip("/")
    if clean.endswith("/v2"):
        clean = clean[:-3]
    return clean


def _print_env_check() -> None:
    api_key = get_env("BROKER_API_KEY", "")
    api_secret = get_env("BROKER_API_SECRET", "")
    base_url = get_env("BROKER_BASE_URL", "")
    print("BROKER_API_KEY:", _masked_status(api_key))
    print("BROKER_API_SECRET:", _masked_status(api_secret))
    print("BROKER_BASE_URL:", base_url or "MISSING")


def _masked_status(value: str) -> str:
    if not value:
        return "MISSING"
    return (
        f"set len={len(value)} "
        f"prefix={value[:3]} "
        f"outer_spaces={value != value.strip()}"
    )


def _print_account(snapshot: AlpacaAccountSnapshot) -> None:
    print("Alpaca PAPER account OK")
    data = asdict(snapshot)
    data["account_id"] = data["account_id"][:8] + "..."
    for key, value in data.items():
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpaca paper broker helper")
    parser.add_argument("--check", action="store_true", help="Validate account connection")
    parser.add_argument("--positions", action="store_true", help="Print paper positions")
    parser.add_argument("--orders", action="store_true", help="Print recent paper orders")
    parser.add_argument("--env-check", action="store_true", help="Print masked env diagnostics")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.env_check:
        _print_env_check()
        return

    broker = AlpacaPaperBroker.from_env()

    if args.positions:
        for position in broker.get_positions():
            print(position)
        return

    if args.orders:
        for order in broker.get_orders(args.limit):
            print(order)
        return

    try:
        _print_account(broker.get_account_snapshot())
    except Exception as exc:
        message = str(exc)
        if "unauthorized" in message.lower() or "401" in message:
            print("Alpaca PAPER connection failed: unauthorized.")
            print("Check that .env contains the real PAPER key/secret pair, not placeholders.")
            print("Also verify BROKER_BASE_URL=https://paper-api.alpaca.markets/v2")
            return
        raise


if __name__ == "__main__":
    main()
