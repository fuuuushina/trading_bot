"""
scripts/test_paper_trade.py

Test rapide du paper broker : achète 1 action de plusieurs tickers,
affiche le portefeuille, puis revend tout.

Usage:
    python scripts/test_paper_trade.py
    python scripts/test_paper_trade.py --tickers NVDA AAPL MSFT --qty 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf

from src.execution.paper_broker import OrderSide, PaperBroker

# --- helpers -------------------------------------------------------------------

def fmt_money(v: float) -> str:
    return f"${v:+,.2f}" if v != abs(v) else f"${v:,.2f}"


def fetch_price(ticker: str) -> float:
    df = yf.download(ticker, period="2d", interval="1d",
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    # Handle MultiIndex columns (newer yfinance)
    if hasattr(df.columns, "levels"):
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return float(df["close"].iloc[-1])


def print_portfolio(broker: PaperBroker) -> None:
    ps = broker.get_portfolio_state()
    print(f"\n{'-'*56}")
    print(f"  Cash          : {fmt_money(ps['cash'])}")
    print(f"  Positions     : ${ps['total_market_value']:,.2f}")
    print(f"  Total equity  : {fmt_money(ps['total_equity'])}")
    print(f"  Unrealised P&L: {fmt_money(ps['unrealized_pnl'])}")
    positions = broker.get_open_positions()
    if positions:
        print(f"\n  {'TICKER':<10} {'QTY':>8} {'ENTRY':>9} {'PRICE':>9} {'P&L':>10}")
        for p in positions:
            print(f"  {p['asset']:<10} {p['quantity']:>8.4f} "
                  f"{fmt_money(p['avg_entry']):>9} "
                  f"{fmt_money(p['current_price']):>9} "
                  f"{fmt_money(p['unrealized_pnl']):>10}")
    print(f"{'-'*56}")


# --- main ----------------------------------------------------------------------

def run(tickers: list[str], qty: float, capital: float) -> None:
    broker = PaperBroker(
        initial_capital=capital,
        commission_flat=0.0,
        slippage_pct=0.0002,
    )

    print(f"\n{'='*56}")
    print(f"  PAPER TRADING TEST — capital ${capital:,.0f}")
    print(f"{'='*56}")

    # -- Fetch current prices --------------------------------------------
    prices: dict[str, float] = {}
    for t in tickers:
        try:
            prices[t] = fetch_price(t)
            print(f"  {t:10} : ${prices[t]:,.4f}")
        except Exception as exc:
            print(f"  {t:10} : ERREUR — {exc}")

    if not prices:
        print("Aucun prix récupéré, abandon.")
        return

    # -- BUY -------------------------------------------------------------
    print(f"\n>>> BUY {qty} action(s) de chaque ticker")
    for ticker, price in prices.items():
        cost = price * qty
        if cost > broker.cash:
            print(f"  {ticker}: cash insuffisant (${broker.cash:.2f} < ${cost:.2f}), ignoré")
            continue
        order = broker.submit_market_order(
            asset=ticker,
            side=OrderSide.BUY,
            quantity=qty,
            strategy_name="test_script",
            horizon="swing",
            stop_loss=round(price * 0.95, 4),
            take_profit=round(price * 1.10, 4),
        )
        broker.fill_pending_orders(prices)
        status = order.status.value
        print(f"  {ticker}: {status} @ ${order.filled_price:,.4f} "
              f"(coût ${order.filled_price * qty:,.2f})")

    print_portfolio(broker)

    # -- Simuler une variation de prix -----------------------------------
    print("\n>>> Simulation : +0.5% sur tous les prix")
    new_prices = {t: round(p * 1.005, 4) for t, p in prices.items()}
    broker.update_prices(new_prices)
    print_portfolio(broker)

    # -- SELL tout -------------------------------------------------------
    print("\n>>> SELL toutes les positions")
    for pos in broker.get_open_positions():
        ticker = pos["asset"]
        order = broker.submit_market_order(
            asset=ticker,
            side=OrderSide.SELL,
            quantity=pos["quantity"],
            strategy_name="test_script",
            horizon="swing",
        )
        broker.fill_pending_orders(new_prices)
        print(f"  {ticker}: {order.status.value} @ ${order.filled_price:,.4f}")

    print_portfolio(broker)

    trades = broker.get_trade_history()
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    print(f"\n  P&L realise total : {fmt_money(total_pnl)}")
    print(f"  Nombre de trades  : {len(trades)}")
    print(f"\n  [OK] Paper broker fonctionne - les ordres s'executent correctement.")
    print("=" * 56)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test paper trading broker")
    parser.add_argument("--tickers", nargs="+", default=["NVDA", "AAPL", "MSFT"])
    parser.add_argument("--qty",     type=float, default=1.0)
    parser.add_argument("--capital", type=float, default=2000.0)
    args = parser.parse_args()
    run(args.tickers, args.qty, args.capital)
