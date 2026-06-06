"""
Alpaca paper trading engine.

Uses a real Alpaca PAPER account for orders and positions, while enforcing a
software budget cap from config/settings.yaml.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytz

from config.loader import get_risk_config, get_settings, get_strategy_config
from src.data.yfinance_helpers import configure_yfinance_cache, normalize_yfinance_columns
from src.execution.alpaca_paper_broker import AlpacaPaperBroker
from src.features.indicators import compute_all_features
from src.features.regime_detector import MarketRegimeDetector
from src.strategies.base import SignalType
from src.strategies.breakout import BreakoutStrategy
from src.strategies.tactical_dca import TacticalDCAStrategy
from src.strategies.true_dca import TrueDCAStrategy
from src.strategies.rsi_dip_buyer import RSIDipBuyerStrategy
from src.strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)

STATE_FILE = Path("data/paper_trading/alpaca_state.json")
_ET = pytz.timezone("America/New_York")
_REGIME_DET = MarketRegimeDetector()


@dataclass
class AlpacaPaperState:
    initial_capital: float
    equity_history: list[dict[str, Any]] = field(default_factory=list)
    signals_log: list[dict[str, Any]] = field(default_factory=list)
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)
    closed_events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, initial_capital: float) -> "AlpacaPaperState":
        if not STATE_FILE.exists():
            return cls(initial_capital=initial_capital)
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return cls(
                initial_capital=float(data.get("initial_capital", initial_capital)),
                equity_history=list(data.get("equity_history", [])),
                signals_log=list(data.get("signals_log", [])),
                targets=dict(data.get("targets", {})),
                closed_events=list(data.get("closed_events", [])),
            )
        except Exception as exc:
            logger.warning("Could not load Alpaca paper state: %s", exc)
            return cls(initial_capital=initial_capital)

    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps(
                {
                    "initial_capital": self.initial_capital,
                    "equity_history": self.equity_history[-5000:],
                    "signals_log": self.signals_log[-500:],
                    "targets": self.targets,
                    "closed_events": self.closed_events[-500:],
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def log_signal(self, ticker: str, strategy: str, signal: str, price: float, reason: str) -> None:
        self.signals_log.append(
            {
                "time": _now_str(),
                "ticker": ticker,
                "strategy": strategy,
                "signal": signal,
                "price": round(price, 2),
                "reason": reason[:160],
            }
        )
        self.signals_log = self.signals_log[-500:]

    def snapshot_equity(self, equity: float) -> None:
        self.equity_history.append({"time": _now_str(), "equity": round(equity, 2)})
        self.equity_history = self.equity_history[-5000:]

    def realized_pnl(self) -> float:
        return round(sum(float(x.get("pnl", 0.0)) for x in self.closed_events), 2)


class AlpacaPaperTradingEngine:
    """Runs strategy signals and submits orders to Alpaca PAPER only."""

    def __init__(
        self,
        watchlist: list[str],
        initial_capital: float,
        min_position_usd: float,
        max_position_usd: float,
        risk_pct_trade: float,
        slippage_pct: float,
        execution_enabled: bool,
        refresh_seconds: int,
    ) -> None:
        self.watchlist = [x.upper() for x in watchlist]
        self.initial_capital = float(initial_capital)
        self.min_position_usd = float(min_position_usd)
        self.max_position_usd = float(max_position_usd)
        self.risk_pct_trade = float(risk_pct_trade)
        self.slippage_pct = float(slippage_pct)
        self.execution_enabled = bool(execution_enabled)
        self.refresh_seconds = int(refresh_seconds)
        self.state = AlpacaPaperState.load(self.initial_capital)
        self.broker = AlpacaPaperBroker.from_env()
        self._strategies = self._build_strategies()
        self._running = False
        self._thread: threading.Thread | None = None
        self.last_update = ""
        self.last_error = ""

    @classmethod
    def from_config(cls) -> "AlpacaPaperTradingEngine":
        settings = get_settings()
        risk_cfg = get_risk_config()
        paper_cfg = settings.get("broker", {}).get("paper", {})
        sizing_cfg = risk_cfg.get("position_sizing", {})
        return cls(
            watchlist=paper_cfg.get("watchlist", ["SPY", "QQQ"]),
            initial_capital=paper_cfg.get("initial_capital", 500.0),
            min_position_usd=paper_cfg.get("min_position_usd", sizing_cfg.get("min_position_usd", 10.0)),
            max_position_usd=paper_cfg.get("max_position_usd", sizing_cfg.get("max_position_usd", 100.0)),
            risk_pct_trade=paper_cfg.get("risk_pct_trade", 0.02),
            slippage_pct=paper_cfg.get("slippage_pct", 0.001),
            execution_enabled=paper_cfg.get("execution_enabled", False),
            refresh_seconds=paper_cfg.get("refresh_seconds", 60),
        )

    def _build_strategies(self) -> dict[str, Any]:
        scfg = get_strategy_config().get("strategies", {})
        strategies: dict[str, Any] = {}
        if scfg.get("true_dca", {}).get("enabled", False):
            strategies["true_dca"] = TrueDCAStrategy(scfg.get("true_dca", {}))
        if scfg.get("tactical_dca", {}).get("enabled", False):
            strategies["tactical_dca"] = TacticalDCAStrategy(scfg.get("tactical_dca", {}))
        if scfg.get("rsi_dip_buyer", {}).get("enabled", False):
            strategies["rsi_dip_buyer"] = RSIDipBuyerStrategy(scfg.get("rsi_dip_buyer", {}))
        if scfg.get("trend_following", {}).get("enabled", False):
            strategies["trend_following"] = TrendFollowingStrategy(scfg.get("trend_following", {}))
        if scfg.get("breakout", {}).get("enabled", False):
            strategies["breakout"] = BreakoutStrategy(scfg.get("breakout", {}))
        return strategies

    def _get_bars(self, ticker: str) -> pd.DataFrame | None:
        import yfinance as yf

        configure_yfinance_cache()
        start = (date.today() - timedelta(days=420)).isoformat()
        try:
            df = yf.download(
                ticker,
                start=start,
                auto_adjust=True,
                progress=False,
                timeout=20,
                threads=False,
            )
            df = normalize_yfinance_columns(df)
            return df if len(df) >= 250 else None
        except Exception as exc:
            logger.warning("Data error %s: %s", ticker, exc)
            return None

    def _live_price(self, ticker: str) -> float | None:
        import yfinance as yf

        try:
            configure_yfinance_cache()
            info = yf.Ticker(ticker).fast_info
            price = info.get("last_price") or info.get("regularMarketPrice")
            return float(price) if price else None
        except Exception:
            return None

    def update(self) -> None:
        account = self.broker.get_account_snapshot()
        positions = self.broker.get_positions()
        position_map = {p["symbol"]: p for p in positions}
        symbols = set(self.watchlist) | set(position_map)
        prices = {s: self._live_price(s) for s in symbols}
        prices = {s: p for s, p in prices.items() if p and p > 0}

        self._sync_targets(position_map)
        if self.execution_enabled and is_market_open():
            self._check_exits(position_map, prices)
            self._open_new_positions(position_map, prices)

        positions = self.broker.get_positions()
        open_pnl = sum(float(p.get("unrealized_pl", 0.0)) for p in positions)
        bot_equity = self.initial_capital + self.state.realized_pnl() + open_pnl
        self.state.snapshot_equity(bot_equity)
        self.state.save()
        self.last_update = _now_str()
        logger.info(
            "Alpaca paper update: account=%.2f bot_equity=%.2f positions=%d",
            account.portfolio_value,
            bot_equity,
            len(positions),
        )

    def _sync_targets(self, position_map: dict[str, dict[str, Any]]) -> None:
        for symbol in list(self.state.targets):
            if symbol not in position_map:
                self.state.targets.pop(symbol, None)

    def _check_exits(self, position_map: dict[str, dict[str, Any]], prices: dict[str, float]) -> None:
        for symbol, target in list(self.state.targets.items()):
            if symbol not in position_map or symbol not in prices:
                continue
            price = prices[symbol]
            stop_loss = float(target.get("stop_loss", 0.0))
            take_profit = float(target.get("take_profit", 0.0))
            if stop_loss and price <= stop_loss:
                self._close_symbol(symbol, price, "stop_loss")
            elif take_profit and price >= take_profit:
                self._close_symbol(symbol, price, "take_profit")

    def _close_symbol(self, symbol: str, price: float, reason: str) -> None:
        position = next((p for p in self.broker.get_positions() if p["symbol"] == symbol), None)
        if not position:
            return
        qty = float(position.get("qty", 0.0))
        if qty <= 0:
            return
        order = self.broker.submit_market_order(symbol=symbol, qty=qty, side="sell")
        target = self.state.targets.pop(symbol, {})
        entry = float(target.get("entry_price", position.get("avg_entry_price", price)))
        pnl = round((price * (1 - self.slippage_pct) - entry) * qty, 2)
        self.state.closed_events.append(
            {
                "ticker": symbol,
                "strategy": target.get("strategy", "unknown"),
                "quantity": round(qty, 6),
                "entry_price": round(entry, 4),
                "exit_price": round(price, 4),
                "exit_time": _now_str(),
                "exit_reason": reason,
                "pnl": pnl,
                "pnl_pct": round((price / entry - 1) * 100, 2) if entry else 0.0,
                "order_id": order.get("id"),
            }
        )
        self.state.log_signal(symbol, target.get("strategy", "unknown"), f"SELL {reason}", price, f"pnl={pnl:+.2f}")

    def _open_new_positions(self, position_map: dict[str, dict[str, Any]], prices: dict[str, float]) -> None:
        if len(position_map) >= 3:
            return

        used_budget = sum(float(p.get("market_value", 0.0)) for p in position_map.values())
        available_budget = max(0.0, self.initial_capital - used_budget)
        if available_budget < self.min_position_usd:
            return

        for ticker in self.watchlist:
            if ticker in position_map or ticker in self.state.targets:
                continue
            df_raw = self._get_bars(ticker)
            if df_raw is None:
                continue
            df = compute_all_features(df_raw)
            try:
                regime = _REGIME_DET.detect(df).regime.value
            except Exception:
                regime = "unknown"

            for strategy_name, strategy in self._strategies.items():
                strategy_assets = set(strategy.config.get("assets", []))
                if strategy_assets and ticker not in strategy_assets:
                    continue
                try:
                    signal = strategy.generate_signal(df, ticker, regime)
                except Exception as exc:
                    logger.debug("Signal error %s/%s: %s", ticker, strategy_name, exc)
                    continue
                if signal.signal != SignalType.BUY:
                    continue

                live_price = prices.get(ticker, float(df["close"].iloc[-1]))
                entry = live_price * (1 + self.slippage_pct)

                is_long_term = getattr(signal.horizon, "value", signal.horizon) == "long_term"
                fixed_pct = float(signal.metadata.get("requested_size_pct", 0.0) or 0.0)
                if fixed_pct > 0:
                    position_usd = self.initial_capital * fixed_pct
                else:
                    stop_for_size = float(signal.stop_loss or entry * 0.97)
                    stop_dist = abs(entry - stop_for_size) or entry * 0.02
                    risk_usd = self.initial_capital * self.risk_pct_trade
                    position_usd = risk_usd / stop_dist * entry

                position_usd = min(position_usd, self.max_position_usd, available_budget)
                if position_usd < self.min_position_usd:
                    continue

                qty = round(position_usd / entry, 6)
                if qty <= 0:
                    continue

                if is_long_term and signal.stop_loss is None and signal.take_profit is None:
                    stop = 0.0
                    take_profit = 0.0
                else:
                    stop = float(signal.stop_loss or entry * 0.97)
                    take_profit = float(signal.take_profit or entry * 1.05)

                order = self.broker.submit_market_order(symbol=ticker, qty=qty, side="buy")
                self.state.targets[ticker] = {
                    "strategy": strategy_name,
                    "quantity": qty,
                    "entry_price": round(entry, 4),
                    "entry_time": _now_str(),
                    "stop_loss": round(stop, 4),
                    "take_profit": round(take_profit, 4),
                    "order_id": order.get("id"),
                }
                self.state.log_signal(ticker, strategy_name, "BUY", live_price, signal.reason)
                logger.info("Alpaca paper BUY %s %s qty=%.6f notional=%.2f", ticker, strategy_name, qty, position_usd)
                return

    def get_dashboard_snapshot(self) -> dict[str, Any]:
        account = self.broker.get_account_snapshot()
        positions = self.broker.get_positions()
        orders = self.broker.get_orders(limit=50)
        open_pnl = sum(float(p.get("unrealized_pl", 0.0)) for p in positions)
        bot_equity = self.initial_capital + self.state.realized_pnl() + open_pnl
        return {
            "account": account,
            "positions": positions,
            "orders": orders,
            "state": self.state,
            "bot_equity": round(bot_equity, 2),
            "open_pnl": round(open_pnl, 2),
            "realized_pnl": self.state.realized_pnl(),
            "running": self.running,
            "last_update": self.last_update,
            "last_error": self.last_error,
        }

    def start(self, interval: int | None = None) -> None:
        if self._running:
            return
        self._running = True
        interval = int(interval or self.refresh_seconds)

        def _loop() -> None:
            while self._running:
                try:
                    self.update()
                    self.last_error = ""
                except Exception as exc:
                    self.last_error = str(exc)
                    logger.error("Alpaca paper engine error: %s", exc)
                for _ in range(interval):
                    if not self._running:
                        break
                    time.sleep(1)

        self._thread = threading.Thread(target=_loop, daemon=True, name="AlpacaPaperTrader")
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running


def _now_str() -> str:
    return datetime.now(_ET).strftime("%Y-%m-%d %H:%M:%S ET")


def is_market_open() -> bool:
    now = datetime.now(_ET)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def market_status() -> str:
    now = datetime.now(_ET)
    if now.weekday() >= 5:
        return "CLOSED weekend"
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if market_open <= now <= market_close:
        return "OPEN"
    if now < market_open:
        delta = market_open - now
        hours, rem = divmod(int(delta.total_seconds()), 3600)
        minutes = rem // 60
        return f"PRE-MARKET opens in {hours}h{minutes:02d}"
    return "CLOSED"
