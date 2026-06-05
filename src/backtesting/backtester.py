"""
src/backtesting/backtester.py

Event-driven backtester.
Iterates bar-by-bar, runs the signal pipeline, simulates fills.
Supports walk-forward analysis and multi-strategy comparison.

IMPORTANT: Never validate a strategy on in-sample data alone.
Always use out-of-sample splits and stress tests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.features.indicators import compute_all_features
from src.features.regime_detector import MarketRegimeDetector
from src.strategies.base import BaseStrategy, Horizon, Signal, SignalType
from src.backtesting.metrics import compute_metrics

logger = logging.getLogger(__name__)

COMMISSION_FLAT = 1.0
SLIPPAGE_PCT = 0.001


@dataclass
class BacktestTrade:
    asset: str
    strategy: str
    side: str
    entry_bar: int
    entry_price: float
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    quantity: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    exit_reason: str = ""
    regime_at_entry: str = ""


@dataclass
class BacktestResult:
    strategy_name: str
    asset: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    trades: list[BacktestTrade]
    equity_curve: pd.Series
    metrics: dict[str, Any]

    def summary(self) -> str:
        m = self.metrics
        return (
            f"\n{'='*60}\n"
            f"Strategy : {self.strategy_name} | Asset: {self.asset}\n"
            f"Period   : {self.start_date} → {self.end_date}\n"
            f"Capital  : ${self.initial_capital:,.2f} → ${self.final_capital:,.2f}\n"
            f"Return   : {m.get('total_return_pct', 0):.2f}%\n"
            f"Sharpe   : {m.get('sharpe_ratio', 0):.3f}\n"
            f"Sortino  : {m.get('sortino_ratio', 0):.3f}\n"
            f"Max DD   : {m.get('max_drawdown_pct', 0):.2f}%\n"
            f"Win Rate : {m.get('win_rate_pct', 0):.1f}%\n"
            f"Trades   : {m.get('total_trades', 0)} "
            f"(W:{m.get('winning_trades',0)} / L:{m.get('losing_trades',0)})\n"
            f"Profit F : {m.get('profit_factor', 0):.3f}\n"
            f"Expectancy: ${m.get('expectancy_usd', 0):.2f}\n"
            f"vs B&H   : {m.get('vs_buy_and_hold_pct', 0):.2f}%\n"
            f"{'='*60}"
        )


class Backtester:
    """
    Bar-by-bar event-driven backtester.

    Usage:
        bt = Backtester(initial_capital=100_000)
        result = bt.run(strategy, df, asset="SPY")
        print(result.summary())
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_flat: float = COMMISSION_FLAT,
        slippage_pct: float = SLIPPAGE_PCT,
        risk_pct_per_trade: float = 0.005,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_flat = commission_flat
        self.slippage_pct = slippage_pct
        self.risk_pct_per_trade = risk_pct_per_trade
        self.regime_detector = MarketRegimeDetector()

    def run(
        self,
        strategy: BaseStrategy,
        df_raw: pd.DataFrame,
        asset: str = "ASSET",
        warmup_bars: int = 210,
    ) -> BacktestResult:
        """
        Run backtest for a single strategy on a single asset.

        Parameters
        ----------
        strategy    : Instantiated strategy object
        df_raw      : Raw OHLCV DataFrame indexed by date
        asset       : Asset ticker name
        warmup_bars : Bars to skip before generating signals
        """
        df = compute_all_features(df_raw.copy())
        n = len(df)

        if n < warmup_bars + 10:
            raise ValueError(f"Need at least {warmup_bars + 10} bars, got {n}.")

        capital = self.initial_capital
        cash = capital
        equity_curve: list[float] = [capital] * warmup_bars
        trades: list[BacktestTrade] = []
        open_trade: Optional[BacktestTrade] = None

        for i in range(warmup_bars, n):
            bar_df = df.iloc[: i + 1]
            current_bar = df.iloc[i]
            price = float(current_bar["close"])

            # Mark-to-market open position
            position_value = 0.0
            if open_trade:
                position_value = open_trade.quantity * price
                equity_curve.append(cash + position_value)

                # Check stop loss / take profit
                if open_trade.stop_loss and price <= open_trade.stop_loss:
                    cash, open_trade = self._close_trade(
                        open_trade, price, i, "stop_loss", cash
                    )
                    trades.append(open_trade)
                    open_trade = None
                elif open_trade.take_profit and price >= open_trade.take_profit:
                    cash, open_trade = self._close_trade(
                        open_trade, price, i, "take_profit", cash
                    )
                    trades.append(open_trade)
                    open_trade = None
            else:
                equity_curve.append(cash)

            if open_trade:
                continue

            # Detect regime
            try:
                regime = self.regime_detector.detect(bar_df).regime.value
            except Exception:
                regime = "unknown"

            # Generate signal
            try:
                signal: Signal = strategy.generate_signal(bar_df, asset, regime)
            except Exception as exc:
                logger.debug("Strategy error at bar %d: %s", i, exc)
                continue

            if signal.signal not in (SignalType.BUY, SignalType.SELL):
                continue

            # Size position
            entry_price = price * (
                1 + self.slippage_pct if signal.signal == SignalType.BUY
                else 1 - self.slippage_pct
            )
            commission = self.commission_flat + entry_price * 0.0
            risk_usd = capital * self.risk_pct_per_trade
            stop_dist = abs(entry_price - signal.stop_loss) if signal.stop_loss else entry_price * 0.02
            quantity = risk_usd / stop_dist if stop_dist > 0 else 0.0
            cost = quantity * entry_price + commission

            if cost > cash or quantity <= 0:
                continue

            cash -= cost

            open_trade = BacktestTrade(
                asset=asset,
                strategy=strategy.name,
                side="long" if signal.signal == SignalType.BUY else "short",
                entry_bar=i,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                commission=commission,
                regime_at_entry=regime,
            )

        # Close any open trade at end
        if open_trade:
            final_price = float(df.iloc[-1]["close"])
            cash, open_trade = self._close_trade(
                open_trade, final_price, n - 1, "end_of_backtest", cash
            )
            trades.append(open_trade)

        final_capital = cash
        equity_series = pd.Series(
            equity_curve, index=df.index[: len(equity_curve)]
        )

        # Buy-and-hold comparison
        bah_return = (
            float(df["close"].iloc[-1]) / float(df["close"].iloc[warmup_bars]) - 1
        ) * 100

        metrics = compute_metrics(equity_series, trades, bah_return)

        return BacktestResult(
            strategy_name=strategy.name,
            asset=asset,
            start_date=str(df.index[warmup_bars].date()),
            end_date=str(df.index[-1].date()),
            initial_capital=self.initial_capital,
            final_capital=round(final_capital, 2),
            trades=trades,
            equity_curve=equity_series,
            metrics=metrics,
        )

    def walk_forward(
        self,
        strategy: BaseStrategy,
        df_raw: pd.DataFrame,
        asset: str,
        n_splits: int = 5,
        train_pct: float = 0.70,
        warmup_bars: int = 210,
    ) -> list[BacktestResult]:
        """
        Walk-forward analysis: train on in-sample, test on out-of-sample.
        Never peek at future data.
        """
        n = len(df_raw)
        window = n // n_splits
        results = []

        for i in range(n_splits):
            start = i * window
            end = start + window
            if end > n:
                break
            segment = df_raw.iloc[start:end]
            split = int(len(segment) * train_pct)
            oos = segment.iloc[split:]  # out-of-sample only
            if len(oos) < warmup_bars + 10:
                continue
            result = self.run(strategy, oos, asset, warmup_bars)
            result.strategy_name += f"_fold_{i+1}"
            results.append(result)

        return results

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _close_trade(
        trade: BacktestTrade,
        price: float,
        bar_idx: int,
        reason: str,
        cash: float,
    ) -> tuple[float, BacktestTrade]:
        slippage_adj = price * (1 - SLIPPAGE_PCT) if trade.side == "long" else price * (1 + SLIPPAGE_PCT)
        commission = COMMISSION_FLAT
        direction = 1 if trade.side == "long" else -1
        gross_pnl = direction * (slippage_adj - trade.entry_price) * trade.quantity
        net_pnl = gross_pnl - commission - trade.commission

        trade.exit_bar = bar_idx
        trade.exit_price = round(slippage_adj, 4)
        trade.exit_reason = reason
        trade.pnl = round(net_pnl, 4)
        trade.pnl_pct = round(net_pnl / (trade.entry_price * trade.quantity), 4)

        cash += slippage_adj * trade.quantity - commission
        return cash, trade
