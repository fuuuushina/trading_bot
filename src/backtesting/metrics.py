"""
src/backtesting/metrics.py

Portfolio performance metrics.
All functions are pure — no side effects.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_metrics(
    equity_curve: pd.Series,
    trades: list,
    bah_return_pct: float = 0.0,
    risk_free_rate: float = 0.04,
) -> dict[str, Any]:
    """
    Compute the full set of performance metrics.

    Parameters
    ----------
    equity_curve    : Daily portfolio equity values
    trades          : List of BacktestTrade objects
    bah_return_pct  : Buy-and-hold return % for the same period
    risk_free_rate  : Annual risk-free rate for Sharpe/Sortino
    """
    metrics: dict[str, Any] = {}

    if equity_curve.empty or len(equity_curve) < 2:
        return {"error": "Insufficient data for metrics."}

    # ---- Returns ----
    daily_returns = equity_curve.pct_change().dropna()
    total_return_pct = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    metrics["total_return_pct"] = round(total_return_pct, 4)

    # Annualised return
    n_days = len(equity_curve)
    years = n_days / 252
    cagr = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / max(years, 0.01)) - 1) * 100
    metrics["cagr_pct"] = round(cagr, 4)

    # ---- Sharpe Ratio ----
    if len(daily_returns) > 1:
        daily_rf = risk_free_rate / 252
        excess = daily_returns - daily_rf
        sharpe = (excess.mean() / (excess.std() + 1e-10)) * np.sqrt(252)
        metrics["sharpe_ratio"] = round(float(sharpe), 4)
    else:
        metrics["sharpe_ratio"] = 0.0

    # ---- Sortino Ratio ----
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1:
        downside_std = downside.std() * np.sqrt(252)
        annual_return = daily_returns.mean() * 252
        sortino = (annual_return - risk_free_rate) / (downside_std + 1e-10)
        metrics["sortino_ratio"] = round(float(sortino), 4)
    else:
        metrics["sortino_ratio"] = 0.0

    # ---- Drawdown ----
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = float(drawdowns.min()) * 100
    metrics["max_drawdown_pct"] = round(max_dd, 4)

    # Calmar ratio
    if max_dd != 0:
        metrics["calmar_ratio"] = round(cagr / abs(max_dd), 4)
    else:
        metrics["calmar_ratio"] = None

    # ---- Trade-level metrics ----
    if not trades:
        metrics.update({
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_usd": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "max_consecutive_losses": 0,
            "avg_trade_duration_bars": 0,
        })
    else:
        pnls = [t.pnl for t in trades if t.exit_price is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total = len(pnls)
        n_wins = len(wins)
        n_losses = len(losses)

        metrics["total_trades"] = total
        metrics["winning_trades"] = n_wins
        metrics["losing_trades"] = n_losses
        metrics["win_rate_pct"] = round(n_wins / total * 100, 2) if total > 0 else 0.0

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        if gross_loss > 0:
            metrics["profit_factor"] = round(gross_profit / gross_loss, 4)
        elif gross_profit > 0:
            metrics["profit_factor"] = float("inf")
        else:
            metrics["profit_factor"] = 0.0

        metrics["expectancy_usd"] = round(np.mean(pnls) if pnls else 0.0, 4)
        metrics["avg_win_usd"] = round(np.mean(wins) if wins else 0.0, 4)
        metrics["avg_loss_usd"] = round(np.mean(losses) if losses else 0.0, 4)

        # Avg win/loss as pct
        win_pcts = [t.pnl_pct for t in trades if t.pnl > 0 and t.exit_price]
        loss_pcts = [t.pnl_pct for t in trades if t.pnl <= 0 and t.exit_price]
        metrics["avg_win_pct"] = round(np.mean(win_pcts) * 100 if win_pcts else 0.0, 2)
        metrics["avg_loss_pct"] = round(np.mean(loss_pcts) * 100 if loss_pcts else 0.0, 2)

        # Max consecutive losses
        consecutive = 0
        max_consec = 0
        for p in pnls:
            if p < 0:
                consecutive += 1
                max_consec = max(max_consec, consecutive)
            else:
                consecutive = 0
        metrics["max_consecutive_losses"] = max_consec

        # Avg duration
        durations = [
            t.exit_bar - t.entry_bar
            for t in trades
            if t.exit_bar is not None
        ]
        metrics["avg_trade_duration_bars"] = round(np.mean(durations) if durations else 0, 1)

    # ---- vs Buy & Hold ----
    metrics["buy_and_hold_return_pct"] = round(bah_return_pct, 4)
    metrics["vs_buy_and_hold_pct"] = round(total_return_pct - bah_return_pct, 4)

    # ---- Volatility ----
    ann_vol = float(daily_returns.std()) * np.sqrt(252) * 100
    metrics["annual_volatility_pct"] = round(ann_vol, 4)

    return metrics
