"""
Interactive backtest dashboard.

Run:
    python -m src.dashboard.backtest_dashboard
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf  # type: ignore
from dash import Dash, Input, Output, State, dash_table, dcc, html
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import get_risk_config, get_settings, get_strategy_config
from src.backtesting.backtester import Backtester, BacktestResult
from src.data.yfinance_helpers import configure_yfinance_cache, normalize_yfinance_columns
from src.strategies.breakout import BreakoutStrategy
from src.strategies.dca import DCAStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.volatility_compression import VolatilityCompressionStrategy


STRATEGY_ORDER = [
    "dca_etf",
    "breakout",
    "trend_following",
    "mean_reversion",
    "momentum",
    "volatility_compression",
]

CHART_COLORS = {
    "dca_etf": "#2563eb",
    "breakout": "#059669",
    "trend_following": "#d97706",
    "mean_reversion": "#dc2626",
    "momentum": "#7c3aed",
    "volatility_compression": "#0891b2",
    "buy_hold": "#111827",
}


def strategy_registry() -> dict[str, Any]:
    scfg = get_strategy_config().get("strategies", {})
    return {
        "trend_following": TrendFollowingStrategy(scfg.get("trend_following", {})),
        "mean_reversion": MeanReversionStrategy(scfg.get("mean_reversion", {})),
        "breakout": BreakoutStrategy(scfg.get("breakout", {})),
        "dca_etf": DCAStrategy(scfg.get("dca_etf", {})),
        "momentum": MomentumStrategy(scfg.get("momentum", {})),
        "volatility_compression": VolatilityCompressionStrategy(
            scfg.get("volatility_compression", {})
        ),
    }


def enabled_strategy_names() -> list[str]:
    scfg = get_strategy_config().get("strategies", {})
    enabled = [name for name in STRATEGY_ORDER if scfg.get(name, {}).get("enabled", False)]
    return enabled or ["dca_etf"]


def build_backtester() -> Backtester:
    settings = get_settings()
    risk_cfg = get_risk_config()
    broker_cfg = settings.get("broker", {}).get("paper", {})
    sizing_cfg = risk_cfg.get("position_sizing", {})
    return Backtester(
        initial_capital=broker_cfg.get("initial_capital", 500.0),
        commission_flat=broker_cfg.get("commission_per_trade", 0.0),
        slippage_pct=broker_cfg.get("slippage_pct", 0.001),
        risk_pct_per_trade=sizing_cfg.get("fixed_risk_pct", 0.005),
        min_position_usd=sizing_cfg.get("min_position_usd", 0.0),
        max_position_usd=sizing_cfg.get("max_position_usd"),
    )


def download_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    configure_yfinance_cache()
    end_plus_one = (pd.to_datetime(end_date).date() + timedelta(days=1)).isoformat()
    df = yf.download(
        ticker.upper().strip(),
        start=start_date,
        end=end_plus_one,
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=20,
    )
    df = normalize_yfinance_columns(df)
    if df.empty:
        raise ValueError(f"No market data returned for {ticker}.")
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns from data: {', '.join(sorted(missing))}.")
    return df


def run_selected_backtests(
    ticker: str,
    df: pd.DataFrame,
    strategy_names: list[str],
) -> list[BacktestResult]:
    registry = strategy_registry()
    results: list[BacktestResult] = []
    for name in strategy_names:
        if name not in registry:
            continue
        bt = build_backtester()
        results.append(bt.run(registry[name], df, asset=ticker.upper().strip()))
    return results


def buy_and_hold_curve(df: pd.DataFrame, result: BacktestResult) -> pd.Series:
    start = pd.to_datetime(result.start_date)
    prices = df.loc[df.index >= start, "close"]
    if prices.empty:
        return pd.Series(dtype=float)
    return result.initial_capital * (prices / float(prices.iloc[0]))


def drawdown_curve(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    return (equity / equity.cummax() - 1.0) * 100


def result_equity_after_warmup(result: BacktestResult) -> pd.Series:
    start = pd.to_datetime(result.start_date)
    return result.equity_curve.loc[result.equity_curve.index >= start]


def make_empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=390,
        margin={"l": 54, "r": 24, "t": 54, "b": 42},
        font={"family": "Inter, Segoe UI, Arial", "size": 12},
        xaxis={"showgrid": False},
        yaxis={"gridcolor": "#e5e7eb"},
    )
    return fig


def make_equity_figure(df: pd.DataFrame, results: list[BacktestResult]) -> go.Figure:
    fig = make_empty_figure("Courbes equity")
    if not results:
        return fig

    for result in results:
        equity = result_equity_after_warmup(result)
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                line={"width": 2.4, "color": CHART_COLORS.get(result.strategy_name)},
                name=result.strategy_name,
            )
        )

    bah = buy_and_hold_curve(df, results[0])
    if not bah.empty:
        fig.add_trace(
            go.Scatter(
                x=bah.index,
                y=bah.values,
                mode="lines",
                line={"width": 1.8, "dash": "dash", "color": CHART_COLORS["buy_hold"]},
                name="buy_hold",
            )
        )

    fig.update_layout(legend={"orientation": "h", "y": 1.1, "x": 0})
    fig.update_yaxes(title_text="Capital")
    return fig


def make_drawdown_figure(results: list[BacktestResult]) -> go.Figure:
    fig = make_empty_figure("Drawdown")
    for result in results:
        equity = result_equity_after_warmup(result)
        dd = drawdown_curve(equity)
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                mode="lines",
                fill="tozeroy",
                line={"width": 1.8, "color": CHART_COLORS.get(result.strategy_name)},
                name=result.strategy_name,
            )
        )
    fig.update_layout(legend={"orientation": "h", "y": 1.1, "x": 0})
    fig.update_yaxes(title_text="%")
    return fig


def metric_rows(results: list[BacktestResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        m = result.metrics
        rows.append(
            {
                "strategy": result.strategy_name,
                "initial": round(result.initial_capital, 2),
                "final": round(result.final_capital, 2),
                "return_pct": round(m.get("total_return_pct", 0.0), 2),
                "sharpe": round(m.get("sharpe_ratio", 0.0), 3),
                "max_dd_pct": round(m.get("max_drawdown_pct", 0.0), 2),
                "trades": m.get("total_trades", 0),
                "win_rate_pct": round(m.get("win_rate_pct", 0.0), 1),
                "profit_factor": round(m.get("profit_factor", 0.0), 3),
                "vs_bh_pct": round(m.get("vs_buy_and_hold_pct", 0.0), 2),
            }
        )
    return rows


def trade_rows(df: pd.DataFrame, results: list[BacktestResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for trade in result.trades:
            entry_date = ""
            exit_date = ""
            if trade.entry_bar < len(df.index):
                entry_date = str(df.index[trade.entry_bar].date())
            if trade.exit_bar is not None and trade.exit_bar < len(df.index):
                exit_date = str(df.index[trade.exit_bar].date())
            rows.append(
                {
                    "strategy": result.strategy_name,
                    "side": trade.side,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry": round(trade.entry_price, 4),
                    "exit": round(trade.exit_price or 0.0, 4),
                    "qty": round(trade.quantity, 4),
                    "pnl": round(trade.pnl, 2),
                    "pnl_pct": round(trade.pnl_pct * 100, 2),
                    "reason": trade.exit_reason,
                }
            )
    return rows


def metric_cards(results: list[BacktestResult]) -> list[Any]:
    if not results:
        return []
    best = max(results, key=lambda r: r.metrics.get("total_return_pct", -9999))
    worst_dd = min(r.metrics.get("max_drawdown_pct", 0.0) for r in results)
    total_trades = sum(int(r.metrics.get("total_trades", 0)) for r in results)
    capital = results[0].initial_capital
    cards = [
        ("Capital", f"${capital:,.2f}"),
        ("Meilleur return", f"{best.strategy_name} {best.metrics.get('total_return_pct', 0):.2f}%"),
        ("Max DD", f"{worst_dd:.2f}%"),
        ("Trades", str(total_trades)),
    ]
    return [
        html.Div(
            [html.Div(label, className="metric-label"), html.Div(value, className="metric-value")],
            className="metric-card",
        )
        for label, value in cards
    ]


def create_app() -> Dash:
    settings = get_settings()
    default_start = (date.today() - timedelta(days=365 * 5)).isoformat()
    default_end = date.today().isoformat()
    default_strategies = enabled_strategy_names()
    strategy_options = [
        {"label": f"{name}{' *' if name in default_strategies else ''}", "value": name}
        for name in STRATEGY_ORDER
    ]

    app = Dash(__name__, title="Backtests")

    app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      :root {
        --bg: #f8fafc;
        --panel: #ffffff;
        --line: #d8dee8;
        --text: #111827;
        --muted: #64748b;
        --accent: #2563eb;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font-family: Inter, Segoe UI, Arial, sans-serif;
      }
      .page { min-height: 100vh; }
      .topbar {
        display: grid;
        grid-template-columns: 180px 160px 260px minmax(280px, 1fr) 120px;
        gap: 12px;
        align-items: end;
        padding: 16px 18px;
        background: var(--panel);
        border-bottom: 1px solid var(--line);
        position: sticky;
        top: 0;
        z-index: 5;
      }
      .brand {
        font-size: 22px;
        font-weight: 700;
        line-height: 38px;
      }
      .field-label {
        display: block;
        color: var(--muted);
        font-size: 12px;
        font-weight: 650;
        margin: 0 0 6px;
      }
      .ticker-input, .run-button {
        width: 100%;
        height: 38px;
        border: 1px solid var(--line);
        border-radius: 6px;
        background: #fff;
        color: var(--text);
        font-size: 14px;
      }
      .ticker-input { padding: 0 10px; text-transform: uppercase; }
      .run-button {
        border-color: var(--accent);
        background: var(--accent);
        color: #fff;
        font-weight: 700;
        cursor: pointer;
      }
      .content { padding: 16px 18px 24px; }
      .status {
        min-height: 26px;
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 10px;
      }
      .loading-wrap {
        min-height: 520px;
      }
      .metrics {
        display: grid;
        grid-template-columns: repeat(4, minmax(120px, 1fr));
        gap: 12px;
        margin-bottom: 14px;
      }
      .metric-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px 14px;
      }
      .metric-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 650;
      }
      .metric-value {
        margin-top: 6px;
        font-size: 20px;
        font-weight: 750;
      }
      .chart-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-bottom: 14px;
      }
      .chart-panel, .table-panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
      }
      .table-grid {
        display: grid;
        grid-template-columns: 1fr 1.2fr;
        gap: 14px;
      }
      .table-title {
        padding: 12px 14px;
        font-size: 14px;
        font-weight: 750;
        border-bottom: 1px solid var(--line);
      }
      @media (max-width: 980px) {
        .topbar { grid-template-columns: 1fr; align-items: stretch; }
        .brand { line-height: 1.2; }
        .metrics, .chart-grid, .table-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

    table_style = {
        "style_table": {"overflowX": "auto", "maxHeight": "360px", "overflowY": "auto"},
        "style_header": {
            "backgroundColor": "#f1f5f9",
            "fontWeight": "700",
            "border": "1px solid #d8dee8",
        },
        "style_cell": {
            "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
            "fontSize": "12px",
            "padding": "8px",
            "border": "1px solid #e5e7eb",
            "whiteSpace": "nowrap",
        },
        "style_data_conditional": [
            {"if": {"row_index": "odd"}, "backgroundColor": "#fbfdff"},
        ],
    }

    app.layout = html.Div(
        className="page",
        children=[
            html.Div(
                className="topbar",
                children=[
                    html.Div("Backtests", className="brand"),
                    html.Label(
                        [
                            html.Span("Ticker", className="field-label"),
                            dcc.Input(
                                id="ticker-input",
                                value="SPY",
                                className="ticker-input",
                                debounce=True,
                            ),
                        ]
                    ),
                    html.Label(
                        [
                            html.Span("Periode", className="field-label"),
                            dcc.DatePickerRange(
                                id="date-range",
                                start_date=default_start,
                                end_date=default_end,
                                display_format="YYYY-MM-DD",
                            ),
                        ]
                    ),
                    html.Label(
                        [
                            html.Span("Strategies", className="field-label"),
                            dcc.Dropdown(
                                id="strategy-select",
                                options=strategy_options,
                                value=default_strategies,
                                multi=True,
                                clearable=False,
                            ),
                        ]
                    ),
                    html.Button("Lancer", id="run-button", className="run-button"),
                ],
            ),
            html.Div(
                className="content",
                children=[
                    dcc.Loading(
                        type="circle",
                        color="#2563eb",
                        parent_className="loading-wrap",
                        children=[
                            html.Div(
                                id="status-line",
                                className="status",
                                children="Pret. Choisis une periode puis clique sur Lancer.",
                            ),
                            html.Div(id="metric-cards", className="metrics"),
                            html.Div(
                                className="chart-grid",
                                children=[
                                    html.Div(dcc.Graph(id="equity-chart", config={"displayModeBar": False}), className="chart-panel"),
                                    html.Div(dcc.Graph(id="drawdown-chart", config={"displayModeBar": False}), className="chart-panel"),
                                ],
                            ),
                            html.Div(
                                className="table-grid",
                                children=[
                                    html.Div(
                                        className="table-panel",
                                        children=[
                                            html.Div("Metriques", className="table-title"),
                                            dash_table.DataTable(
                                                id="metrics-table",
                                                columns=[
                                                    {"name": "Strategie", "id": "strategy"},
                                                    {"name": "Initial", "id": "initial"},
                                                    {"name": "Final", "id": "final"},
                                                    {"name": "Return %", "id": "return_pct"},
                                                    {"name": "Sharpe", "id": "sharpe"},
                                                    {"name": "Max DD %", "id": "max_dd_pct"},
                                                    {"name": "Trades", "id": "trades"},
                                                    {"name": "Win %", "id": "win_rate_pct"},
                                                    {"name": "PF", "id": "profit_factor"},
                                                    {"name": "vs B&H %", "id": "vs_bh_pct"},
                                                ],
                                                data=[],
                                                **table_style,
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="table-panel",
                                        children=[
                                            html.Div("Trades", className="table-title"),
                                            dash_table.DataTable(
                                                id="trades-table",
                                                columns=[
                                                    {"name": "Strategie", "id": "strategy"},
                                                    {"name": "Side", "id": "side"},
                                                    {"name": "Entry", "id": "entry_date"},
                                                    {"name": "Exit", "id": "exit_date"},
                                                    {"name": "Prix in", "id": "entry"},
                                                    {"name": "Prix out", "id": "exit"},
                                                    {"name": "Qty", "id": "qty"},
                                                    {"name": "PnL", "id": "pnl"},
                                                    {"name": "PnL %", "id": "pnl_pct"},
                                                    {"name": "Raison", "id": "reason"},
                                                ],
                                                data=[],
                                                page_size=12,
                                                **table_style,
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("status-line", "children"),
        Output("metric-cards", "children"),
        Output("equity-chart", "figure"),
        Output("drawdown-chart", "figure"),
        Output("metrics-table", "data"),
        Output("trades-table", "data"),
        Input("run-button", "n_clicks"),
        State("ticker-input", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date"),
        State("strategy-select", "value"),
    )
    def update_dashboard(
        _: int | None,
        ticker: str,
        start_date: str,
        end_date: str,
        strategy_names: list[str],
    ) -> tuple[Any, Any, go.Figure, go.Figure, list[dict[str, Any]], list[dict[str, Any]]]:
        ticker = (ticker or "SPY").upper().strip()
        strategy_names = strategy_names or enabled_strategy_names()

        try:
            df = download_prices(ticker, start_date, end_date)
            results = run_selected_backtests(ticker, df, strategy_names)
            status = (
                f"{ticker} | {df.index[0].date()} - {df.index[-1].date()} | "
                f"{len(df)} bougies | capital ${build_backtester().initial_capital:,.2f}"
            )
            return (
                status,
                metric_cards(results),
                make_equity_figure(df, results),
                make_drawdown_figure(results),
                metric_rows(results),
                trade_rows(df, results),
            )
        except Exception as exc:
            status = f"Erreur: {exc}"
            return (
                status,
                [],
                make_empty_figure("Courbes equity"),
                make_empty_figure("Drawdown"),
                [],
                [],
            )

    return app


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    default_port = int(settings.get("monitoring", {}).get("dashboard_port", 8050))
    parser = argparse.ArgumentParser(description="Backtest dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
