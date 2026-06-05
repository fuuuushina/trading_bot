"""
Live paper-trading dashboard.

Run:
    python -m src.dashboard.live_dashboard --port 8051

The dashboard can use either:
  - local: a simulated in-repo portfolio
  - alpaca: the Alpaca PAPER account, with a local software budget cap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import get_settings
from src.trading.alpaca_paper_trader import (
    AlpacaPaperState,
    AlpacaPaperTradingEngine,
    STATE_FILE as ALPACA_STATE_FILE,
    is_market_open as alpaca_is_market_open,
    market_status as alpaca_market_status,
)
from src.trading.paper_trader import (
    DEFAULT_CAPITAL,
    STATE_FILE as LOCAL_STATE_FILE,
    WATCHLIST as LOCAL_WATCHLIST,
    PaperPortfolio,
    PaperTradingEngine,
    is_market_open as local_is_market_open,
    market_status as local_market_status,
)


SETTINGS = get_settings()
PAPER_CFG = SETTINGS.get("broker", {}).get("paper", {})
PROVIDER = str(PAPER_CFG.get("provider", "local")).lower()
BOT_CAPITAL = float(PAPER_CFG.get("initial_capital", DEFAULT_CAPITAL))
WATCHLIST = [str(x).upper() for x in PAPER_CFG.get("watchlist", LOCAL_WATCHLIST)]

PNL_POS = "#16a34a"
PNL_NEG = "#dc2626"
MUTED = "#64748b"

_engine: AlpacaPaperTradingEngine | PaperTradingEngine | None = None
_engine_error = ""


def _create_engine() -> AlpacaPaperTradingEngine | PaperTradingEngine:
    if PROVIDER == "alpaca":
        return AlpacaPaperTradingEngine.from_config()
    return PaperTradingEngine(watchlist=WATCHLIST, initial_capital=BOT_CAPITAL)


try:
    _engine = _create_engine()
    _engine.start()
except Exception as exc:  # Dashboard should still open and show the issue.
    _engine_error = str(exc)


def _style() -> dict[str, Any]:
    return {
        "template": "plotly_white",
        "margin": {"l": 50, "r": 12, "t": 42, "b": 34},
        "font": {"family": "Inter, system-ui, sans-serif", "size": 12},
        "plot_bgcolor": "#fff",
        "paper_bgcolor": "#fff",
        "xaxis": {"showgrid": False},
        "yaxis": {"gridcolor": "#e5e7eb"},
    }


def _pnl_color(value: float) -> str:
    return PNL_POS if value >= 0 else PNL_NEG


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _num(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if value == 0:
        return "-"
    return f"{value:.{digits}f}"


def _kpi(label: str, value: str, color: str = "#0f172a") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value", style={"color": color}),
        ],
        className="kpi-card",
    )


def _table_style() -> dict[str, Any]:
    return {
        "style_table": {"overflowX": "auto", "maxHeight": "320px", "overflowY": "auto"},
        "style_header": {
            "backgroundColor": "#f8fafc",
            "fontWeight": "700",
            "border": "1px solid #e2e8f0",
            "fontSize": "12px",
        },
        "style_cell": {
            "fontFamily": "Inter, system-ui, sans-serif",
            "fontSize": "12px",
            "padding": "7px 10px",
            "border": "1px solid #f1f5f9",
            "whiteSpace": "nowrap",
        },
        "style_data_conditional": [{"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"}],
    }


def _equity_fig(history: list[dict[str, Any]], initial: float, title: str) -> go.Figure:
    fig = go.Figure()
    if not history:
        fig.update_layout(title=f"{title} (en attente de donnees)", height=310, **_style())
        fig.add_hline(y=initial, line_dash="dash", line_color="#94a3b8", line_width=1)
        fig.update_yaxes(title_text="Capital ($)")
        return fig

    times = [row.get("time") for row in history]
    values = [row.get("equity") for row in history]
    fig.add_trace(
        go.Scatter(
            x=times,
            y=values,
            mode="lines",
            line={"width": 2.5, "color": "#2563eb"},
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.07)",
            name="Equity",
        )
    )
    fig.add_hline(y=initial, line_dash="dash", line_color="#94a3b8", line_width=1)
    fig.update_layout(title=title, height=310, showlegend=False, **_style())
    fig.update_yaxes(title_text="Capital ($)")
    return fig


def _empty_error_snapshot() -> tuple[list[html.Div], go.Figure, list, list, list, str, str, str, str]:
    kpis = [
        _kpi("Provider", PROVIDER.upper(), MUTED),
        _kpi("Budget bot", _money(BOT_CAPITAL)),
        _kpi("Etat", "Erreur", PNL_NEG),
        _kpi("P&L ouvert", "$0.00"),
        _kpi("P&L realise", "$0.00"),
    ]
    fig = _equity_fig([], BOT_CAPITAL, "Equity paper")
    status = _engine_error or "Moteur non initialise"
    return kpis, fig, [], [], [], "PAUSE", "live-badge", status, status


def _local_refresh() -> tuple[list[html.Div], go.Figure, list, list, list, str, str, str, str]:
    assert isinstance(_engine, PaperTradingEngine)
    portfolio = PaperPortfolio.load(BOT_CAPITAL)
    total_equity = portfolio.equity({})
    open_pnl = portfolio.open_pnl({})
    realized_pnl = portfolio.closed_pnl()
    return_pct = (total_equity / portfolio.initial_capital - 1) * 100

    kpis = [
        _kpi("Capital total", _money(total_equity), _pnl_color(total_equity - portfolio.initial_capital)),
        _kpi("Cash disponible", _money(portfolio.cash)),
        _kpi("P&L ouvert", f"{open_pnl:+.2f}$", _pnl_color(open_pnl)),
        _kpi("P&L realise", f"{realized_pnl:+.2f}$", _pnl_color(realized_pnl)),
        _kpi("Return total", f"{return_pct:+.2f}%", _pnl_color(return_pct)),
    ]

    positions = [
        {
            "ticker": p.ticker,
            "strategy": p.strategy,
            "quantity": round(p.quantity, 6),
            "entry_price": p.entry_price,
            "stop_loss": p.stop_loss,
            "take_profit": p.take_profit,
            "entry_time": p.entry_time,
        }
        for p in portfolio.positions
    ]
    signals = list(reversed(portfolio.signals_log[-30:]))
    trades = [
        {
            "ticker": t.ticker,
            "strategy": t.strategy,
            "quantity": round(t.quantity, 6),
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "exit_time": t.exit_time,
            "exit_reason": t.exit_reason,
            "pnl": t.pnl,
            "pnl_pct": f"{t.pnl_pct:+.2f}%",
        }
        for t in reversed(portfolio.closed_trades[-100:])
    ]

    is_open = local_is_market_open()
    badge = "LOCAL PAPER" if _engine.running else "PAUSE"
    badge_class = ("live-badge open" if is_open else "live-badge") if _engine.running else "live-badge"
    last_update = f"Derniere MAJ : {_engine.last_update or '-'}"
    if _engine.last_error:
        last_update += f" | Erreur: {_engine.last_error[:120]}"

    fig = _equity_fig(
        portfolio.equity_history,
        portfolio.initial_capital,
        f"Equity temps reel - capital initial ${portfolio.initial_capital:,.0f}",
    )
    return kpis, fig, positions, signals, trades, badge, badge_class, local_market_status(), last_update


def _alpaca_refresh() -> tuple[list[html.Div], go.Figure, list, list, list, str, str, str, str]:
    assert isinstance(_engine, AlpacaPaperTradingEngine)
    snap = _engine.get_dashboard_snapshot()
    account = snap["account"]
    state = snap["state"]
    positions_raw = snap["positions"]
    used_budget = sum(float(p.get("market_value", 0.0)) for p in positions_raw)
    available_budget = max(0.0, _engine.initial_capital - used_budget)
    bot_equity = float(snap["bot_equity"])
    open_pnl = float(snap["open_pnl"])
    realized_pnl = float(snap["realized_pnl"])
    return_pct = (bot_equity / _engine.initial_capital - 1) * 100

    kpis = [
        _kpi("Budget bot", _money(bot_equity), _pnl_color(bot_equity - _engine.initial_capital)),
        _kpi("Disponible bot", _money(available_budget)),
        _kpi("P&L ouvert", f"{open_pnl:+.2f}$", _pnl_color(open_pnl)),
        _kpi("P&L realise", f"{realized_pnl:+.2f}$", _pnl_color(realized_pnl)),
        _kpi("Compte Alpaca", _money(account.portfolio_value), MUTED),
    ]

    positions = []
    for pos in positions_raw:
        symbol = pos.get("symbol", "")
        target = state.targets.get(symbol, {})
        positions.append(
            {
                "ticker": symbol,
                "strategy": target.get("strategy", "alpaca"),
                "quantity": round(float(pos.get("qty", 0.0)), 6),
                "entry_price": _num(target.get("entry_price") or pos.get("avg_entry_price")),
                "stop_loss": _num(target.get("stop_loss")),
                "take_profit": _num(target.get("take_profit")),
                "entry_time": target.get("entry_time", ""),
            }
        )

    signals = list(reversed(state.signals_log[-30:]))
    if not signals:
        signals = [
            {
                "time": str(order.get("submitted_at", ""))[:19],
                "ticker": order.get("symbol", ""),
                "strategy": "alpaca_order",
                "signal": f"{str(order.get('side', '')).upper()} {order.get('status', '')}",
                "price": "",
                "reason": f"order {str(order.get('id', ''))[:8]}",
            }
            for order in snap["orders"][:20]
        ]

    trades = [
        {
            "ticker": row.get("ticker"),
            "strategy": row.get("strategy"),
            "quantity": row.get("quantity"),
            "entry_price": row.get("entry_price"),
            "exit_price": row.get("exit_price"),
            "exit_time": row.get("exit_time"),
            "exit_reason": row.get("exit_reason"),
            "pnl": row.get("pnl"),
            "pnl_pct": f"{float(row.get('pnl_pct', 0.0)):+.2f}%",
        }
        for row in reversed(state.closed_events[-100:])
    ]

    is_open = alpaca_is_market_open()
    badge = "ALPACA PAPER" if _engine.running else "PAUSE"
    badge_class = ("live-badge open" if is_open else "live-badge") if _engine.running else "live-badge"
    exec_status = "execution ON" if _engine.execution_enabled else "execution OFF"
    market = f"{alpaca_market_status()} | {exec_status} | budget ${_engine.initial_capital:,.0f}"
    last_update = f"Derniere MAJ : {snap['last_update'] or '-'}"
    if snap["last_error"]:
        last_update += f" | Erreur: {snap['last_error'][:120]}"

    fig = _equity_fig(
        state.equity_history,
        _engine.initial_capital,
        f"Equity Alpaca Paper - budget bot ${_engine.initial_capital:,.0f}",
    )
    return kpis, fig, positions, signals, trades, badge, badge_class, market, last_update


def create_app() -> Dash:
    app = Dash(__name__, title="Paper Trading Live")
    app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
      :root {
        --bg: #f1f5f9; --panel: #fff; --border: #e2e8f0;
        --text: #0f172a; --muted: #64748b;
        --accent: #2563eb; --green: #16a34a; --red: #dc2626;
        --warning: #b45309; --radius: 8px;
      }
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body {
        background: var(--bg); color: var(--text);
        font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif;
        font-size: 14px;
      }
      .topbar {
        background: var(--panel); border-bottom: 1px solid var(--border);
        padding: 10px 20px; display: flex; align-items: center;
        gap: 14px; position: sticky; top: 0; z-index: 10;
      }
      .brand { font-size: 17px; font-weight: 800; }
      .live-badge {
        background: #fee2e2; color: var(--red); border-radius: 999px;
        font-size: 11px; font-weight: 800; padding: 4px 10px;
        letter-spacing: 0.3px;
      }
      .live-badge.open { background: #dcfce7; color: var(--green); }
      .market-status { font-size: 12px; color: var(--muted); }
      .last-update { font-size: 11px; color: var(--muted); margin-left: auto; }
      .ctrl-btn {
        padding: 7px 14px; border-radius: 6px; font-size: 12px; font-weight: 750;
        cursor: pointer; border: 1.5px solid var(--border);
        background: #fff; color: var(--muted); transition: all .15s;
      }
      .ctrl-btn:hover { border-color: var(--accent); color: var(--accent); }
      .ctrl-btn.stop { border-color: var(--red); color: var(--red); }
      .ctrl-btn.reset { border-color: #f59e0b; color: var(--warning); }
      .content { padding: 16px 20px 32px; max-width: 1600px; }
      .kpi-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 14px; }
      .kpi-card {
        background: var(--panel); border: 1px solid var(--border);
        border-radius: var(--radius); padding: 14px 16px;
      }
      .kpi-label {
        font-size: 11px; font-weight: 700; color: var(--muted);
        text-transform: uppercase; letter-spacing: 0.4px;
      }
      .kpi-value { margin-top: 6px; font-size: 20px; font-weight: 850; }
      .chart-panel {
        background: var(--panel); border: 1px solid var(--border);
        border-radius: var(--radius); overflow: hidden; margin-bottom: 14px;
      }
      .tables-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }
      .table-panel, .trade-log {
        background: var(--panel); border: 1px solid var(--border);
        border-radius: var(--radius); overflow: hidden;
      }
      .table-title {
        padding: 10px 16px; font-size: 12px; font-weight: 800;
        text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted);
        border-bottom: 1px solid var(--border);
      }
      @media (max-width: 900px) {
        .kpi-row, .tables-row { grid-template-columns: 1fr; }
        .topbar { flex-wrap: wrap; }
        .last-update { margin-left: 0; width: 100%; }
      }
    </style>
  </head>
  <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""

    table_style = _table_style()
    reset_label = "Reset affichage" if PROVIDER == "alpaca" else "Reset portefeuille"

    app.layout = html.Div(
        [
            dcc.Interval(id="live-interval", interval=60_000, n_intervals=0),
            dcc.Store(id="engine-state", data="running" if _engine else "error"),
            html.Div(
                className="topbar",
                children=[
                    html.Div("Paper Trading", className="brand"),
                    html.Div("LIVE", id="live-badge", className="live-badge"),
                    html.Div(id="market-status-txt", className="market-status"),
                    html.Button("Pause", id="btn-toggle", className="ctrl-btn stop"),
                    html.Button(reset_label, id="btn-reset", className="ctrl-btn reset"),
                    html.Div(id="last-update-txt", className="last-update"),
                ],
            ),
            html.Div(
                className="content",
                children=[
                    html.Div(id="kpi-row", className="kpi-row"),
                    html.Div(dcc.Graph(id="eq-live", config={"displayModeBar": False}), className="chart-panel"),
                    html.Div(
                        className="tables-row",
                        children=[
                            html.Div(
                                className="table-panel",
                                children=[
                                    html.Div("Positions ouvertes", className="table-title"),
                                    dash_table.DataTable(
                                        id="positions-tbl",
                                        columns=[
                                            {"name": "Ticker", "id": "ticker"},
                                            {"name": "Strategie", "id": "strategy"},
                                            {"name": "Qte", "id": "quantity"},
                                            {"name": "Prix in", "id": "entry_price"},
                                            {"name": "Stop", "id": "stop_loss"},
                                            {"name": "Target", "id": "take_profit"},
                                            {"name": "Entree", "id": "entry_time"},
                                        ],
                                        data=[],
                                        page_size=8,
                                        **table_style,
                                    ),
                                ],
                            ),
                            html.Div(
                                className="table-panel",
                                children=[
                                    html.Div("Signaux recents / ordres", className="table-title"),
                                    dash_table.DataTable(
                                        id="signals-tbl",
                                        columns=[
                                            {"name": "Heure", "id": "time"},
                                            {"name": "Ticker", "id": "ticker"},
                                            {"name": "Strategie", "id": "strategy"},
                                            {"name": "Signal", "id": "signal"},
                                            {"name": "Prix", "id": "price"},
                                            {"name": "Raison", "id": "reason"},
                                        ],
                                        data=[],
                                        page_size=10,
                                        **table_style,
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="trade-log",
                        children=[
                            html.Div("Historique des trades fermes", className="table-title"),
                            dash_table.DataTable(
                                id="trades-tbl",
                                columns=[
                                    {"name": "Ticker", "id": "ticker"},
                                    {"name": "Strategie", "id": "strategy"},
                                    {"name": "Qte", "id": "quantity"},
                                    {"name": "Prix in", "id": "entry_price"},
                                    {"name": "Prix out", "id": "exit_price"},
                                    {"name": "Sortie", "id": "exit_time"},
                                    {"name": "Raison", "id": "exit_reason"},
                                    {"name": "PnL $", "id": "pnl"},
                                    {"name": "PnL %", "id": "pnl_pct"},
                                ],
                                data=[],
                                page_size=15,
                                style_table=table_style["style_table"],
                                style_header=table_style["style_header"],
                                style_cell=table_style["style_cell"],
                                style_data_conditional=[
                                    {"if": {"filter_query": "{pnl} > 0", "column_id": "pnl"}, "color": PNL_POS, "fontWeight": "700"},
                                    {"if": {"filter_query": "{pnl} < 0", "column_id": "pnl"}, "color": PNL_NEG, "fontWeight": "700"},
                                    {"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"},
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ]
    )

    @app.callback(
        Output("engine-state", "data"),
        Output("btn-toggle", "children"),
        Output("btn-toggle", "className"),
        Input("btn-toggle", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_engine(_n: int):
        if _engine is None:
            return "error", "Erreur", "ctrl-btn"
        if _engine.running:
            _engine.stop()
            return "paused", "Reprendre", "ctrl-btn"
        _engine.start()
        return "running", "Pause", "ctrl-btn stop"

    @app.callback(
        Output("engine-state", "data", allow_duplicate=True),
        Input("btn-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_display(_n: int):
        if _engine is None:
            return "error"
        if PROVIDER == "alpaca" and isinstance(_engine, AlpacaPaperTradingEngine):
            _engine.state.equity_history = []
            _engine.state.signals_log = []
            _engine.state.closed_events = []
            _engine.state.save()
            return "reset-display"
        if LOCAL_STATE_FILE.exists():
            LOCAL_STATE_FILE.unlink()
        if isinstance(_engine, PaperTradingEngine):
            _engine.portfolio = PaperPortfolio(BOT_CAPITAL)
        return "reset"

    @app.callback(
        Output("kpi-row", "children"),
        Output("eq-live", "figure"),
        Output("positions-tbl", "data"),
        Output("signals-tbl", "data"),
        Output("trades-tbl", "data"),
        Output("live-badge", "children"),
        Output("live-badge", "className"),
        Output("market-status-txt", "children"),
        Output("last-update-txt", "children"),
        Input("live-interval", "n_intervals"),
        Input("engine-state", "data"),
    )
    def refresh(_n: int, _state: str):
        if _engine is None:
            return _empty_error_snapshot()
        try:
            if PROVIDER == "alpaca":
                return _alpaca_refresh()
            return _local_refresh()
        except Exception as exc:
            global _engine_error
            _engine_error = str(exc)
            return _empty_error_snapshot()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper trading live dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Paper trading dashboard -> http://{args.host}:{args.port}")
    print(f"Provider: {PROVIDER} | Watchlist: {WATCHLIST} | Bot budget: ${BOT_CAPITAL:,.0f}")
    if PROVIDER == "alpaca":
        status = "ON" if PAPER_CFG.get("execution_enabled", False) else "OFF"
        print(f"Alpaca PAPER execution: {status}")
    if _engine_error:
        print(f"Engine error: {_engine_error}")
    print("Ctrl+C pour arreter.")
    create_app().run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
