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

from config.loader import get_risk_config, get_settings, get_strategy_config
from src.backtesting.backtester import Backtester, BacktestResult
from src.data.yfinance_helpers import configure_yfinance_cache, normalize_yfinance_columns
from src.strategies.breakout import BreakoutStrategy
from src.strategies.rsi_dip_buyer import RSIDipBuyerStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.true_dca import TrueDCAStrategy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

STRATEGY_ORDER = [
    "true_dca", "trend_following", "breakout", "rsi_dip_buyer",
]

CHART_COLORS = {
    "true_dca":        "#0b84a5",
    "trend_following": "#d97706",
    "breakout":        "#059669",
    "rsi_dip_buyer":   "#7c3aed",
    "buy_hold":        "#374151",
}

DCA_STRATEGIES = {"true_dca"}

# ── Parameter definitions ─────────────────────────────────────────────────────

STRATEGY_PARAMS: dict[str, list[dict[str, Any]]] = {
    "true_dca": [
        {"key": "monthly_size_pct",     "label": "Taille mensuelle",  "min": 0.005, "max": 0.10,  "step": 0.005, "default": 0.022, "unit": "%", "scale": 100},
        {"key": "dip_size_pct",         "label": "Taille sur dip",    "min": 0.005, "max": 0.15,  "step": 0.005, "default": 0.033, "unit": "%", "scale": 100},
        {"key": "dip_threshold_pct",    "label": "Seuil dip",         "min": -0.20, "max": -0.01, "step": 0.005, "default": -0.05, "unit": "%", "scale": 100},
        {"key": "max_exposure_pct",     "label": "Exposition max",    "min": 0.50,  "max": 1.0,   "step": 0.05,  "default": 0.90,  "unit": "%", "scale": 100},
        {"key": "min_cash_reserve_pct", "label": "Réserve cash",      "min": 0.02,  "max": 0.20,  "step": 0.02,  "default": 0.05,  "unit": "%", "scale": 100},
    ],
    "trend_following": [
        {"key": "min_adx",           "label": "ADX minimum",        "min": 15,  "max": 40,   "step": 5,    "default": 25,  "unit": "",  "scale": 1},
        {"key": "atr_multiplier_sl", "label": "Stop Loss (×ATR)",   "min": 1.0, "max": 4.0,  "step": 0.25, "default": 2.0, "unit": "×", "scale": 1},
        {"key": "atr_multiplier_tp", "label": "Take Profit (×ATR)", "min": 2.0, "max": 10.0, "step": 0.5,  "default": 3.5, "unit": "×", "scale": 1},
    ],
    "breakout": [
        {"key": "lookback_period",   "label": "Lookback (jours)",    "min": 5,   "max": 60,  "step": 5,    "default": 20,  "unit": "j", "scale": 1},
        {"key": "atr_multiplier_sl", "label": "Stop Loss (×ATR)",    "min": 0.5, "max": 3.0, "step": 0.25, "default": 1.5, "unit": "×", "scale": 1},
        {"key": "atr_multiplier_tp", "label": "Take Profit (×ATR)",  "min": 1.0, "max": 8.0, "step": 0.5,  "default": 3.0, "unit": "×", "scale": 1},
    ],
    "rsi_dip_buyer": [
        {"key": "rsi2_entry_threshold",  "label": "RSI(2) seuil entrée", "min": 5.0, "max": 25.0, "step": 1.0,  "default": 15.0, "unit": "",  "scale": 1},
        {"key": "rsi2_strong_threshold", "label": "RSI(2) extrême",      "min": 2.0, "max": 15.0, "step": 1.0,  "default":  8.0, "unit": "",  "scale": 1},
        {"key": "position_size_pct",     "label": "Taille position",     "min": 0.10, "max": 0.50, "step": 0.05, "default":  0.25, "unit": "%", "scale": 100},
        {"key": "atr_multiplier_sl",     "label": "Stop Loss (×ATR)",    "min": 0.5, "max": 3.0,  "step": 0.25, "default":  1.5, "unit": "×", "scale": 1},
        {"key": "atr_multiplier_tp",     "label": "Take Profit (×ATR)",  "min": 1.0, "max": 6.0,  "step": 0.5,  "default":  2.5, "unit": "×", "scale": 1},
    ],
}

ALL_SLIDER_IDS = [
    f"sl-{s}-{p['key']}"
    for s in STRATEGY_ORDER
    for p in STRATEGY_PARAMS.get(s, [])
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = {**base}
    for k, v in overrides.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged


def strategy_registry(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    scfg = get_strategy_config().get("strategies", {})
    if overrides:
        scfg = merge_dicts(scfg, overrides)
    return {
        "true_dca":        TrueDCAStrategy(scfg.get("true_dca", {})),
        "trend_following": TrendFollowingStrategy(scfg.get("trend_following", {})),
        "breakout":        BreakoutStrategy(scfg.get("breakout", {})),
        "rsi_dip_buyer":   RSIDipBuyerStrategy(scfg.get("rsi_dip_buyer", {})),
    }


def enabled_strategy_names() -> list[str]:
    scfg = get_strategy_config().get("strategies", {})
    enabled = [n for n in STRATEGY_ORDER if scfg.get(n, {}).get("enabled", False)]
    return enabled or ["true_dca"]


def build_backtester(risk_override: dict[str, Any] | None = None) -> Backtester:
    settings = get_settings()
    risk_cfg = get_risk_config()
    if risk_override:
        risk_cfg = merge_dicts(risk_cfg, {"risk": risk_override})
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
        raise ValueError(f"Pas de données pour {ticker}.")
    missing = {"open", "high", "low", "close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {', '.join(sorted(missing))}.")
    return df


def run_selected_backtests(
    ticker: str,
    df: pd.DataFrame,
    strategy_names: list[str],
    config_overrides: dict[str, Any] | None = None,
) -> list[BacktestResult]:
    strat_override = config_overrides.get("strategies") if config_overrides else None
    risk_override = config_overrides.get("risk") if config_overrides else None
    registry = strategy_registry(strat_override)
    results: list[BacktestResult] = []
    for name in strategy_names:
        if name not in registry:
            continue
        bt = build_backtester(risk_override)
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


def compute_trade_stats(result: BacktestResult) -> dict[str, float]:
    buy_sizes = [t.quantity * t.entry_price for t in result.trades if t.side == "long"]
    total_invested = sum(buy_sizes)
    return {
        "avg_position_size": round(total_invested / len(buy_sizes), 2) if buy_sizes else 0.0,
        "total_invested": round(total_invested, 2),
    }


def _win_rate_display(result: BacktestResult) -> str:
    m = result.metrics
    h = m.get("horizon_win_rate_pct")
    n = m.get("horizon_total", 0)
    if result.strategy_name in DCA_STRATEGIES and h is not None:
        return f"{h:.0f}% (1A/{n})"
    v = m.get("win_rate_pct", 0.0)
    return f"{v:.1f}%" if v else "—"


# ── Chart builders ────────────────────────────────────────────────────────────

def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title, template="plotly_white", height=340,
        margin={"l": 50, "r": 16, "t": 44, "b": 36},
        font={"family": "Inter, system-ui, sans-serif", "size": 12},
        xaxis={"showgrid": False},
        yaxis={"gridcolor": "#e5e7eb"},
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
    )
    return fig


def make_equity_figure(df: pd.DataFrame, results: list[BacktestResult]) -> go.Figure:
    fig = _empty_fig("Courbes equity")
    for r in results:
        eq = result_equity_after_warmup(r)
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, mode="lines",
            line={"width": 2, "color": CHART_COLORS.get(r.strategy_name, "#888")},
            name=r.strategy_name,
        ))
    if results:
        bah = buy_and_hold_curve(df, results[0])
        if not bah.empty:
            fig.add_trace(go.Scatter(
                x=bah.index, y=bah.values, mode="lines",
                line={"width": 1.5, "dash": "dash", "color": CHART_COLORS["buy_hold"]},
                name="buy & hold",
            ))
    fig.update_layout(legend={"orientation": "h", "y": 1.12, "x": 0, "font": {"size": 11}})
    fig.update_yaxes(title_text="Capital ($)")
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_drawdown_figure(results: list[BacktestResult]) -> go.Figure:
    fig = _empty_fig("Drawdown")
    for r in results:
        eq = result_equity_after_warmup(r)
        dd = drawdown_curve(eq)
        color = CHART_COLORS.get(r.strategy_name, "#888888")
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, mode="lines", fill="tozeroy",
            line={"width": 1.5, "color": color},
            fillcolor=_hex_to_rgba(color),
            name=r.strategy_name,
        ))
    fig.update_layout(legend={"orientation": "h", "y": 1.12, "x": 0, "font": {"size": 11}})
    fig.update_yaxes(title_text="%")
    return fig


# ── Metric rows / cards ───────────────────────────────────────────────────────

def metric_rows(results: list[BacktestResult]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        m = r.metrics
        rows.append({
            "strategie":    r.strategy_name,
            "initial":      f"${r.initial_capital:,.0f}",
            "final":        f"${r.final_capital:,.2f}",
            "return_pct":   f"{m.get('total_return_pct', 0):.2f}%",
            "cagr":         f"{m.get('cagr_pct', 0):.2f}%",
            "sharpe":       f"{m.get('sharpe_ratio', 0) or 0:.3f}",
            "max_dd":       f"{m.get('max_drawdown_pct', 0):.2f}%",
            "trades":       m.get("total_trades", 0),
            "win_rate":     _win_rate_display(r),
            "pf":           f"{m.get('profit_factor', 0):.2f}" if m.get("profit_factor") != float("inf") else "∞",
            "vs_bh":        f"{m.get('vs_buy_and_hold_pct', 0):+.2f}%",
            "total_invest": f"${m.get('total_invested', 0):,.2f}",
        })
    return rows


def trade_rows(df: pd.DataFrame, results: list[BacktestResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        for t in r.trades:
            entry_date = str(df.index[t.entry_bar].date()) if t.entry_bar < len(df.index) else ""
            exit_date  = str(df.index[t.exit_bar].date())  if t.exit_bar  is not None and t.exit_bar < len(df.index) else ""
            rows.append({
                "strategie":  r.strategy_name,
                "side":       t.side,
                "entry_date": entry_date,
                "exit_date":  exit_date,
                "entry":      round(t.entry_price, 2),
                "exit":       round(t.exit_price or 0.0, 2),
                "qty":        round(t.quantity, 4),
                "pnl":        round(t.pnl, 2),
                "pnl_pct":    f"{t.pnl_pct * 100:.2f}%",
                "raison":     t.exit_reason,
            })
    return rows


def summary_cards(results: list[BacktestResult]) -> list[Any]:
    if not results:
        return []
    best    = max(results, key=lambda r: r.metrics.get("total_return_pct", -9999))
    worst   = min(r.metrics.get("max_drawdown_pct", 0.0) for r in results)
    n_total = sum(int(r.metrics.get("total_trades", 0)) for r in results)
    capital = results[0].initial_capital
    items = [
        ("Capital", f"${capital:,.0f}"),
        ("Meilleur return", f"{best.strategy_name}  {best.metrics.get('total_return_pct', 0):.1f}%"),
        ("Max Drawdown",    f"{worst:.2f}%"),
        ("Total trades",    str(n_total)),
    ]
    return [
        html.Div([
            html.Div(lbl, className="kpi-label"),
            html.Div(val, className="kpi-value"),
        ], className="kpi-card")
        for lbl, val in items
    ]


# ── Slider builders ───────────────────────────────────────────────────────────

def _slider_marks(p: dict[str, Any]) -> dict:
    lo, hi, sc, unit = p["min"], p["max"], p["scale"], p["unit"]
    steps = 4
    span = hi - lo
    marks = {}
    for i in range(steps + 1):
        v = round(lo + i * span / steps, 4)
        disp_v = round(v * sc, 1)
        marks[v] = f"{disp_v}{unit}"
    return marks


def _build_param_panel(strategy_name: str) -> html.Div:
    params = STRATEGY_PARAMS.get(strategy_name, [])
    if not params:
        return html.Div(
            f"Aucun paramètre configurable pour {strategy_name}.",
            style={"color": "#94a3b8", "padding": "16px 0", "fontSize": "13px"},
        )
    rows = []
    for p in params:
        slider_id = f"sl-{strategy_name}-{p['key']}"
        sc, unit = p["scale"], p["unit"]
        disp_val = round(p["default"] * sc, 1)
        rows.append(html.Div([
            html.Div([
                html.Span(p["label"], className="slider-label"),
                html.Span(
                    f"{disp_val}{unit}",
                    id=f"val-{strategy_name}-{p['key']}",
                    className="slider-value",
                ),
            ], className="slider-header"),
            dcc.Slider(
                id=slider_id,
                min=p["min"], max=p["max"], step=p["step"],
                value=p["default"],
                marks=_slider_marks(p),
                tooltip={"placement": "bottom", "always_visible": False},
                className="param-slider",
            ),
        ], className="slider-row"))
    rows.append(html.Button(
        "Réinitialiser", id=f"reset-{strategy_name}",
        className="reset-btn",
    ))
    return html.Div(rows)


# ── App ───────────────────────────────────────────────────────────────────────

def create_app() -> Dash:
    default_start = (date.today() - timedelta(days=365 * 4)).isoformat()
    default_end   = date.today().isoformat()
    default_strategies = enabled_strategy_names()
    strategy_options = [
        {"label": name + (" ★" if name in default_strategies else ""), "value": name}
        for name in STRATEGY_ORDER
    ]

    app = Dash(__name__, title="Backtests")

    app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
      :root {
        --bg: #f1f5f9;
        --panel: #ffffff;
        --border: #e2e8f0;
        --text: #0f172a;
        --muted: #64748b;
        --accent: #2563eb;
        --accent-light: #eff6ff;
        --success: #16a34a;
        --danger: #dc2626;
        --radius: 10px;
      }
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body { background: var(--bg); color: var(--text); font-family: Inter, system-ui, sans-serif; font-size: 14px; }

      /* ── Topbar ── */
      .topbar {
        background: var(--panel);
        border-bottom: 1px solid var(--border);
        padding: 12px 20px;
        display: flex;
        align-items: flex-end;
        gap: 16px;
        position: sticky;
        top: 0;
        z-index: 10;
      }
      .brand { font-size: 18px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 2px; flex-shrink: 0; }
      .field { display: flex; flex-direction: column; gap: 4px; }
      .field-label { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; }
      .ticker-input { height: 36px; width: 90px; border: 1px solid var(--border); border-radius: 6px; padding: 0 10px; font-size: 14px; font-weight: 600; text-transform: uppercase; }
      .run-btn {
        height: 36px; padding: 0 20px;
        background: var(--accent); color: #fff;
        border: none; border-radius: 6px;
        font-size: 13px; font-weight: 700; cursor: pointer;
        transition: opacity .15s;
        margin-bottom: 0;
      }
      .run-btn:hover { opacity: .88; }

      /* ── Content ── */
      .content { padding: 16px 20px 32px; max-width: 1600px; }
      .status-bar { min-height: 22px; color: var(--muted); font-size: 12px; margin-bottom: 12px; }

      /* ── KPI Cards ── */
      .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 14px; }
      .kpi-card { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 16px; }
      .kpi-label { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.4px; }
      .kpi-value { margin-top: 6px; font-size: 22px; font-weight: 800; letter-spacing: -0.5px; }

      /* ── Charts ── */
      .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }
      .chart-panel { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }

      /* ── Param panel ── */
      .param-panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 16px 20px 20px;
        margin-bottom: 14px;
      }
      .param-title { font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
      .tab-strip { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 18px; }
      .tab-strip label {
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        border: 1.5px solid var(--border);
        color: var(--muted);
        background: #fff;
        transition: all .15s;
        user-select: none;
      }
      .tab-strip input { display: none; }
      .tab-strip input:checked + label {
        background: var(--accent-light);
        border-color: var(--accent);
        color: var(--accent);
      }
      .slider-row { margin-bottom: 18px; }
      .slider-header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
      .slider-label { font-size: 12px; font-weight: 600; color: var(--text); }
      .slider-value { font-size: 13px; font-weight: 700; color: var(--accent); }
      .param-slider { margin: 0 2px; }
      .reset-btn {
        margin-top: 8px;
        padding: 5px 14px;
        border: 1.5px solid var(--border);
        border-radius: 6px;
        background: #fff;
        font-size: 12px;
        font-weight: 600;
        color: var(--muted);
        cursor: pointer;
        transition: all .15s;
      }
      .reset-btn:hover { border-color: var(--accent); color: var(--accent); }

      /* ── Tables ── */
      .tables-row { display: grid; grid-template-columns: 1fr 1.3fr; gap: 12px; }
      .table-panel { background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
      .table-title { padding: 12px 16px; font-size: 13px; font-weight: 700; border-bottom: 1px solid var(--border); }

      /* ── Misc ── */
      .loading-wrap { min-height: 400px; }
      @media (max-width: 900px) {
        .kpi-row, .chart-row, .tables-row { grid-template-columns: 1fr; }
        .topbar { flex-wrap: wrap; }
      }
    </style>
  </head>
  <body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""

    _ts = {
        "style_table": {"overflowX": "auto", "maxHeight": "340px", "overflowY": "auto"},
        "style_header": {"backgroundColor": "#f8fafc", "fontWeight": "700", "border": "1px solid #e2e8f0", "fontSize": "12px"},
        "style_cell":   {"fontFamily": "Inter, system-ui, sans-serif", "fontSize": "12px", "padding": "7px 10px", "border": "1px solid #f1f5f9", "whiteSpace": "nowrap"},
        "style_data_conditional": [{"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"}],
    }

    # ── Build param panels (one per strategy) ─────────────────────────────────
    param_panels = []
    for s in STRATEGY_ORDER:
        param_panels.append(
            html.Div(
                id=f"pp-{s}",
                children=_build_param_panel(s),
                style={"display": "block" if s == STRATEGY_ORDER[0] else "none"},
            )
        )

    # ── Tab strip options ─────────────────────────────────────────────────────
    tab_options = [{"label": s.replace("_", " "), "value": s} for s in STRATEGY_ORDER]

    app.layout = html.Div([

        # Topbar
        html.Div(className="topbar", children=[
            html.Div("Backtests", className="brand"),
            html.Div(className="field", children=[
                html.Span("Ticker", className="field-label"),
                dcc.Input(id="ticker", value="SPY", className="ticker-input", debounce=True),
            ]),
            html.Div(className="field", children=[
                html.Span("Période", className="field-label"),
                dcc.DatePickerRange(
                    id="date-range", start_date=default_start, end_date=default_end,
                    display_format="YYYY-MM-DD",
                ),
            ]),
            html.Div(className="field", style={"flex": "1", "minWidth": "220px"}, children=[
                html.Span("Stratégies", className="field-label"),
                dcc.Dropdown(
                    id="strat-select", options=strategy_options,
                    value=default_strategies, multi=True, clearable=False,
                ),
            ]),
            html.Button("Lancer", id="run-btn", className="run-btn"),
        ]),

        html.Div(className="content", children=[

            # Param panel
            html.Div(className="param-panel", children=[
                html.Div("Paramètres de stratégie", className="param-title"),
                dcc.RadioItems(
                    id="param-tab",
                    options=tab_options,
                    value=STRATEGY_ORDER[0],
                    className="tab-strip",
                    inputStyle={"display": "none"},
                    labelStyle={},
                ),
                html.Div(id="param-panels-container", children=param_panels),
            ]),

            dcc.Loading(type="circle", color="#2563eb", parent_className="loading-wrap", children=[
                html.Div(id="status-bar", className="status-bar",
                         children="Choisis une période et clique Lancer."),
                html.Div(id="kpi-row", className="kpi-row"),
                html.Div(className="chart-row", children=[
                    html.Div(dcc.Graph(id="eq-chart",  config={"displayModeBar": False}), className="chart-panel"),
                    html.Div(dcc.Graph(id="dd-chart",  config={"displayModeBar": False}), className="chart-panel"),
                ]),
                html.Div(className="tables-row", children=[
                    html.Div(className="table-panel", children=[
                        html.Div("Métriques", className="table-title"),
                        dash_table.DataTable(
                            id="metrics-tbl",
                            columns=[
                                {"name": "Stratégie",     "id": "strategie"},
                                {"name": "Initial",       "id": "initial"},
                                {"name": "Final",         "id": "final"},
                                {"name": "Return %",      "id": "return_pct"},
                                {"name": "CAGR",          "id": "cagr"},
                                {"name": "Sharpe",        "id": "sharpe"},
                                {"name": "Max DD",        "id": "max_dd"},
                                {"name": "Trades",        "id": "trades"},
                                {"name": "Win % (1A DCA)","id": "win_rate"},
                                {"name": "PF",            "id": "pf"},
                                {"name": "vs B&H",        "id": "vs_bh"},
                                {"name": "Total investi", "id": "total_invest"},
                            ],
                            data=[], **_ts,
                        ),
                    ]),
                    html.Div(className="table-panel", children=[
                        html.Div("Trades", className="table-title"),
                        dash_table.DataTable(
                            id="trades-tbl",
                            columns=[
                                {"name": "Stratégie", "id": "strategie"},
                                {"name": "Côté",      "id": "side"},
                                {"name": "Entrée",    "id": "entry_date"},
                                {"name": "Sortie",    "id": "exit_date"},
                                {"name": "Prix in",   "id": "entry"},
                                {"name": "Prix out",  "id": "exit"},
                                {"name": "Qté",       "id": "qty"},
                                {"name": "PnL",       "id": "pnl"},
                                {"name": "PnL %",     "id": "pnl_pct"},
                                {"name": "Raison",    "id": "raison"},
                            ],
                            data=[], page_size=15, **_ts,
                        ),
                    ]),
                ]),
            ]),
        ]),
    ])

    # ── Callbacks ─────────────────────────────────────────────────────────────

    # Show/hide param panels when tab changes
    @app.callback(
        [Output(f"pp-{s}", "style") for s in STRATEGY_ORDER],
        Input("param-tab", "value"),
    )
    def toggle_param_panels(selected: str) -> list[dict]:
        return [{"display": "block" if s == selected else "none"} for s in STRATEGY_ORDER]

    # Update slider value labels live
    for _s in STRATEGY_ORDER:
        for _p in STRATEGY_PARAMS.get(_s, []):
            _sid = f"sl-{_s}-{_p['key']}"
            _vid = f"val-{_s}-{_p['key']}"
            _sc  = _p["scale"]
            _unit = _p["unit"]

            @app.callback(
                Output(_vid, "children"),
                Input(_sid, "value"),
            )
            def _update_label(val: float, sc: int = _sc, unit: str = _unit) -> str:
                if val is None:
                    return "—"
                disp = round(val * sc, 2)
                return f"{disp}{unit}"

    # Reset buttons (one per strategy)
    for _s in STRATEGY_ORDER:
        _params = STRATEGY_PARAMS.get(_s, [])
        if not _params:
            continue

        @app.callback(
            [Output(f"sl-{_s}-{p['key']}", "value") for p in _params],
            Input(f"reset-{_s}", "n_clicks"),
            prevent_initial_call=True,
        )
        def _reset(_, s: str = _s, ps: list = _params) -> list:
            return [p["default"] for p in ps]

    _empty_eq = _empty_fig("Courbes equity")
    _empty_dd = _empty_fig("Drawdown")
    _default_out = ("Choisis une période et clique Lancer.", [], _empty_eq, _empty_dd, [], [])

    # Main run callback.
    # Triggered by: button click, ticker change (debounced), date change, strategy selection or slider updates.
    @app.callback(
        Output("status-bar",  "children"),
        Output("kpi-row",     "children"),
        Output("eq-chart",    "figure"),
        Output("dd-chart",    "figure"),
        Output("metrics-tbl", "data"),
        Output("trades-tbl",  "data"),
        Input("run-btn",      "n_clicks"),
        Input("ticker",       "value"),
        Input("date-range",   "start_date"),
        Input("date-range",   "end_date"),
        Input("strat-select", "value"),
        *[Input(sid, "value") for sid in ALL_SLIDER_IDS],
        prevent_initial_call=True,
    )
    def run(
        _n_clicks: int | None,
        ticker: str,
        start_date: str | None,
        end_date: str | None,
        strategy_names: list[str],
        *slider_values: float,
    ) -> tuple:

        # Guard: incomplete inputs — wait silently
        if not start_date or not end_date or not ticker or not ticker.strip():
            return _default_out
        try:
            if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
                return ("Dates invalides : début ≥ fin.", [], _empty_eq, _empty_dd, [], [])
        except Exception:
            return _default_out

        ticker = ticker.upper().strip()
        strategy_names = strategy_names or enabled_strategy_names()

        # Build param overrides from slider values
        idx = 0
        strat_overrides: dict[str, Any] = {}
        for s in STRATEGY_ORDER:
            params = STRATEGY_PARAMS.get(s, [])
            strat_overrides[s] = {}
            for p in params:
                v = slider_values[idx]
                strat_overrides[s][p["key"]] = float(v if v is not None else p["default"])
                idx += 1
        overrides = {"strategies": strat_overrides}

        try:
            df = download_prices(ticker, start_date, end_date)
            results = run_selected_backtests(ticker, df, strategy_names, overrides)
            for r in results:
                r.metrics.update(compute_trade_stats(r))
            status = (
                f"{ticker}  ·  {df.index[0].date()} → {df.index[-1].date()}"
                f"  ·  {len(df)} barres"
                f"  ·  capital ${build_backtester().initial_capital:,.0f}"
            )
            return (
                status,
                summary_cards(results),
                make_equity_figure(df, results),
                make_drawdown_figure(results),
                metric_rows(results),
                trade_rows(df, results),
            )
        except Exception as exc:
            return (f"Erreur : {exc}", [], _empty_eq, _empty_dd, [], [])

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    settings = get_settings()
    default_port = int(settings.get("monitoring", {}).get("dashboard_port", 8050))
    p = argparse.ArgumentParser(description="Backtest dashboard")
    p.add_argument("--host",  default="127.0.0.1")
    p.add_argument("--port",  type=int, default=default_port)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    create_app().run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
