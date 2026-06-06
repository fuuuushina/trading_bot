"""
src/dashboard/live_dashboard.py

Tableau de bord multi-pages pour le bot de trading.
Affiche à la fois :
  - Le bot EUR/USD 5min (données dans data/dashboard/bot_state.json)
  - Le compte Alpaca Paper (données dans data/paper_trading/alpaca_state.json)

Pages :
  1  Vue globale    — aperçu complet + compte Alpaca
  2  EUR/USD Live   — graphique chandeliers 5 min + indicateurs
  3  Stratégies     — état des 3 stratégies avec explications
  4  Portefeuille   — Alpaca + bot EUR/USD, trades, courbe d'équité
  5  Régime & IA    — régime de marché, analyse LLM, actualités

Démarrage :
    python -m src.dashboard.live_dashboard --port 8051
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import (
    Dash, Input, Output, State,
    callback_context, dash_table, dcc, html, no_update,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BOT_STATE_FILE    = PROJECT_ROOT / "data" / "dashboard" / "bot_state.json"
ALPACA_STATE_FILE = PROJECT_ROOT / "data" / "paper_trading" / "alpaca_state.json"
REFRESH_MS = 15_000

# ── Palettes ──────────────────────────────────────────────────────────────────
C_BG     = "#f0f4f8"
C_PANEL  = "#ffffff"
C_BORDER = "#e2e8f0"
C_TEXT   = "#1a202c"
C_MUTED  = "#718096"
C_ACCENT = "#3182ce"
C_GREEN  = "#38a169"
C_RED    = "#e53e3e"
C_ORANGE = "#dd6b20"
C_YELLOW = "#d69e2e"
C_PURPLE = "#805ad5"
C_TEAL   = "#319795"
C_ALPACA = "#7c3aed"

REGIME_COLOR = {
    "bull_trend": C_GREEN, "bear_trend": C_RED, "range": C_YELLOW,
    "high_volatility": C_ORANGE, "low_volatility": C_TEAL, "panic": "#742a2a",
    "euphoric": "#97266d", "compression": C_PURPLE,
    "breakout_expansion": C_ACCENT, "unknown": C_MUTED,
}
REGIME_ICON = {
    "bull_trend": "↑", "bear_trend": "↓", "range": "↔",
    "high_volatility": "⚡", "low_volatility": "〜", "panic": "⚠",
    "euphoric": "★", "compression": "⏸", "breakout_expansion": "▶", "unknown": "?",
}
RISK_COLOR = {"low": C_GREEN, "medium": C_YELLOW, "high": C_ORANGE, "extreme": C_RED}

STRATEGY_INFO = {
    "intraday_ema_cross": {
        "name": "Croisement EMA", "icon": "✕", "color": "#3182ce",
        "desc": "Achète quand la moyenne courte (EMA 9) passe au-dessus de la moyenne longue (EMA 21). Idéal en marché directionnel.",
        "when": "Tous régimes sauf panique",
    },
    "intraday_bollinger_rsi": {
        "name": "Bollinger + RSI", "icon": "〜", "color": "#805ad5",
        "desc": "Achète quand le prix touche la bande basse de Bollinger ET que le RSI < 35. Parie sur un retour vers la moyenne. Idéal en range.",
        "when": "Range / Basse volatilité",
    },
    "intraday_session_breakout": {
        "name": "Cassure de Session", "icon": "▶", "color": "#38a169",
        "desc": "Entre dans le sens de la cassure du range pré-session. Fenêtres actives : Londres 07h00-09h30 UTC, New York 13h30-16h00 UTC.",
        "when": "Ouverture London / NY",
    },
}

TABS = [
    ("overview",   "Vue globale"),
    ("eurusd",     "EUR/USD Live"),
    ("strategies", "Strategies"),
    ("portfolio",  "Portefeuille"),
    ("regime",     "Regime et IA"),
]

# Cache yfinance pour ne pas refetcher a chaque refresh
_eur_yf_cache: dict = {"ts": 0.0, "data": {}}


# ── Lecture des sources de données ───────────────────────────────────────────

def read_bot_state() -> dict:
    try:
        if BOT_STATE_FILE.exists():
            with open(BOT_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def read_alpaca_state() -> dict:
    try:
        if ALPACA_STATE_FILE.exists():
            with open(ALPACA_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def bot_freshness(state: dict) -> str:
    lu = state.get("last_update")
    if not lu:
        return "offline"
    try:
        ts = datetime.fromisoformat(lu).replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if age < 120:
            return "live"
        if age < 3600:
            return "delayed"
    except Exception:
        pass
    return "offline"


def fetch_eurusd_live() -> dict:
    """Fetch EUR/USD 5min depuis yfinance. Résultat mis en cache 60s."""
    global _eur_yf_cache
    if time.time() - _eur_yf_cache["ts"] < 60 and _eur_yf_cache["data"]:
        return _eur_yf_cache["data"]
    try:
        import yfinance as yf
        from src.data.yfinance_helpers import normalize_yfinance_columns
        from src.features.indicators import ema as _ema, rsi as _rsi, atr as _atr

        df = yf.download("EURUSD=X", period="2d", interval="5m",
                         auto_adjust=True, progress=False)
        if df.empty:
            return {}
        df = normalize_yfinance_columns(df)
        close = df["close"]
        e9    = _ema(close, 9)
        e21   = _ema(close, 21)
        r14   = _rsi(close, 14)
        a14   = _atr(df, 14)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std(ddof=0)
        bb_u  = sma20 + 2.0 * std20
        bb_l  = sma20 - 2.0 * std20

        cur  = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else cur
        n    = min(80, len(df))
        ohlcv = []
        for i in range(-n, 0):
            bbu_v = float(bb_u.iloc[i])
            bbl_v = float(bb_l.iloc[i])
            ohlcv.append({
                "t":   str(df.index[i]),
                "o":   round(float(df["open"].iloc[i]), 5),
                "h":   round(float(df["high"].iloc[i]), 5),
                "l":   round(float(df["low"].iloc[i]), 5),
                "c":   round(float(df["close"].iloc[i]), 5),
                "e9":  round(float(e9.iloc[i]), 5) if not isinstance(e9.iloc[i], float) or e9.iloc[i] == e9.iloc[i] else None,
                "e21": round(float(e21.iloc[i]), 5) if not isinstance(e21.iloc[i], float) or e21.iloc[i] == e21.iloc[i] else None,
                "bbu": round(bbu_v, 5) if bbu_v == bbu_v else None,
                "bbl": round(bbl_v, 5) if bbl_v == bbl_v else None,
            })
        result = {
            "price":     round(cur, 5),
            "change_pct": round((cur / prev - 1) * 100, 4),
            "ema_9":     round(float(e9.iloc[-1]), 5),
            "ema_21":    round(float(e21.iloc[-1]), 5),
            "rsi_14":    round(float(r14.iloc[-1]), 2),
            "atr_14":    round(float(a14.iloc[-1]), 6),
            "bb_upper":  round(float(bb_u.iloc[-1]), 5),
            "bb_middle": round(float(sma20.iloc[-1]), 5),
            "bb_lower":  round(float(bb_l.iloc[-1]), 5),
            "ohlcv":     ohlcv,
        }
        _eur_yf_cache = {"ts": time.time(), "data": result}
        return result
    except Exception:
        return {}


def get_eurusd_data(state: dict) -> dict:
    if bot_freshness(state) == "live" and state.get("eurusd"):
        return state["eurusd"]
    return fetch_eurusd_live()


# ── Composants UI ─────────────────────────────────────────────────────────────

def tip(text: str, tooltip: str) -> html.Span:
    return html.Span(text, title=tooltip,
                     style={"borderBottom": f"1px dotted {C_MUTED}", "cursor": "help"})


def kpi_card(label: str, value: str, subtitle: str = "",
             color: str = C_TEXT, tooltip: str = "") -> html.Div:
    label_el = tip(label, tooltip) if tooltip else html.Span(label)
    children = [
        html.Div(label_el, style={
            "fontSize": "11px", "fontWeight": "700", "color": C_MUTED,
            "textTransform": "uppercase", "letterSpacing": "0.5px",
        }),
        html.Div(str(value), style={
            "fontSize": "22px", "fontWeight": "800", "color": color,
            "marginTop": "5px", "lineHeight": "1.15", "wordBreak": "break-word",
        }),
    ]
    if subtitle:
        children.append(html.Div(str(subtitle),
                                 style={"fontSize": "12px", "color": C_MUTED, "marginTop": "3px"}))
    return html.Div(children, style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "10px", "padding": "14px 16px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
    })


def badge(text: str, color: str = C_ACCENT, bg: str = "") -> html.Span:
    bg_col = bg or f"{color}18"
    return html.Span(str(text), style={
        "background": bg_col, "color": color,
        "borderRadius": "999px", "fontSize": "11px", "fontWeight": "800",
        "padding": "3px 10px",
    })


def signal_badge_el(sig: str) -> html.Span:
    sig = str(sig).upper()
    colors = {
        "BUY": (C_GREEN, "#f0fff4"), "SELL": (C_RED, "#fff5f5"),
        "EXECUTE": (C_GREEN, "#f0fff4"), "BLOCK": (C_ORANGE, "#fffaf0"),
        "NO_TRADE": (C_MUTED, "#f7fafc"), "HOLD": (C_YELLOW, "#fffff0"),
    }
    c, bg = colors.get(sig, (C_MUTED, "#f7fafc"))
    return badge(sig, c, bg)


def section_header(title: str, color: str = C_ACCENT) -> html.Div:
    return html.Div(title, style={
        "fontSize": "13px", "fontWeight": "900", "color": color,
        "textTransform": "uppercase", "letterSpacing": "0.8px",
        "padding": "10px 0 8px",
        "borderBottom": f"2px solid {color}30",
        "marginBottom": "12px", "marginTop": "20px",
    })


def hex_rgba(hex_color: str, alpha: float = 0.07) -> str:
    """Convertit #rrggbb en rgba(r,g,b,alpha) compatible Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def empty_fig(msg: str = "En attente de données...") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font={"size": 13, "color": C_MUTED})
    fig.update_layout(
        height=280, margin={"l": 30, "r": 20, "t": 30, "b": 20},
        paper_bgcolor=C_PANEL, plot_bgcolor=C_PANEL,
        xaxis={"visible": False}, yaxis={"visible": False},
    )
    return fig


def _chart_base() -> dict:
    return dict(
        template="plotly_white", paper_bgcolor=C_PANEL, plot_bgcolor=C_PANEL,
        font={"family": "Inter, system-ui, sans-serif", "size": 11, "color": C_TEXT},
        margin={"l": 55, "r": 15, "t": 35, "b": 35},
        xaxis={"showgrid": False, "zeroline": False},
        yaxis={"gridcolor": "#edf2f7", "zeroline": False},
    )


def card_wrap(children, border_color: str = C_BORDER, padding: str = "16px 18px") -> html.Div:
    return html.Div(children, style={
        "background": C_PANEL, "border": f"1px solid {border_color}",
        "borderRadius": "10px", "padding": padding,
        "boxShadow": "0 1px 3px rgba(0,0,0,0.06)", "marginBottom": "14px",
    })


def grid2(left, right, left_width: str = "1fr", right_width: str = "220px") -> html.Div:
    return html.Div([left, right], style={
        "display": "grid",
        "gridTemplateColumns": f"{left_width} {right_width}",
        "gap": "12px", "alignItems": "start",
    })


# ── Graphiques ────────────────────────────────────────────────────────────────

def equity_chart(history: list[dict], initial: float,
                 title: str = "Equite", color: str | None = None) -> go.Figure:
    if not history:
        return empty_fig("Pas encore de données d'équité")
    times  = [r.get("time", "") for r in history]
    values = [float(r.get("equity", initial)) for r in history]
    c = color or (C_GREEN if values[-1] >= initial else C_RED)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=values, mode="lines",
        line={"width": 2.5, "color": c},
        fill="tozeroy", fillcolor=hex_rgba(c, 0.07),
        name="Equite",
        hovertemplate="%{y:$,.2f}<extra></extra>",
    ))
    fig.add_hline(y=initial, line_dash="dash", line_color="#a0aec0", line_width=1,
                  annotation_text=f"Depart ${initial:,.0f}",
                  annotation_font_color=C_MUTED)
    fig.update_layout(title=title, height=250, showlegend=False, **_chart_base())
    fig.update_yaxes(tickprefix="$")
    return fig


def eurusd_chart(ohlcv: list[dict]) -> go.Figure:
    if not ohlcv:
        return empty_fig("EUR/USD — En attente de données 5min...")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
        subplot_titles=("EUR/USD — 5 minutes", "RSI 14"),
    )
    times  = [r["t"]  for r in ohlcv]
    opens  = [r["o"]  for r in ohlcv]
    highs  = [r["h"]  for r in ohlcv]
    lows   = [r["l"]  for r in ohlcv]
    closes = [r["c"]  for r in ohlcv]
    e9v    = [r.get("e9")  for r in ohlcv]
    e21v   = [r.get("e21") for r in ohlcv]
    bbu    = [r.get("bbu") for r in ohlcv]
    bbl    = [r.get("bbl") for r in ohlcv]

    if all(v is not None for v in bbu + bbl):
        fig.add_trace(go.Scatter(
            x=times + times[::-1], y=bbu + bbl[::-1],
            fill="toself", fillcolor="rgba(49,130,206,0.07)",
            line={"color": "rgba(0,0,0,0)"}, showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=bbu, line={"color": "#3182ce", "width": 0.9, "dash": "dot"},
                                 showlegend=False, hovertemplate="%{y:.5f}<extra>BB+</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=bbl, line={"color": "#3182ce", "width": 0.9, "dash": "dot"},
                                 showlegend=False, hovertemplate="%{y:.5f}<extra>BB-</extra>"), row=1, col=1)

    fig.add_trace(go.Candlestick(
        x=times, open=opens, high=highs, low=lows, close=closes,
        increasing_line_color=C_GREEN, decreasing_line_color=C_RED,
        increasing_fillcolor=C_GREEN, decreasing_fillcolor=C_RED,
        name="EUR/USD",
    ), row=1, col=1)

    if any(v is not None for v in e9v):
        fig.add_trace(go.Scatter(x=times, y=e9v, line={"color": "#e53e3e", "width": 1.5},
                                 name="EMA 9", hovertemplate="%{y:.5f}<extra>EMA9</extra>"), row=1, col=1)
    if any(v is not None for v in e21v):
        fig.add_trace(go.Scatter(x=times, y=e21v, line={"color": "#3182ce", "width": 1.5},
                                 name="EMA 21", hovertemplate="%{y:.5f}<extra>EMA21</extra>"), row=1, col=1)

    # RSI sur sous-graphique (calculer depuis les closes)
    try:
        import numpy as np
        c_arr = [v for v in closes if v is not None]
        if len(c_arr) >= 14:
            delta = np.diff(c_arr)
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)
            avg_g = np.convolve(gain, np.ones(14)/14, mode='valid')
            avg_l = np.convolve(loss, np.ones(14)/14, mode='valid')
            rs = avg_g / np.where(avg_l == 0, 1e-9, avg_l)
            rsi_vals = 100 - (100 / (1 + rs))
            pad = len(closes) - len(rsi_vals)
            rsi_full = [None] * pad + list(rsi_vals)
            fig.add_trace(go.Scatter(
                x=times, y=rsi_full, line={"color": C_PURPLE, "width": 1.5},
                name="RSI", hovertemplate="%{y:.1f}<extra>RSI</extra>",
            ), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color=C_RED,   line_width=1, row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color=C_GREEN, line_width=1, row=2, col=1)
    except Exception:
        pass

    fig.update_layout(
        height=500, xaxis_rangeslider_visible=False,
        paper_bgcolor=C_PANEL, plot_bgcolor=C_PANEL,
        font={"family": "Inter, system-ui, sans-serif", "size": 11},
        margin={"l": 55, "r": 15, "t": 40, "b": 25},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#edf2f7", row=1, col=1, tickformat=".5f")
    fig.update_yaxes(gridcolor="#edf2f7", row=2, col=1, range=[0, 100])
    return fig


def rsi_gauge(rsi_val: float) -> go.Figure:
    color = C_RED if rsi_val > 70 else (C_GREEN if rsi_val < 30 else C_ACCENT)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_val,
        title={"text": "RSI (14)", "font": {"size": 12, "color": C_MUTED}},
        number={"font": {"size": 26, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": C_MUTED},
            "bar": {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0, 30], "color": "#f0fff4"},
                {"range": [30, 70], "color": "#ebf8ff"},
                {"range": [70, 100], "color": "#fff5f5"},
            ],
        },
    ))
    fig.update_layout(height=180, margin={"l": 20, "r": 20, "t": 25, "b": 5},
                      paper_bgcolor=C_PANEL)
    return fig


# ── Page : Vue Globale ────────────────────────────────────────────────────────

def page_overview(state: dict, alpaca: dict) -> html.Div:
    fresh = bot_freshness(state)
    port  = state.get("portfolio", {})
    mkt   = state.get("market", {})
    hist  = state.get("equity_history", [])
    sigs  = state.get("recent_signals", [])
    eur   = state.get("eurusd", {})

    # Alpaca
    alp_initial  = float(alpaca.get("initial_capital", 0))
    alp_hist     = alpaca.get("equity_history", [])
    alp_targets  = alpaca.get("targets", {})
    alp_closed   = alpaca.get("closed_events", [])
    alp_equity   = float(alp_hist[-1]["equity"]) if alp_hist else alp_initial
    alp_realized = sum(float(x.get("pnl", 0)) for x in alp_closed)
    alp_open_est = alp_equity - alp_initial  # approximation
    has_alpaca   = alp_initial > 0

    # Status banner
    status_info = {
        "live":    ("Bot EUR/USD actif — données en temps réel", C_GREEN, "#f0fff4"),
        "delayed": ("Données récentes (< 1h)", C_YELLOW, "#fffff0"),
        "offline": ("Bot EUR/USD hors ligne — lancez src/main.py", C_RED, "#fff5f5"),
    }
    st_text, st_col, st_bg = status_info[fresh]
    banner = html.Div(st_text, style={
        "background": st_bg, "color": st_col, "border": f"1px solid {st_col}30",
        "borderRadius": "8px", "padding": "10px 16px",
        "fontSize": "13px", "fontWeight": "600", "marginBottom": "14px",
    })

    # KPIs EUR/USD bot
    initial = port.get("initial_capital", 10_000)
    equity  = port.get("total_equity", initial)
    pnl     = port.get("total_pnl", 0)
    ret     = port.get("return_pct", 0)
    eur_price = eur.get("price", 0)
    eur_chg   = eur.get("change_pct", 0)
    regime    = mkt.get("regime", "unknown")
    reg_fr    = mkt.get("regime_fr", "Inconnu")
    reg_col   = REGIME_COLOR.get(regime, C_MUTED)
    reg_icon  = REGIME_ICON.get(regime, "?")

    eur_kpis = html.Div([
        html.Div("Bot EUR/USD (Intraday 5min)", style={
            "fontSize": "11px", "fontWeight": "900", "color": C_ACCENT,
            "textTransform": "uppercase", "letterSpacing": "0.8px",
            "marginBottom": "10px",
        }),
        html.Div([
            kpi_card("Capital total", f"${equity:,.2f}",
                     subtitle=f"Depart ${initial:,.0f}",
                     color=C_GREEN if equity >= initial else C_RED,
                     tooltip="Cash + positions ouvertes"),
            kpi_card("P&L total", f"{'+' if pnl >= 0 else ''}{pnl:,.2f}$",
                     subtitle=f"{ret:+.2f}%",
                     color=C_GREEN if pnl >= 0 else C_RED,
                     tooltip="Gains/pertes réalisés + non réalisés"),
            kpi_card("EUR/USD", f"{eur_price:.5f}" if eur_price else "—",
                     subtitle=f"{eur_chg:+.4f}%" if eur_price else "offline",
                     color=C_GREEN if eur_chg >= 0 else C_RED,
                     tooltip="Cours actuel EUR/USD 5min"),
            kpi_card(f"{reg_icon} Regime", reg_fr,
                     subtitle=f"Confiance {mkt.get('confidence', 0)*100:.0f}%",
                     color=reg_col,
                     tooltip="Regime de marché détecté"),
            kpi_card("Niveau de risque", mkt.get("risk_fr", "—"),
                     color=RISK_COLOR.get(mkt.get("risk_level", ""), C_MUTED),
                     tooltip="Niveau de risque calculé par le gestionnaire de risque"),
            kpi_card("Positions", str(port.get("num_positions", 0)),
                     subtitle=f"Exposition {port.get('exposure_pct', 0)*100:.1f}%",
                     tooltip="Trades EUR/USD actuellement ouverts"),
        ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, 1fr)",
            "gap": "10px",
        }),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "10px", "padding": "14px 16px", "marginBottom": "14px",
    })

    # KPIs Alpaca
    alpaca_block = None
    if has_alpaca:
        alp_positions_rows = []
        for sym, tgt in alp_targets.items():
            ep   = float(tgt.get("entry_price", 0))
            qty  = float(tgt.get("quantity", 0))
            strat= tgt.get("strategy", "—")
            alp_positions_rows.append(html.Div([
                html.Span(sym, style={"fontWeight": "800", "color": C_TEXT, "minWidth": "60px"}),
                badge(strat.replace("_", " "), C_ALPACA),
                html.Span(f"Entree ${ep:.2f} × {qty:.4f}", style={"color": C_MUTED, "fontSize": "12px"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "10px",
                      "padding": "6px 0", "borderBottom": f"1px solid {C_BORDER}"}))

        alpaca_block = html.Div([
            html.Div("Compte Alpaca Paper", style={
                "fontSize": "11px", "fontWeight": "900", "color": C_ALPACA,
                "textTransform": "uppercase", "letterSpacing": "0.8px",
                "marginBottom": "10px",
            }),
            html.Div([
                kpi_card("Capital Alpaca", f"${alp_equity:,.2f}",
                         subtitle=f"Depart ${alp_initial:,.0f}",
                         color=C_GREEN if alp_equity >= alp_initial else C_RED,
                         tooltip="Équité estimée compte Alpaca Paper"),
                kpi_card("P&L realisé", f"${alp_realized:+,.2f}",
                         color=C_GREEN if alp_realized >= 0 else C_RED,
                         tooltip="Gains/pertes définitivement clôturés"),
                kpi_card("Positions ouvertes", str(len(alp_targets)),
                         subtitle=", ".join(alp_targets.keys()) if alp_targets else "aucune",
                         tooltip="Positions actuellement trackées par le bot Alpaca"),
                kpi_card("Trades fermes", str(len(alp_closed)),
                         tooltip="Nombre de trades complets (entrée + sortie)"),
            ], style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "10px",
            }),
            html.Div([
                html.Div("Positions Alpaca", style={
                    "fontSize": "12px", "fontWeight": "700", "marginBottom": "8px",
                    "color": C_MUTED,
                }),
                html.Div(
                    alp_positions_rows if alp_positions_rows else
                    [html.Div("Aucune position ouverte.", style={"color": C_MUTED, "fontSize": "13px"})],
                ),
            ], style={"marginTop": "12px"}) if alp_targets or not alp_positions_rows else None,
        ], style={
            "background": C_PANEL, "border": f"2px solid {C_ALPACA}30",
            "borderRadius": "10px", "padding": "14px 16px", "marginBottom": "14px",
        })

    # Courbe équité (Alpaca en priorité si EUR/USD vide)
    eq_history = hist or alp_hist
    eq_initial = initial if hist else alp_initial
    eq_color   = C_ACCENT if hist else C_ALPACA
    eq_title   = "Équité EUR/USD Bot" if hist else "Équité Alpaca Paper"
    eq_fig = equity_chart(eq_history, eq_initial, eq_title, eq_color)
    eq_panel = card_wrap(
        dcc.Graph(figure=eq_fig, config={"displayModeBar": False}),
        padding="0"
    )

    # Signaux récents
    events_rows = []
    for sig in (sigs or [])[:8]:
        t      = str(sig.get("time", "—"))[:16]
        action = sig.get("action", "")
        strat  = sig.get("strategy", "").replace("intraday_", "")
        reason = str(sig.get("reason", ""))[:80]
        events_rows.append(html.Div([
            html.Span(t, style={"color": C_MUTED, "fontSize": "11px", "minWidth": "100px"}),
            signal_badge_el(action or "—"),
            html.Span(strat, style={"color": C_ACCENT, "fontSize": "12px", "fontWeight": "700"}),
            html.Span(reason, style={"color": C_MUTED, "fontSize": "12px",
                                     "overflow": "hidden", "textOverflow": "ellipsis",
                                     "whiteSpace": "nowrap"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px",
                  "padding": "6px 0", "borderBottom": f"1px solid {C_BORDER}"}))

    # Alpaca signals
    alp_sigs = alpaca.get("signals_log", [])
    for sig in list(reversed(alp_sigs))[:5]:
        t      = str(sig.get("time", "—"))[:16]
        ticker = sig.get("ticker", "")
        signal = sig.get("signal", "")
        reason = str(sig.get("reason", ""))[:80]
        events_rows.append(html.Div([
            html.Span(t, style={"color": C_MUTED, "fontSize": "11px", "minWidth": "100px"}),
            badge("ALPACA", C_ALPACA),
            signal_badge_el(signal.split()[0] if signal else "—"),
            html.Span(ticker, style={"color": C_ALPACA, "fontSize": "12px", "fontWeight": "700"}),
            html.Span(reason, style={"color": C_MUTED, "fontSize": "12px",
                                     "overflow": "hidden", "textOverflow": "ellipsis",
                                     "whiteSpace": "nowrap"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px",
                  "padding": "6px 0", "borderBottom": f"1px solid {C_BORDER}"}))

    if not events_rows:
        events_rows = [html.Div("Aucun signal récent.", style={"color": C_MUTED, "padding": "14px"})]

    events_panel = html.Div([
        html.Div("Derniers signaux et décisions", style={
            "padding": "10px 16px", "fontWeight": "800", "fontSize": "13px",
            "borderBottom": f"1px solid {C_BORDER}",
        }),
        html.Div(events_rows, style={"padding": "6px 16px"}),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "10px", "overflow": "hidden", "marginBottom": "14px",
    })

    children = [banner, eur_kpis]
    if alpaca_block:
        children.append(alpaca_block)
    children += [eq_panel, events_panel]
    return html.Div(children)


# ── Page : EUR/USD Live ───────────────────────────────────────────────────────

def page_eurusd(state: dict) -> html.Div:
    eur = get_eurusd_data(state)

    if not eur:
        return html.Div([
            html.Div("Chargement des données EUR/USD...", style={
                "textAlign": "center", "padding": "60px",
                "color": C_MUTED, "fontSize": "15px",
            }),
            html.Div("Les données yfinance se chargent automatiquement.",
                     style={"textAlign": "center", "color": C_MUTED, "fontSize": "13px"}),
        ])

    price   = eur.get("price", 0)
    chg     = eur.get("change_pct", 0)
    rsi_val = float(eur.get("rsi_14", 50) or 50)
    ema9    = eur.get("ema_9", 0)
    ema21   = eur.get("ema_21", 0)
    atr     = eur.get("atr_14", 0)
    bb_u    = eur.get("bb_upper", 0)
    bb_l    = eur.get("bb_lower", 0)
    ohlcv   = eur.get("ohlcv", [])
    chg_col = C_GREEN if chg >= 0 else C_RED

    # Interprétations
    if ema9 and ema21:
        ema_lbl  = "EMA 9 > EMA 21 — Haussier" if ema9 > ema21 else "EMA 9 < EMA 21 — Baissier"
        ema_col  = C_GREEN if ema9 > ema21 else C_RED
    else:
        ema_lbl, ema_col = "—", C_MUTED

    if rsi_val > 70:
        rsi_lbl, rsi_col = "Surachat >70", C_RED
    elif rsi_val < 30:
        rsi_lbl, rsi_col = "Survente <30", C_GREEN
    else:
        rsi_lbl, rsi_col = "Zone neutre", C_MUTED

    if price and bb_u and bb_l and (bb_u - bb_l) > 0:
        bb_pct = (price - bb_l) / (bb_u - bb_l) * 100
        if bb_pct > 90:   bb_lbl = f"Bande haute ({bb_pct:.0f}%)"
        elif bb_pct < 10: bb_lbl = f"Bande basse ({bb_pct:.0f}%)"
        else:             bb_lbl = f"Centre bande ({bb_pct:.0f}%)"
    else:
        bb_lbl = "—"

    kpis = html.Div([
        kpi_card("Prix EUR/USD", f"{price:.5f}" if price else "—",
                 subtitle=f"{chg:+.4f}%" if chg else "",
                 color=chg_col, tooltip="Dernier prix EUR/USD 5 minutes"),
        kpi_card("RSI (14)", f"{rsi_val:.1f}",
                 subtitle=rsi_lbl, color=rsi_col,
                 tooltip="0-30 = survente / 70-100 = surachat"),
        kpi_card("EMA 9 / 21", f"{ema9:.5f}/{ema21:.5f}" if ema9 and ema21 else "—",
                 subtitle=ema_lbl, color=ema_col,
                 tooltip="EMA9 > EMA21 = tendance haussière"),
        kpi_card("ATR 14", f"{atr*10000:.1f} pips" if atr else "—",
                 subtitle="Volatilite moyenne",
                 tooltip="1 pip = 0.0001. ATR mesure l'amplitude moyenne des mouvements."),
        kpi_card("Bollinger", bb_lbl, color=C_ACCENT,
                 tooltip="Position dans les bandes de Bollinger (0% = bas, 100% = haut)"),
    ], style={
        "display": "grid", "gridTemplateColumns": "repeat(5, 1fr)",
        "gap": "10px", "marginBottom": "14px",
    })

    # Sessions
    now_utc   = datetime.now(timezone.utc)
    total_min = now_utc.hour * 60 + now_utc.minute
    # Fenêtres exactes utilisées par intraday_session_breakout.py
    sessions  = [
        ("Londres",  7*60,      9*60+30,  "#38a169"),   # 07:00-09:30 UTC
        ("New York", 13*60+30,  16*60,    "#3182ce"),   # 13:30-16:00 UTC
        ("Asie",     22*60,     8*60,     "#805ad5"),   # 22:00-08:00 UTC (chevauche minuit)
    ]
    sess_items = []
    for name, start, end, col in sessions:
        if start < end:
            active = start <= total_min < end
        else:  # session traverse minuit (ex: Asie 22h-08h)
            active = total_min >= start or total_min < end
        sess_items.append(html.Div([
            html.Div("●", style={"color": col if active else "#cbd5e0", "fontSize": "20px"}),
            html.Div(name, style={"fontWeight": "700", "fontSize": "13px",
                                   "color": C_TEXT if active else C_MUTED}),
            html.Div("ACTIVE" if active else "fermée",
                     style={"fontSize": "10px", "color": col if active else C_MUTED, "fontWeight": "800"}),
        ], style={
            "textAlign": "center", "padding": "10px 16px",
            "background": f"{col}12" if active else C_PANEL,
            "border": f"1px solid {col}50" if active else f"1px solid {C_BORDER}",
            "borderRadius": "8px",
        }))
    sessions_panel = card_wrap([
        html.Div("Sessions de trading", style={"fontWeight": "800", "marginBottom": "10px", "fontSize": "13px"}),
        html.Div(sess_items, style={"display": "flex", "gap": "10px"}),
        html.Div(f"UTC actuel : {now_utc.strftime('%H:%M')}  —  Chevauchement London+NY : 13h30-16h00 UTC",
                 style={"fontSize": "12px", "color": C_MUTED, "marginTop": "8px"}),
    ])

    chart_el = card_wrap(
        dcc.Graph(figure=eurusd_chart(ohlcv), config={"displayModeBar": False, "scrollZoom": True}),
        padding="0",
    )
    gauge_el = card_wrap(
        dcc.Graph(figure=rsi_gauge(rsi_val), config={"displayModeBar": False}),
        padding="0",
    )

    return html.Div([
        kpis, sessions_panel,
        grid2(chart_el, gauge_el, left_width="1fr", right_width="210px"),
    ])


# ── Page : Stratégies ─────────────────────────────────────────────────────────

def page_strategies(state: dict) -> html.Div:
    sigs = state.get("recent_signals", [])
    last_sig: dict[str, dict] = {}
    for s in reversed(sigs):
        strat = s.get("strategy", "")
        if strat not in last_sig:
            last_sig[strat] = s

    regime    = state.get("market", {}).get("regime", "unknown")
    regime_fr = state.get("market", {}).get("regime_fr", "Inconnu")

    regime_map = {
        "bull_trend":        ["Croisement EMA", "Cassure de Session"],
        "bear_trend":        ["Croisement EMA", "Cassure de Session"],
        "range":             ["Bollinger + RSI", "Croisement EMA"],
        "high_volatility":   ["Croisement EMA"],
        "low_volatility":    ["Bollinger + RSI", "Croisement EMA"],
        "panic":             ["Aucune — pause"],
        "compression":       ["Bollinger + RSI", "Cassure de Session"],
        "breakout_expansion":["Cassure de Session", "Croisement EMA"],
    }
    active_strats = regime_map.get(regime, ["Selon régime"])

    regime_info = card_wrap([
        html.Div("Strategies actives maintenant", style={"fontWeight": "800", "fontSize": "13px", "marginBottom": "8px"}),
        html.Div([
            html.Span(f"{REGIME_ICON.get(regime,'?')} {regime_fr}  → ", style={"color": C_MUTED}),
            *[badge(s, C_ACCENT) for s in active_strats],
        ], style={"display": "flex", "alignItems": "center", "gap": "6px", "flexWrap": "wrap"}),
    ])

    cards = []
    for strat_id, info in STRATEGY_INFO.items():
        last     = last_sig.get(strat_id, {})
        action   = last.get("action", "")
        last_t   = str(last.get("time", "—"))[:16]
        reason   = str(last.get("reason", ""))[:100]
        conf     = float(last.get("confidence", 0) or 0)
        is_exec  = action == "EXECUTE"
        is_block = action == "BLOCK"
        st_color = C_GREEN if is_exec else (C_ORANGE if is_block else C_MUTED)
        st_text  = "Ordre execute" if is_exec else ("Signal bloque" if is_block else "En surveillance")

        cards.append(html.Div([
            html.Div([
                html.Span(info["icon"], style={"fontSize": "22px", "color": info["color"]}),
                html.Div([
                    html.Div(info["name"], style={"fontWeight": "800", "fontSize": "15px"}),
                    html.Div(f"Actif en : {info['when']}", style={"fontSize": "11px", "color": C_MUTED}),
                ]),
                html.Div(st_text, style={
                    "marginLeft": "auto", "fontSize": "11px", "fontWeight": "800",
                    "color": st_color, "background": f"{st_color}15",
                    "borderRadius": "999px", "padding": "3px 10px",
                }),
            ], style={"display": "flex", "alignItems": "center", "gap": "12px",
                      "marginBottom": "10px", "paddingBottom": "10px",
                      "borderBottom": f"1px solid {C_BORDER}"}),

            html.Div(info["desc"], style={"fontSize": "13px", "color": C_MUTED,
                                           "lineHeight": "1.65", "marginBottom": "10px"}),

            html.Div([
                html.Div([
                    html.Span("Dernier signal : ", style={"fontWeight": "700", "fontSize": "12px"}),
                    signal_badge_el(action or "—"),
                    html.Span(f"  {last_t}", style={"color": C_MUTED, "fontSize": "12px"}),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
                html.Div(reason, style={
                    "fontSize": "12px", "color": C_MUTED, "marginTop": "4px",
                    "fontFamily": "monospace",
                }) if reason else None,
                html.Div(f"Confiance : {conf*100:.0f}%", style={
                    "fontSize": "12px", "color": C_ACCENT, "marginTop": "4px", "fontWeight": "700",
                }) if conf > 0 else None,
            ], style={
                "background": "#f7fafc", "borderRadius": "8px", "padding": "10px 12px",
            }),
        ], style={
            "background": C_PANEL, "border": f"1px solid {C_BORDER}",
            "borderLeft": f"4px solid {info['color']}",
            "borderRadius": "10px", "padding": "14px 16px",
        }))

    return html.Div([
        regime_info,
        html.Div(cards, style={"display": "flex", "flexDirection": "column", "gap": "12px"}),
    ])


# ── Page : Portefeuille ───────────────────────────────────────────────────────

def _safe_pct(val: float, base: float) -> str:
    try:
        return f"{val / base * 100:+.3f}%" if base else "—"
    except Exception:
        return "—"


def page_portfolio(state: dict, alpaca: dict) -> html.Div:
    port      = state.get("portfolio", {})
    hist      = state.get("equity_history", [])
    pos_list  = state.get("positions", [])
    trades    = state.get("recent_trades", [])

    alp_initial = float(alpaca.get("initial_capital", 0))
    alp_hist    = alpaca.get("equity_history", [])
    alp_targets = alpaca.get("targets", {})
    alp_closed  = alpaca.get("closed_events", [])
    alp_signals = alpaca.get("signals_log", [])
    alp_equity  = float(alp_hist[-1]["equity"]) if alp_hist else alp_initial
    alp_realized= sum(float(x.get("pnl", 0)) for x in alp_closed)
    alp_open_pnl= alp_equity - alp_initial - alp_realized
    has_alpaca  = alp_initial > 0

    sections = []

    # ─── SECTION ALPACA ────────────────────────────────────────────────────
    if has_alpaca:
        sections.append(section_header("Compte Alpaca Paper", C_ALPACA))

        alp_kpis = html.Div([
            kpi_card("Capital Alpaca", f"${alp_equity:,.2f}",
                     subtitle=f"Depart ${alp_initial:,.0f}",
                     color=C_GREEN if alp_equity >= alp_initial else C_RED,
                     tooltip="Estimation: capital initial + P&L réalisé + P&L ouvert"),
            kpi_card("P&L realise", f"${alp_realized:+,.2f}",
                     color=C_GREEN if alp_realized >= 0 else C_RED,
                     tooltip="Somme des P&L des trades clôturés"),
            kpi_card("P&L ouvert (estim.)", f"${alp_open_pnl:+,.2f}",
                     color=C_GREEN if alp_open_pnl >= 0 else C_RED,
                     tooltip="Estimation basée sur l'equity trackée"),
            kpi_card("Return total", _safe_pct(alp_equity - alp_initial, alp_initial),
                     color=C_GREEN if alp_equity >= alp_initial else C_RED),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)",
                  "gap": "10px", "marginBottom": "14px"})

        alp_eq_fig = equity_chart(alp_hist, alp_initial, "Courbe equite Alpaca", C_ALPACA)
        alp_eq_panel = card_wrap(
            dcc.Graph(figure=alp_eq_fig, config={"displayModeBar": True}),
            border_color=f"{C_ALPACA}30", padding="0",
        )

        # Positions Alpaca (targets dict)
        alp_pos_rows = []
        for sym, tgt in alp_targets.items():
            ep    = float(tgt.get("entry_price", 0))
            qty   = float(tgt.get("quantity", 0))
            strat = tgt.get("strategy", "—")
            sl    = float(tgt.get("stop_loss", 0))
            tp    = float(tgt.get("take_profit", 0))
            et    = str(tgt.get("entry_time", "—"))[:16]
            alp_pos_rows.append({
                "symbol": sym, "strategie": strat, "quantite": f"{qty:.4f}",
                "entree": f"${ep:.2f}", "stop": f"${sl:.2f}" if sl else "—",
                "objectif": f"${tp:.2f}" if tp else "—", "date": et,
            })

        alp_pos_panel = html.Div([
            html.Div(f"Positions Alpaca ouvertes ({len(alp_targets)})", style={
                "padding": "10px 16px", "fontWeight": "800", "fontSize": "13px",
                "borderBottom": f"1px solid {C_BORDER}",
            }),
            dash_table.DataTable(
                columns=[
                    {"name": "Titre",     "id": "symbol"},
                    {"name": "Stratégie", "id": "strategie"},
                    {"name": "Qté",       "id": "quantite"},
                    {"name": "Entrée",    "id": "entree"},
                    {"name": "Stop",      "id": "stop"},
                    {"name": "Objectif",  "id": "objectif"},
                    {"name": "Date entrée","id": "date"},
                ],
                data=alp_pos_rows,
                page_size=20,
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#f3f0ff", "fontWeight": "700",
                               "fontSize": "11px", "border": f"1px solid {C_ALPACA}20"},
                style_cell={"fontSize": "12px", "padding": "7px 10px",
                             "fontFamily": "Inter, sans-serif"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#faf5ff"}],
            ),
        ], style={"background": C_PANEL, "border": f"1px solid {C_ALPACA}30",
                  "borderRadius": "10px", "overflow": "hidden", "marginBottom": "14px"})

        # Trades Alpaca clôturés
        alp_trade_rows = []
        for t in reversed(alp_closed):
            pnl_v = float(t.get("pnl", 0) or 0)
            ep    = float(t.get("entry_price", 1) or 1)
            qty   = float(t.get("quantity", 1) or 1)
            alp_trade_rows.append({
                "ticker":    t.get("ticker", "—"),
                "strategie": t.get("strategy", "—"),
                "entree":    f"${float(t.get('entry_price', 0)):.2f}",
                "sortie":    f"${float(t.get('exit_price', 0)):.2f}",
                "raison":    t.get("exit_reason", "—"),
                "pnl":       f"${pnl_v:+,.2f}",
                "pct":       _safe_pct(pnl_v, ep * qty),
                "date":      str(t.get("exit_time", "—"))[:16],
            })

        alp_trades_panel = html.Div([
            html.Div(f"Historique trades Alpaca ({len(alp_closed)})", style={
                "padding": "10px 16px", "fontWeight": "800", "fontSize": "13px",
                "borderBottom": f"1px solid {C_BORDER}",
            }),
            dash_table.DataTable(
                columns=[
                    {"name": "Titre",     "id": "ticker"},
                    {"name": "Stratégie", "id": "strategie"},
                    {"name": "Entrée",    "id": "entree"},
                    {"name": "Sortie",    "id": "sortie"},
                    {"name": "Raison",    "id": "raison"},
                    {"name": "P&L $",     "id": "pnl"},
                    {"name": "P&L %",     "id": "pct"},
                    {"name": "Date",      "id": "date"},
                ],
                data=alp_trade_rows,
                page_size=15,
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#f3f0ff", "fontWeight": "700",
                               "fontSize": "11px", "border": f"1px solid {C_ALPACA}20"},
                style_cell={"fontSize": "12px", "padding": "7px 10px",
                             "fontFamily": "Inter, sans-serif"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#faf5ff"}],
            ),
        ], style={"background": C_PANEL, "border": f"1px solid {C_ALPACA}30",
                  "borderRadius": "10px", "overflow": "hidden", "marginBottom": "14px"})

        sections += [alp_kpis, alp_eq_panel, alp_pos_panel, alp_trades_panel]

    # ─── SECTION EUR/USD BOT ───────────────────────────────────────────────
    sections.append(section_header("Bot EUR/USD Intraday", C_ACCENT))

    initial = float(port.get("initial_capital", 10_000))
    equity  = float(port.get("total_equity", initial))
    pnl_col = C_GREEN if port.get("total_pnl", 0) >= 0 else C_RED

    eur_kpis = html.Div([
        kpi_card("Capital total", f"${equity:,.2f}",
                 color=C_GREEN if equity >= initial else C_RED,
                 tooltip="Cash + positions EUR/USD ouvertes"),
        kpi_card("Cash dispo", f"${float(port.get('cash', 0)):,.2f}"),
        kpi_card("P&L non realise", f"${float(port.get('open_pnl', 0)):+,.2f}",
                 color=C_GREEN if float(port.get('open_pnl', 0)) >= 0 else C_RED),
        kpi_card("P&L realise", f"${float(port.get('realized_pnl', 0)):+,.2f}",
                 color=C_GREEN if float(port.get('realized_pnl', 0)) >= 0 else C_RED),
        kpi_card("Return", f"{float(port.get('return_pct', 0)):+.2f}%",
                 color=pnl_col),
        kpi_card("Drawdown max", f"{float(port.get('drawdown_pct', 0))*100:.2f}%",
                 color=C_RED if float(port.get('drawdown_pct', 0)) < -0.05 else C_MUTED,
                 tooltip="Perte maximale depuis le plus haut atteint"),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
              "gap": "10px", "marginBottom": "14px"})

    eur_eq_fig = equity_chart(hist, initial, "Courbe equite EUR/USD Bot", C_ACCENT)
    eur_eq_panel = card_wrap(
        dcc.Graph(figure=eur_eq_fig, config={"displayModeBar": True}),
        padding="0",
    )

    # Positions EUR/USD
    pos_data = []
    for p in (pos_list or []):
        pnl_pct_raw = float(p.get("unrealized_pnl_pct", 0) or 0)
        pos_data.append({
            "asset":    p.get("asset", "—"),
            "side":     p.get("side", "—"),
            "qty":      str(p.get("quantity", "—")),
            "entree":   str(p.get("avg_entry", "—")),
            "prix":     str(p.get("current_price", "—")),
            "valeur":   str(p.get("market_value", "—")),
            "pnl":      str(p.get("unrealized_pnl", "—")),
            "pnl_pct":  f"{pnl_pct_raw*100:+.3f}%",
            "stop":     str(p.get("stop_loss", "—")),
            "objectif": str(p.get("take_profit", "—")),
            "strat":    str(p.get("strategy", "—")),
        })

    eur_pos_panel = html.Div([
        html.Div(f"Positions EUR/USD ouvertes ({len(pos_list or [])})", style={
            "padding": "10px 16px", "fontWeight": "800", "fontSize": "13px",
            "borderBottom": f"1px solid {C_BORDER}",
        }),
        dash_table.DataTable(
            columns=[
                {"name": "Actif",     "id": "asset"},
                {"name": "Cote",      "id": "side"},
                {"name": "Qte",       "id": "qty"},
                {"name": "Entree",    "id": "entree"},
                {"name": "Prix act.", "id": "prix"},
                {"name": "Valeur $",  "id": "valeur"},
                {"name": "P&L $",     "id": "pnl"},
                {"name": "P&L %",     "id": "pnl_pct"},
                {"name": "Stop",      "id": "stop"},
                {"name": "Objectif",  "id": "objectif"},
                {"name": "Strategie", "id": "strat"},
            ],
            data=pos_data,
            page_size=20,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#ebf8ff", "fontWeight": "700",
                           "fontSize": "11px", "border": f"1px solid {C_ACCENT}20"},
            style_cell={"fontSize": "12px", "padding": "7px 10px", "fontFamily": "Inter, sans-serif"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f7fbff"}],
        ),
    ], style={"background": C_PANEL, "border": f"1px solid {C_ACCENT}30",
              "borderRadius": "10px", "overflow": "hidden", "marginBottom": "14px"})

    # Trades EUR/USD
    trade_data = []
    for t in reversed(trades or []):
        ep   = float(t.get("entry_price") or 1)
        qty  = float(t.get("quantity") or 1)
        pnl_v= float(t.get("pnl", 0) or 0)
        trade_data.append({
            "asset":  t.get("asset", "—"),
            "side":   t.get("side", "—"),
            "qty":    str(t.get("quantity", "—")),
            "entree": str(t.get("entry_price", "—")),
            "sortie": str(t.get("exit_price", "—")),
            "raison": t.get("exit_reason", "—"),
            "pnl":    f"${pnl_v:+,.4f}",
            "pct":    _safe_pct(pnl_v, ep * qty),
            "date":   str(t.get("closed_at", "—"))[:16],
            "strat":  t.get("strategy", "—"),
        })

    eur_trades_panel = html.Div([
        html.Div(f"Historique trades EUR/USD ({len(trades or [])})", style={
            "padding": "10px 16px", "fontWeight": "800", "fontSize": "13px",
            "borderBottom": f"1px solid {C_BORDER}",
        }),
        dash_table.DataTable(
            columns=[
                {"name": "Actif",    "id": "asset"},
                {"name": "Cote",     "id": "side"},
                {"name": "Qte",      "id": "qty"},
                {"name": "Entree",   "id": "entree"},
                {"name": "Sortie",   "id": "sortie"},
                {"name": "Raison",   "id": "raison"},
                {"name": "P&L $",    "id": "pnl"},
                {"name": "P&L %",    "id": "pct"},
                {"name": "Date",     "id": "date"},
                {"name": "Strategie","id": "strat"},
            ],
            data=trade_data,
            page_size=15,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#ebf8ff", "fontWeight": "700",
                           "fontSize": "11px", "border": f"1px solid {C_ACCENT}20"},
            style_cell={"fontSize": "12px", "padding": "7px 10px", "fontFamily": "Inter, sans-serif"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f7fbff"}],
        ),
    ], style={"background": C_PANEL, "border": f"1px solid {C_ACCENT}30",
              "borderRadius": "10px", "overflow": "hidden"})

    sections += [eur_kpis, eur_eq_panel, eur_pos_panel, eur_trades_panel]
    return html.Div(sections)


# ── Page : Régime & IA ────────────────────────────────────────────────────────

def page_regime(state: dict) -> html.Div:
    mkt  = state.get("market", {})
    news = state.get("news", [])

    regime    = mkt.get("regime", "unknown")
    regime_fr = mkt.get("regime_fr", "Inconnu")
    reg_col   = REGIME_COLOR.get(regime, C_MUTED)
    reg_icon  = REGIME_ICON.get(regime, "?")

    regime_detail = {
        "bull_trend":        ("Tendance haussière soutenue. Les acheteurs dominent.",
                              "Le bot privilégie les achats (BUY).",
                              "Risque de retournement si surréchauffé."),
        "bear_trend":        ("Tendance baissière. Les vendeurs dominent.",
                              "Le bot peut vendre (SELL).",
                              "Risque de rebond technique."),
        "range":             ("Marché dans un couloir horizontal.",
                              "Stratégies de retour à la moyenne actives.",
                              "Faux signaux possible si cassure."),
        "high_volatility":   ("Forte volatilité — prix agités.",
                              "Le bot est plus prudent, stops plus larges.",
                              "Faux signaux fréquents."),
        "low_volatility":    ("Marché calme, faible amplitude.",
                              "Stratégies de range actives.",
                              "Rendements limités."),
        "panic":             ("Panique de marché — chute brutale.",
                              "Le bot suspend toutes les entrées.",
                              "RISQUE EXTREME."),
        "compression":       ("Marché qui se comprime avant une cassure.",
                              "Le bot attend la direction de la cassure.",
                              "Direction de la cassure imprévisible."),
        "breakout_expansion":("Cassure en cours avec forte impulsion.",
                              "Le bot suit la cassure.",
                              "Faux breakouts possibles."),
        "euphoric":          ("Marché suracheté, euphorie.",
                              "Le bot est défensif.",
                              "Risque de correction soudaine."),
    }
    expl, action, risk = regime_detail.get(regime, ("Régime inconnu.", "—", "—"))

    regime_card = card_wrap([
        html.Div([
            html.Span(reg_icon, style={"fontSize": "32px"}),
            html.Div([
                html.Div(regime_fr, style={"fontWeight": "900", "fontSize": "20px", "color": reg_col}),
                html.Div(f"Confiance : {mkt.get('confidence', 0)*100:.0f}%  |  Source : {mkt.get('source', '?')}",
                         style={"fontSize": "12px", "color": C_MUTED}),
            ]),
        ], style={"display": "flex", "alignItems": "center", "gap": "14px", "marginBottom": "14px"}),
        html.Div(expl, style={"fontSize": "14px", "lineHeight": "1.7", "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Div("Ce que fait le bot", style={"fontWeight": "700", "fontSize": "12px", "color": C_MUTED}),
                html.Div(action, style={"fontSize": "13px", "marginTop": "4px"}),
            ], style={"background": "#f0fff4", "borderRadius": "8px", "padding": "12px", "flex": "1"}),
            html.Div([
                html.Div("Risques", style={"fontWeight": "700", "fontSize": "12px", "color": C_MUTED}),
                html.Div(risk, style={"fontSize": "13px", "color": C_RED, "marginTop": "4px"}),
            ], style={"background": "#fff5f5", "borderRadius": "8px", "padding": "12px", "flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),
    ], border_color=reg_col)

    # LLM analysis
    analyst = mkt.get("analyst_summary", "")
    risks_list  = mkt.get("key_risks", [])
    opps_list   = mkt.get("opportunities", [])
    trend_map   = {"positive": "Haussier", "negative": "Baissier", "neutral": "Neutre"}
    trend_txt   = trend_map.get(mkt.get("trend", "neutral"), "Neutre")
    trend_col   = C_GREEN if trend_txt == "Haussier" else (C_RED if trend_txt == "Baissier" else C_MUTED)

    analysis_card = card_wrap([
        html.Div("Analyse IA — Groq / LLM", style={"fontWeight": "800", "fontSize": "13px",
                                                     "marginBottom": "10px", "color": C_PURPLE}),
        html.Div([
            html.Span("Tendance détectée : ", style={"color": C_MUTED, "fontSize": "13px"}),
            html.Span(trend_txt, style={"color": trend_col, "fontWeight": "800", "fontSize": "13px"}),
        ], style={"marginBottom": "8px"}),
        html.Div(analyst or "Analyse IA non disponible (se déclenche 1×/jour avec clé Anthropic/Groq).",
                 style={"fontSize": "13px", "lineHeight": "1.7", "color": C_TEXT, "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Div("Risques identifiés", style={"fontWeight": "700", "color": C_MUTED, "fontSize": "12px"}),
                html.Ul([html.Li(r, style={"fontSize": "13px", "marginTop": "3px"}) for r in (risks_list or ["—"])]),
            ], style={"background": "#fff5f5", "borderRadius": "8px", "padding": "12px", "flex": "1"}),
            html.Div([
                html.Div("Opportunités", style={"fontWeight": "700", "color": C_MUTED, "fontSize": "12px"}),
                html.Ul([html.Li(o, style={"fontSize": "13px", "marginTop": "3px"}) for o in (opps_list or ["—"])]),
            ], style={"background": "#f0fff4", "borderRadius": "8px", "padding": "12px", "flex": "1"}),
        ], style={"display": "flex", "gap": "12px"}),
    ])

    # News
    impact_icons = {"strongly_positive": "↑↑", "positive": "↑", "slightly_positive": "↗",
                    "neutral": "→", "slightly_negative": "↘", "negative": "↓",
                    "strongly_negative": "↓↓", "high_risk": "⚠", "low_risk": "✓"}
    news_rows = []
    for art in (news or [])[:15]:
        sentiment = float(art.get("sentiment", 0) or 0)
        impact    = art.get("impact", "neutral")
        icon      = impact_icons.get(impact, "→")
        sc        = C_GREEN if sentiment > 0.1 else (C_RED if sentiment < -0.1 else C_MUTED)
        topics    = art.get("topics", [])
        news_rows.append(html.Div([
            html.Span(icon, style={"fontSize": "16px", "color": sc, "minWidth": "24px"}),
            html.Div([
                html.Div(art.get("asset", "Général"),
                         style={"fontSize": "11px", "fontWeight": "700", "color": C_ACCENT}),
                html.Div(", ".join(topics[:3]) if topics else "—",
                         style={"fontSize": "12px"}),
            ], style={"flex": "1"}),
            html.Span(f"risque {float(art.get('risk_score', 0)):.0%}",
                      style={"fontSize": "11px",
                             "color": C_ORANGE if float(art.get("risk_score", 0) or 0) > 0.5 else C_MUTED}),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px",
                  "padding": "6px 0", "borderBottom": f"1px solid {C_BORDER}"}))

    news_card = html.Div([
        html.Div("Actualites et impact", style={"padding": "10px 16px", "fontWeight": "800",
                                                  "fontSize": "13px",
                                                  "borderBottom": f"1px solid {C_BORDER}"}),
        html.Div(
            news_rows or [html.Div("Aucune actualité.", style={"padding": "16px", "color": C_MUTED})],
            style={"padding": "6px 16px"},
        ),
    ], style={"background": C_PANEL, "border": f"1px solid {C_BORDER}",
              "borderRadius": "10px"})

    return html.Div([regime_card, analysis_card, news_card])


# ── Rendu d'un onglet ─────────────────────────────────────────────────────────

def render_tab(tab: str, state: dict, alpaca: dict) -> html.Div:
    try:
        if tab == "overview":
            return page_overview(state, alpaca)
        if tab == "eurusd":
            return page_eurusd(state)
        if tab == "strategies":
            return page_strategies(state)
        if tab == "portfolio":
            return page_portfolio(state, alpaca)
        if tab == "regime":
            return page_regime(state)
    except Exception as exc:
        return html.Div([
            html.Div("Erreur lors du rendu de la page", style={
                "fontWeight": "800", "color": C_RED, "marginBottom": "8px",
            }),
            html.Pre(str(exc), style={
                "background": "#fff5f5", "padding": "12px",
                "borderRadius": "8px", "fontSize": "12px", "color": C_RED,
            }),
        ])
    return html.Div("Page inconnue")


# ── Application Dash ──────────────────────────────────────────────────────────

def create_app() -> Dash:
    app = Dash(__name__, title="Trading Bot Dashboard",
               suppress_callback_exceptions=True)

    app.index_string = """<!DOCTYPE html>
<html lang="fr">
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:#f0f4f8;color:#1a202c;font-family:'Inter',system-ui,sans-serif;font-size:14px}
.topbar{
  position:sticky;top:0;z-index:100;
  background:#1a202c;color:#fff;
  padding:0 20px;height:50px;
  display:flex;align-items:center;gap:14px;
  box-shadow:0 2px 6px rgba(0,0,0,.3);
}
.brand{font-size:15px;font-weight:900;color:#fff;letter-spacing:-.3px}
.brand span{color:#63b3ed}
.tb-badge{border-radius:999px;font-size:11px;font-weight:800;padding:3px 10px}
.badge-live   {background:#276749;color:#68d391}
.badge-delayed{background:#744210;color:#f6ad55}
.badge-offline{background:#742a2a;color:#fc8181}
.tb-price {font-size:13px;color:#90cdf4;font-weight:700}
.tb-regime{font-size:12px;color:#a0aec0}
.tb-time  {margin-left:auto;font-size:11px;color:#718096}
.nav{
  background:#fff;border-bottom:2px solid #e2e8f0;
  padding:0 20px;display:flex;gap:2px;
  position:sticky;top:50px;z-index:90;
}
.tab-btn{
  padding:11px 16px;font-size:13px;font-weight:700;cursor:pointer;
  border:none;background:transparent;color:#718096;
  border-bottom:3px solid transparent;margin-bottom:-2px;
  transition:color .12s,border-color .12s;
  white-space:nowrap;
}
.tab-btn:hover{color:#3182ce}
.tab-btn.active{color:#3182ce;border-bottom-color:#3182ce}
.content{padding:16px 20px 48px;max-width:1420px;margin:0 auto}
@media(max-width:960px){
  .nav{overflow-x:auto}
  .tab-btn{padding:9px 12px;font-size:12px}
}
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""

    app.layout = html.Div([
        dcc.Interval(id="interval", interval=REFRESH_MS, n_intervals=0),
        dcc.Store(id="active-tab", data="overview"),

        # Topbar
        html.Div(className="topbar", children=[
            html.Div([html.Span("Trading"), html.Span(" Bot")], className="brand"),
            html.Div("OFFLINE", id="tb-badge", className="tb-badge badge-offline"),
            html.Div("EUR/USD —", id="tb-price", className="tb-price"),
            html.Div("—", id="tb-regime", className="tb-regime"),
            html.Div("—", id="tb-time", className="tb-time"),
        ]),

        # Navigation tabs
        html.Div([
            html.Button(label, id=f"tab-{t_id}", className="tab-btn" + (" active" if t_id == "overview" else ""))
            for t_id, label in TABS
        ], className="nav"),

        # Page content
        html.Div(id="content", className="content"),
    ])

    # ── Callback principal : navigation + rendu + topbar ──────────────────
    @app.callback(
        Output("content",    "children"),
        Output("active-tab", "data"),
        Output("tb-badge",   "children"),
        Output("tb-badge",   "className"),
        Output("tb-price",   "children"),
        Output("tb-regime",  "children"),
        Output("tb-time",    "children"),
        [Input(f"tab-{t_id}", "n_clicks") for t_id, _ in TABS],
        Input("interval", "n_intervals"),
        State("active-tab", "data"),
        prevent_initial_call=False,
    )
    def main_update(*args):
        n_tabs   = len(TABS)
        # args[0..n_tabs-1] = n_clicks des boutons
        # args[n_tabs]      = n_intervals
        # args[n_tabs+1]    = active-tab (State)
        current_tab = args[n_tabs + 1] or "overview"

        ctx = callback_context
        triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""

        if triggered.startswith("tab-"):
            active_tab = triggered[4:]   # "tab-portfolio" → "portfolio"
        else:
            active_tab = current_tab

        state  = read_bot_state()
        alpaca = read_alpaca_state()
        fresh  = bot_freshness(state)

        mkt   = state.get("market", {})
        eur   = state.get("eurusd", {})
        price = float(eur.get("price", 0) or 0)
        chg   = float(eur.get("change_pct", 0) or 0)

        badge_cls  = f"tb-badge badge-{fresh}"
        badge_txt  = {"live": "LIVE", "delayed": "DELAI", "offline": "OFFLINE"}[fresh]
        price_txt  = f"EUR/USD  {price:.5f}  {chg:+.4f}%" if price else "EUR/USD —"
        regime_txt = mkt.get("regime_fr", "—")
        time_txt   = (f"MàJ : {state.get('last_update', '—')} | "
                      f"Cycle #{state.get('cycle_count', 0)}")

        page = render_tab(active_tab, state, alpaca)
        return page, active_tab, badge_txt, badge_cls, price_txt, regime_txt, time_txt

    # ── Callback styles des onglets ────────────────────────────────────────
    @app.callback(
        [Output(f"tab-{t_id}", "className") for t_id, _ in TABS],
        Input("active-tab", "data"),
    )
    def update_tab_styles(active: str) -> list[str]:
        active = active or "overview"
        return ["tab-btn active" if t_id == active else "tab-btn"
                for t_id, _ in TABS]

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bot trading dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8051)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Dashboard -> http://{args.host}:{args.port}")
    print("Ctrl+C pour arrêter.")
    Path("data/dashboard").mkdir(parents=True, exist_ok=True)
    create_app().run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
