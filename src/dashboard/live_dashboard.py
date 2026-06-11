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
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import feedparser as _feedparser
    _FEEDPARSER_OK = True
except ImportError:
    _FEEDPARSER_OK = False

try:
    from groq import Groq as _GroqClient
    _GROQ_OK = True
except ImportError:
    _GROQ_OK = False

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import (
    Dash, Input, Output, State,
    callback_context, dash_table, dcc, html, no_update,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

BOT_STATE_FILE    = PROJECT_ROOT / "data" / "dashboard" / "bot_state.json"
ALPACA_STATE_FILE = PROJECT_ROOT / "data" / "paper_trading" / "alpaca_state.json"
ASSETS_FILE       = PROJECT_ROOT / "config" / "assets.yaml"
REFRESH_MS = 15_000

# ── Palettes ──────────────────────────────────────────────────────────────────
C_BG     = "#F5F4F0"   # Washi — warm paper off-white
C_PANEL  = "#FEFEFE"   # Near-white cards
C_BORDER = "#E8E6E1"   # Subtle divider
C_TEXT   = "#1A1916"   # Sumi ink (warm black)
C_MUTED  = "#8C8A84"   # Stone grey
C_ACCENT = "#375F7A"   # Deep water blue
C_GREEN  = "#357A52"   # Forest (matcha-iro)
C_RED    = "#A83C3C"   # Lacquer red
C_ORANGE = "#A8712A"   # Amber (kohaku)
C_YELLOW = "#8E7A32"   # Gold (kin)
C_PURPLE = "#5E5496"   # Indigo (ai-iro)
C_TEAL   = "#317070"   # Sea (seiheki)
C_ALPACA = "#5E5496"   # Indigo

REGIME_COLOR = {
    "bull_trend": C_GREEN, "bear_trend": C_RED, "range": C_YELLOW,
    "high_volatility": C_ORANGE, "low_volatility": C_TEAL, "panic": "#6A2A2A",
    "euphoric": "#7A3A60", "compression": C_PURPLE,
    "breakout_expansion": C_ACCENT, "unknown": C_MUTED,
}
REGIME_ICON = {
    "bull_trend": "↑", "bear_trend": "↓", "range": "↔",
    "high_volatility": "⚡", "low_volatility": "〜", "panic": "⚠",
    "euphoric": "★", "compression": "⏸", "breakout_expansion": "▶", "unknown": "?",
}
RISK_COLOR = {"low": C_GREEN, "medium": C_YELLOW, "high": C_ORANGE, "extreme": C_RED}

STRATEGY_INFO = {
    # ── Swing / Actions ─────────────────────────────────────────────────────
    "trend_following": {
        "name": "Tendance (EMA)", "icon": "↗", "color": "#357A52",
        "desc": "Achète sur une tendance haussière confirmée : EMA20 > EMA50 > EMA200 + ADX ≥ 15. Stop/TP basés sur l'ATR.",
        "when": "Bull trend / Breakout / Unknown",
    },
    "breakout": {
        "name": "Cassure N-jours", "icon": "▲", "color": "#375F7A",
        "desc": "Entre sur une cassure du plus haut ou plus bas sur 20 jours, alignée avec l'EMA200. Capture les grandes tendances.",
        "when": "Bull trend / Breakout expansion",
    },
    "rsi_dip_buyer": {
        "name": "Dip RSI(2)", "icon": "↩", "color": "#5E5496",
        "desc": "Achète quand le RSI(2) passe sous 20 (survente court terme) au-dessus de la SMA200. Pari sur un rebond rapide.",
        "when": "Range / Bull trend / Unknown",
    },
    "thematic_momentum": {
        "name": "Momentum Sectoriel", "icon": "🏭", "color": "#A8712A",
        "desc": "Analyse les news par secteur via Groq LLM → achète les actions des secteurs haussiers identifiés (pharma, IA, semi…).",
        "when": "Tous régimes (score secteur > 0.25)",
    },
    "ema_cross_swing": {
        "name": "EMA Cross Rapide", "icon": "⚡", "color": "#A83C3C",
        "desc": "Achete des que EMA9 passe au-dessus de EMA21 (daily). Tres reactif, se declenche sur n'importe quel changement de tendance court terme.",
        "when": "Tous regimes sauf panique",
    },
    "momentum_burst": {
        "name": "Momentum Burst", "icon": "🚀", "color": "#A8712A",
        "desc": "Achete les actions avec momentum positif sur 5 jours (prix > EMA20 + retour > 0.5%). Capture les hausses rapides.",
        "when": "Tous regimes sauf panique",
    },
    # ── Intraday Forex ───────────────────────────────────────────────────────
    "intraday_ema_cross": {
        "name": "Croisement EMA (Forex)", "icon": "✕", "color": "#375F7A",
        "desc": "Achète EUR/USD quand l'EMA 9 passe au-dessus de l'EMA 21 sur 5 min. Idéal en marché directionnel.",
        "when": "Tous régimes sauf panique (lun-ven)",
    },
    "intraday_bollinger_rsi": {
        "name": "Bollinger + RSI (Forex)", "icon": "〜", "color": "#5E5496",
        "desc": "Achète EUR/USD quand le prix touche la bande basse Bollinger ET RSI < 35. Retour à la moyenne. Idéal en range.",
        "when": "Range / Basse volatilité (lun-ven)",
    },
    "intraday_session_breakout": {
        "name": "Cassure de Session (Forex)", "icon": "▶", "color": "#357A52",
        "desc": "Entre dans le sens de la cassure du range pré-session. Londres 07h-09h30 UTC, New York 13h30-16h UTC.",
        "when": "Ouverture London / NY (lun-ven)",
    },
    "intraday_trend_scalp": {
        "name": "Trend Scalp Multi-assets", "icon": "TS", "color": "#317070",
        "desc": "Scalpe la tendance courte sur les instruments intraday geres par l'orchestrator : forex, crypto et or.",
        "when": "Forex / crypto / or, 5min",
    },
    "intraday_macd": {
        "name": "MACD Intraday", "icon": "M", "color": "#A8712A",
        "desc": "Confirme les impulsions intraday avec croisement MACD, momentum et filtres de confiance par asset.",
        "when": "Forex / crypto / or, 5min",
    },
}

TABS = [
    ("overview",   "Vue globale"),
    ("eurusd",     "Live Multi-assets"),
    ("strategies", "Strategies"),
    ("positions",  "Positions Live"),
    ("portfolio",  "Portefeuille"),
    ("regime",     "Regime et IA"),
    ("analyse",    "Analyse Groq"),
    ("themes",     "Themes & Secteurs"),
    ("metrics",    "Metriques & ML"),
]

_BROKER_STATE_FILE = PROJECT_ROOT / "data" / "paper_trading" / "broker_state.json"
_PERF_TRACKER_FILE = PROJECT_ROOT / "data" / "models" / "performance_tracker.json"

ASSET_DISPLAY_ORDER = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "BTC-USD", "ETH-USD", "GC=F"]
ASSET_DISPLAY_NAME = {
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD",
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "GC=F": "GC=F (Or)",
}

SYSTEM_UPDATE_CARDS = [
    ("Registre assets", "config/assets.yaml", "6 instruments tradables centralises avec levier, budget et strategies."),
    ("Servo central", "TradingOrchestrator", "Distribution du capital et garde-fou contre la surconcentration par asset."),
    ("RulesEngine", "min_volume_intraday=0", "Le filtre a ete corrige pour laisser passer la valeur zero explicitement."),
    ("Liquidite", "crypto/futures OK", "Le check volume ne bloque plus les instruments dont le volume differe des actions."),
    ("Stop-loss", "Prix exact du SL", "Les sorties SL n'appliquent plus le slippage qui creusait la perte."),
]

# Cache yfinance pour ne pas refetcher a chaque refresh
_eur_yf_cache: dict = {"ts": 0.0, "data": {}}
_live_asset_cache: dict[str, dict] = {}
_equity_chart_cache: dict = {"ts": 0.0, "data": {}}


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


def read_broker_state() -> dict:
    try:
        if _BROKER_STATE_FILE.exists():
            with open(_BROKER_STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def read_perf_tracker() -> list:
    try:
        if _PERF_TRACKER_FILE.exists():
            with open(_PERF_TRACKER_FILE, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def read_assets_config() -> dict:
    try:
        if ASSETS_FILE.exists():
            import yaml
            with open(ASSETS_FILE, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def managed_assets_config(enabled_only: bool = True) -> dict:
    raw = read_assets_config()
    assets = raw.get("assets", raw) if isinstance(raw, dict) else {}
    if not isinstance(assets, dict):
        return {}
    if enabled_only:
        return {k: v for k, v in assets.items() if isinstance(v, dict) and v.get("enabled", False)}
    return {k: v for k, v in assets.items() if isinstance(v, dict)}


_NEWS_FEEDS = [
    ("FXStreet",       "https://www.fxstreet.com/rss/news"),
    ("Investing.com",  "https://www.investing.com/rss/news_301.rss"),
]
_NEWS_KEYWORDS = [
    "eur", "usd", "euro", "dollar", "ecb", "fed", "bce", "rate", "taux",
    "inflation", "forex", "eurusd", "eur/usd", "interest",
]


def fetch_forex_news(max_items: int = 10) -> list[dict]:
    """Fetch recent EUR/USD-relevant news from free RSS feeds."""
    if not _FEEDPARSER_OK:
        return []
    articles: list[dict] = []
    for source, url in _NEWS_FEEDS:
        try:
            d = _feedparser.parse(url)
            for entry in d.entries[:25]:
                title = entry.get("title", "")
                if any(kw in title.lower() for kw in _NEWS_KEYWORDS):
                    articles.append({
                        "title": title,
                        "source": source,
                        "published": entry.get("published", entry.get("updated", "")),
                        "url": entry.get("link", ""),
                        "summary": (entry.get("summary", "") or "")[:200],
                    })
        except Exception:
            pass
    articles.sort(key=lambda x: x.get("published", ""), reverse=True)
    return articles[:max_items]


def call_groq_analysis(news_items: list[dict], eur_data: dict, market_data: dict) -> str:
    """Call Groq API: synthesise current EUR/USD technicals + news headlines."""
    if not _GROQ_OK:
        return "Module Groq non disponible (pip install groq)."
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "GROQ_API_KEY absente de l'environnement."

    price = float(eur_data.get("price", 0) or 0)
    rsi   = float(eur_data.get("rsi_14", 50) or 50)
    ema9  = float(eur_data.get("ema_9", 0) or 0)
    ema21 = float(eur_data.get("ema_21", 0) or 0)
    atr   = float(eur_data.get("atr_14", 0) or 0)
    bb_u  = float(eur_data.get("bb_upper", 0) or 0)
    bb_l  = float(eur_data.get("bb_lower", 0) or 0)
    regime = market_data.get("regime", "unknown")

    trend_txt = "haussière" if ema9 > ema21 else "baissière"
    news_text = "\n".join(
        f"- [{a['source']}] {a['title']}" for a in news_items
    ) or "Aucune news disponible."

    prompt = f"""Tu es un analyste forex expert EUR/USD. Analyse les données suivantes et donne une interprétation pratique pour un trader intraday (horizons 1–4h).

=== SITUATION TECHNIQUE (5 min) ===
Prix EUR/USD : {price:.5f}
EMA 9 / 21  : {ema9:.5f} / {ema21:.5f}  → tendance {trend_txt}
RSI(14)     : {rsi:.1f}
ATR(14)     : {atr*10000:.1f} pips
Bollinger   : haut {bb_u:.5f}  bas {bb_l:.5f}
Régime bot  : {regime}

=== NEWS RÉCENTES (EUR/USD) ===
{news_text}

Réponds en français, sois direct et concis :
1. Impact des news sur EUR/USD (haussier / baissier / neutre) — explique pourquoi en 1-2 phrases
2. Biais directionnel à court terme
3. Niveau-clé à surveiller (support ou résistance)
4. Ce que le trader doit faire maintenant (attendre, acheter sur repli, vendre, etc.)"""

    try:
        client = _GroqClient(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"Erreur Groq : {str(exc)[:200]}"


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


def asset_price_decimals(asset_id: str) -> int:
    if asset_id.endswith("=X"):
        return 5
    if asset_id in {"BTC-USD", "ETH-USD", "GC=F"}:
        return 2
    return 2


def _safe_round(value, digits: int):
    try:
        value = float(value)
        return round(value, digits) if value == value else None
    except Exception:
        return None


def live_snapshot_from_df(asset_id: str, df) -> dict:
    if df is None or len(df) < 25:
        return {}
    try:
        from src.features.indicators import ema as _ema, rsi as _rsi, atr as _atr

        dec = asset_price_decimals(asset_id)
        close = df["close"]
        e9 = _ema(close, 9)
        e21 = _ema(close, 21)
        r14 = _rsi(close, 14)
        a14 = _atr(df, 14)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std(ddof=0)
        bb_u = sma20 + 2.0 * std20
        bb_l = sma20 - 2.0 * std20

        cur = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else cur
        n = min(80, len(df))
        ohlcv = []
        for i in range(-n, 0):
            ohlcv.append({
                "t": str(df.index[i]),
                "o": _safe_round(df["open"].iloc[i], dec),
                "h": _safe_round(df["high"].iloc[i], dec),
                "l": _safe_round(df["low"].iloc[i], dec),
                "c": _safe_round(df["close"].iloc[i], dec),
                "e9": _safe_round(e9.iloc[i], dec),
                "e21": _safe_round(e21.iloc[i], dec),
                "bbu": _safe_round(bb_u.iloc[i], dec),
                "bbl": _safe_round(bb_l.iloc[i], dec),
            })

        return {
            "asset": asset_id,
            "price": _safe_round(cur, dec),
            "change_pct": round((cur / prev - 1) * 100, 4) if prev else 0,
            "ema_9": _safe_round(e9.iloc[-1], dec),
            "ema_21": _safe_round(e21.iloc[-1], dec),
            "rsi_14": _safe_round(r14.iloc[-1], 2),
            "atr_14": _safe_round(a14.iloc[-1], 6),
            "bb_upper": _safe_round(bb_u.iloc[-1], dec),
            "bb_middle": _safe_round(sma20.iloc[-1], dec),
            "bb_lower": _safe_round(bb_l.iloc[-1], dec),
            "ohlcv": ohlcv,
            "last_bar": str(df.index[-1]),
        }
    except Exception:
        return {}


def fetch_live_asset_data(asset_id: str, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    cached = _live_asset_cache.get(asset_id)
    if cached and time.time() - cached.get("ts", 0) < 60:
        return cached.get("data", {})
    try:
        import yfinance as yf
        from src.data.yfinance_helpers import normalize_yfinance_columns

        period = str(cfg.get("data_period", "5d"))
        interval = str(cfg.get("data_interval", "5m"))
        df = yf.download(asset_id, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
            return {}
        df = normalize_yfinance_columns(df)
        data = live_snapshot_from_df(asset_id, df)
        if data:
            _live_asset_cache[asset_id] = {"ts": time.time(), "data": data}
        return data
    except Exception:
        return {}


def get_live_assets_data(state: dict) -> dict[str, dict]:
    assets_cfg = managed_assets_config(enabled_only=True)
    live_assets = dict(state.get("live_assets", {}) or {})
    if state.get("eurusd") and "EURUSD=X" not in live_assets:
        live_assets["EURUSD=X"] = state["eurusd"]

    fresh = bot_freshness(state)
    for asset_id in ASSET_DISPLAY_ORDER:
        cfg = assets_cfg.get(asset_id)
        if not cfg:
            continue
        if fresh == "live" and live_assets.get(asset_id):
            continue
        fetched = fetch_live_asset_data(asset_id, cfg)
        if fetched:
            live_assets[asset_id] = fetched
    return live_assets


# ── Composants UI ─────────────────────────────────────────────────────────────

def tip(text: str, tooltip: str) -> html.Span:
    return html.Span(text, title=tooltip,
                     style={"borderBottom": f"1px dotted {C_MUTED}", "cursor": "help"})


def kpi_card(label: str, value: str, subtitle: str = "",
             color: str = C_TEXT, tooltip: str = "") -> html.Div:
    label_el = tip(label, tooltip) if tooltip else html.Span(label)
    children = [
        html.Div(label_el, style={
            "fontSize": "10px", "fontWeight": "600", "color": C_MUTED,
            "textTransform": "uppercase", "letterSpacing": "0.08em",
        }),
        html.Div(str(value), style={
            "fontSize": "21px", "fontWeight": "700", "color": color,
            "marginTop": "6px", "lineHeight": "1.2", "wordBreak": "break-word",
            "fontVariantNumeric": "tabular-nums",
        }),
    ]
    if subtitle:
        children.append(html.Div(str(subtitle),
                                 style={"fontSize": "11px", "color": C_MUTED, "marginTop": "4px",
                                        "letterSpacing": "0.02em"}))
    return html.Div(children, style={
        "background": C_PANEL,
        "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px", "padding": "16px 18px",
        "boxShadow": "0 1px 4px rgba(26,25,22,0.05)",
        "transition": "box-shadow 0.2s ease",
    })


def badge(text: str, color: str = C_ACCENT, bg: str = "") -> html.Span:
    bg_col = bg or f"{color}18"
    return html.Span(str(text), style={
        "background": bg_col, "color": color,
        "borderRadius": "4px", "fontSize": "10px", "fontWeight": "700",
        "padding": "3px 9px", "letterSpacing": "0.06em",
        "textTransform": "uppercase",
    })


def signal_badge_el(sig: str) -> html.Span:
    sig = str(sig).upper()
    colors = {
        "BUY": (C_GREEN, "#EEF5EE"), "SELL": (C_RED, "#F5EEEE"),
        "EXECUTE": (C_GREEN, "#EEF5EE"), "BLOCK": (C_ORANGE, "#F5F2EA"),
        "NO_TRADE": (C_MUTED, "#F4F3EF"), "HOLD": (C_YELLOW, "#F5F2E7"),
    }
    c, bg = colors.get(sig, (C_MUTED, "#F4F3EF"))
    return badge(sig, c, bg)


def section_header(title: str, color: str = C_ACCENT) -> html.Div:
    return html.Div(title, style={
        "fontSize": "11px", "fontWeight": "700", "color": color,
        "textTransform": "uppercase", "letterSpacing": "0.1em",
        "padding": "10px 0 8px",
        "borderBottom": f"1px solid {color}28",
        "marginBottom": "14px", "marginTop": "22px",
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
        margin={"l": 55, "r": 15, "t": 30, "b": 30},
        xaxis={"showgrid": False, "zeroline": False,
               "tickfont": {"size": 10, "color": C_MUTED},
               "linecolor": C_BORDER},
        yaxis={"gridcolor": "#ECEAE4", "zeroline": False,
               "tickfont": {"size": 10, "color": C_MUTED},
               "linecolor": C_BORDER},
    )


def card_wrap(children, border_color: str = C_BORDER, padding: str = "16px 18px") -> html.Div:
    return html.Div(children, style={
        "background": C_PANEL, "border": f"1px solid {border_color}",
        "borderRadius": "8px", "padding": padding,
        "boxShadow": "0 1px 4px rgba(26,25,22,0.05)", "marginBottom": "14px",
        "transition": "box-shadow 0.2s ease",
    })


def grid2(left, right, left_width: str = "1fr", right_width: str = "220px") -> html.Div:
    return html.Div([left, right], style={
        "display": "grid",
        "gridTemplateColumns": f"{left_width} {right_width}",
        "gap": "12px", "alignItems": "start",
    })


# ── Helpers dashboard multi-assets ────────────────────────────────────────────
def _pv(v, default=0.0):
    """Safe numeric value: None / NaN / Inf → default.
    dict.get(key, default) returns None when key exists with null value —
    use _pv(d.get(key), default) to handle both missing and null keys."""
    import math as _m
    try:
        f = float(v)
        return f if _m.isfinite(f) else default
    except Exception:
        return default


def fmt_money(value: float, decimals: int = 0) -> str:
    try:
        return f"${float(value):,.{decimals}f}"
    except Exception:
        return "$0"


def dashboard_capital_reference(state: dict) -> float:
    port = state.get("portfolio", {}) if isinstance(state, dict) else {}
    for key in ("available_cash", "total_capital", "total_equity", "initial_capital"):
        try:
            value = float(port.get(key, 0) or 0)
        except Exception:
            value = 0.0
        if value > 0:
            return value
    return 0.0


def asset_registry_rows(state: dict) -> list[dict]:
    assets = managed_assets_config(enabled_only=True)
    if not assets:
        return []

    ordered_ids = ASSET_DISPLAY_ORDER + sorted(
        aid for aid in assets.keys() if aid not in ASSET_DISPLAY_ORDER
    )
    capital_ref = dashboard_capital_reference(state)
    rows: list[dict] = []

    for asset_id in ordered_ids:
        cfg = assets.get(asset_id)
        if not cfg:
            continue
        leverage = float(cfg.get("leverage", 1) or 1)
        max_margin_pct = float(cfg.get("max_margin_pct", 0) or 0)
        margin_max = capital_ref * max_margin_pct
        notional_max = margin_max * leverage
        strategies = [str(s).replace("intraday_", "") for s in cfg.get("strategies", [])]
        rows.append({
            "asset": ASSET_DISPLAY_NAME.get(asset_id, cfg.get("name", asset_id)),
            "ticker": asset_id,
            "type": str(cfg.get("type", "-")),
            "levier": f"x{leverage:g}",
            "cap": f"{max_margin_pct * 100:.0f}%",
            "margin": fmt_money(margin_max),
            "notional": fmt_money(notional_max),
            "strategies": ", ".join(strategies),
        })
    return rows


def system_update_panel(state: dict, compact: bool = False) -> html.Div:
    rows = asset_registry_rows(state)
    capital_ref = dashboard_capital_reference(state)
    active_count = len(rows) or 6

    cards = [
        html.Div([
            html.Div(title, style={
                "fontSize": "11px", "fontWeight": "900", "color": C_MUTED,
                "textTransform": "uppercase", "letterSpacing": "0.06em",
            }),
            html.Div(value, style={
                "fontSize": "15px", "fontWeight": "900", "color": C_TEXT,
                "marginTop": "4px",
            }),
            html.Div(desc, style={
                "fontSize": "12px", "color": C_MUTED, "lineHeight": "1.45",
                "marginTop": "5px",
            }),
        ], style={
            "border": f"1px solid {C_BORDER}", "borderRadius": "8px",
            "padding": "10px 12px", "background": "#F4F3EF",
        })
        for title, value, desc in SYSTEM_UPDATE_CARDS
    ]

    return html.Div([
        html.Div([
            html.Div([
                html.Div("Mise a jour systeme", style={
                    "fontSize": "13px", "fontWeight": "900", "color": C_ACCENT,
                    "textTransform": "uppercase", "letterSpacing": "0.08em",
                }),
                html.Div(
                    f"Machinerie multi-assets en place : {active_count} assets tradables simultanement.",
                    style={"fontSize": "13px", "color": C_TEXT, "marginTop": "4px", "fontWeight": "700"},
                ),
            ]),
            badge("Redemarrage requis", C_ORANGE, "#F5F2EA"),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "gap": "12px", "alignItems": "start", "marginBottom": "12px"}),

        html.Div([
            html.Div("A activer au redemarrage", style={
                "fontWeight": "900", "fontSize": "12px", "color": C_ORANGE,
                "marginBottom": "4px",
            }),
            html.Div(
                "Les 2 dernieres positions perdantes viennent de l'ancien code SL encore en memoire. "
                "Apres redemarrage, le fix SL ferme au prix exact du stop et evite le slippage anormal.",
                style={"fontSize": "13px", "color": C_TEXT, "lineHeight": "1.55"},
            ),
        ], style={
            "background": "#F5F2EA", "border": f"1px solid {C_ORANGE}30",
            "borderRadius": "8px", "padding": "10px 12px", "marginBottom": "12px",
        }),

        html.Div([
            html.Span("Budget de reference : ", style={"fontWeight": "800", "color": C_MUTED}),
            html.Span(fmt_money(capital_ref, 2), style={"fontWeight": "900", "color": C_TEXT}),
            html.Span(" | SL max attendu apres fix : environ -$38 au lieu des sorties degradees.",
                      style={"color": C_MUTED}),
        ], style={"fontSize": "12px", "marginBottom": "12px"}) if capital_ref else None,

        html.Div(cards, style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
            "gap": "10px",
        }) if not compact else None,
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_ACCENT}30",
        "borderRadius": "10px", "padding": "14px 16px", "marginBottom": "14px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
    })


def asset_registry_panel(state: dict, compact: bool = False) -> html.Div:
    rows = asset_registry_rows(state)
    if not rows:
        return html.Div()

    columns = [
        {"name": "Asset", "id": "asset"},
        {"name": "Ticker", "id": "ticker"},
        {"name": "Type", "id": "type"},
        {"name": "Levier", "id": "levier"},
        {"name": "Cap marge", "id": "cap"},
        {"name": "Marge max", "id": "margin"},
        {"name": "Notional max", "id": "notional"},
    ]
    if not compact:
        columns.append({"name": "Strategies", "id": "strategies"})

    return html.Div([
        html.Div([
            html.Div("Registre assets tradables", style={
                "fontWeight": "900", "fontSize": "13px", "color": C_TEXT,
            }),
            html.Div(
                "Le capital est reparti par asset pour eviter la surconcentration.",
                style={"fontSize": "12px", "color": C_MUTED, "marginTop": "3px"},
            ),
        ], style={"padding": "12px 16px", "borderBottom": f"1px solid {C_BORDER}"}),
        dash_table.DataTable(
            columns=columns,
            data=rows,
            page_size=6,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#EDF2F6", "fontWeight": "800",
                          "fontSize": "11px", "border": f"1px solid {C_ACCENT}20"},
            style_cell={"fontSize": "12px", "padding": "7px 10px",
                        "fontFamily": "Inter, sans-serif", "whiteSpace": "normal",
                        "height": "auto"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#EFF4F8"},
                {"if": {"column_id": "notional"}, "fontWeight": "800", "color": C_ACCENT},
                {"if": {"column_id": "margin"}, "fontWeight": "800"},
            ],
        ),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_ACCENT}30",
        "borderRadius": "10px", "overflow": "hidden", "marginBottom": "14px",
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
    fig.add_hline(y=initial, line_dash="dash", line_color="#A0A09A", line_width=1,
                  annotation_text=f"Depart ${initial:,.0f}",
                  annotation_font_color=C_MUTED)
    fig.update_layout(title=title, height=250, showlegend=False, **_chart_base())
    fig.update_yaxes(tickprefix="$")
    return fig


def eurusd_chart(
    ohlcv: list[dict],
    trades: list[dict] | None = None,
    open_position: dict | None = None,
) -> go.Figure:
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
        fig.add_trace(go.Scatter(x=times, y=bbu, line={"color": "#375F7A", "width": 0.9, "dash": "dot"},
                                 showlegend=False, hovertemplate="%{y:.5f}<extra>BB+</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=times, y=bbl, line={"color": "#375F7A", "width": 0.9, "dash": "dot"},
                                 showlegend=False, hovertemplate="%{y:.5f}<extra>BB-</extra>"), row=1, col=1)

    fig.add_trace(go.Candlestick(
        x=times, open=opens, high=highs, low=lows, close=closes,
        increasing_line_color=C_GREEN, decreasing_line_color=C_RED,
        increasing_fillcolor=C_GREEN, decreasing_fillcolor=C_RED,
        name="EUR/USD",
    ), row=1, col=1)

    if any(v is not None for v in e9v):
        fig.add_trace(go.Scatter(x=times, y=e9v, line={"color": "#A83C3C", "width": 1.5},
                                 name="EMA 9", hovertemplate="%{y:.5f}<extra>EMA9</extra>"), row=1, col=1)
    if any(v is not None for v in e21v):
        fig.add_trace(go.Scatter(x=times, y=e21v, line={"color": "#375F7A", "width": 1.5},
                                 name="EMA 21", hovertemplate="%{y:.5f}<extra>EMA21</extra>"), row=1, col=1)

    # Plage de temps des bougies — ne tracer les marqueurs que dans cette fenêtre
    t_start = times[0][:19] if times else ""
    t_end   = times[-1][:19] if times else ""

    # ── Marqueurs des trades (seulement dans la fenêtre visible) ─────────────
    if trades:
        buy_t, buy_p, buy_lbl = [], [], []
        sell_t, sell_p, sell_lbl = [], [], []
        close_t, close_p, close_lbl = [], [], []
        for tr in trades:
            if tr.get("asset") != "EURUSD=X":
                continue
            opened = str(tr.get("opened_at") or tr.get("created_at", ""))[:19]
            closed = str(tr.get("closed_at", ""))[:19]
            entry  = float(tr.get("entry_price") or tr.get("avg_entry") or tr.get("fill_price") or 0)
            exit_p = float(tr.get("exit_price") or tr.get("close_price") or 0)
            pnl    = float(tr.get("pnl") or 0)
            side   = tr.get("side", "long")
            strat  = tr.get("strategy", "")
            # N'afficher que les trades dont l'heure est dans la plage du graphique
            if opened and t_start and opened < t_start:
                continue
            if entry > 0:
                if side in ("long", "BUY"):
                    buy_t.append(opened)
                    buy_p.append(entry)
                    buy_lbl.append(f"ACHAT @ {entry:.5f}<br>{strat}<br>PnL: {pnl:+.2f}$")
                else:
                    sell_t.append(opened)
                    sell_p.append(entry)
                    sell_lbl.append(f"VENTE @ {entry:.5f}<br>{strat}<br>PnL: {pnl:+.2f}$")
            if closed and exit_p > 0 and closed >= t_start:
                pnl_sym = "+" if pnl >= 0 else ""
                close_t.append(closed)
                close_p.append(exit_p)
                close_lbl.append(f"CLÔTURE @ {exit_p:.5f}<br>PnL: {pnl_sym}{pnl:.2f}$")

        if buy_t:
            fig.add_trace(go.Scatter(
                x=buy_t, y=buy_p, mode="markers",
                marker={"symbol": "triangle-up", "size": 14, "color": "#357A52",
                        "line": {"color": "#fff", "width": 1}},
                name="Achat", hovertext=buy_lbl, hoverinfo="text",
            ), row=1, col=1)
        if sell_t:
            fig.add_trace(go.Scatter(
                x=sell_t, y=sell_p, mode="markers",
                marker={"symbol": "triangle-down", "size": 14, "color": "#A83C3C",
                        "line": {"color": "#fff", "width": 1}},
                name="Vente", hovertext=sell_lbl, hoverinfo="text",
            ), row=1, col=1)
        if close_t:
            fig.add_trace(go.Scatter(
                x=close_t, y=close_p, mode="markers",
                marker={"symbol": "circle-open", "size": 10, "color": "#5E5496",
                        "line": {"color": "#5E5496", "width": 2}},
                name="Clôture", hovertext=close_lbl, hoverinfo="text",
            ), row=1, col=1)

    # ── Position ouverte : shape SL/TP + marqueur entrée ────────────────────
    if open_position:
        entry = float(open_position.get("avg_entry") or open_position.get("current_price") or 0)
        sl    = open_position.get("stop_loss")
        tp    = open_position.get("take_profit")
        side  = open_position.get("side", "long")
        opened = str(open_position.get("opened_at", ""))[:19]
        entry_color = "#357A52" if side == "long" else "#A83C3C"

        # Lignes horizontales via add_shape (plus stable que add_hline avec subplots)
        yref = "y"
        if entry:
            fig.add_shape(type="line", x0=t_start, x1=t_end, y0=entry, y1=entry,
                          line={"color": entry_color, "width": 1.5, "dash": "solid"},
                          xref="x", yref=yref)
            fig.add_annotation(x=t_end, y=entry, text=f"Entrée {entry:.5f}",
                               xref="x", yref=yref, showarrow=False,
                               font={"size": 9, "color": entry_color},
                               xanchor="right", yanchor="bottom")
        if sl:
            sl = float(sl)
            fig.add_shape(type="line", x0=t_start, x1=t_end, y0=sl, y1=sl,
                          line={"color": "#A83C3C", "width": 1, "dash": "dot"},
                          xref="x", yref=yref)
            fig.add_annotation(x=t_end, y=sl, text=f"SL {sl:.5f}",
                               xref="x", yref=yref, showarrow=False,
                               font={"size": 9, "color": "#A83C3C"},
                               xanchor="right", yanchor="top")
        if tp:
            tp = float(tp)
            fig.add_shape(type="line", x0=t_start, x1=t_end, y0=tp, y1=tp,
                          line={"color": "#357A52", "width": 1, "dash": "dot"},
                          xref="x", yref=yref)
            fig.add_annotation(x=t_end, y=tp, text=f"TP {tp:.5f}",
                               xref="x", yref=yref, showarrow=False,
                               font={"size": 9, "color": "#357A52"},
                               xanchor="right", yanchor="bottom")
        # Triangle d'entrée sur le graphique
        if opened and opened >= t_start and entry:
            marker_sym = "triangle-up" if side == "long" else "triangle-down"
            fig.add_trace(go.Scatter(
                x=[opened], y=[entry], mode="markers",
                marker={"symbol": marker_sym, "size": 16, "color": entry_color,
                        "line": {"color": "#fff", "width": 2}},
                name="Position ouverte",
                hovertext=[f"OUVERT {'Long' if side=='long' else 'Short'} @ {entry:.5f}"],
                hoverinfo="text",
            ), row=1, col=1)

    # RSI sous-graphique
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
    fig.update_yaxes(gridcolor="#ECEAE4", row=1, col=1, tickformat=".5f")
    fig.update_yaxes(gridcolor="#ECEAE4", row=2, col=1, range=[0, 100])
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
                {"range": [0, 30], "color": "#EEF5EE"},
                {"range": [30, 70], "color": "#EDF2F6"},
                {"range": [70, 100], "color": "#F5EEEE"},
            ],
        },
    ))
    fig.update_layout(height=180, margin={"l": 20, "r": 20, "t": 25, "b": 5},
                      paper_bgcolor=C_PANEL)
    return fig


def format_live_price(asset_id: str, value) -> str:
    try:
        dec = asset_price_decimals(asset_id)
        return f"{float(value):,.{dec}f}"
    except Exception:
        return "-"


def format_live_atr(asset_id: str, value) -> str:
    try:
        val = float(value or 0)
        if asset_id.endswith("=X"):
            return f"{val * 10000:.1f} pips"
        return f"{val:,.2f}"
    except Exception:
        return "-"


def asset_sparkline(asset_id: str, data: dict) -> go.Figure:
    ohlcv = (data or {}).get("ohlcv", [])[-48:]
    fig = go.Figure()
    if not ohlcv:
        fig.add_annotation(text="Pas de donnees", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font={"size": 11, "color": C_MUTED})
    else:
        x_vals = [r.get("t", "") for r in ohlcv]
        y_vals = [r.get("c") for r in ohlcv]
        chg = float((data or {}).get("change_pct", 0) or 0)
        color = C_GREEN if chg >= 0 else C_RED
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="lines",
            line={"color": color, "width": 2},
            hovertemplate="%{y}<extra></extra>",
        ))
    fig.update_layout(
        height=90, margin={"l": 6, "r": 6, "t": 6, "b": 6},
        paper_bgcolor=C_BG, plot_bgcolor=C_BG,
        showlegend=False,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def latest_signal_for_asset(signals: list[dict], asset_id: str) -> dict:
    for sig in signals or []:
        if sig.get("asset") == asset_id:
            return sig
    return {}


def live_asset_card(asset_id: str, data: dict, position: dict | None, signal: dict | None) -> html.Div:
    cfg = managed_assets_config(enabled_only=True).get(asset_id, {})
    label = ASSET_DISPLAY_NAME.get(asset_id, cfg.get("name", asset_id))
    asset_type = str(cfg.get("type", "asset")).upper()
    price = data.get("price") if data else None
    chg = float(data.get("change_pct", 0) or 0) if data else 0.0
    rsi_val = data.get("rsi_14") if data else None
    ema9 = data.get("ema_9") if data else None
    ema21 = data.get("ema_21") if data else None
    atr_val = data.get("atr_14") if data else None
    chg_col = C_GREEN if chg >= 0 else C_RED

    if ema9 and ema21:
        trend = "EMA haussiere" if float(ema9) > float(ema21) else "EMA baissiere"
        trend_col = C_GREEN if float(ema9) > float(ema21) else C_RED
    else:
        trend, trend_col = "EMA -", C_MUTED

    pos_line = "Aucune position"
    pos_col = C_MUTED
    if position:
        side = str(position.get("side", "")).upper()
        pnl = float(position.get("unrealized_pnl", 0) or 0)
        pnl_pct = float(position.get("unrealized_pnl_pct", 0) or 0) * 100
        pos_line = f"{side} | {pnl:+.2f}$ ({pnl_pct:+.2f}%)"
        pos_col = C_GREEN if pnl >= 0 else C_RED

    sig_action = (signal or {}).get("action") or (signal or {}).get("signal") or "-"
    sig_reason = str((signal or {}).get("reason", ""))[:58]

    return html.Div([
        html.Div([
            html.Div([
                html.Div(label, style={"fontWeight": "900", "fontSize": "15px", "color": C_TEXT}),
                html.Div(asset_type, style={"fontSize": "10px", "fontWeight": "900", "color": C_MUTED,
                                            "letterSpacing": "0.08em"}),
            ]),
            signal_badge_el(sig_action),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "start",
                  "gap": "10px", "marginBottom": "8px"}),

        html.Div([
            html.Div(format_live_price(asset_id, price) if price is not None else "-",
                     style={"fontWeight": "900", "fontSize": "24px", "color": C_TEXT,
                            "lineHeight": "1.05"}),
            html.Div(f"{chg:+.4f}%", style={"fontWeight": "900", "fontSize": "13px", "color": chg_col}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "baseline",
                  "gap": "10px", "marginBottom": "6px"}),

        dcc.Graph(figure=asset_sparkline(asset_id, data or {}), config={"displayModeBar": False}),

        html.Div([
            _mini_stat("RSI", f"{float(rsi_val):.1f}" if rsi_val is not None else "-", C_PURPLE),
            _mini_stat("Tendance", trend, trend_col),
            _mini_stat("ATR", format_live_atr(asset_id, atr_val), C_ACCENT),
        ], style={"display": "grid", "gridTemplateColumns": "0.65fr 1fr 0.8fr",
                  "gap": "8px", "marginTop": "8px"}),

        html.Div(pos_line, style={"fontSize": "12px", "fontWeight": "800", "color": pos_col,
                                  "marginTop": "9px"}),
        html.Div(sig_reason or "En attente du prochain signal.",
                 style={"fontSize": "11px", "color": C_MUTED, "marginTop": "4px",
                        "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
    ], style={
        "background": C_PANEL,
        "border": f"1px solid {C_BORDER}",
        "borderLeft": f"4px solid {chg_col}",
        "borderRadius": "10px",
        "padding": "12px 14px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
        "minWidth": 0,
    })


def all_assets_live_panel(state: dict, live_assets: dict[str, dict]) -> html.Div:
    positions = {p.get("asset"): p for p in state.get("positions", [])}
    signals = state.get("recent_signals", [])
    assets_cfg = managed_assets_config(enabled_only=True)
    cards = []
    for asset_id in ASSET_DISPLAY_ORDER:
        if asset_id not in assets_cfg:
            continue
        cards.append(live_asset_card(
            asset_id=asset_id,
            data=live_assets.get(asset_id, {}),
            position=positions.get(asset_id),
            signal=latest_signal_for_asset(signals, asset_id),
        ))

    return html.Div([
        html.Div([
            html.Div("Live tous instruments", style={
                "fontWeight": "900", "fontSize": "13px", "color": C_TEXT,
            }),
            html.Div("Forex, crypto et or en 5 minutes, depuis le bot ou le fallback yfinance.",
                     style={"fontSize": "12px", "color": C_MUTED, "marginTop": "3px"}),
        ], style={"marginBottom": "10px"}),
        html.Div(cards, style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))",
            "gap": "12px",
        }),
    ], style={"marginBottom": "14px"})


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
        "live":    ("Bot Paper Trading actif — données en temps réel", C_GREEN, "#EEF5EE"),
        "delayed": ("Données récentes (< 1h)", C_YELLOW, "#F5F2E7"),
        "offline": ("Bot hors ligne — lancez src/main.py", C_RED, "#F5EEEE"),
    }
    st_text, st_col, st_bg = status_info[fresh]
    banner = html.Div(st_text, style={
        "background": st_bg, "color": st_col, "border": f"1px solid {st_col}30",
        "borderRadius": "8px", "padding": "10px 16px",
        "fontSize": "13px", "fontWeight": "600", "marginBottom": "14px",
    })

    # KPIs EUR/USD bot
    initial   = _pv(port.get("initial_capital"), 10_000)
    equity    = _pv(port.get("total_equity"),    initial)
    pnl       = _pv(port.get("total_pnl"),       0.0)
    ret       = _pv(port.get("return_pct"),      0.0)
    eur_price = _pv(eur.get("price"),            0.0)
    eur_chg   = _pv(eur.get("change_pct"),       0.0)
    regime    = mkt.get("regime", "unknown")
    reg_fr    = mkt.get("regime_fr", "Inconnu")
    reg_col   = REGIME_COLOR.get(regime, C_MUTED)
    reg_icon  = REGIME_ICON.get(regime, "?")

    eur_kpis = html.Div([
        html.Div("Bot Paper Trading (EUR/USD Intraday)", style={
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
                     subtitle=f"Confiance {_pv(mkt.get('confidence'), 0)*100:.0f}%",
                     color=reg_col,
                     tooltip="Regime de marché détecté"),
            kpi_card("Niveau de risque", mkt.get("risk_fr", "—"),
                     color=RISK_COLOR.get(mkt.get("risk_level", ""), C_MUTED),
                     tooltip="Niveau de risque calculé par le gestionnaire de risque"),
            kpi_card("Positions", str(port.get("num_positions", 0) or 0),
                     subtitle=f"Exposition {_pv(port.get('exposure_pct'), 0)*100:.1f}%",
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

    children = [banner, system_update_panel(state), asset_registry_panel(state), eur_kpis]
    if alpaca_block:
        children.append(alpaca_block)
    children += [eq_panel, events_panel]
    return html.Div(children)


# ── Page : EUR/USD Live ───────────────────────────────────────────────────────

def page_eurusd(state: dict) -> html.Div:
    live_assets = get_live_assets_data(state)
    eur = live_assets.get("EURUSD=X") or get_eurusd_data(state)

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

    # ── Trades et position ouverte EUR/USD ───────────────────────────────────
    all_trades    = state.get("recent_trades", [])
    all_positions = state.get("positions", [])
    eur_trades    = [t for t in all_trades    if t.get("asset") == "EURUSD=X"]
    eur_position  = next((p for p in all_positions if p.get("asset") == "EURUSD=X"), None)

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

    # Sessions — alignées avec les fenêtres de intraday_session_breakout.py
    now_utc   = datetime.now(timezone.utc)
    total_min = now_utc.hour * 60 + now_utc.minute
    sessions  = [
        ("Asie/Tokyo",  0,         3*60,     "#5E5496"),   # 00:00-03:00 UTC
        ("Londres",     7*60,      9*60+30,  "#357A52"),   # 07:00-09:30 UTC
        ("New York",    13*60+30,  16*60,    "#375F7A"),   # 13:30-16:00 UTC
    ]
    sess_items = []
    for name, start, end, col in sessions:
        active = start <= total_min < end
        sess_items.append(html.Div([
            html.Div("●", style={"color": col if active else "#C6C4BC", "fontSize": "20px"}),
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
        html.Div(f"UTC actuel : {now_utc.strftime('%H:%M')}  —  Forex ouvert 00:01-21:45 UTC (24/5)  |  Chevauchement London+NY : 13h30-16h00 UTC",
                 style={"fontSize": "12px", "color": C_MUTED, "marginTop": "8px"}),
    ])

    chart_el = card_wrap(
        dcc.Graph(
            figure=eurusd_chart(ohlcv, trades=eur_trades, open_position=eur_position),
            config={"displayModeBar": False, "scrollZoom": True},
        ),
        padding="0",
    )

    # ── Panneau droit : RSI gauge + position live ────────────────────────────
    right_children = [
        dcc.Graph(figure=rsi_gauge(rsi_val), config={"displayModeBar": False}),
        html.Hr(style={"margin": "8px 0", "borderColor": C_BORDER}),
    ]

    if eur_position:
        side      = eur_position.get("side", "long")
        entry_p   = float(eur_position.get("avg_entry") or eur_position.get("current_price") or 0)
        cur_p     = float(eur_position.get("current_price") or entry_p)
        upnl      = float(eur_position.get("unrealized_pnl") or 0)
        upnl_pct  = float(eur_position.get("unrealized_pnl_pct") or 0)
        sl        = eur_position.get("stop_loss")
        tp        = eur_position.get("take_profit")
        qty       = float(eur_position.get("quantity") or 0)
        strat     = eur_position.get("strategy", "—")
        side_lbl  = "LONG ▲" if side == "long" else "SHORT ▼"
        side_col  = C_GREEN if side == "long" else C_RED
        pnl_col   = C_GREEN if upnl >= 0 else C_RED
        right_children += [
            html.Div("Position ouverte", style={
                "fontWeight": "800", "fontSize": "11px", "color": C_MUTED,
                "textTransform": "uppercase", "letterSpacing": "0.05em",
                "marginBottom": "6px",
            }),
            html.Div(side_lbl, style={
                "fontWeight": "900", "fontSize": "18px", "color": side_col,
                "marginBottom": "4px",
            }),
            html.Div(f"{qty:.4f} lots", style={"fontSize": "11px", "color": C_MUTED, "marginBottom": "8px"}),
            html.Div([
                html.Div("Entrée", style={"fontSize": "10px", "color": C_MUTED}),
                html.Div(f"{entry_p:.5f}", style={"fontWeight": "700", "fontSize": "13px"}),
            ], style={"marginBottom": "4px"}),
            html.Div([
                html.Div("Prix actuel", style={"fontSize": "10px", "color": C_MUTED}),
                html.Div(f"{cur_p:.5f}", style={"fontWeight": "700", "fontSize": "13px"}),
            ], style={"marginBottom": "6px"}),
            html.Div([
                html.Div("P&L non réalisé", style={"fontSize": "10px", "color": C_MUTED}),
                html.Div(f"{upnl:+.2f}$", style={
                    "fontWeight": "900", "fontSize": "16px", "color": pnl_col,
                }),
                html.Div(f"({upnl_pct:+.2%})", style={"fontSize": "11px", "color": pnl_col}),
            ], style={"marginBottom": "6px"}),
            html.Div([
                html.Span("SL ", style={"fontSize": "10px", "color": "#A83C3C", "fontWeight": "700"}),
                html.Span(f"{float(sl):.5f}" if sl else "—", style={"fontSize": "11px"}),
            ], style={"marginBottom": "2px"}),
            html.Div([
                html.Span("TP ", style={"fontSize": "10px", "color": "#357A52", "fontWeight": "700"}),
                html.Span(f"{float(tp):.5f}" if tp else "—", style={"fontSize": "11px"}),
            ], style={"marginBottom": "6px"}),
            html.Div(strat[:20], style={"fontSize": "10px", "color": C_MUTED, "fontStyle": "italic"}),
        ]
    else:
        right_children += [
            html.Div("Aucune position", style={
                "textAlign": "center", "color": C_MUTED,
                "fontSize": "12px", "padding": "12px 0",
            }),
            html.Div("En attente de signal...", style={
                "textAlign": "center", "color": C_MUTED,
                "fontSize": "11px", "fontStyle": "italic",
            }),
        ]

    right_panel = card_wrap(right_children, padding="12px")

    # ── Tableau des trades EUR/USD (pleine largeur, en bas) ──────────────────
    trades_card = None
    if eur_trades:
        rows = []
        for tr in reversed(eur_trades[-15:]):
            side_t   = tr.get("side", "long")
            entry_t  = float(tr.get("entry_price") or tr.get("avg_entry") or tr.get("fill_price") or 0)
            exit_t   = tr.get("exit_price") or tr.get("close_price")
            pnl_t    = float(tr.get("pnl") or 0)
            strat_t  = tr.get("strategy", "—")
            opened_t = str(tr.get("opened_at", ""))[:16]
            pnl_c    = C_GREEN if pnl_t >= 0 else C_RED
            side_c   = C_GREEN if side_t in ("long", "BUY") else C_RED
            rows.append(html.Tr([
                html.Td(opened_t, style={"fontSize": "11px", "color": C_MUTED, "padding": "4px 8px"}),
                html.Td(html.Span("▲ Long" if side_t in ("long", "BUY") else "▼ Short",
                                  style={"color": side_c, "fontWeight": "700", "fontSize": "11px"}),
                        style={"padding": "4px 8px"}),
                html.Td(f"{entry_t:.5f}" if entry_t else "—",
                        style={"fontSize": "11px", "padding": "4px 8px"}),
                html.Td(f"{float(exit_t):.5f}" if exit_t else "ouvert",
                        style={"fontSize": "11px", "color": C_MUTED, "padding": "4px 8px"}),
                html.Td(f"{pnl_t:+.2f}$",
                        style={"fontWeight": "700", "fontSize": "11px", "color": pnl_c, "padding": "4px 8px"}),
                html.Td(strat_t[:24], style={"fontSize": "10px", "color": C_MUTED, "padding": "4px 8px"}),
            ]))
        trades_card = card_wrap([
            html.Div("Derniers trades EUR/USD", style={
                "fontWeight": "800", "fontSize": "13px", "marginBottom": "8px",
            }),
            html.Table([
                html.Thead(html.Tr([
                    html.Th(h, style={"fontSize": "10px", "color": C_MUTED, "fontWeight": "600",
                                      "padding": "4px 8px", "borderBottom": f"1px solid {C_BORDER}",
                                      "textAlign": "left"})
                    for h in ["Heure", "Sens", "Entrée", "Sortie", "P&L", "Stratégie"]
                ])),
                html.Tbody(rows),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
        ])

    bottom_els = [trades_card] if trades_card else []

    return html.Div([
        kpis,
        all_assets_live_panel(state, live_assets),
        sessions_panel,
        grid2(chart_el, right_panel, left_width="1fr", right_width="210px"),
        *bottom_els,
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

    # Check if forex market is likely closed (no intraday signals recently)
    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc)
    is_weekend = now_utc.weekday() >= 5  # Saturday=5, Sunday=6
    has_intraday = any(s.get("strategy", "").startswith("intraday_") for s in sigs)
    forex_closed = is_weekend or not has_intraday

    regime_map = {
        "bull_trend":        ["Tendance (EMA)", "Cassure N-jours", "Momentum Sectoriel", "Trend Scalp Multi-assets", "MACD Intraday", "Croisement EMA (Forex)", "Cassure de Session (Forex)"],
        "bear_trend":        ["Momentum Sectoriel", "Trend Scalp Multi-assets", "MACD Intraday", "Croisement EMA (Forex)"],
        "range":             ["Dip RSI(2)", "Momentum Sectoriel", "Bollinger + RSI (Forex)", "MACD Intraday", "Croisement EMA (Forex)"],
        "high_volatility":   ["Dip RSI(2)", "Trend Scalp Multi-assets", "MACD Intraday", "Croisement EMA (Forex)"],
        "low_volatility":    ["Tendance (EMA)", "Momentum Sectoriel", "Bollinger + RSI (Forex)", "Trend Scalp Multi-assets", "Croisement EMA (Forex)"],
        "panic":             ["Aucune — pause"],
        "compression":       ["Bollinger + RSI (Forex)", "MACD Intraday", "Cassure de Session (Forex)"],
        "breakout_expansion":["Cassure N-jours", "Tendance (EMA)", "Trend Scalp Multi-assets", "MACD Intraday", "Cassure de Session (Forex)", "Croisement EMA (Forex)"],
        "unknown":           ["Tendance (EMA)", "Dip RSI(2)", "Momentum Sectoriel", "Trend Scalp Multi-assets", "MACD Intraday", "Croisement EMA (Forex)"],
    }
    active_strats = regime_map.get(regime, ["Selon régime"])

    regime_info = card_wrap([
        html.Div("Strategies actives maintenant", style={"fontWeight": "800", "fontSize": "13px", "marginBottom": "8px"}),
        html.Div([
            html.Span(f"{REGIME_ICON.get(regime,'?')} {regime_fr}  → ", style={"color": C_MUTED}),
            *[badge(s, C_ACCENT) for s in active_strats],
        ], style={"display": "flex", "alignItems": "center", "gap": "6px", "flexWrap": "wrap"}),
        html.Div("⚠ Forex fermé ce week-end — stratégies forex en pause; crypto/or restent surveillés si les données sont disponibles.", style={
            "marginTop": "8px", "fontSize": "12px", "color": C_ORANGE, "fontWeight": "600",
        }) if forex_closed else None,
    ])

    swing_ids    = {"trend_following", "breakout", "rsi_dip_buyer", "thematic_momentum",
                    "ema_cross_swing", "momentum_burst"}
    intraday_ids = {
        "intraday_ema_cross",
        "intraday_bollinger_rsi",
        "intraday_session_breakout",
        "intraday_trend_scalp",
        "intraday_macd",
    }

    cards = []
    prev_group = None
    for strat_id, info in STRATEGY_INFO.items():
        # Section separator between swing and intraday groups
        group = "swing" if strat_id in swing_ids else "intraday"
        if group != prev_group:
            label = "Actions & ETF (Swing)" if group == "swing" else "Multi-assets Intraday 5min"
            cards.append(html.Div(label, style={
                "fontSize": "11px", "fontWeight": "800", "textTransform": "uppercase",
                "letterSpacing": "0.08em", "color": C_MUTED,
                "padding": "6px 0 4px",
            }))
            prev_group = group

        last     = last_sig.get(strat_id, {})
        action   = last.get("action", "")
        last_t   = str(last.get("time", "—"))[:16]
        reason   = str(last.get("reason", ""))[:100]
        conf     = float(last.get("confidence", 0) or 0)
        is_exec  = action == "EXECUTE"
        is_block = action == "BLOCK"
        is_notrade = action == "NO_TRADE"
        st_color = C_GREEN if is_exec else (C_ORANGE if is_block else (C_MUTED if is_notrade else C_MUTED))
        st_text  = "Ordre execute" if is_exec else ("Signal bloque" if is_block else ("Analysé / NO_TRADE" if is_notrade else "En surveillance"))

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
                "background": "#F4F3EF", "borderRadius": "8px", "padding": "10px 12px",
            }),
        ], style={
            "background": C_PANEL, "border": f"1px solid {C_BORDER}",
            "borderLeft": f"4px solid {info['color']}",
            "borderRadius": "10px", "padding": "14px 16px",
        }))

    return html.Div([
        system_update_panel(state, compact=True),
        asset_registry_panel(state, compact=True),
        regime_info,
        html.Div(cards, style={"display": "flex", "flexDirection": "column", "gap": "12px"}),
    ])


# ── Page : Positions Live ─────────────────────────────────────────────────────

def _fetch_position_charts(tickers: list) -> dict:
    """Fetch 6-month daily OHLCV + EMA for position charts. Cached 10min."""
    import time as _t
    now = _t.time()
    if now - _equity_chart_cache["ts"] < 600 and _equity_chart_cache["data"]:
        return _equity_chart_cache["data"]
    try:
        import yfinance as yf
        from src.data.yfinance_helpers import normalize_yfinance_columns
        from src.features.indicators import ema as _ema, rsi as _rsi, atr as _atr
    except ImportError:
        return {}

    data = {}
    for t in (tickers or []):
        if t.endswith("=X") or t.startswith("^"):
            continue
        try:
            df = yf.download(t, period="6mo", interval="1d", auto_adjust=True, progress=False)
            if df.empty or len(df) < 10:
                continue
            df = normalize_yfinance_columns(df)
            c = df["close"]
            df = df.copy()
            df["ema20"]  = _ema(c, 20)
            df["ema50"]  = _ema(c, 50)
            df["ema200"] = _ema(c, 200)
            df["rsi14"]  = _rsi(c, 14)
            data[t] = df
        except Exception:
            pass

    _equity_chart_cache["ts"] = now
    _equity_chart_cache["data"] = data
    return data


def _position_chart(df, entry_price: float, sl, tp, opened_at: str) -> "go.Figure":
    """Build price chart for one position with EMA + entry/SL/TP lines."""
    import plotly.graph_objects as go

    df90 = df.iloc[-90:].copy()
    dates = df90.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=df90["close"], name="Prix",
        line={"color": "#375F7A", "width": 2},
        hovertemplate="%{y:.2f}<extra></extra>",
    ))
    if "ema20" in df90.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df90["ema20"], name="EMA20",
            line={"color": "#357A52", "width": 1.5, "dash": "dot"},
            hovertemplate="%{y:.2f}<extra>EMA20</extra>",
        ))
    if "ema50" in df90.columns:
        fig.add_trace(go.Scatter(
            x=dates, y=df90["ema50"], name="EMA50",
            line={"color": "#A8712A", "width": 1.5, "dash": "dot"},
            hovertemplate="%{y:.2f}<extra>EMA50</extra>",
        ))

    # Horizontal levels
    fig.add_hline(y=entry_price, line_dash="dash", line_color="#375F7A", line_width=1.5,
                  annotation_text=f"Entree ${entry_price:.2f}",
                  annotation_font={"size": 10, "color": "#375F7A"})
    if sl:
        fig.add_hline(y=float(sl), line_dash="dash", line_color="#A83C3C", line_width=1.2,
                      annotation_text=f"Stop ${float(sl):.2f}",
                      annotation_font={"size": 10, "color": "#A83C3C"})
    if tp:
        fig.add_hline(y=float(tp), line_dash="dash", line_color="#357A52", line_width=1.2,
                      annotation_text=f"Objectif ${float(tp):.2f}",
                      annotation_font={"size": 10, "color": "#357A52"})

    # Vertical line at entry date
    if opened_at and len(opened_at) >= 10:
        try:
            import pandas as pd
            entry_date = pd.Timestamp(opened_at[:10])
            if entry_date >= dates[0]:
                fig.add_vline(x=entry_date, line_dash="dash",
                              line_color="#5E5496", line_width=1.5, opacity=0.7,
                              annotation_text="Achat", annotation_font={"size": 10, "color": "#5E5496"})
        except Exception:
            pass

    fig.update_layout(
        height=260, margin={"l": 30, "r": 10, "t": 10, "b": 30},
        paper_bgcolor=C_PANEL, plot_bgcolor=C_BG,
        showlegend=True,
        legend={"orientation": "h", "y": -0.18, "x": 0, "font": {"size": 11}},
        xaxis={"gridcolor": "#E8E6E1", "showgrid": True},
        yaxis={"gridcolor": "#E8E6E1", "showgrid": True},
        hovermode="x unified",
    )
    return fig


def _get_sector_info(asset: str, themes: dict) -> tuple:
    """Return (sector_label, score, reason, top_picks) for an asset."""
    try:
        from src.analysis.sector_universe import SECTOR_UNIVERSE
        for sk, info in SECTOR_UNIVERSE.items():
            if asset in info.get("tickers", []):
                theme = themes.get(sk, {})
                score  = float(theme.get("score", 0)) if isinstance(theme, dict) else 0.0
                reason = theme.get("reason", "") if isinstance(theme, dict) else ""
                picks  = theme.get("top_picks", []) if isinstance(theme, dict) else []
                return info.get("label", sk), score, reason, picks
    except Exception:
        pass
    return None, 0.0, "", []


def _trend_summary(df, theme_score: float) -> tuple:
    """Return (trend_text, color) based on EMA alignment + theme score."""
    if df is None or len(df) < 20:
        return "Donnees insuffisantes", C_MUTED
    try:
        last = df.iloc[-1]
        e20  = float(last.get("ema20", 0))
        e50  = float(last.get("ema50", 0))
        price = float(last["close"])
        if e20 > e50 and price > e20:
            ema_trend = "hausse"
        elif e20 < e50 and price < e20:
            ema_trend = "baisse"
        else:
            ema_trend = "neutre"

        rsi14 = float(last.get("rsi14", 50))
        rsi_txt = "suracheté" if rsi14 > 70 else ("survendu" if rsi14 < 30 else f"RSI={rsi14:.0f}")

        if ema_trend == "hausse" and theme_score >= 0:
            return f"Haussier ({rsi_txt})", C_GREEN
        if ema_trend == "baisse" and theme_score <= 0:
            return f"Baissier ({rsi_txt})", C_RED
        return f"Mixte — EMA {ema_trend} ({rsi_txt})", C_YELLOW
    except Exception:
        return "Analyse indisponible", C_MUTED


def page_positions(state: dict) -> html.Div:
    """Live positions page: one card per held stock with chart + analysis."""
    import plotly.graph_objects as go

    pos_list = state.get("positions", [])
    trades   = state.get("recent_trades", [])
    themes   = state.get("themes", {})
    port     = state.get("portfolio", {})
    news     = state.get("news", [])

    managed_cfg = managed_assets_config(enabled_only=True)

    def _is_managed_intraday_position(position: dict) -> bool:
        asset = str(position.get("asset", ""))
        try:
            lev = float(position.get("leverage", 1) or 1)
        except Exception:
            lev = 1.0
        return asset in managed_cfg or asset.endswith("=X") or lev > 1.0

    managed_pos = [p for p in pos_list if _is_managed_intraday_position(p)]
    equity_pos  = [p for p in pos_list if not _is_managed_intraday_position(p)]

    # Fetch charts
    tickers = [p["asset"] for p in equity_pos]
    chart_data = _fetch_position_charts(tickers)

    # KPI bar
    total_unreal = sum(_pv(p.get("unrealized_pnl")) for p in pos_list)
    total_val    = sum(_pv(p.get("market_value"))   for p in pos_list)
    total_real   = sum(_pv(t.get("pnl"))            for t in trades)
    exposure     = _pv(port.get("total_exposure"))
    total_val    = exposure or total_val
    cash         = _pv(port.get("available_cash") or port.get("cash"))

    kpis = html.Div([
        kpi_card("Positions ouvertes", str(len(pos_list)),
                 color=C_ACCENT if pos_list else C_MUTED),
        kpi_card("Exposition brute", f"${total_val:,.2f}", color=C_ACCENT),
        kpi_card("P&L non realise", f"${total_unreal:+,.2f}",
                 color=C_GREEN if total_unreal >= 0 else C_RED),
        kpi_card("P&L realise (total)", f"${total_real:+,.2f}",
                 color=C_GREEN if total_real >= 0 else C_RED),
        kpi_card("Cash disponible", f"${cash:,.2f}"),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(5,1fr)",
              "gap": "10px", "marginBottom": "14px"})

    # Empty state
    if not pos_list:
        empty = html.Div([
            html.Div("Aucune position ouverte", style={
                "fontWeight": "800", "fontSize": "16px", "marginBottom": "8px",
            }),
            html.Div(
                "Le bot surveille le marche et placera des ordres des que les conditions "
                "de signal sont reunies (RSI oversold, tendance sectorielle positive, "
                "EMA alignee).",
                style={"color": C_MUTED, "fontSize": "13px", "lineHeight": "1.6",
                       "maxWidth": "480px", "margin": "0 auto"},
            ),
        ], style={
            "textAlign": "center", "padding": "60px 20px",
            "background": C_PANEL, "borderRadius": "12px",
            "border": f"2px dashed {C_BORDER}",
        })
        # Still show trade history even when no open positions
        return html.Div([kpis, empty, _closed_trades_panel(trades)])

    cards = []

    # ── Equity position cards ─────────────────────────────────────────────────
    for p in equity_pos:
        asset   = p["asset"]
        entry   = _pv(p.get("avg_entry"),           0.0)
        current = _pv(p.get("current_price"),        entry)
        qty     = _pv(p.get("quantity"),             0.0)
        pnl     = _pv(p.get("unrealized_pnl"),      0.0)
        pnl_pct = _pv(p.get("unrealized_pnl_pct"),  0.0) * 100
        sl      = p.get("stop_loss")
        tp      = p.get("take_profit")
        strat   = p.get("strategy",  "—")
        opened  = p.get("opened_at", "—")
        side    = p.get("side", "long")

        is_up      = pnl >= 0
        pnl_color  = C_GREEN if is_up else C_RED
        arrow      = "+" if is_up else ""

        # R:R
        rr_txt = "—"
        if sl and tp and entry > 0:
            try:
                risk   = abs(entry - float(sl))
                reward = abs(float(tp) - entry)
                if risk > 0:
                    rr_txt = f"{reward/risk:.1f}x"
            except Exception:
                pass

        # Sector + theme
        sector_label, theme_score, theme_reason, top_picks = _get_sector_info(asset, themes)
        trend_text, trend_color = _trend_summary(chart_data.get(asset), theme_score)

        # Chart
        df = chart_data.get(asset)
        if df is not None and len(df) > 10:
            fig = _position_chart(df, entry, sl, tp, str(opened))
            chart_el = dcc.Graph(figure=fig, config={"displayModeBar": True,
                                                      "modeBarButtonsToRemove": ["lasso2d","select2d"]})
        else:
            chart_el = html.Div(
                "Graphique indisponible — les donnees seront chargees au prochain cycle.",
                style={"color": C_MUTED, "padding": "30px", "textAlign": "center",
                       "fontSize": "12px"},
            )

        # News snippets for this sector
        sector_news = []
        if sector_label:
            for art in (news or [])[:50]:
                headline = (art.get("headline") or art.get("title") or "")
                if any(w.lower() in headline.lower()
                       for w in [asset] + (top_picks or [])[:3]):
                    sector_news.append(headline[:100])
                    if len(sector_news) >= 2:
                        break

        card = html.Div([
            # ── Header ───────────────────────────────────────────────────────
            html.Div([
                html.Div([
                    html.Span(asset, style={
                        "fontWeight": "900", "fontSize": "22px", "marginRight": "8px",
                    }),
                    html.Span(side.upper(), style={
                        "background": C_GREEN if side == "long" else C_RED,
                        "color": "white", "fontSize": "10px", "fontWeight": "800",
                        "borderRadius": "4px", "padding": "2px 7px",
                    }),
                    html.Span(f" {qty:.4f} actions",
                              style={"color": C_MUTED, "fontSize": "12px", "marginLeft": "6px"}),
                ], style={"display": "flex", "alignItems": "center"}),
                html.Div([
                    html.Span(f"{arrow}{pnl_pct:.2f}%", style={
                        "fontWeight": "900", "fontSize": "20px", "color": pnl_color,
                    }),
                    html.Span(f"  ${pnl:+,.2f}", style={
                        "fontWeight": "600", "fontSize": "14px", "color": pnl_color,
                    }),
                ], style={"display": "flex", "alignItems": "baseline", "gap": "4px"}),
            ], style={
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                "padding": "12px 16px 10px",
                "background": f"linear-gradient(135deg, {pnl_color}10, white)",
                "borderBottom": f"1px solid {C_BORDER}",
            }),

            # ── Price strip ───────────────────────────────────────────────────
            html.Div([
                _mini_stat("Entree",   f"${entry:,.2f}"),
                _mini_stat("Actuel",   f"${current:,.2f}", pnl_color),
                _mini_stat("Stop",     f"${float(sl):,.2f}" if sl else "—", C_RED),
                _mini_stat("Objectif", f"${float(tp):,.2f}" if tp else "—", C_GREEN),
                _mini_stat("R:R",      rr_txt, C_ACCENT),
                _mini_stat("Strategie", strat),
                _mini_stat("Ouvert le", str(opened)[:10]),
            ], style={
                "display": "flex", "gap": "18px", "flexWrap": "wrap",
                "padding": "8px 16px", "borderBottom": f"1px solid {C_BORDER}",
                "background": "#F4F3EF",
            }),

            # ── Chart ─────────────────────────────────────────────────────────
            chart_el,

            # ── Analysis footer ───────────────────────────────────────────────
            html.Div([
                html.Div([
                    # Trend badge
                    html.Div([
                        html.Span("Tendance : ", style={"color": C_MUTED, "fontSize": "12px"}),
                        html.Span(trend_text, style={
                            "fontWeight": "700", "fontSize": "12px", "color": trend_color,
                        }),
                    ], style={"marginBottom": "6px"}),

                    # Sector + score
                    html.Div([
                        html.Span("Secteur : ", style={"color": C_MUTED, "fontSize": "12px"}),
                        html.Span(sector_label or "Non classe",
                                  style={"fontWeight": "700", "fontSize": "12px"}),
                        html.Span(
                            f"  Score LLM : {theme_score:+.2f}",
                            style={
                                "marginLeft": "10px", "fontSize": "12px",
                                "color": C_GREEN if theme_score > 0.2 else (C_RED if theme_score < -0.2 else C_MUTED),
                                "fontWeight": "700",
                            },
                        ) if theme_score != 0 else None,
                    ], style={"marginBottom": "4px"}) if sector_label else None,

                    # Theme reason
                    html.Div(
                        theme_reason[:160] if theme_reason else
                        "Analyse sectorielle: en attente de la prochaine analyse Groq.",
                        style={
                            "fontSize": "12px", "color": C_MUTED,
                            "fontStyle": "italic", "marginBottom": "6px",
                        },
                    ),

                    # News snippets
                    *[html.Div(f"  {n}", style={
                        "fontSize": "11px", "color": C_MUTED,
                        "padding": "2px 0", "borderLeft": f"3px solid {C_ACCENT}30",
                        "paddingLeft": "8px",
                    }) for n in sector_news],
                ]),
            ], style={
                "padding": "10px 16px",
                "borderTop": f"1px solid {C_BORDER}",
                "background": "#F4F3EF",
            }),
        ], style={
            "background": C_PANEL,
            "border": f"1px solid {C_BORDER}",
            "borderLeft": f"5px solid {pnl_color}",
            "borderRadius": "12px",
            "overflow": "hidden",
            "marginBottom": "14px",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
        })
        cards.append(card)

    # ── Managed intraday positions table ──────────────────────────────────────
    if managed_pos:
        fx_rows = [{
            "asset": ASSET_DISPLAY_NAME.get(p["asset"], p["asset"]),
            "ticker": p["asset"],
            "type": str(managed_cfg.get(p["asset"], {}).get("type", "intraday")),
            "levier": f"x{_pv(p.get('leverage', managed_cfg.get(p['asset'], {}).get('leverage', 1)), 1):g}",
            "qty": str(p.get("quantity", "—")),
            "entree": f"${_pv(p.get('avg_entry')):,.4f}",
            "actuel": f"${_pv(p.get('current_price'), _pv(p.get('avg_entry'))):,.4f}",
            "marge": f"${_pv(p.get('margin') or p.get('market_value')):,.2f}",
            "notional": f"${_pv(p.get('notional') or p.get('market_value')):,.2f}",
            "pnl": f"${_pv(p.get('unrealized_pnl')):+,.4f}",
            "strat": p.get("strategy", "—"),
        } for p in managed_pos]

        cards.append(html.Div([
            html.Div("Positions multi-assets intraday", style={
                "padding": "8px 16px", "fontWeight": "800", "fontSize": "13px",
                "borderBottom": f"1px solid {C_BORDER}",
            }),
            dash_table.DataTable(
                columns=[
                    {"name": "Actif",      "id": "asset"},
                    {"name": "Ticker",     "id": "ticker"},
                    {"name": "Type",       "id": "type"},
                    {"name": "Levier",     "id": "levier"},
                    {"name": "Qte",        "id": "qty"},
                    {"name": "Entree",     "id": "entree"},
                    {"name": "Actuel",     "id": "actuel"},
                    {"name": "Marge",      "id": "marge"},
                    {"name": "Notional",   "id": "notional"},
                    {"name": "P&L",        "id": "pnl"},
                    {"name": "Strategie",  "id": "strat"},
                ],
                data=fx_rows, page_size=10,
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#EDF2F6", "fontWeight": "700", "fontSize": "11px"},
                style_cell={"fontSize": "12px", "padding": "6px 10px"},
            ),
        ], style={"background": C_PANEL, "border": f"1px solid {C_BORDER}",
                  "borderRadius": "10px", "overflow": "hidden", "marginBottom": "14px"}))

    return html.Div([kpis] + cards + [_closed_trades_panel(trades)])


def _mini_stat(label: str, value: str, color: str = C_TEXT) -> html.Div:
    """Small label+value block used in the price strip."""
    return html.Div([
        html.Div(label, style={"color": C_MUTED, "fontSize": "10px", "fontWeight": "600",
                               "textTransform": "uppercase", "letterSpacing": "0.04em"}),
        html.Div(value, style={"fontWeight": "700", "fontSize": "13px", "color": color}),
    ])


def _closed_trades_panel(trades: list) -> html.Div:
    """Compact table of recently closed trades."""
    if not trades:
        return html.Div()
    rows = []
    for t in reversed(trades[-30:]):
        pnl = float(t.get("pnl", 0) or 0)
        ep  = float(t.get("entry_price", 1) or 1)
        qty = float(t.get("quantity", 1) or 1)
        rows.append({
            "asset":  t.get("asset", "—"),
            "side":   t.get("side", "—"),
            "entree": f"${_pv(t.get('entry_price')):,.4f}",
            "sortie": f"${_pv(t.get('exit_price')):,.4f}",
            "pnl":    f"${pnl:+,.4f}",
            "pct":    _safe_pct(pnl, ep * qty) if ep * qty else "—",
            "raison": t.get("exit_reason", "—"),
            "strat":  t.get("strategy", "—"),
            "date":   str(t.get("closed_at", "—"))[:16],
        })
    return html.Div([
        html.Div(f"Historique des trades fermes ({len(trades)})", style={
            "padding": "10px 16px", "fontWeight": "800", "fontSize": "13px",
            "borderBottom": f"1px solid {C_BORDER}",
        }),
        dash_table.DataTable(
            columns=[
                {"name": "Actif",      "id": "asset"},
                {"name": "Cote",       "id": "side"},
                {"name": "Entree",     "id": "entree"},
                {"name": "Sortie",     "id": "sortie"},
                {"name": "P&L $",      "id": "pnl"},
                {"name": "P&L %",      "id": "pct"},
                {"name": "Raison",     "id": "raison"},
                {"name": "Strategie",  "id": "strat"},
                {"name": "Date",       "id": "date"},
            ],
            data=rows, page_size=15,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#EDF2F6", "fontWeight": "700",
                          "fontSize": "11px", "border": f"1px solid {C_ACCENT}20"},
            style_cell={"fontSize": "12px", "padding": "7px 10px"},
            style_data_conditional=[
                {"if": {"filter_query": '{pnl} contains "+"', "column_id": "pnl"},
                 "color": C_GREEN, "fontWeight": "700"},
                {"if": {"filter_query": '{pnl} contains "-"', "column_id": "pnl"},
                 "color": C_RED, "fontWeight": "700"},
                {"if": {"row_index": "odd"}, "backgroundColor": "#EFF4F8"},
            ],
        ),
    ], style={"background": C_PANEL, "border": f"1px solid {C_ACCENT}30",
              "borderRadius": "10px", "overflow": "hidden", "marginTop": "14px"})


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
                style_header={"backgroundColor": "#EDEAF6", "fontWeight": "700",
                               "fontSize": "11px", "border": f"1px solid {C_ALPACA}20"},
                style_cell={"fontSize": "12px", "padding": "7px 10px",
                             "fontFamily": "Inter, sans-serif"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#F0EBF8"}],
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
                "entree":    f"${_pv(t.get('entry_price')):.2f}",
                "sortie":    f"${_pv(t.get('exit_price')):.2f}",
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
                style_header={"backgroundColor": "#EDEAF6", "fontWeight": "700",
                               "fontSize": "11px", "border": f"1px solid {C_ALPACA}20"},
                style_cell={"fontSize": "12px", "padding": "7px 10px",
                             "fontFamily": "Inter, sans-serif"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#F0EBF8"}],
            ),
        ], style={"background": C_PANEL, "border": f"1px solid {C_ALPACA}30",
                  "borderRadius": "10px", "overflow": "hidden", "marginBottom": "14px"})

        sections += [alp_kpis, alp_eq_panel, alp_pos_panel, alp_trades_panel]

    # ─── SECTION BOT PAPER TRADING ────────────────────────────────────────
    sections.append(section_header("Bot Paper Trading (Actions & Forex)", C_ACCENT))

    initial = _pv(port.get("initial_capital"), 10_000)
    equity  = _pv(port.get("total_equity"),    initial)
    _pnl    = _pv(port.get("total_pnl"))
    pnl_col = C_GREEN if _pnl >= 0 else C_RED

    _cash   = _pv(port.get("available_cash") or port.get("cash"))
    _opnl   = _pv(port.get("open_pnl"))
    _rpnl   = _pv(port.get("realized_pnl"))
    _ret    = _pv(port.get("return_pct"))
    _dd     = _pv(port.get("drawdown_pct"))

    eur_kpis = html.Div([
        kpi_card("Capital total", f"${equity:,.2f}",
                 color=C_GREEN if equity >= initial else C_RED,
                 tooltip="Cash + toutes positions ouvertes"),
        kpi_card("Cash dispo", f"${_cash:,.2f}"),
        kpi_card("P&L non realise", f"${_opnl:+,.2f}",
                 color=C_GREEN if _opnl >= 0 else C_RED),
        kpi_card("P&L realise", f"${_rpnl:+,.2f}",
                 color=C_GREEN if _rpnl >= 0 else C_RED),
        kpi_card("Return", f"{_ret:+.2f}%",
                 color=pnl_col),
        kpi_card("Drawdown max", f"{_dd * 100:.2f}%",
                 color=C_RED if _dd < -0.05 else C_MUTED,
                 tooltip="Perte maximale depuis le plus haut atteint"),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
              "gap": "10px", "marginBottom": "14px"})

    eur_eq_fig = equity_chart(hist, initial, "Courbe equite Paper Bot", C_ACCENT)
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
        html.Div(f"Positions ouvertes ({len(pos_list or [])})", style={
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
            style_header={"backgroundColor": "#EDF2F6", "fontWeight": "700",
                           "fontSize": "11px", "border": f"1px solid {C_ACCENT}20"},
            style_cell={"fontSize": "12px", "padding": "7px 10px", "fontFamily": "Inter, sans-serif"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#EFF4F8"}],
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
        html.Div(f"Historique trades ({len(trades or [])})", style={
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
            style_header={"backgroundColor": "#EDF2F6", "fontWeight": "700",
                           "fontSize": "11px", "border": f"1px solid {C_ACCENT}20"},
            style_cell={"fontSize": "12px", "padding": "7px 10px", "fontFamily": "Inter, sans-serif"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#EFF4F8"}],
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
            ], style={"background": "#EEF5EE", "borderRadius": "8px", "padding": "12px", "flex": "1"}),
            html.Div([
                html.Div("Risques", style={"fontWeight": "700", "fontSize": "12px", "color": C_MUTED}),
                html.Div(risk, style={"fontSize": "13px", "color": C_RED, "marginTop": "4px"}),
            ], style={"background": "#F5EEEE", "borderRadius": "8px", "padding": "12px", "flex": "1"}),
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
            ], style={"background": "#F5EEEE", "borderRadius": "8px", "padding": "12px", "flex": "1"}),
            html.Div([
                html.Div("Opportunités", style={"fontWeight": "700", "color": C_MUTED, "fontSize": "12px"}),
                html.Ul([html.Li(o, style={"fontSize": "13px", "marginTop": "3px"}) for o in (opps_list or ["—"])]),
            ], style={"background": "#EEF5EE", "borderRadius": "8px", "padding": "12px", "flex": "1"}),
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
            html.Span(f"risque {_pv(art.get('risk_score')):.0%}",
                      style={"fontSize": "11px",
                             "color": C_ORANGE if _pv(art.get("risk_score")) > 0.5 else C_MUTED}),
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


# ── Page : Analyse Groq ──────────────────────────────────────────────────────

def page_analyse(state: dict, alpaca: dict, groq_live: str | None = None) -> html.Div:
    """
    Page dédiée à l'analyse Groq/LLM avec :
      - Commentaire Groq complet et mis en valeur
      - Niveaux techniques clés EUR/USD
      - Journal des décisions du bot (pourquoi il trade ou pas)
      - Prochaines fenêtres de trading
    """
    mkt  = state.get("market", {})
    eur  = state.get("eurusd", {})
    sigs = state.get("recent_signals", [])
    port = state.get("portfolio", {})
    alp_targets = alpaca.get("targets", {})

    regime    = mkt.get("regime", "unknown")
    regime_fr = mkt.get("regime_fr", "Inconnu")
    reg_col   = REGIME_COLOR.get(regime, C_MUTED)
    reg_icon  = REGIME_ICON.get(regime, "?")
    fresh     = bot_freshness(state)
    cycle     = state.get("cycle_count", 0)

    # ── 1. Barre de statut bot ────────────────────────────────────────────
    status_colors = {
        "live":    (C_GREEN, "#EEF5EE", "Bot actif"),
        "delayed": (C_YELLOW, "#F5F2E7", "Données récentes"),
        "offline": (C_RED, "#F5EEEE", "Bot hors ligne"),
    }
    sc, sbg, stxt = status_colors[fresh]
    status_bar = html.Div([
        html.Span(f"● {stxt}", style={"fontWeight": "800", "color": sc}),
        html.Span(f"  |  Cycle #{cycle}", style={"color": C_MUTED}),
        html.Span(f"  |  Régime : {reg_icon} {regime_fr}", style={"color": reg_col, "fontWeight": "700"}),
        html.Span(
            f"  |  MàJ : {state.get('last_update', '—')}",
            style={"color": C_MUTED, "fontSize": "12px"},
        ),
    ], style={
        "background": sbg, "border": f"1px solid {sc}30",
        "borderRadius": "8px", "padding": "10px 16px", "marginBottom": "16px",
        "fontSize": "13px", "display": "flex", "flexWrap": "wrap", "gap": "4px",
    })

    # ── 2. News EUR/USD en direct ─────────────────────────────────────────
    news_items = fetch_forex_news(max_items=10)

    def sentiment_badge(title: str) -> tuple[str, str]:
        t = title.lower()
        bull = any(w in t for w in ["rise", "surge", "gain", "strong", "hawkish", "hausse", "monte", "haussier", "up"])
        bear = any(w in t for w in ["fall", "drop", "weak", "dovish", "baisse", "chute", "bearish", "down", "fear", "crisis"])
        if bull and not bear:
            return "HAUSSIER", C_GREEN
        if bear and not bull:
            return "BAISSIER", C_RED
        return "NEUTRE", C_MUTED

    news_cards = []
    for art in news_items:
        label, lcol = sentiment_badge(art["title"])
        pub_raw = art.get("published", "")
        pub_short = pub_raw[5:22] if len(pub_raw) > 10 else pub_raw
        news_cards.append(html.Div([
            html.Div([
                html.Span(art["source"], style={
                    "fontSize": "10px", "fontWeight": "700", "color": C_ACCENT,
                    "background": "#EDF2F6", "borderRadius": "4px",
                    "padding": "2px 7px", "marginRight": "8px",
                }),
                html.Span(pub_short, style={"fontSize": "10px", "color": C_MUTED}),
                html.Span(label, style={
                    "fontSize": "10px", "fontWeight": "800", "color": lcol,
                    "marginLeft": "8px", "background": f"{lcol}15",
                    "borderRadius": "4px", "padding": "2px 7px",
                }),
            ], style={"marginBottom": "4px"}),
            html.Div(art["title"], style={
                "fontSize": "13px", "lineHeight": "1.5", "color": C_TEXT, "fontWeight": "500",
            }),
        ], style={
            "padding": "10px 14px", "borderBottom": f"1px solid {C_BORDER}",
        }))

    if not news_cards:
        news_cards = [html.Div(
            "Aucune news EUR/USD trouvée (vérifiez la connexion internet).",
            style={"color": C_MUTED, "padding": "20px", "textAlign": "center", "fontSize": "13px"},
        )]

    news_section = html.Div([
        html.Div([
            html.Div([
                html.Span("Actualités EUR/USD", style={
                    "fontSize": "16px", "fontWeight": "900", "color": C_TEXT,
                }),
                html.Span(f"  {len(news_items)} titres", style={
                    "fontSize": "12px", "color": C_MUTED, "marginLeft": "8px",
                }),
            ]),
            html.Div("mis à jour toutes les 15s", style={
                "fontSize": "11px", "color": C_MUTED, "fontStyle": "italic",
            }),
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "marginBottom": "12px", "paddingBottom": "10px", "borderBottom": f"1px solid {C_BORDER}",
        }),
        html.Div(news_cards, style={"maxHeight": "340px", "overflowY": "auto"}),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px", "padding": "18px 20px", "marginBottom": "16px",
    })

    # ── 3. Analyse Groq interactive ───────────────────────────────────────
    groq_display = groq_live or ""
    groq_btn_section = html.Div([
        html.Div([
            html.Div([
                html.Span("Analyse IA en temps réel", style={
                    "fontSize": "16px", "fontWeight": "900", "color": C_PURPLE,
                }),
                html.Span("  Groq · Llama-3.3-70B", style={
                    "fontSize": "12px", "color": C_MUTED, "marginLeft": "8px",
                }),
            ]),
            html.Button("Analyser avec Groq", id="btn-groq-analyse",
                n_clicks=0,
                style={
                    "background": C_PURPLE, "color": "#fff", "border": "none",
                    "borderRadius": "4px", "padding": "9px 20px",
                    "fontWeight": "600", "fontSize": "12px", "cursor": "pointer",
                    "letterSpacing": "0.04em",
                    "boxShadow": "0 1px 4px rgba(94,84,150,0.2)",
                    "transition": "opacity 0.18s ease",
                }),
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "marginBottom": "14px", "paddingBottom": "10px", "borderBottom": f"1px solid {C_BORDER}",
        }),
        html.Div(
            groq_display or "Cliquez sur « Analyser avec Groq » pour obtenir une interprétation IA des news et de la situation technique actuelle.",
            id="groq-live-display",
            style={
                "fontSize": "15px", "lineHeight": "1.9", "color": C_TEXT if groq_display else C_MUTED,
                "background": "#F3F1F9" if groq_display else "#F4F3EF",
                "borderRadius": "10px", "padding": "18px 20px",
                "borderLeft": f"4px solid {C_PURPLE}",
                "fontStyle": "normal" if groq_display else "italic",
                "whiteSpace": "pre-wrap",
            },
        ),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px", "padding": "20px 22px", "marginBottom": "16px",
        "boxShadow": "0 1px 4px rgba(94,84,150,0.07)",
    })

    # ── 4. Analyse Groq quotidienne (du bot) ─────────────────────────────
    analyst     = mkt.get("analyst_summary", "")
    risks_list  = mkt.get("key_risks", [])
    opps_list   = mkt.get("opportunities", [])
    trend_map   = {"positive": ("Haussier", C_GREEN), "negative": ("Baissier", C_RED),
                   "neutral": ("Neutre", C_MUTED)}
    trend_txt, trend_col = trend_map.get(mkt.get("trend", "neutral"), ("Neutre", C_MUTED))

    groq_section = html.Div([
        # En-tête
        html.Div([
            html.Div([
                html.Span("Analyse IA", style={
                    "fontSize": "18px", "fontWeight": "900", "color": C_PURPLE,
                }),
                html.Span(" — Groq Llama 3.3 70B", style={
                    "fontSize": "13px", "color": C_MUTED, "marginLeft": "8px",
                }),
            ]),
            html.Div([
                html.Span("Tendance : ", style={"color": C_MUTED, "fontSize": "13px"}),
                html.Span(trend_txt, style={
                    "color": trend_col, "fontWeight": "900", "fontSize": "15px",
                }),
            ]),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "center", "marginBottom": "14px",
                  "paddingBottom": "10px", "borderBottom": f"1px solid {C_BORDER}"}),

        # Texte principal
        html.Div(
            analyst or "L'analyse Groq n'est pas encore disponible. Elle se déclenche automatiquement une fois par jour. Relancez le bot pour forcer un cycle.",
            style={
                "fontSize": "15px", "lineHeight": "1.85", "color": C_TEXT,
                "background": "#F3F1F9", "borderRadius": "10px",
                "padding": "18px 20px", "marginBottom": "16px",
                "borderLeft": f"4px solid {C_PURPLE}",
                "fontStyle": "normal" if analyst else "italic",
                "color": C_TEXT if analyst else C_MUTED,
            },
        ),

        # Risques et Opportunités
        html.Div([
            html.Div([
                html.Div([
                    html.Span("⚠ ", style={"fontSize": "16px"}),
                    html.Span("Risques identifiés", style={"fontWeight": "800", "fontSize": "13px"}),
                ], style={"marginBottom": "10px", "color": C_RED}),
                html.Div(
                    [html.Div([
                        html.Span("• ", style={"color": C_RED, "fontWeight": "800"}),
                        html.Span(r, style={"fontSize": "13px", "lineHeight": "1.6"}),
                    ], style={"marginBottom": "6px"}) for r in (risks_list or ["Aucun risque identifié"])],
                ),
            ], style={
                "background": "#F5EEEE", "borderRadius": "10px", "padding": "16px",
                "flex": "1", "border": f"1px solid {C_RED}20",
            }),
            html.Div([
                html.Div([
                    html.Span("✓ ", style={"fontSize": "16px"}),
                    html.Span("Opportunités", style={"fontWeight": "800", "fontSize": "13px"}),
                ], style={"marginBottom": "10px", "color": C_GREEN}),
                html.Div(
                    [html.Div([
                        html.Span("• ", style={"color": C_GREEN, "fontWeight": "800"}),
                        html.Span(o, style={"fontSize": "13px", "lineHeight": "1.6"}),
                    ], style={"marginBottom": "6px"}) for o in (opps_list or ["Aucune opportunité identifiée"])],
                ),
            ], style={
                "background": "#EEF5EE", "borderRadius": "10px", "padding": "16px",
                "flex": "1", "border": f"1px solid {C_GREEN}20",
            }),
        ], style={"display": "flex", "gap": "14px"}),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px", "padding": "20px 22px", "marginBottom": "16px",
        "boxShadow": "0 1px 4px rgba(94,84,150,0.07)",
    })

    # ── 5. Niveaux techniques EUR/USD ─────────────────────────────────────
    price  = float(eur.get("price", 0) or 0)
    ema9   = float(eur.get("ema_9", 0) or 0)
    ema21  = float(eur.get("ema_21", 0) or 0)
    rsi_v  = float(eur.get("rsi_14", 50) or 50)
    atr_v  = float(eur.get("atr_14", 0) or 0)
    bb_u   = float(eur.get("bb_upper", 0) or 0)
    bb_m   = float(eur.get("bb_middle", 0) or 0)
    bb_l   = float(eur.get("bb_lower", 0) or 0)

    def level_row(label: str, value: str, note: str, color: str = C_TEXT) -> html.Div:
        return html.Div([
            html.Div(label, style={"fontSize": "12px", "color": C_MUTED, "minWidth": "140px"}),
            html.Div(value, style={"fontSize": "14px", "fontWeight": "800", "color": color, "minWidth": "120px"}),
            html.Div(note, style={"fontSize": "12px", "color": C_MUTED}),
        ], style={"display": "flex", "alignItems": "center", "gap": "16px",
                  "padding": "8px 0", "borderBottom": f"1px solid {C_BORDER}"})

    ema_col  = C_GREEN if ema9 > ema21 else C_RED
    ema_note = "EMA 9 au-dessus — tendance haussière court terme" if ema9 > ema21 else "EMA 9 en-dessous — tendance baissière court terme"
    rsi_note = "Surachat — possible retournement baissier" if rsi_v > 70 else ("Survente — possible rebond haussier" if rsi_v < 30 else "Zone neutre — pas d'extrême")
    rsi_col  = C_RED if rsi_v > 70 else (C_GREEN if rsi_v < 30 else C_MUTED)
    bb_note  = "Position dans les bandes de Bollinger"
    if price and bb_u and bb_l:
        bb_pct = (price - bb_l) / (bb_u - bb_l) * 100 if (bb_u - bb_l) > 0 else 50
        bb_note = f"Dans la bande à {bb_pct:.0f}% (0%=bas, 100%=haut)"
    atr_pips = atr_v * 10000 if atr_v else 0

    tech_section = html.Div([
        html.Div("Niveaux techniques EUR/USD", style={
            "fontWeight": "900", "fontSize": "14px", "color": C_TEXT,
            "marginBottom": "12px",
        }),
        level_row("Prix actuel",         f"{price:.5f}" if price else "—",    "Dernier prix 5 minutes"),
        level_row("EMA 9 / EMA 21",      f"{ema9:.5f} / {ema21:.5f}" if ema9 else "—", ema_note, ema_col),
        level_row("RSI (14)",            f"{rsi_v:.1f}",                        rsi_note, rsi_col),
        level_row("ATR (14)",            f"{atr_pips:.1f} pips" if atr_pips else "—", "Amplitude moyenne des mouvements"),
        level_row("Bollinger haut",      f"{bb_u:.5f}" if bb_u else "—",       "Résistance dynamique — zone de vente potentielle"),
        level_row("Bollinger moyen",     f"{bb_m:.5f}" if bb_m else "—",       "Moyenne 20 périodes — objectif de retour"),
        level_row("Bollinger bas",       f"{bb_l:.5f}" if bb_l else "—",       "Support dynamique — zone d'achat potentielle"),
        level_row("Stop suggéré (ATR×1.5)", f"{atr_v*1.5*10000:.1f} pips" if atr_v else "—", "Distance stop-loss type intraday"),
        level_row("Objectif (ATR×2.5)", f"{atr_v*2.5*10000:.1f} pips" if atr_v else "—", "Distance take-profit type intraday"),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px", "padding": "18px 20px", "marginBottom": "16px",
    })

    # ── 6. Journal des décisions du bot ──────────────────────────────────
    def decision_icon(action: str) -> tuple[str, str]:
        return {
            "EXECUTE": ("✓", C_GREEN),
            "BLOCK":   ("✗", C_ORANGE),
            "NO_TRADE":("—", C_MUTED),
            "KILL":    ("⚠", C_RED),
        }.get(action.upper(), ("?", C_MUTED))

    journal_rows = []
    for sig in reversed(sigs[-20:] if sigs else []):
        action = sig.get("action", "")
        icon, icol = decision_icon(action)
        t      = str(sig.get("time", "—"))[:16]
        strat  = sig.get("strategy", "").replace("intraday_", "")
        signal = sig.get("signal", "")
        reason = str(sig.get("reason", ""))
        conf   = float(sig.get("confidence", 0) or 0)

        journal_rows.append(html.Div([
            html.Span(icon, style={
                "fontSize": "16px", "color": icol,
                "minWidth": "22px", "textAlign": "center",
            }),
            html.Div([
                html.Div([
                    html.Span(t, style={"color": C_MUTED, "fontSize": "11px", "marginRight": "10px"}),
                    signal_badge_el(action or signal or "—"),
                    html.Span(f"  {strat}", style={"color": C_ACCENT, "fontWeight": "700", "fontSize": "12px"}),
                    html.Span(f"  conf: {conf*100:.0f}%", style={"color": C_MUTED, "fontSize": "11px"}) if conf else None,
                ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "4px"}),
                html.Div(reason, style={
                    "fontSize": "12px", "color": C_MUTED, "marginTop": "3px",
                    "fontFamily": "monospace", "lineHeight": "1.5",
                }),
            ], style={"flex": "1"}),
        ], style={
            "display": "flex", "gap": "12px", "padding": "10px 0",
            "borderBottom": f"1px solid {C_BORDER}", "alignItems": "flex-start",
        }))

    if not journal_rows:
        journal_rows = [html.Div([
            html.Div("Aucune décision enregistrée pour le moment.", style={
                "color": C_MUTED, "padding": "20px", "textAlign": "center", "fontSize": "13px",
            }),
            html.Div("Le bot génère des décisions à chaque cycle (toutes les 5 min).", style={
                "color": C_MUTED, "textAlign": "center", "fontSize": "12px",
            }),
        ])]

    journal_section = html.Div([
        html.Div(f"Journal des décisions ({len(sigs)} entrées)", style={
            "fontWeight": "900", "fontSize": "14px", "color": C_TEXT,
            "marginBottom": "12px",
        }),
        html.Div(journal_rows),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px", "padding": "18px 20px", "marginBottom": "16px",
    })

    # ── 7. Prochaines fenêtres de trading ────────────────────────────────
    from datetime import timedelta
    now_utc = datetime.now(timezone.utc)
    h, m = now_utc.hour, now_utc.minute
    total_min = h * 60 + m

    def _mins_to_next(target_min: int) -> str:
        delta = target_min - total_min
        if delta < 0:
            delta += 24 * 60
        if delta == 0:
            return "maintenant"
        hrs, mins = divmod(delta, 60)
        return f"dans {hrs}h{mins:02d}" if hrs else f"dans {mins} min"

    windows = [
        ("Londres",  7*60,      9*60+30,  "#357A52",
         "Première heure — volatilité élevée, cassures fréquentes"),
        ("New York", 13*60+30,  16*60,    "#375F7A",
         "Ouverture US — chevauchement London+NY, meilleure liquidité"),
    ]

    session_rows = []
    for name, start, end, col, desc in windows:
        if start <= total_min < end:
            status = "ACTIVE MAINTENANT"
            time_txt = f"Ferme {_mins_to_next(end)}"
        else:
            status = _mins_to_next(start)
            time_txt = f"Durée : {(end-start)//60}h{(end-start)%60:02d}"
        is_active = start <= total_min < end
        session_rows.append(html.Div([
            html.Div([
                html.Div(name, style={"fontWeight": "800", "fontSize": "14px", "color": col}),
                html.Div(desc, style={"fontSize": "12px", "color": C_MUTED}),
            ], style={"flex": "1"}),
            html.Div([
                html.Div(status, style={
                    "fontWeight": "900", "fontSize": "13px",
                    "color": col if is_active else C_TEXT,
                    "background": f"{col}15" if is_active else "#F4F3EF",
                    "borderRadius": "8px", "padding": "6px 14px",
                    "border": f"1px solid {col}30",
                }),
                html.Div(time_txt, style={"fontSize": "11px", "color": C_MUTED, "marginTop": "4px", "textAlign": "center"}),
            ], style={"textAlign": "center"}),
        ], style={
            "display": "flex", "alignItems": "center", "gap": "16px",
            "padding": "12px 0", "borderBottom": f"1px solid {C_BORDER}",
        }))

    # Explication pourquoi bot ne trade pas maintenant
    ema9_ok   = ema9 and ema21 and abs(ema9 - ema21) > 0  # EMA cross happened
    rsi_ok    = rsi_v < 35 or rsi_v > 65
    session_active = any(s <= total_min < e for (_, s, e, _, _) in windows)
    bb_ok     = price and bb_l and bb_u and (price <= bb_l * 1.001 or price >= bb_u * 0.999)

    blockers = []
    if not session_active:
        next_s = min(
            (_mins_to_next(s), name)
            for name, s, e, _, _ in windows
            if not (s <= total_min < e)
        )
        blockers.append(f"Aucune session active — {next_s[1]} ouvre {next_s[0]}")
    if ema9 and ema21:
        blockers.append(f"EMA cross : EMA9={ema9:.5f} vs EMA21={ema21:.5f} — le prochain signal se déclenchera au prochain croisement")
    if not rsi_ok:
        blockers.append(f"RSI={rsi_v:.1f} en zone neutre (besoin de <35 ou >65 pour Bollinger RSI)")
    if not blockers:
        blockers.append("Conditions en cours d'évaluation...")

    why_section = html.Div([
        html.Div("Pourquoi le bot attend-il ?", style={
            "fontWeight": "900", "fontSize": "14px", "color": C_TEXT, "marginBottom": "10px",
        }),
        html.Div([
            html.Div([
                html.Span("⏳ ", style={"fontSize": "14px"}),
                html.Span(b, style={"fontSize": "13px", "lineHeight": "1.6", "color": C_TEXT}),
            ], style={"marginBottom": "6px"})
            for b in blockers
        ], style={
            "background": "#F4F3EF", "borderRadius": "8px",
            "padding": "14px 16px", "marginBottom": "14px",
        }),
        html.Div([
            html.Div("Prochaines fenêtres d'opportunité (UTC)", style={
                "fontWeight": "800", "fontSize": "13px",
                "color": C_TEXT, "marginBottom": "8px",
            }),
            html.Div(session_rows),
        ]),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px", "padding": "18px 20px", "marginBottom": "16px",
    })

    # ── 8. Etat bot résumé (Alpaca) ───────────────────────────────────────
    alp_initial = float(alpaca.get("initial_capital", 0))
    alp_equity  = float(alpaca.get("equity_history", [{}])[-1].get("equity", alp_initial)) if alpaca.get("equity_history") else alp_initial
    alp_pnl     = sum(float(x.get("pnl", 0)) for x in alpaca.get("closed_events", []))

    bot_status_section = html.Div([
        html.Div("État des comptes", style={
            "fontWeight": "900", "fontSize": "14px", "color": C_TEXT,
            "marginBottom": "12px",
        }),
        html.Div([
            html.Div([
                html.Div("EUR/USD Bot (intraday)", style={"fontWeight": "700", "color": C_ACCENT, "marginBottom": "6px"}),
                html.Div(f"Capital : ${_pv(port.get('total_equity'), 10000):,.2f}", style={"fontSize": "13px"}),
                html.Div(f"Positions : {port.get('num_positions', 0)}", style={"fontSize": "13px"}),
                html.Div(f"Cash : ${_pv(port.get('available_cash') or port.get('cash')):,.2f}", style={"fontSize": "13px"}),
            ], style={
                "background": "#EDF2F6", "borderRadius": "10px",
                "padding": "14px 16px", "flex": "1", "border": f"1px solid {C_ACCENT}20",
            }),
            html.Div([
                html.Div("Compte Alpaca Paper", style={"fontWeight": "700", "color": C_ALPACA, "marginBottom": "6px"}),
                html.Div(f"Capital : ${alp_equity:,.2f}", style={"fontSize": "13px"}),
                html.Div(f"Positions : {len(alp_targets)} ({', '.join(alp_targets.keys()) if alp_targets else 'aucune'})", style={"fontSize": "13px"}),
                html.Div(f"P&L réalisé : ${alp_pnl:+,.2f}", style={"fontSize": "13px", "color": C_GREEN if alp_pnl >= 0 else C_RED}),
            ], style={
                "background": "#EDEAF6", "borderRadius": "10px",
                "padding": "14px 16px", "flex": "1", "border": f"1px solid {C_ALPACA}20",
            }),
        ], style={"display": "flex", "gap": "14px"}),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px", "padding": "18px 20px",
    })

    return html.Div([
        status_bar,
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}, children=[
            html.Div([news_section, groq_section]),
            html.Div([groq_btn_section, tech_section, why_section, bot_status_section]),
        ]),
        journal_section,
    ])


# ── Page Thèmes & Secteurs ────────────────────────────────────────────────────

def page_themes(state: dict) -> html.Div:
    """Thematic investing tab — sector trend scores from LLM analysis."""
    themes_data = state.get("themes", {})
    sectors     = themes_data.get("sectors", {})
    narrative   = themes_data.get("narrative", "")
    active_uni  = themes_data.get("active_universe", [])
    last_ts     = themes_data.get("last_analysis", 0)

    # Header
    if last_ts:
        try:
            from datetime import datetime, timezone
            age_min = (time.time() - float(last_ts)) / 60
            ts_label = f"Dernière analyse : il y a {age_min:.0f} min"
        except Exception:
            ts_label = ""
    else:
        ts_label = "Aucune analyse disponible — le bot doit tourner"

    # ── Narrative box ──────────────────────────────────────────────────────
    narrative_box = html.Div([
        section_header("Analyse narrative (IA)", C_PURPLE),
        html.Div(
            narrative or "En attente de la première analyse thématique…",
            style={"fontSize": "13px", "lineHeight": "1.6", "color": C_TEXT,
                   "whiteSpace": "pre-wrap"},
        ),
        html.Div(ts_label, style={"fontSize": "11px", "color": C_MUTED, "marginTop": "8px"}),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px", "padding": "18px 20px", "marginBottom": "16px",
    })

    # ── Score bar chart ────────────────────────────────────────────────────
    if sectors:
        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1].get("score", 0), reverse=True)
        labels = [v.get("label", k) for k, v in sorted_sectors]
        scores = [v.get("score", 0.0) for _, v in sorted_sectors]
        bar_colors = [
            C_GREEN if s > 0.3 else (C_RED if s < -0.3 else C_YELLOW)
            for s in scores
        ]
        fig_bar = go.Figure(go.Bar(
            x=scores, y=labels, orientation="h",
            marker_color=bar_colors,
            text=[f"{s:+.2f}" for s in scores],
            textposition="outside",
        ))
        fig_bar.update_layout(
            margin=dict(l=10, r=60, t=20, b=20),
            paper_bgcolor=C_PANEL, plot_bgcolor=C_PANEL,
            height=280,
            xaxis=dict(range=[-1.1, 1.1], zeroline=True,
                       zerolinecolor=C_BORDER, tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=12)),
            font=dict(family="Inter, system-ui", size=11),
        )
        chart_section = html.Div([
            section_header("Score de tendance par secteur", C_ACCENT),
            dcc.Graph(figure=fig_bar, config={"displayModeBar": False}),
        ], style={
            "background": C_PANEL, "border": f"1px solid {C_BORDER}",
            "borderRadius": "8px", "padding": "18px 20px", "marginBottom": "16px",
        })
    else:
        chart_section = html.Div(
            "Les scores sectoriels apparaîtront ici après le premier cycle.",
            style={"color": C_MUTED, "fontSize": "13px", "padding": "20px",
                   "background": C_PANEL, "borderRadius": "12px",
                   "border": f"1px solid {C_BORDER}", "marginBottom": "16px"},
        )

    # ── Sector detail cards ────────────────────────────────────────────────
    sector_cards = []
    for k, v in sorted(sectors.items(), key=lambda x: -x[1].get("score", 0)):
        score   = float(v.get("score", 0))
        label   = v.get("label", k)
        reason  = v.get("reason", "")
        picks   = v.get("top_picks", [])
        mom     = v.get("momentum", "neutral")
        count   = v.get("article_count", 0)
        color   = C_GREEN if score > 0.3 else (C_RED if score < -0.3 else C_MUTED)
        mom_icon = {"rising": "↑", "falling": "↓", "neutral": "→"}.get(mom, "→")

        card = html.Div([
            html.Div([
                html.Span(label, style={"fontWeight": "800", "fontSize": "13px"}),
                html.Span(
                    f"{score:+.2f} {mom_icon}",
                    style={"fontWeight": "900", "fontSize": "15px",
                           "color": color, "marginLeft": "auto"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            html.Div(reason, style={"fontSize": "12px", "color": C_TEXT,
                                    "lineHeight": "1.5", "marginBottom": "8px"}),
            html.Div([
                html.Span("Titres suivis : ", style={"fontSize": "11px", "color": C_MUTED}),
                *[badge(t, C_ACCENT) for t in picks],
                html.Span(f"  {count} articles", style={"fontSize": "11px",
                                                          "color": C_MUTED, "marginLeft": "8px"}),
            ]),
        ], style={
            "background": C_PANEL,
            "border": f"2px solid {color}30",
            "borderRadius": "10px", "padding": "14px 16px",
            "boxShadow": "0 1px 3px rgba(0,0,0,.05)",
        })
        sector_cards.append(card)

    cards_section = html.Div([
        section_header("Détail par secteur", C_TEXT),
        html.Div(sector_cards, style={
            "display": "grid", "gridTemplateColumns": "1fr 1fr",
            "gap": "12px",
        }),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px", "padding": "18px 20px", "marginBottom": "16px",
    }) if sector_cards else html.Div()

    # ── Active universe ────────────────────────────────────────────────────
    universe_section = html.Div([
        section_header("Univers actif (titres sélectionnés)", C_GREEN),
        html.Div(
            [badge(t, C_GREEN) for t in active_uni] if active_uni
            else [html.Span("Aucun titre sélectionné", style={"color": C_MUTED, "fontSize": "13px"})],
            style={"display": "flex", "flexWrap": "wrap", "gap": "8px", "padding": "4px 0"},
        ),
        html.Div(
            f"{len(active_uni)} titres issus des secteurs avec score ≥ 0.35",
            style={"fontSize": "11px", "color": C_MUTED, "marginTop": "8px"},
        ),
    ], style={
        "background": C_PANEL, "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px", "padding": "18px 20px", "marginBottom": "16px",
    })

    return html.Div([
        narrative_box,
        chart_section,
        universe_section,
        cards_section,
    ])


# ── Rendu d'un onglet ─────────────────────────────────────────────────────────

def page_metrics(state: dict) -> html.Div:
    """Page Metriques — graphiques simples et lisibles."""
    import plotly.graph_objects as go

    broker      = read_broker_state()
    tracker     = read_perf_tracker()
    trades      = broker.get("trade_history", [])
    eq_hist     = state.get("equity_history", [])
    initial_cap = broker.get("initial_capital", 2000.0)

    # ── Calculs de base ──────────────────────────────────────────────────
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    wins      = [t for t in trades if t.get("pnl", 0) > 0]
    losses    = [t for t in trades if t.get("pnl", 0) < 0]
    n_trades  = len(trades)
    win_rate  = len(wins) / max(n_trades, 1) * 100
    avg_win   = sum(t["pnl"] for t in wins)  / max(len(wins), 1)
    avg_loss  = sum(abs(t["pnl"]) for t in losses) / max(len(losses), 1)
    rr        = avg_win / max(avg_loss, 0.01)

    current_cap = broker.get("cash", initial_cap)
    for p in broker.get("positions", []):
        current_cap += p.get("current_price", p.get("avg_entry", 0)) * p.get("quantity", 0)

    by_asset: dict[str, dict] = {}
    for t in trades:
        a = t.get("asset", "?")
        if a not in by_asset:
            by_asset[a] = {"pnl": 0.0, "wins": 0, "total": 0}
        by_asset[a]["pnl"]   += t.get("pnl", 0)
        by_asset[a]["total"] += 1
        if t.get("pnl", 0) > 0:
            by_asset[a]["wins"] += 1

    close_reasons: dict[str, int] = {}
    for t in trades:
        r = t.get("close_reason", "signal") or "signal"
        close_reasons[r] = close_reasons.get(r, 0) + 1

    # ── KPI row ──────────────────────────────────────────────────────────
    pnl_color = C_GREEN if total_pnl >= 0 else C_RED
    wr_color  = C_GREEN if win_rate >= 45 else (C_ORANGE if win_rate >= 30 else C_RED)
    rr_color  = C_GREEN if rr >= 2.0 else (C_ORANGE if rr >= 1.0 else C_RED)

    kpi_row = html.Div(style={
        "display": "grid", "gridTemplateColumns": "repeat(5, 1fr)",
        "gap": "12px", "marginBottom": "24px",
    }, children=[
        kpi_card("P&L Total",        f"${total_pnl:+.2f}",    color=pnl_color,
                 subtitle="gains/pertes depuis le debut"),
        kpi_card("Taux de reussite", f"{win_rate:.0f}%",       color=wr_color,
                 subtitle=f"{len(wins)} gagnants / {len(losses)} perdants"),
        kpi_card("R:R reel",         f"{rr:.2f}x",             color=rr_color,
                 subtitle=f"gain moy ${avg_win:.2f}  perte moy ${avg_loss:.2f}"),
        kpi_card("Capital actuel",   f"${current_cap:.0f}",    color=C_ACCENT,
                 subtitle=f"depart : ${initial_cap:.0f}"),
        kpi_card("Trades fermes",    str(n_trades),            color=C_TEXT,
                 subtitle="positions cloturees"),
    ])

    # ── Helpers graphiques ───────────────────────────────────────────────
    def _layout(height=280, pie=False) -> dict:
        base = dict(
            paper_bgcolor=C_PANEL, plot_bgcolor=C_PANEL,
            font=dict(family="Inter, system-ui", size=11, color=C_TEXT),
            margin=dict(l=55, r=20, t=60, b=40),
            height=height, showlegend=False,
        )
        if not pie:
            base["xaxis"] = dict(showgrid=False, zeroline=False,
                                 tickfont=dict(size=10, color=C_MUTED), linecolor=C_BORDER)
            base["yaxis"] = dict(gridcolor="#ECEAE4", zeroline=False,
                                 tickfont=dict(size=10, color=C_MUTED), linecolor=C_BORDER)
        return base

    def _title(main: str, sub: str = "") -> dict:
        text = f"<b>{main}</b>"
        if sub:
            text += f"<br><span style='font-size:10px;color:{C_MUTED}'>{sub}</span>"
        return dict(text=text, x=0.01, xanchor="left", yanchor="top",
                    font=dict(size=13, color=C_TEXT))

    def _empty(msg="Pas encore de donnees") -> html.Div:
        return html.Div(msg, style={
            "color": C_MUTED, "fontSize": "13px", "textAlign": "center",
            "padding": "40px 0",
        })

    def _panel(content) -> html.Div:
        return html.Div(content, style={
            "background": C_PANEL, "border": f"1px solid {C_BORDER}",
            "borderRadius": "8px", "padding": "4px 8px",
        })

    def _graph(fig) -> html.Div:
        return _panel(dcc.Graph(figure=fig, config={"displayModeBar": False}))

    # ── GRAPHIQUE 1 : Courbe du capital ───────────────────────────────────
    # Question : "Est-ce que le bot gagne de l'argent ?"
    def _chart_capital() -> go.Figure:
        fig = go.Figure()
        xs: list = []
        ys: list = []

        if len(eq_hist) >= 2:
            xs = [e.get("time", "") for e in eq_hist]
            ys = [float(e.get("equity", initial_cap)) for e in eq_hist]
        elif trades:
            running = initial_cap
            for t in sorted(trades, key=lambda x: x.get("closed_at", "")):
                running += t.get("pnl", 0)
                xs.append(t.get("closed_at", "")[:16])
                ys.append(running)

        if ys:
            color = C_GREEN if ys[-1] >= initial_cap else C_RED
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=color, width=2.5),
                fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.10)",
            ))
            fig.add_hline(y=initial_cap,
                          line_dash="dash", line_color=C_MUTED, line_width=1.5,
                          annotation_text=f"Depart ${initial_cap:.0f}",
                          annotation_font_size=10, annotation_font_color=C_MUTED,
                          annotation_position="bottom right")
        else:
            fig.add_annotation(text="Aucun trade ferme", x=0.5, y=0.5,
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(size=13, color=C_MUTED))

        lo = _layout(300)
        lo["yaxis"]["title"] = "Capital ($)"
        fig.update_layout(**lo, title=_title(
            "Capital de la plateforme",
            "La courbe monte = le bot gagne. La ligne en pointilles = capital de depart."
        ))
        return fig

    # ── GRAPHIQUE 2 : Victoires vs Defaites ──────────────────────────────
    # Question : "Combien de trades sur deux sont gagnants ?"
    def _chart_wins() -> go.Figure:
        fig = go.Figure()
        if n_trades == 0:
            fig.add_annotation(text="Aucun trade", x=0.5, y=0.5,
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(size=13, color=C_MUTED))
        else:
            n_neutral = n_trades - len(wins) - len(losses)
            labels = ["Gagnants", "Perdants"]
            values = [len(wins), len(losses)]
            colors = [C_GREEN, C_RED]
            if n_neutral > 0:
                labels.append("Nul")
                values.append(n_neutral)
                colors.append(C_MUTED)
            fig.add_trace(go.Pie(
                labels=labels, values=values,
                marker_colors=colors, hole=0.58,
                textinfo="label+percent+value",
                textfont_size=12, textposition="outside",
                insidetextorientation="auto",
            ))
        fig.update_layout(
            paper_bgcolor=C_PANEL, plot_bgcolor=C_PANEL,
            font=dict(family="Inter, system-ui", size=11, color=C_TEXT),
            margin=dict(l=30, r=30, t=65, b=30),
            height=280, showlegend=False,
            title=_title("Victoires vs Defaites",
                         f"Sur {n_trades} trades fermes au total"),
        )
        return fig

    # ── GRAPHIQUE 3 : SL vs TP ────────────────────────────────────────────
    # Question : "Le bot atteint-il ses objectifs ou est-il souvent stoppe ?"
    def _chart_close_reasons() -> go.Figure:
        fig = go.Figure()
        label_map = {
            "stop_loss":   "Stop Loss\n(perte stoppee)",
            "take_profit": "Take Profit\n(objectif atteint)",
            "signal":      "Signal\n(fermeture manuelle)",
        }
        if not close_reasons:
            fig.add_annotation(text="Aucun trade", x=0.5, y=0.5,
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(size=13, color=C_MUTED))
        else:
            labels, values, colors = [], [], []
            for k, v in sorted(close_reasons.items(), key=lambda x: -x[1]):
                labels.append(label_map.get(k, k))
                values.append(v)
                if "stop" in k or k.lower() == "sl":
                    colors.append(C_RED)
                elif "take" in k or k.lower() == "tp":
                    colors.append(C_GREEN)
                else:
                    colors.append(C_ACCENT)
            fig.add_trace(go.Bar(
                x=labels, y=values, marker_color=colors,
                text=values, textposition="outside",
                textfont=dict(size=14, color=C_TEXT),
                width=0.5,
            ))
        lo = _layout(280)
        lo["yaxis"]["title"] = "Nombre de trades"
        fig.update_layout(**lo, title=_title(
            "Comment les trades se ferment",
            "Vert = TP atteint (bon). Rouge = SL touche (perte limitee)."
        ))
        return fig

    # ── GRAPHIQUE 4 : P&L par actif ───────────────────────────────────────
    # Question : "Quel marche rapporte le plus (ou perd le plus) ?"
    def _chart_asset_pnl() -> go.Figure:
        fig = go.Figure()
        if not by_asset:
            fig.add_annotation(text="Aucun trade", x=0.5, y=0.5,
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(size=13, color=C_MUTED))
        else:
            items = sorted(by_asset.items(), key=lambda x: x[1]["pnl"])
            names  = [a for a, _ in items]
            pnls   = [d["pnl"] for _, d in items]
            labels = [
                f"${p:+.2f}  ({d['wins']}/{d['total']} gagnants)"
                for (_, d), p in zip(items, pnls)
            ]
            colors = [C_GREEN if p >= 0 else C_RED for p in pnls]
            fig.add_trace(go.Bar(
                y=names, x=pnls, orientation="h",
                marker_color=colors,
                text=labels, textposition="outside",
                textfont=dict(size=11, color=C_TEXT),
            ))
            fig.add_vline(x=0, line_color=C_BORDER, line_width=1.5)

        h = max(240, len(by_asset) * 52 + 90) if by_asset else 240
        lo = _layout(h)
        lo["xaxis"]["title"] = "P&L ($)"
        fig.update_layout(**lo, title=_title(
            "Quel marche rapporte le plus ?",
            "Vert = rentable. Rouge = perd de l'argent. Format : gain ($) et taux de reussite."
        ))
        return fig

    # ── GRAPHIQUE 5 : Le bot apprend-il ? ────────────────────────────────
    # Question : "Le taux de reussite augmente-t-il avec le temps ?"
    def _chart_learning() -> go.Figure:
        fig = go.Figure()
        source = sorted(
            [t for t in (tracker or trades) if t.get("pnl") is not None],
            key=lambda x: x.get("closed_at", ""),
        )
        n = len(source)
        if n < 5:
            fig.add_annotation(
                text=f"Il faut au moins 5 trades pour afficher cette courbe ({n} actuellement).",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                font=dict(size=12, color=C_MUTED),
            )
        else:
            window = min(10, max(3, n // 4))
            labels, roll_wr = [], []
            for i in range(n):
                w_trades = source[max(0, i - window + 1): i + 1]
                w = sum(1 for t in w_trades if t.get("pnl", 0) > 0)
                roll_wr.append(w / len(w_trades) * 100)
                labels.append(f"#{i+1}")

            point_colors = [C_GREEN if v >= 40 else (C_ORANGE if v >= 25 else C_RED)
                            for v in roll_wr]
            fig.add_trace(go.Scatter(
                x=labels, y=roll_wr, mode="lines+markers",
                line=dict(color=C_ACCENT, width=2),
                marker=dict(size=6, color=point_colors,
                            line=dict(color=C_PANEL, width=1)),
            ))
            fig.add_hrect(y0=40, y1=100, fillcolor=C_GREEN, opacity=0.05, line_width=0)
            fig.add_hline(y=33, line_dash="dot", line_color=C_RED, line_width=1.5,
                          annotation_text="Minimum viable (33%)",
                          annotation_font_size=9, annotation_font_color=C_RED,
                          annotation_position="bottom right")
            fig.add_hline(y=50, line_dash="dot", line_color=C_GREEN, line_width=1,
                          annotation_text="50%",
                          annotation_font_size=9, annotation_font_color=C_GREEN,
                          annotation_position="top right")

        lo = _layout(280)
        lo["yaxis"]["title"] = "% de trades gagnants"
        lo["yaxis"]["range"] = [0, 100]
        fig.update_layout(**lo, title=_title(
            "Le bot s'ameliore-t-il ?",
            f"Win rate glissant sur les {min(10, max(3, n//4) if n >= 5 else 0)} derniers trades. Zone verte = objectif."
        ))
        return fig

    # ── GRAPHIQUE 6 : Gain moyen vs Perte moyenne ─────────────────────────
    # Question : "Quand le bot gagne, gagne-t-il plus qu'il ne perd ?"
    def _chart_rr() -> go.Figure:
        fig = go.Figure()
        if not wins and not losses:
            fig.add_annotation(text="Aucun trade", x=0.5, y=0.5,
                               xref="paper", yref="paper", showarrow=False,
                               font=dict(size=13, color=C_MUTED))
        else:
            fig.add_trace(go.Bar(
                x=["Gain moyen quand on gagne", "Perte moyenne quand on perd"],
                y=[avg_win, avg_loss],
                marker_color=[C_GREEN, C_RED],
                text=[f"${avg_win:.2f}", f"${avg_loss:.2f}"],
                textposition="outside",
                textfont=dict(size=16, color=C_TEXT),
                width=0.45,
            ))
            rr_label = f"Ratio gain/perte = {rr:.2f}x"
            rr_color = C_GREEN if rr >= 2 else (C_ORANGE if rr >= 1 else C_RED)
            fig.add_annotation(
                text=f"<b style='color:{rr_color}'>{rr_label}</b>"
                     f"  —  au-dessus de 1x le bot gagne plus qu'il ne perd",
                x=0.5, y=1.12, xref="paper", yref="paper", showarrow=False,
                font=dict(size=11, color=C_MUTED), align="center",
            )
        lo = _layout(260)
        lo["yaxis"]["title"] = "Montant moyen ($)"
        fig.update_layout(**lo, title=_title(
            "Gain moyen vs Perte moyenne",
            "Si la barre verte depasse la rouge, le bot est profitable meme avec moins de 50% de victoires."
        ))
        return fig

    # ── Assembler la page ─────────────────────────────────────────────────
    return html.Div([
        html.H2("Metriques & Performance", style={
            "fontSize": "18px", "fontWeight": "700", "color": C_TEXT,
            "marginBottom": "20px", "marginTop": "0",
        }),

        kpi_row,

        # Courbe du capital — pleine largeur
        _graph(_chart_capital()),
        html.Div(style={"height": "16px"}),

        # Victoires vs Defaites  |  Comment les trades se ferment
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                        "gap": "16px", "marginBottom": "16px"}, children=[
            _graph(_chart_wins()),
            _graph(_chart_close_reasons()),
        ]),

        # P&L par actif  |  Courbe d'apprentissage
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                        "gap": "16px", "marginBottom": "16px"}, children=[
            _graph(_chart_asset_pnl()),
            _graph(_chart_learning()),
        ]),

        # Gain moyen vs Perte moyenne — pleine largeur
        _graph(_chart_rr()),

    ], style={"padding": "24px"})


def render_tab(tab: str, state: dict, alpaca: dict, groq_live: str | None = None) -> html.Div:
    try:
        if tab == "overview":
            return page_overview(state, alpaca)
        if tab == "eurusd":
            return page_eurusd(state)
        if tab == "strategies":
            return page_strategies(state)
        if tab == "positions":
            return page_positions(state)
        if tab == "portfolio":
            return page_portfolio(state, alpaca)
        if tab == "regime":
            return page_regime(state)
        if tab == "analyse":
            return page_analyse(state, alpaca, groq_live=groq_live)
        if tab == "themes":
            return page_themes(state)
        if tab == "metrics":
            return page_metrics(state)
    except Exception as exc:
        return html.Div([
            html.Div("Erreur lors du rendu de la page", style={
                "fontWeight": "800", "color": C_RED, "marginBottom": "8px",
            }),
            html.Pre(str(exc), style={
                "background": "#F5EEEE", "padding": "12px",
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
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  background:#F5F4F0;
  color:#1A1916;
  font-family:'Inter','Noto Sans JP',system-ui,sans-serif;
  font-size:14px;
  line-height:1.6;
  -webkit-font-smoothing:antialiased;
  -moz-osx-font-smoothing:grayscale;
}
.topbar{
  position:sticky;top:0;z-index:100;
  background:#1A1916;color:#F0EEE8;
  padding:0 24px;height:52px;
  display:flex;align-items:center;gap:16px;
  border-bottom:1px solid rgba(255,255,255,0.04);
}
.brand{
  font-size:13px;font-weight:700;
  color:#F0EEE8;letter-spacing:0.08em;
  text-transform:uppercase;
}
.brand span{color:#8BBBD4}
.tb-badge{
  border-radius:3px;font-size:10px;font-weight:700;
  padding:3px 8px;letter-spacing:0.06em;text-transform:uppercase;
}
.badge-live   {background:#1E3A2A;color:#74B893}
.badge-delayed{background:#3A2E12;color:#C9A864}
.badge-offline{background:#3A1E1E;color:#C47878}
.tb-price {font-size:12px;color:#8BBBD4;font-weight:500;letter-spacing:0.02em}
.tb-regime{font-size:11px;color:#6A6862;letter-spacing:0.02em}
.tb-time  {margin-left:auto;font-size:10px;color:#5A5852;letter-spacing:0.03em}
.nav{
  background:#FEFEFE;
  border-bottom:1px solid #E8E6E1;
  padding:0 24px;display:flex;gap:0;
  position:sticky;top:52px;z-index:90;
  overflow-x:auto;
}
.nav::-webkit-scrollbar{display:none}
.tab-btn{
  padding:12px 16px;font-size:11px;font-weight:600;cursor:pointer;
  border:none;background:transparent;color:#8C8A84;
  border-bottom:2px solid transparent;margin-bottom:-1px;
  transition:color 0.18s ease,border-color 0.18s ease;
  white-space:nowrap;letter-spacing:0.06em;
  text-transform:uppercase;
}
.tab-btn:hover{color:#1A1916}
.tab-btn.active{color:#375F7A;border-bottom-color:#375F7A}
.content{padding:20px 24px 60px;max-width:1440px;margin:0 auto}
@media(max-width:960px){
  .tab-btn{padding:10px 12px;font-size:10px}
  .content{padding:14px 14px 48px}
}
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""

    app.layout = html.Div([
        dcc.Interval(id="interval", interval=REFRESH_MS, n_intervals=0),
        dcc.Store(id="active-tab", data="overview"),
        dcc.Store(id="groq-live-store", data=""),

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
        State("groq-live-store", "data"),
        prevent_initial_call=False,
    )
    def main_update(*args):
        n_tabs   = len(TABS)
        # args[0..n_tabs-1] = n_clicks des boutons
        # args[n_tabs]      = n_intervals
        # args[n_tabs+1]    = active-tab (State)
        # args[n_tabs+2]    = groq-live-store (State)
        current_tab = args[n_tabs + 1] or "overview"
        groq_result = args[n_tabs + 2] or ""

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

        page = render_tab(active_tab, state, alpaca, groq_live=groq_result)
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

    # ── Callback bouton Groq ───────────────────────────────────────────────
    @app.callback(
        Output("groq-live-store", "data"),
        Input("btn-groq-analyse", "n_clicks"),
        prevent_initial_call=True,
    )
    def trigger_groq(n_clicks):
        if not n_clicks:
            return no_update
        state  = read_bot_state()
        news   = fetch_forex_news(max_items=10)
        result = call_groq_analysis(
            news_items=news,
            eur_data=state.get("eurusd", {}),
            market_data=state.get("market", {}),
        )
        return result

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
