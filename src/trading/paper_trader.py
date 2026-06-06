"""
src/trading/paper_trader.py

Moteur de paper trading (argent fictif, données réelles).

Fonctionnement :
  1. Télécharge les barres journalières via yfinance
  2. Exécute les stratégies pour générer des signaux
  3. Simule les ordres (achat/vente) sur le portefeuille virtuel
  4. Vérifie les stop-loss / take-profit en temps réel
  5. Persiste l'état dans data/paper_trading/state.json

Lancer en standalone :
    python -m src.trading.paper_trader
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytz

from config.loader import get_strategy_config
from src.data.yfinance_helpers import configure_yfinance_cache, normalize_yfinance_columns
from src.features.indicators import compute_all_features
from src.features.regime_detector import MarketRegimeDetector
from src.strategies.base import SignalType
from src.strategies.breakout import BreakoutStrategy
from src.strategies.rsi_dip_buyer import RSIDipBuyerStrategy
from src.strategies.tactical_dca import TacticalDCAStrategy
from src.strategies.true_dca import TrueDCAStrategy
from src.strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)

STATE_FILE  = Path("data/paper_trading/state.json")
_ET         = pytz.timezone("America/New_York")
_REGIME_DET = MarketRegimeDetector()

WATCHLIST        = ["SPY", "QQQ"]
DEFAULT_CAPITAL  = 500.0
MAX_POSITIONS    = 3
RISK_PCT_TRADE   = 0.02     # 2% du capital par trade
MAX_POSITION_PCT = 0.40     # jamais plus de 40% du capital sur un seul trade
SLIPPAGE         = 0.001
REFRESH_SECONDS  = 60


# ── Structures de données ──────────────────────────────────────────────────────

@dataclass
class PaperPosition:
    ticker:      str
    strategy:    str
    quantity:    float
    entry_price: float
    entry_time:  str
    stop_loss:   float
    take_profit: float

    def unrealized_pnl(self, price: float) -> float:
        return round((price - self.entry_price) * self.quantity, 2)

    def pnl_pct(self, price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return round((price - self.entry_price) / self.entry_price * 100, 2)


@dataclass
class PaperTrade:
    ticker:      str
    strategy:    str
    quantity:    float
    entry_price: float
    entry_time:  str
    exit_price:  float
    exit_time:   str
    exit_reason: str
    pnl:         float
    pnl_pct:     float


# ── Portefeuille ───────────────────────────────────────────────────────────────

class PaperPortfolio:
    """Portefeuille virtuel avec persistance JSON."""

    def __init__(self, initial_capital: float = DEFAULT_CAPITAL) -> None:
        self.initial_capital = initial_capital
        self.cash            = initial_capital
        self.positions:      list[PaperPosition] = []
        self.closed_trades:  list[PaperTrade]    = []
        self.equity_history: list[dict]          = []
        self.signals_log:    list[dict]          = []

    # ── Calculs ────────────────────────────────────────────────────────────────

    def equity(self, prices: dict[str, float]) -> float:
        pos_val = sum(
            p.quantity * prices.get(p.ticker, p.entry_price)
            for p in self.positions
        )
        return round(self.cash + pos_val, 2)

    def open_pnl(self, prices: dict[str, float]) -> float:
        return round(sum(
            p.unrealized_pnl(prices.get(p.ticker, p.entry_price))
            for p in self.positions
        ), 2)

    def closed_pnl(self) -> float:
        return round(sum(t.pnl for t in self.closed_trades), 2)

    # ── Journalisation ─────────────────────────────────────────────────────────

    def snapshot(self, prices: dict[str, float]) -> None:
        self.equity_history.append({
            "time":   _now_str(),
            "equity": self.equity(prices),
        })
        if len(self.equity_history) > 10_000:
            self.equity_history = self.equity_history[-10_000:]

    def log_signal(
        self, ticker: str, strategy: str,
        signal: str, price: float, reason: str,
    ) -> None:
        self.signals_log.append({
            "time":     _now_str(),
            "ticker":   ticker,
            "strategy": strategy,
            "signal":   signal,
            "price":    round(price, 2),
            "reason":   reason[:140],
        })
        if len(self.signals_log) > 500:
            self.signals_log = self.signals_log[-500:]

    # ── Persistance ────────────────────────────────────────────────────────────

    def save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "initial_capital": self.initial_capital,
            "cash":            round(self.cash, 4),
            "positions":       [asdict(p) for p in self.positions],
            "closed_trades":   [asdict(t) for t in self.closed_trades[-1_000:]],
            "equity_history":  self.equity_history[-5_000:],
            "signals_log":     self.signals_log[-300:],
        }
        STATE_FILE.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, initial_capital: float = DEFAULT_CAPITAL) -> "PaperPortfolio":
        p = cls(initial_capital)
        if not STATE_FILE.exists():
            logger.info("Nouveau portefeuille de paper trading — capital $%.0f", initial_capital)
            return p
        try:
            data = json.loads(STATE_FILE.read_text())
            p.initial_capital = data.get("initial_capital", initial_capital)
            p.cash            = data.get("cash", initial_capital)
            p.positions       = [PaperPosition(**x) for x in data.get("positions", [])]
            p.closed_trades   = [PaperTrade(**x)    for x in data.get("closed_trades", [])]
            p.equity_history  = data.get("equity_history", [])
            p.signals_log     = data.get("signals_log", [])
            logger.info(
                "Portefeuille chargé — capital $%.2f | %d positions | %d trades fermés",
                p.equity({}), len(p.positions), len(p.closed_trades),
            )
        except Exception as exc:
            logger.warning("Impossible de charger l'état : %s — nouveau portefeuille.", exc)
        return p


# ── Moteur de trading ──────────────────────────────────────────────────────────

class PaperTradingEngine:
    """
    Télécharge les données, exécute les stratégies, met à jour le portefeuille.
    S'exécute en boucle toutes les `REFRESH_SECONDS` secondes dans un thread.
    """

    def __init__(
        self,
        watchlist: list[str] | None = None,
        initial_capital: float = DEFAULT_CAPITAL,
    ) -> None:
        self.watchlist  = [t.upper() for t in (watchlist or WATCHLIST)]
        self.portfolio  = PaperPortfolio.load(initial_capital)
        self._strategies = self._build_strategies()
        self._running   = False
        self._thread: threading.Thread | None = None
        self.last_update: str = ""
        self.last_error:  str = ""

    # ── Construction des stratégies ────────────────────────────────────────────

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

    # ── Données ────────────────────────────────────────────────────────────────

    def _get_bars(self, ticker: str) -> pd.DataFrame | None:
        import yfinance as yf
        configure_yfinance_cache()
        start = (date.today() - timedelta(days=420)).isoformat()
        try:
            df = yf.download(
                ticker, start=start, auto_adjust=True,
                progress=False, timeout=20,
            )
            df = normalize_yfinance_columns(df)
            return df if len(df) >= 250 else None
        except Exception as exc:
            logger.warning("Erreur données %s : %s", ticker, exc)
            return None

    def _live_price(self, ticker: str) -> float | None:
        import yfinance as yf
        try:
            fi = yf.Ticker(ticker).fast_info
            price = fi.get("last_price") or fi.get("regularMarketPrice")
            return float(price) if price else None
        except Exception:
            return None

    # ── Cycle principal ────────────────────────────────────────────────────────

    def update(self) -> None:
        """Un cycle complet : prix → SL/TP → signaux → snapshot."""

        # 1. Prix en direct
        prices: dict[str, float] = {}
        all_tickers = set(self.watchlist) | {p.ticker for p in self.portfolio.positions}
        for ticker in all_tickers:
            p = self._live_price(ticker)
            if p and p > 0:
                prices[ticker] = p

        # 2. Vérification stop-loss / take-profit
        remaining: list[PaperPosition] = []
        for pos in self.portfolio.positions:
            price = prices.get(pos.ticker)
            if price is None:
                remaining.append(pos)
                continue

            stop_hit   = price <= pos.stop_loss
            target_hit = price >= pos.take_profit

            if stop_hit or target_hit:
                reason      = "stop_loss" if stop_hit else "take_profit"
                exit_price  = round(price * (1 - SLIPPAGE), 4)
                pnl         = round((exit_price - pos.entry_price) * pos.quantity, 2)
                pnl_pct_val = round((exit_price / pos.entry_price - 1) * 100, 2)

                self.portfolio.cash += exit_price * pos.quantity
                self.portfolio.closed_trades.append(PaperTrade(
                    ticker=pos.ticker, strategy=pos.strategy,
                    quantity=pos.quantity, entry_price=pos.entry_price,
                    entry_time=pos.entry_time, exit_price=exit_price,
                    exit_time=_now_str(), exit_reason=reason,
                    pnl=pnl, pnl_pct=pnl_pct_val,
                ))
                self.portfolio.log_signal(
                    pos.ticker, pos.strategy,
                    f"FERMÉ ({reason})", price,
                    f"pnl={pnl:+.2f}$ ({pnl_pct_val:+.2f}%)",
                )
                logger.info("Fermé %s/%s @ %.2f | %s | PnL $%.2f",
                            pos.ticker, pos.strategy, exit_price, reason, pnl)
            else:
                remaining.append(pos)
        self.portfolio.positions = remaining

        # 3. Génération de signaux (si slots disponibles)
        open_tickers = {p.ticker for p in self.portfolio.positions}
        if len(self.portfolio.positions) < MAX_POSITIONS:
            for ticker in self.watchlist:
                if ticker in open_tickers:
                    continue
                df_raw = self._get_bars(ticker)
                if df_raw is None:
                    continue
                df = compute_all_features(df_raw)
                try:
                    regime = _REGIME_DET.detect(df).regime.value
                except Exception:
                    regime = "unknown"

                for strat_name, strategy in self._strategies.items():
                    already = any(
                        p.ticker == ticker and p.strategy == strat_name
                        for p in self.portfolio.positions
                    )
                    if already:
                        continue

                    try:
                        sig = strategy.generate_signal(df, ticker, regime)
                    except Exception as exc:
                        logger.debug("Signal error %s/%s : %s", ticker, strat_name, exc)
                        continue

                    if sig.signal not in (SignalType.BUY, SignalType.SELL):
                        continue

                    live_p     = prices.get(ticker, float(df["close"].iloc[-1]))
                    entry_p    = live_p * (1 + SLIPPAGE)
                    stop_dist  = abs(entry_p - sig.stop_loss) if sig.stop_loss else entry_p * 0.02
                    risk_usd   = self.portfolio.cash * RISK_PCT_TRADE
                    qty        = risk_usd / stop_dist if stop_dist > 0 else 0.0
                    pos_usd    = qty * entry_p

                    # Filtres de taille
                    min_usd = 15.0
                    max_usd = self.portfolio.cash * MAX_POSITION_PCT
                    if pos_usd < min_usd or pos_usd > max_usd:
                        continue
                    if pos_usd > self.portfolio.cash * 0.95:
                        continue

                    self.portfolio.cash -= pos_usd
                    pos = PaperPosition(
                        ticker=ticker, strategy=strat_name,
                        quantity=round(qty, 6),
                        entry_price=round(entry_p, 4),
                        entry_time=_now_str(),
                        stop_loss=round(sig.stop_loss  or entry_p * 0.97, 4),
                        take_profit=round(sig.take_profit or entry_p * 1.05, 4),
                    )
                    self.portfolio.positions.append(pos)
                    self.portfolio.log_signal(
                        ticker, strat_name, "ACHAT", live_p,
                        sig.reason or f"conf={sig.confidence:.2f}",
                    )
                    logger.info("Ouvert %s/%s @ %.2f qty=%.4f SL=%.2f TP=%.2f",
                                ticker, strat_name, entry_p, qty, pos.stop_loss, pos.take_profit)
                    break  # Un signal par ticker par cycle

        # 4. Snapshot equity + sauvegarde
        self.portfolio.snapshot(prices)
        self.portfolio.save()
        self.last_update = _now_str()

    # ── Contrôle du thread ─────────────────────────────────────────────────────

    def start(self, interval: int = REFRESH_SECONDS) -> None:
        if self._running:
            return
        self._running = True

        def _loop() -> None:
            while self._running:
                try:
                    self.update()
                    self.last_error = ""
                except Exception as exc:
                    self.last_error = str(exc)
                    logger.error("Erreur moteur : %s", exc)
                for _ in range(interval):
                    if not self._running:
                        break
                    time.sleep(1)

        self._thread = threading.Thread(target=_loop, daemon=True, name="PaperTrader")
        self._thread.start()
        logger.info("Paper trader démarré — cycle toutes les %ds", interval)

    def stop(self) -> None:
        self._running = False
        logger.info("Paper trader arrêté.")

    @property
    def running(self) -> bool:
        return self._running


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_str() -> str:
    return datetime.now(_ET).strftime("%Y-%m-%d %H:%M:%S ET")


def is_market_open() -> bool:
    now = datetime.now(_ET)
    if now.weekday() >= 5:
        return False
    mo = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    mc = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return mo <= now <= mc


def market_status() -> str:
    now = datetime.now(_ET)
    if now.weekday() >= 5:
        return "FERMÉ (week-end)"
    mo = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    mc = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    if mo <= now <= mc:
        return "OUVERT"
    if now < mo:
        delta = mo - now
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m = rem // 60
        return f"Pré-marché (ouverture dans {h}h{m:02d})"
    return "FERMÉ"


# ── Point d'entrée standalone ──────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    engine = PaperTradingEngine()
    engine.start()
    print(f"Paper trader démarré sur {engine.watchlist}. Ctrl+C pour arrêter.")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        engine.stop()
        print("Arrêté.")
