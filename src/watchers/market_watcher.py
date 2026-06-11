"""
src/watchers/market_watcher.py

Market Watcher — module central de surveillance des marchés.

Il orchestre l'ensemble du pipeline à plusieurs fréquences :

  REALTIME  (toutes les secondes)   : surveillance stops/limites (pas de fetch données)
  INTRADAY  (toutes les 5 min)      : forex, bots intraday, spikes de volatilité
  HOURLY    (toutes les 60 min)     : SPY, QQQ, VIX, positions ouvertes
  DAILY     (1x/jour)               : régime, rotation ETF, macro, analyse LLM complète

Pipeline par cycle :
  1. Fetch market data (yfinance)
  2. FeatureEngine.compute() → MarketSnapshot
  3. RegimeModel.predict()   → MarketRegimePrediction
  4. MarketAnalyst.analyze() → MarketAnalysis (LLM ou rule-based)
  5. NewsManager (si activé) → risk_scores
  6. SignalAggregator override news
  7. DecisionEngine.run_cycle() → decisions
  8. PortfolioWatcher.watch() → WatchedPortfolioState
  9. AlertManager → alertes si seuils dépassés

MVP : 1 utilisateur, 5 actifs, cycle 15 min, LLM 1x/jour, paper only.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, Thread
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# MarketState — sortie du watcher à chaque cycle
# ------------------------------------------------------------------ #

@dataclass
class MarketState:
    """
    État de marché produit après chaque cycle complet.

    C'est la sortie principale du Market Watcher.
    """
    market: str                   # "US_EQUITIES" | "FOREX" | "MIXED"
    regime: str                   # label régime
    regime_confidence: float      # 0.0 – 1.0
    regime_source: str            # "ml" | "rules"
    risk_level: str               # "low" | "medium" | "high" | "extreme"
    vix: Optional[float]
    trend: str                    # "positive" | "negative" | "neutral"
    recommended_exposure: float   # 0.0 – 1.0
    cycle_type: str               # "daily" | "hourly" | "intraday" | "realtime"
    computed_at: str              # ISO 8601
    asset_features: dict          = field(default_factory=dict)  # {ticker: features_dict}
    analyst_summary: str          = ""
    key_risks: list[str]          = field(default_factory=list)
    opportunities: list[str]      = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "market": self.market,
            "regime": self.regime,
            "regime_confidence": round(self.regime_confidence, 3),
            "regime_source": self.regime_source,
            "risk_level": self.risk_level,
            "vix": self.vix,
            "trend": self.trend,
            "recommended_exposure": round(self.recommended_exposure, 3),
            "cycle_type": self.cycle_type,
            "computed_at": self.computed_at,
            "analyst_summary": self.analyst_summary,
            "key_risks": self.key_risks,
            "opportunities": self.opportunities,
        }


def _infer_asset_type(asset: str) -> str:
    """Map an asset symbol to a KellySizer asset_type string."""
    if asset in {"BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LTC-USD"}:
        return "crypto"
    if asset in {"GC=F", "CL=F", "SI=F", "NG=F", "HG=F"}:
        return "commodity"
    if asset.endswith("=X"):
        return "forex"
    return "equity"


# ------------------------------------------------------------------ #
# Market Watcher
# ------------------------------------------------------------------ #

class MarketWatcher:
    """
    Watcher principal. S'exécute en arrière-plan avec un cycle configurable.

    Hiérarchie des fréquences (MVP : tout sur 15 min, évolutif vers multi-freq) :

      Cycle DAILY    → analyse LLM complète, train ML, rapport
      Cycle HOURLY   → features + régime ML + news_risk
      Cycle INTRADAY → fetch intraday, bots 5 min
      Cycle REALTIME → portfolio watch (stops/limits), pas de fetch

    Usage :
        watcher = MarketWatcher(cfg, broker, engine, alert_manager, ...)
        watcher.start()
        # Le watcher tourne en background
        state = watcher.current_state
    """

    def __init__(
        self,
        cfg: dict,
        broker,                        # PaperBroker ou AlpacaPaperTrader
        decision_engine,               # DecisionEngine
        alert_manager,                 # AlertManager
        portfolio_watcher,             # PortfolioWatcher
        feature_engine,                # FeatureEngine
        regime_model,                  # RegimeModel
        market_analyst,                # MarketAnalyst
        news_manager=None,             # NewsManager (optionnel)
        client_profile=None,           # ClientProfile
        universe=None,                 # MarketUniverse
    ) -> None:
        self.cfg = cfg
        self.broker = broker
        self.engine = decision_engine
        self.alerts = alert_manager
        self.portfolio_watcher = portfolio_watcher
        self.feature_engine = feature_engine
        self.regime_model = regime_model
        self.analyst = market_analyst
        self.news_manager = news_manager
        self.profile = client_profile
        self.universe = universe

        # TradingOrchestrator — registre des assets multi-instruments
        from src.engine.trading_orchestrator import TradingOrchestrator
        assets_cfg   = self._load_assets_cfg()
        risk_cfg     = cfg.get("risk", {})
        settings_cfg = cfg
        self.orchestrator = TradingOrchestrator(assets_cfg, risk_cfg, settings_cfg)

        # ML pipeline — partagé entre le tracker et le decision engine
        from src.ml.signal_filter import SignalQualityFilter
        from src.ml.performance_tracker import PerformanceTracker
        from src.risk.kelly_sizer import KellySizer

        self._signal_filter = SignalQualityFilter()
        self._perf_tracker  = PerformanceTracker(self._signal_filter)
        self._kelly_sizer   = KellySizer()
        # Inject shared filter into decision engine (no restart needed)
        self.engine.signal_filter = self._signal_filter

        # Fréquences (en secondes) — MVP : tout à 15 min
        watcher_cfg = cfg.get("market_watcher", {})
        self.cycle_intraday_s  = watcher_cfg.get("intraday_interval_seconds", 300)    # 5 min
        self.cycle_hourly_s    = watcher_cfg.get("hourly_interval_seconds", 3600)     # 1h
        self.cycle_daily_s     = watcher_cfg.get("daily_interval_seconds", 86400)     # 24h
        self.mvp_cycle_s       = watcher_cfg.get("mvp_cycle_seconds", 900)            # 15 min MVP

        self.mvp_mode: bool = watcher_cfg.get("mvp_mode", True)
        self.llm_interval_s    = watcher_cfg.get("llm_interval_seconds", 86400)       # LLM 1x/jour

        # État interne
        self._current_state: Optional[MarketState] = None
        self._last_daily_ts: float = 0.0
        self._last_hourly_ts: float = 0.0
        self._last_llm_ts: float = 0.0
        self._last_regime: str = "unknown"
        self._cycle_count: int = 0

        # Dashboard state tracking
        self._equity_history: list[dict] = []   # {time, equity}
        self._recent_signals: list[dict] = []   # last 100 signal decisions

        # Thematic analysis state
        self._theme_analyzer = self._build_theme_analyzer()
        self._theme_scores: dict = {}      # {sector_key: ThemeScore}
        self._theme_universe: list[str] = []  # dynamic tickers from top themes

        # Threads
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

    @property
    def current_state(self) -> Optional[MarketState]:
        return self._current_state

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._thread = Thread(target=self._run, daemon=True, name="market-watcher")
        self._thread.start()
        logger.info(
            "MarketWatcher started (mvp_mode=%s, cycle=%ds, llm_interval=%ds)",
            self.mvp_mode, self.mvp_cycle_s, self.llm_interval_s,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("MarketWatcher stopped.")

    def run_once(self) -> Optional[MarketState]:
        """Lance un cycle unique de façon synchrone (utile pour les tests)."""
        return self._cycle()

    # ------------------------------------------------------------------ #
    # Boucle principale
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        """Boucle de fond."""
        # Premier cycle immédiat
        try:
            self._cycle()
        except Exception as exc:
            logger.error("Initial cycle failed: %s", exc, exc_info=True)

        while not self._stop_event.is_set():
            sleep = self.mvp_cycle_s if self.mvp_mode else self.cycle_intraday_s
            self._stop_event.wait(timeout=sleep)
            if not self._stop_event.is_set():
                try:
                    self._cycle()
                except Exception as exc:
                    logger.error("Watcher cycle error: %s", exc, exc_info=True)

    def _cycle(self) -> Optional[MarketState]:
        """Un cycle complet du Market Watcher."""
        now = time.time()
        self._cycle_count += 1

        # Déterminer le type de cycle
        is_daily  = (now - self._last_daily_ts) >= self.cycle_daily_s
        is_hourly = (now - self._last_hourly_ts) >= self.cycle_hourly_s
        cycle_type = "daily" if is_daily else ("hourly" if is_hourly else "intraday")

        logger.info("=== Market Watcher Cycle #%d [%s] ===", self._cycle_count, cycle_type)

        # 1. Fetch données
        tickers = self._get_tickers(cycle_type)
        if not tickers:
            # Even without daily tickers, intraday tickers may still need fetching
            tickers = []
        
        # CRITICAL: Always fetch prices for all open positions to keep them updated
        open_positions = self.broker.get_open_positions()
        position_tickers = set(p.get("asset") for p in open_positions if p.get("asset"))
        tickers = list(set(tickers) | position_tickers)  # Union of tickers and position assets
        
        if not tickers:
            logger.warning("No tickers and no open positions — skipping cycle")
            return None

        data_map, intraday_map, vix_series = self._fetch_data(tickers)

        # 2. Feature Engine — daily data only (intraday_map has 5min bars, wrong scale)
        if data_map:
            snapshot = self.feature_engine.compute(data_map, vix_series)
        else:
            from src.features.feature_engine import MarketSnapshot
            snapshot = MarketSnapshot(computed_at=now)

        # 3. ML Regime Model
        spy_df = data_map.get("SPY")

        # Entraîner le ML si pas encore fait (daily cycle uniquement)
        if is_daily and not self.regime_model.is_trained and spy_df is not None:
            logger.info("Training ML regime model...")
            self.regime_model.train_from_history(spy_df, vix_series)

        regime_prediction = self.regime_model.predict(snapshot, spy_df, vix_series)

        # Alerte changement de régime
        if regime_prediction.label != self._last_regime and self._last_regime != "unknown":
            self.alerts.regime_change(
                self._last_regime, regime_prediction.label, regime_prediction.confidence
            )
        self._last_regime = regime_prediction.label

        # 4. News Manager
        news_summary = ""
        if self.news_manager and self.news_manager.enabled:
            news_summary = self._build_news_summary()
            news_risk_scores = self.news_manager.get_risk_scores()
            # Alertes news risk
            for asset, score in news_risk_scores.items():
                if score > 0.65:
                    impact = self.news_manager.get_impact(asset)
                    topics = impact.topics if impact else []
                    self.alerts.news_risk(asset, score, topics)
            # Injection dans le Signal Aggregator
            self.engine.apply_news_risk(news_risk_scores)

        # 4b. Thematic Analysis (every 4h — driven by news articles)
        if self._theme_analyzer is not None:
            try:
                articles = (self.news_manager.get_articles()
                            if self.news_manager and self.news_manager.enabled else [])
                self._theme_scores = self._theme_analyzer.analyze(articles)
                # Update dynamic universe: top scoring tickers
                from src.analysis.sector_universe import get_tickers_for_themes
                raw_scores = {k: v.score for k, v in self._theme_scores.items()}
                self._theme_universe = get_tickers_for_themes(raw_scores)
                logger.info(
                    "ThemeAnalyzer: %d sectors scored — top universe: %s",
                    len(self._theme_scores),
                    self._theme_universe[:6],
                )
                # Sync theme scores into the strategy
                thematic_strat = self.engine._strategies.get("thematic_momentum")
                if thematic_strat is not None:
                    thematic_strat.update_themes(self._theme_scores)
            except Exception as exc:
                logger.warning("Theme analysis failed: %s", exc)

        # 5. LLM Market Analyst (selon intervalle configuré)
        analysis = None
        use_llm = (now - self._last_llm_ts) >= self.llm_interval_s
        if use_llm:
            from src.ai.market_analyst import AnalysisMode
            mode = AnalysisMode.FULL_STRATEGY if is_daily else (
                AnalysisMode.MARKET_SCAN if is_hourly else AnalysisMode.NEWS_QUICK
            )
            profile_dict = self.profile.to_dict() if self.profile else {}
            portfolio_state = self.broker.get_portfolio_state()

            analysis = self.analyst.analyze(
                mode=mode,
                regime=regime_prediction.label,
                features_snapshot=snapshot,
                profile_dict=profile_dict,
                portfolio_state=portfolio_state,
                news_summary=news_summary,
            )
            self._last_llm_ts = now
            logger.info(
                "LLM analysis [%s]: regime=%s risk=%s exposure=%.0f%%",
                mode.value, analysis.regime, analysis.risk_level,
                analysis.recommended_exposure * 100,
            )

        # 6. Decision Engine — run_cycle
        portfolio_state = self.broker.get_portfolio_state()
        open_positions = self.broker.get_open_positions()
        trade_history  = self.broker.get_trade_history()

        asset_strategy_map = (
            self.orchestrator.get_asset_strategy_map()
            if hasattr(self, 'orchestrator') else {}
        )
        decisions = self.engine.run_cycle(
            data_map=data_map,
            intraday_data_map=intraday_map,
            portfolio_state=portfolio_state,
            open_positions=open_positions,
            trade_history=trade_history,
            vix_series=vix_series,
            asset_strategy_map=asset_strategy_map,
        )

        # 7. Exécution des décisions approuvées
        current_prices = {t: float(df["close"].iloc[-1]) for t, df in data_map.items()}
        bar_highs: dict[str, float] = {}
        bar_lows:  dict[str, float] = {}
        for t, df in data_map.items():
            if "high" in df.columns: bar_highs[t] = float(df["high"].iloc[-1])
            if "low"  in df.columns: bar_lows[t]  = float(df["low"].iloc[-1])
        # Add intraday current prices (EURUSD=X last bar)
        for t, df in intraday_map.items():
            current_prices[t] = float(df["close"].iloc[-1])
            if "high" in df.columns: bar_highs[t] = float(df["high"].iloc[-1])
            if "low"  in df.columns: bar_lows[t]  = float(df["low"].iloc[-1])
        self._execute_decisions(decisions, current_prices, bar_highs, bar_lows)

        # Update all open positions with current market prices
        self.broker.update_prices(current_prices)

        # 8. Portfolio Watcher
        watched_state = self.portfolio_watcher.watch(self.broker, current_prices)

        # 9. Construire l'état global
        effective_analysis = analysis or self.analyst.last_analysis
        intraday_tickers = self._get_intraday_tickers()
        has_only_forex = bool(intraday_tickers) and not data_map
        market_type = "FOREX" if has_only_forex else ("MIXED" if intraday_tickers else "US_EQUITIES")
        state = MarketState(
            market=market_type,
            regime=regime_prediction.label,
            regime_confidence=regime_prediction.confidence,
            regime_source=regime_prediction.source,
            risk_level=effective_analysis.risk_level if effective_analysis else "medium",
            vix=snapshot.vix_level,
            trend=effective_analysis.trend if effective_analysis else "neutral",
            recommended_exposure=effective_analysis.recommended_exposure if effective_analysis else 0.65,
            cycle_type=cycle_type,
            computed_at=datetime.now(timezone.utc).isoformat(),
            asset_features={t: f.to_dict() for t, f in snapshot.assets.items()},
            analyst_summary=effective_analysis.summary if effective_analysis else "",
            key_risks=effective_analysis.key_risks if effective_analysis else [],
            opportunities=effective_analysis.opportunities if effective_analysis else [],
        )

        self._current_state = state

        # Timestamps
        if is_daily:
            self._last_daily_ts = now
        if is_hourly:
            self._last_hourly_ts = now

        logger.info(
            "Cycle #%d done: regime=%s risk=%s vix=%s decisions=%d",
            self._cycle_count, state.regime, state.risk_level,
            f"{state.vix:.1f}" if state.vix else "N/A",
            len(decisions),
        )

        # Track equity history + write dashboard state
        portfolio_state = self.broker.get_portfolio_state()
        self._equity_history.append({
            "time": state.computed_at,
            "equity": portfolio_state.get("total_equity", self.broker.initial_capital),
        })
        if len(self._equity_history) > 2000:
            self._equity_history = self._equity_history[-2000:]

        # Track NO_TRADE strategy activity first (so EXECUTE/BLOCK always appear at the top)
        for nt in self.engine.last_no_trade:
            self._recent_signals.append({
                "time": state.computed_at[:19],
                "asset": nt["asset"],
                "strategy": nt["strategy"],
                "signal": "NO_TRADE",
                "confidence": 0.0,
                "action": "NO_TRADE",
                "reason": nt["reason"][:120],
            })

        # Track decisions (EXECUTE / BLOCK) LAST — so they appear first in the [-30:] window
        for dec in decisions:
            self._recent_signals.append({
                "time": state.computed_at[:19],
                "asset": dec.asset,
                "strategy": dec.signal.strategy_name,
                "signal": dec.signal.signal.value,
                "confidence": round(dec.signal.confidence, 3),
                "action": dec.final_action,
                "reason": (dec.signal.reason or dec.risk_verdict.reason or "")[:120],
            })

        if len(self._recent_signals) > 200:
            self._recent_signals = self._recent_signals[-200:]

        self._write_dashboard_state(state, portfolio_state, intraday_map)
        return state

    # ------------------------------------------------------------------ #
    # Fetch données
    # ------------------------------------------------------------------ #

    def _build_theme_analyzer(self):
        """Instantiate ThemeAnalyzer with API key from settings."""
        try:
            import os
            from src.analysis.theme_analyzer import ThemeAnalyzer
            api_key = os.getenv("GROQ_API_KEY", "")
            llm_cfg = self.cfg.get("market_analyst", {})
            model   = llm_cfg.get("model", "llama-3.3-70b-versatile")
            return ThemeAnalyzer(groq_api_key=api_key, model=model)
        except Exception as exc:
            logger.warning("ThemeAnalyzer init failed: %s", exc)
            return None

    def _load_assets_cfg(self) -> dict:
        """Charge config/assets.yaml — retourne un dict vide si absent."""
        import yaml
        import os
        path = os.path.join(os.path.dirname(__file__), "../../config/assets.yaml")
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Could not load assets.yaml: %s", exc)
            return {}

    def _get_tickers(self, cycle_type: str) -> list[str]:
        if self.universe is None:
            return ["SPY", "QQQ", "^VIX"]

        from src.watchers.universe_builder import WatchFrequency
        # Filter out _5M placeholder tickers (universe_builder adds these as markers,
        # but yfinance doesn't know about "SPY_5M" etc.)
        intraday = [t for t in self.universe.tickers_for(WatchFrequency.INTRADAY)
                    if not t.endswith("_5M")]
        hourly   = self.universe.tickers_for(WatchFrequency.HOURLY)
        daily    = self.universe.tickers_for(WatchFrequency.DAILY)
        if cycle_type == "intraday":
            # Always include equity (hourly) tickers so swing strategies have data
            combined = list(dict.fromkeys(intraday + hourly))
            return combined or self.universe.primary_tickers
        if cycle_type in ("hourly", "daily"):
            tickers = list(dict.fromkeys(hourly + daily))
            return tickers or self.universe.all_tickers
        return self.universe.all_tickers

    def _get_intraday_tickers(self) -> list[str]:
        """Returns the list of tickers that should be fetched at 5min resolution."""
        # Assets from TradingOrchestrator (managed intraday assets)
        managed = list(self.orchestrator.tradable_assets) if hasattr(self, 'orchestrator') else []
        # Assets from universe builder (legacy)
        legacy = []
        if self.universe is not None:
            from src.watchers.universe_builder import WatchFrequency
            legacy = [t for t in self.universe.tickers_for(WatchFrequency.INTRADAY)
                      if not t.endswith("_5M") and t not in managed]
        return managed + legacy

    def _fetch_data(
        self,
        tickers: list[str],
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], Optional[pd.Series]]:
        """
        Fetch OHLCV data.

        Returns (daily_map, intraday_map, vix_series) where:
          daily_map    — 1d bars for SPY/QQQ/etc. (used by FeatureEngine & RegimeModel)
          intraday_map — 5m bars for forex tickers like EURUSD=X (used by intraday strategies)
          vix_series   — VIX close series for regime detection
        """
        try:
            import yfinance as yf
            from src.data.yfinance_helpers import normalize_yfinance_columns
            from src.features.indicators import compute_all_features
        except ImportError as exc:
            logger.error("Import error in _fetch_data: %s", exc)
            return {}, {}, None

        watcher_cfg    = self.cfg.get("market_watcher", {})
        intraday_period   = watcher_cfg.get("intraday_period",   "5d")
        intraday_interval = watcher_cfg.get("intraday_interval", "5m")

        intraday_tickers = set(self._get_intraday_tickers())
        data_map: dict[str, pd.DataFrame] = {}
        intraday_map: dict[str, pd.DataFrame] = {}
        vix_series: Optional[pd.Series] = None

        # --- Daily fetch (SPY, QQQ, ^VIX, etc.) ---
        daily_tickers = [t for t in tickers if t not in intraday_tickers]

        # Always include SPY for regime detection when we have intraday tickers
        if intraday_tickers and "SPY" not in daily_tickers:
            daily_tickers.append("SPY")

        # Add dynamic theme universe tickers (deduped)
        for t in self._theme_universe:
            if t not in daily_tickers and t not in intraday_tickers:
                daily_tickers.append(t)

        for ticker in daily_tickers:
            try:
                df = yf.download(
                    ticker, period="2y", interval="1d",
                    auto_adjust=True, progress=False
                )
                if df.empty or len(df) < 20:
                    continue
                df = normalize_yfinance_columns(df)

                if ticker == "^VIX":
                    vix_series = df["close"]
                    continue

                df = compute_all_features(df)
                data_map[ticker] = df
            except Exception as exc:
                logger.warning("Fetch failed for %s: %s", ticker, exc)

        # --- Intraday 5min fetch (EURUSD=X, etc.) ---
        for ticker in intraday_tickers:
            try:
                interval, period = intraday_interval, intraday_period  # defaults
                if hasattr(self, 'orchestrator') and self.orchestrator.is_managed_asset(ticker):
                    interval, period = self.orchestrator.get_data_params(ticker)
                df = yf.download(
                    ticker, period=period, interval=interval,
                    auto_adjust=True, progress=False
                )
                if df.empty or len(df) < 30:
                    logger.warning(
                        "Intraday fetch for %s returned %d bars (need ≥30)",
                        ticker, len(df)
                    )
                    continue
                df = normalize_yfinance_columns(df)
                intraday_map[ticker] = df
                logger.debug(
                    "Intraday %s: %d bars (%s-%s)",
                    ticker, len(df),
                    df.index[0].strftime("%Y-%m-%d %H:%M"),
                    df.index[-1].strftime("%Y-%m-%d %H:%M"),
                )
            except Exception as exc:
                logger.warning("Intraday fetch failed for %s: %s", ticker, exc)

        return data_map, intraday_map, vix_series

    # ------------------------------------------------------------------ #
    # Exécution
    # ------------------------------------------------------------------ #

    def _execution_size_with_current_budget(
        self,
        signal,
        requested_size_usd: float,
        current_price: float,
    ) -> tuple[float, float, str]:
        """Cap an approved decision against live broker exposure before submitting."""
        portfolio_state = self.broker.get_portfolio_state()
        # Pull risk params from the engine's risk_manager (authoritative source)
        risk_manager = getattr(self.engine, "risk_manager", None)
        risk = getattr(risk_manager, "r", {}) or {}
        total_capital = float(
            portfolio_state.get("total_capital")
            or portfolio_state.get("total_equity")
            or getattr(self.broker, "initial_capital", 0.0)
            or 0.0
        )
        if total_capital <= 0 or current_price <= 0:
            return 0.0, 0.0, "invalid capital or price"

        total_exposure = float(portfolio_state.get("total_exposure", 0.0) or 0.0)
        asset_exposure = portfolio_state.get("asset_exposure", {}) or {}
        horizon_exposure = portfolio_state.get("horizon_exposure", {}) or {}

        horizon_key = signal.horizon.value
        horizon_caps = {
            "intraday": risk.get("max_intraday_allocation_pct", 0.30),
            "swing": risk.get("max_swing_allocation_pct", 0.60),
            "long_term": risk.get("max_long_term_allocation_pct", 0.90),
        }

        asset_available = (
            total_capital * risk.get("max_exposure_per_asset_pct", 0.30)
            - float(asset_exposure.get(signal.asset, 0.0) or 0.0)
        )
        horizon_available = (
            total_capital * horizon_caps.get(horizon_key, 0.20)
            - float(horizon_exposure.get(horizon_key, 0.0) or 0.0)
        )

        if horizon_key == "intraday":
            # Intraday has its own dedicated budget — don't compete with swing/LT positions
            allowed_size = min(requested_size_usd, horizon_available, asset_available)
            # Apply orchestrator's per-asset margin cap (e.g. EUR/USD max 25% = $500)
            if hasattr(self, "orchestrator") and self.orchestrator.is_managed_asset(signal.asset):
                open_pos = self.broker.get_open_positions()
                deployed = self.orchestrator.get_deployed_margin(signal.asset, open_pos)
                orch_avail = self.orchestrator.get_available_margin(signal.asset, total_capital, deployed)
                allowed_size = min(allowed_size, orch_avail)
        else:
            total_available = total_capital * risk.get("max_total_exposure_pct", 0.80) - total_exposure
            allowed_size = min(requested_size_usd, total_available, asset_available, horizon_available)

        if allowed_size <= 0:
            total_available_safe = (
                total_capital * risk.get("max_total_exposure_pct", 0.80) - total_exposure
                if horizon_key != "intraday" else horizon_available
            )
            return 0.0, 0.0, (
                "execution budget exhausted "
                f"(available={total_available_safe:.2f}, asset={asset_available:.2f}, horizon={horizon_available:.2f})"
            )

        risk_manager = getattr(self.engine, "risk_manager", None)
        sizing = getattr(risk_manager, "ps", {}) if risk_manager is not None else {}
        min_size = float(
            sizing.get(
                "min_position_usd",
                self.cfg.get("broker", {}).get("paper", {}).get("min_position_usd", 0.0),
            )
            or 0.0
        )
        if allowed_size < min_size:
            # Kelly returned minimum fraction (negative edge) but budget exists — bump to min_size
            budget_available = min(asset_available, horizon_available)
            if budget_available >= min_size:
                logger.debug(
                    "Kelly floor $%.2f below min_size $%.2f — bumping to min_size",
                    allowed_size, min_size,
                )
                allowed_size = min_size
            else:
                return 0.0, 0.0, f"execution size ${allowed_size:.2f} below minimum ${min_size:.2f}"

        # Apply leverage: margin × leverage / price = quantity
        # Managed assets use orchestrator leverage; legacy forex uses config fallback.
        asset_leverage = 1.0
        if hasattr(self, 'orchestrator') and self.orchestrator.is_managed_asset(signal.asset):
            asset_leverage = self.orchestrator.get_leverage(signal.asset)
        elif horizon_key == "intraday" and "=" in signal.asset:
            asset_leverage = float(
                self.cfg.get("market_watcher", {}).get("forex_leverage", 1.0)
            )

        shares = (allowed_size * asset_leverage) / current_price
        return allowed_size, shares, "ok"

    @staticmethod
    def _is_execution_asset(asset: str) -> bool:
        """Skip macro/index symbols used for analysis only."""
        if asset.startswith("^"):
            return False
        return asset not in {"DX-Y.NYB"}

    def _execute_decisions(
        self,
        decisions: list,
        current_prices: dict,
        bar_highs: dict | None = None,
        bar_lows:  dict | None = None,
    ) -> None:
        """Exécute les décisions approuvées via le broker."""
        try:
            from src.execution.paper_broker import OrderSide
        except ImportError:
            return

        self.broker.fill_pending_orders(current_prices)

        # Snapshot trade count BEFORE stop/target checks to detect newly closed positions
        prev_trade_count = len(self.broker.get_trade_history())
        self.broker.check_stops_and_targets(current_prices, bar_highs, bar_lows)

        # Record closed positions in performance tracker
        new_trades = self.broker.get_trade_history()[prev_trade_count:]
        for trade in new_trades:
            asset      = trade.get("asset", "")
            exit_price = float(trade.get("exit_price") or trade.get("filled_price") or 0)
            pnl        = float(trade.get("pnl", 0))
            horizon    = trade.get("horizon", "intraday")
            if asset and exit_price > 0:
                self._perf_tracker.on_close(asset, exit_price, pnl, horizon)

        submitted_orders = 0

        for dec in decisions:
            if dec.final_action != "EXECUTE":
                continue
            signal = dec.signal
            if not self._is_execution_asset(signal.asset):
                logger.info("Skipping execution for non-tradable analysis asset: %s", signal.asset)
                continue
            if signal.asset not in current_prices:
                logger.warning("Skipping execution for %s: no current price.", signal.asset)
                continue
            side = OrderSide.BUY if signal.signal.value == "BUY" else OrderSide.SELL
            current_price = float(current_prices[signal.asset])

            # Guard NaN price (yfinance peut retourner NaN hors-session)
            import math as _math
            if not _math.isfinite(current_price) or current_price <= 0:
                logger.warning("Prix invalide (%.6g) pour %s — exécution ignorée",
                               current_price, signal.asset)
                continue

            # Anti-flip: ne pas retourner une position ouverte sur signal adverse.
            # Laisser SL/TP fermer la position naturellement évite les flip-losses.
            open_pos = {p["asset"]: p["side"] for p in self.broker.get_open_positions()}
            if signal.asset in open_pos:
                existing_side = open_pos[signal.asset]
                new_side = "long" if side == OrderSide.BUY else "short"
                if existing_side != new_side:
                    logger.debug(
                        "Anti-flip: %s déjà %s — signal %s ignoré, attente SL/TP",
                        signal.asset, existing_side, new_side,
                    )
                    continue

            # Kelly-based sizing
            portfolio_now  = self.broker.get_portfolio_state()
            _raw_capital = portfolio_now.get("total_equity") or portfolio_now.get("total_capital")
            total_capital = float(_raw_capital) if (
                _raw_capital is not None and _math.isfinite(float(_raw_capital)) and float(_raw_capital) > 0
            ) else float(getattr(self.broker, "initial_capital", 2000.0))

            kelly_params = self._perf_tracker.kelly_params(asset=signal.asset)
            asset_type   = _infer_asset_type(signal.asset)
            kelly_size   = self._kelly_sizer.margin_usd(
                capital    = total_capital,
                win_rate   = kelly_params["win_rate"],
                avg_win    = kelly_params["avg_win"],
                avg_loss   = kelly_params["avg_loss"],
                confidence = signal.confidence,
                ml_score   = float(signal.metadata.get("ml_score", 0.5)),
                regime     = self._last_regime,
                asset_type = asset_type,
            )
            # Guard NaN kelly_size (kelly_params manquants ou capital NaN)
            if not _math.isfinite(kelly_size) or kelly_size <= 0:
                kelly_size = total_capital * 0.05
                logger.warning("Kelly size invalide — fallback 5%% capital = $%.2f", kelly_size)

            size_usd, shares, budget_reason = self._execution_size_with_current_budget(
                signal,
                kelly_size,
                current_price,
            )
            if shares <= 0:
                logger.info(
                    "Skipping execution for %s %s: %s",
                    signal.asset, signal.signal.value, budget_reason,
                )
                continue

            lev = 1.0
            # Build order metadata — include leverage for managed and legacy forex assets
            order_meta: dict = {}
            if hasattr(self, 'orchestrator') and self.orchestrator.is_managed_asset(signal.asset):
                lev = self.orchestrator.get_leverage(signal.asset)
                if lev > 1.0:
                    order_meta["leverage"] = lev
            elif signal.horizon.value == "intraday" and "=" in signal.asset:
                lev = float(self.cfg.get("market_watcher", {}).get("forex_leverage", 1.0))
                if lev > 1.0:
                    order_meta["leverage"] = lev

            self._kelly_sizer.log_sizing(signal.asset, total_capital, kelly_size / total_capital, lev)

            order = self.broker.submit_market_order(
                asset=signal.asset,
                side=side,
                quantity=shares,
                strategy_name=signal.strategy_name,
                horizon=signal.horizon.value,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                metadata=order_meta if order_meta else None,
            )
            self.broker.fill_pending_orders(current_prices)
            submitted_orders += 1

            # Record position opening in performance tracker
            features = signal.metadata.get("_signal_features", {})
            self._perf_tracker.on_open(
                asset        = signal.asset,
                strategy     = signal.strategy_name,
                features     = features,
                entry_price  = float(order.filled_price or current_price),
                side         = "long" if side == OrderSide.BUY else "short",
            )

            self.alerts.trade_executed(
                asset=signal.asset,
                side=side.value,
                size=shares,
                price=order.filled_price or signal.entry_price or 0.0,
                strategy=signal.strategy_name,
            )

        if hasattr(self.broker, "save_state"):
            try:
                self.broker.save_state()
            except Exception as exc:
                logger.warning("Broker state save failed after execution: %s", exc)

        if submitted_orders:
            logger.info("Execution: %d paper order(s) submitted.", submitted_orders)

    def _write_dashboard_state(
        self,
        state: MarketState,
        portfolio_state: dict,
        intraday_map: dict,
    ) -> None:
        """Write bot state to JSON for the live dashboard. Never raises."""
        try:
            import json
            from pathlib import Path
            from src.features.indicators import ema as _ema, rsi as _rsi, atr as _atr

            out_dir = Path("data/dashboard")
            out_dir.mkdir(parents=True, exist_ok=True)

            def _asset_decimals(asset_id: str) -> int:
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

            def _dashboard_live_snapshot(asset_id: str, df_live) -> dict:
                if df_live is None or len(df_live) < 25:
                    return {}
                try:
                    dec = _asset_decimals(asset_id)
                    close = df_live["close"]
                    e9  = _ema(close, 9)
                    e21 = _ema(close, 21)
                    r14 = _rsi(close, 14)
                    a14 = _atr(df_live, 14)
                    sma20 = close.rolling(20).mean()
                    std20 = close.rolling(20).std(ddof=0)
                    bb_u = sma20 + 2.0 * std20
                    bb_l = sma20 - 2.0 * std20

                    current = float(close.iloc[-1])
                    prev = float(close.iloc[-2]) if len(close) > 1 else current
                    n = min(80, len(df_live))
                    ohlcv = [
                        {
                            "t": str(df_live.index[i]),
                            "o": _safe_round(df_live["open"].iloc[i], dec),
                            "h": _safe_round(df_live["high"].iloc[i], dec),
                            "l": _safe_round(df_live["low"].iloc[i], dec),
                            "c": _safe_round(df_live["close"].iloc[i], dec),
                            "e9": _safe_round(e9.iloc[i], dec),
                            "e21": _safe_round(e21.iloc[i], dec),
                            "bbu": _safe_round(bb_u.iloc[i], dec),
                            "bbl": _safe_round(bb_l.iloc[i], dec),
                        }
                        for i in range(-n, 0)
                    ]
                    return {
                        "asset": asset_id,
                        "price": _safe_round(current, dec),
                        "change_pct": round((current / prev - 1) * 100, 4) if prev else 0,
                        "ema_9": _safe_round(e9.iloc[-1], dec),
                        "ema_21": _safe_round(e21.iloc[-1], dec),
                        "rsi_14": _safe_round(r14.iloc[-1], 2),
                        "atr_14": _safe_round(a14.iloc[-1], 6),
                        "bb_upper": _safe_round(bb_u.iloc[-1], dec),
                        "bb_middle": _safe_round(sma20.iloc[-1], dec),
                        "bb_lower": _safe_round(bb_l.iloc[-1], dec),
                        "ohlcv": ohlcv,
                        "last_bar": str(df_live.index[-1]),
                    }
                except Exception:
                    return {}

            live_assets: dict = {}
            for asset_id, df_live in intraday_map.items():
                snapshot = _dashboard_live_snapshot(asset_id, df_live)
                if snapshot:
                    live_assets[asset_id] = snapshot

            # ---- EUR/USD indicators ----
            eurusd: dict = {}
            df_eur = intraday_map.get("EURUSD=X")
            if df_eur is not None and len(df_eur) >= 25:
                close = df_eur["close"]
                e9  = _ema(close, 9)
                e21 = _ema(close, 21)
                r14 = _rsi(close, 14)
                a14 = _atr(df_eur, 14)
                sma20 = close.rolling(20).mean()
                std20 = close.rolling(20).std(ddof=0)
                bb_u = sma20 + 2.0 * std20
                bb_l = sma20 - 2.0 * std20

                current = float(close.iloc[-1])
                prev    = float(close.iloc[-2]) if len(close) > 1 else current
                n = min(80, len(df_eur))
                ohlcv = [
                    {
                        "t": str(df_eur.index[i]),
                        "o": round(float(df_eur["open"].iloc[i]), 5),
                        "h": round(float(df_eur["high"].iloc[i]), 5),
                        "l": round(float(df_eur["low"].iloc[i]), 5),
                        "c": round(float(df_eur["close"].iloc[i]), 5),
                        "e9":  round(float(e9.iloc[i]), 5),
                        "e21": round(float(e21.iloc[i]), 5),
                        "bbu": round(float(bb_u.iloc[i]), 5) if not __import__("math").isnan(float(bb_u.iloc[i])) else None,
                        "bbl": round(float(bb_l.iloc[i]), 5) if not __import__("math").isnan(float(bb_l.iloc[i])) else None,
                    }
                    for i in range(-n, 0)
                ]
                eurusd = {
                    "price":      round(current, 5),
                    "change_pct": round((current / prev - 1) * 100, 4),
                    "ema_9":      round(float(e9.iloc[-1]), 5),
                    "ema_21":     round(float(e21.iloc[-1]), 5),
                    "rsi_14":     round(float(r14.iloc[-1]), 2),
                    "atr_14":     round(float(a14.iloc[-1]), 6),
                    "bb_upper":   round(float(bb_u.iloc[-1]), 5),
                    "bb_middle":  round(float(sma20.iloc[-1]), 5),
                    "bb_lower":   round(float(bb_l.iloc[-1]), 5),
                    "ohlcv":      ohlcv,
                }

            if eurusd:
                live_assets["EURUSD=X"] = eurusd

            # ---- News ----
            news_list: list[dict] = []
            if self.news_manager and self.news_manager.enabled:
                news_list = self.news_manager.get_latest_impacts()[:15]

            _REGIME_FR = {
                "bull_trend": "Tendance Haussiere",
                "bear_trend": "Tendance Baissiere",
                "range": "Marche Lateral (Range)",
                "high_volatility": "Haute Volatilite",
                "low_volatility": "Basse Volatilite",
                "panic": "Panique de Marche",
                "euphoric": "Euphorie",
                "compression": "Compression (Attente)",
                "breakout_expansion": "Cassure en Cours",
                "unknown": "Inconnu",
            }
            _RISK_FR = {
                "low": "Faible", "medium": "Modere",
                "high": "Eleve", "extreme": "Extreme",
            }

            doc = {
                "last_update":  state.computed_at[:19],
                "is_running":   True,
                "cycle_count":  self._cycle_count,
                "mode":         "paper",
                "market": {
                    "regime":            state.regime,
                    "regime_fr":         _REGIME_FR.get(state.regime, state.regime),
                    "confidence":        round(state.regime_confidence, 3),
                    "source":            state.regime_source,
                    "risk_level":        state.risk_level,
                    "risk_fr":           _RISK_FR.get(state.risk_level, state.risk_level),
                    "vix":               round(state.vix, 2) if state.vix else None,
                    "trend":             state.trend,
                    "exposure":          round(state.recommended_exposure, 3),
                    "analyst_summary":   state.analyst_summary,
                    "key_risks":         state.key_risks,
                    "opportunities":     state.opportunities,
                },
                "portfolio": {
                    "initial_capital":  self.broker.initial_capital,
                    "total_equity":     portfolio_state.get("total_equity", self.broker.initial_capital),
                    "cash":             portfolio_state.get("cash", 0),
                    "available_cash":   portfolio_state.get("available_cash", portfolio_state.get("cash", 0)),
                    "restricted_short_proceeds": portfolio_state.get("restricted_short_proceeds", 0),
                    "total_exposure":   portfolio_state.get("total_exposure", 0),
                    "open_pnl":         portfolio_state.get("unrealized_pnl", 0),
                    "realized_pnl":     portfolio_state.get("realized_pnl", 0),
                    "total_pnl":        portfolio_state.get("unrealized_pnl", 0) + portfolio_state.get("realized_pnl", 0),
                    "return_pct":       round(
                        (portfolio_state.get("total_equity", self.broker.initial_capital) / self.broker.initial_capital - 1) * 100, 4
                    ),
                    "drawdown_pct":     portfolio_state.get("drawdown_pct", 0),
                    "num_positions":    portfolio_state.get("open_positions", 0),
                    "exposure_pct":     portfolio_state.get("total_exposure_pct", 0),
                },
                "equity_history": self._equity_history[-500:],
                "positions":      self.broker.get_open_positions(),
                "recent_trades":  self.broker.get_trade_history()[-50:],
                "recent_signals": list(reversed(self._recent_signals[-30:])),
                "news":           news_list,
                "eurusd":         eurusd,
                "live_assets":    live_assets,
                "themes": {
                    "last_analysis":       getattr(self._theme_analyzer, "_last_run", 0),
                    "narrative":           getattr(self._theme_analyzer, "narrative", ""),
                    "active_universe":     self._theme_universe,
                    "sectors": {
                        k: v.to_dict()
                        for k, v in self._theme_scores.items()
                    } if self._theme_scores else {},
                },
            }

            import math as _math_json

            def _sanitize(obj):
                """Remplace NaN/Inf par None pour JSON valide."""
                if isinstance(obj, float):
                    return None if not _math_json.isfinite(obj) else obj
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_sanitize(v) for v in obj]
                return obj

            tmp_path   = out_dir / "bot_state.json.tmp"
            final_path = out_dir / "bot_state.json"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(_sanitize(doc), fh, indent=2, default=str)
            tmp_path.replace(final_path)

        except Exception as exc:
            logger.warning("Dashboard state write failed: %s", exc)

    def _build_news_summary(self) -> str:
        if not self.news_manager:
            return ""
        lines = []
        for impact_dict in self.news_manager.get_latest_impacts()[:5]:
            lines.append(
                f"{impact_dict['asset']}: sentiment={impact_dict['sentiment']:+.2f} "
                f"risk={impact_dict['risk_score']:.2f} [{impact_dict['impact']}]"
            )
        return "\n".join(lines)
