"""
src/news/news_manager.py

News Manager — orchestrateur du News Layer.

Responsabilités :
  1. Planifier et exécuter les collectes (toutes les N minutes)
  2. Classifier les articles → NewsImpact par asset
  3. Maintenir un cache des impacts courants
  4. Exposer les risk_scores au Signal Aggregator (via DecisionEngine)
  5. Exposer les données à l'API

Pipeline :
    NewsCollector → NewsClassifier → cache impacts
         ↓
    risk_scores dict → SignalAggregator.apply_news_override()
"""
from __future__ import annotations

import logging
import time
from threading import Thread, Event
from typing import Optional

from src.news.collector import NewsCollector, RawArticle
from src.news.classifier import NewsClassifier, NewsImpact

logger = logging.getLogger(__name__)


class NewsManager:
    """
    Gestionnaire du flux d'informations.

    Usage :
        manager = NewsManager(cfg, universe=["AAPL", "SPY", "NVDA"])
        manager.start()   # Lance le collecteur en arrière-plan

        # Dans le cycle de trading :
        risk_scores = manager.get_risk_scores()
        engine.apply_news_risk(risk_scores)
    """

    def __init__(
        self,
        cfg: dict,
        universe: list[str],
    ) -> None:
        self.cfg = cfg
        self.universe = universe
        self.enabled = cfg.get("enabled", False)
        self.refresh_minutes = cfg.get("refresh_minutes", 30)

        self._collector = NewsCollector(cfg)
        self._classifier = NewsClassifier(cfg)

        # Cache des derniers impacts par asset
        self._impacts: dict[str, NewsImpact] = {}
        self._last_refresh: float = 0.0
        self._raw_articles: list[RawArticle] = []

        # Thread de collecte
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Lance la collecte en arrière-plan."""
        if not self.enabled:
            logger.info("NewsManager: disabled (set news.enabled=true to activate).")
            return

        # Collecte initiale synchrone
        self._refresh()

        # Thread de collecte périodique
        self._thread = Thread(target=self._loop, daemon=True, name="news-collector")
        self._thread.start()
        logger.info(
            "NewsManager started — refresh every %d min, universe=%s",
            self.refresh_minutes, self.universe,
        )

    def stop(self) -> None:
        """Arrête la collecte."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("NewsManager stopped.")

    # ------------------------------------------------------------------ #
    # Données exposées
    # ------------------------------------------------------------------ #

    def get_risk_scores(self) -> dict[str, float]:
        """
        Retourne les scores de risque par asset (0.0 à 1.0).
        Utilisé par DecisionEngine.apply_news_risk().
        """
        return {
            ticker: impact.risk_score
            for ticker, impact in self._impacts.items()
        }

    def get_latest_impacts(self) -> list[dict]:
        """Retourne tous les impacts courants (pour l'API)."""
        return [impact.to_dict() for impact in self._impacts.values()]

    def get_headlines(self, asset: str, limit: int = 10) -> list[str]:
        """Retourne les headlines pour un asset."""
        impact = self._impacts.get(asset)
        if impact:
            return impact.headlines[:limit]
        return []

    def get_impact(self, asset: str) -> Optional[NewsImpact]:
        return self._impacts.get(asset)

    def get_articles(self) -> list:
        """Return the raw articles from the last refresh (for ThemeAnalyzer)."""
        return list(self._raw_articles)

    @property
    def last_refresh_age_minutes(self) -> float:
        """Ancienneté de la dernière collecte en minutes."""
        return (time.time() - self._last_refresh) / 60.0

    # ------------------------------------------------------------------ #
    # Collecte interne
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        """Boucle de collecte en arrière-plan."""
        while not self._stop_event.is_set():
            sleep_seconds = self.refresh_minutes * 60
            self._stop_event.wait(timeout=sleep_seconds)
            if not self._stop_event.is_set():
                try:
                    self._refresh()
                except Exception as exc:
                    logger.error("NewsManager refresh failed: %s", exc)

    def _refresh(self) -> None:
        """Collecte + classification."""
        logger.info("NewsManager: refreshing news for %d assets...", len(self.universe))
        max_age = self.cfg.get("max_age_hours", 24.0)

        articles = self._collector.fetch(
            tickers=self.universe,
            max_age_hours=max_age,
        )
        self._raw_articles = articles

        if articles:
            impacts = self._classifier.classify(articles, self.universe)
            self._impacts.update(impacts)

        self._last_refresh = time.time()
        logger.info(
            "NewsManager: %d articles → %d asset impacts",
            len(articles), len(self._impacts),
        )

        # Log les impacts significatifs
        for ticker, impact in self._impacts.items():
            if abs(impact.sentiment) > 0.30 or impact.risk_score > 0.40:
                logger.info(
                    "  %s: sentiment=%.2f risk=%.2f impact=%s topics=%s",
                    ticker, impact.sentiment, impact.risk_score,
                    impact.impact, impact.topics,
                )
