"""
src/news/collector.py

News Collector — récupère les headlines depuis des sources publiques.

Sources supportées (par ordre de priorité / qualité) :
  1. Finnhub       (API gratuite, très bonne qualité financière)
  2. NewsAPI       (API gratuite, couverture générale)
  3. Alpha Vantage (API gratuite, news financières)
  4. RSS           (fallback gratuit, pas d'API key requise)

Chaque source retourne une liste de RawArticle normalisée.
Le Collector tente les sources dans l'ordre et agrège les résultats.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RawArticle:
    """Article brut normalisé avant nettoyage/classification."""
    id: str                       # Hash du titre
    headline: str
    summary: str
    source: str                   # Nom de la source
    url: str
    published_at: float           # Timestamp UNIX
    assets: list[str]             # Tickers mentionnés (si disponibles)
    raw: dict = field(default_factory=dict)  # Données brutes

    @classmethod
    def from_headline(
        cls,
        headline: str,
        summary: str,
        source: str,
        url: str = "",
        published_at: float = 0.0,
        assets: list[str] | None = None,
        raw: dict | None = None,
    ) -> "RawArticle":
        article_id = hashlib.md5(headline.encode()).hexdigest()[:12]
        return cls(
            id=article_id,
            headline=headline,
            summary=summary,
            source=source,
            url=url,
            published_at=published_at or time.time(),
            assets=assets or [],
            raw=raw or {},
        )


class NewsCollector:
    """
    Agrège les headlines de plusieurs sources.

    Usage :
        collector = NewsCollector(cfg)
        articles = collector.fetch(tickers=["AAPL", "NVDA"], max_age_hours=24)
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.enabled_sources: list[str] = cfg.get("sources", ["finnhub", "rss"])
        self._cache: dict[str, list[RawArticle]] = {}
        self._cache_ts: dict[str, float] = {}
        self._cache_ttl = cfg.get("cache_ttl_seconds", 900)  # 15 min

    def fetch(
        self,
        tickers: list[str] | None = None,
        max_age_hours: float = 24.0,
    ) -> list[RawArticle]:
        """
        Récupère les articles pour les tickers donnés.
        Retourne une liste dédupliquée (par id) de RawArticle.
        """
        all_articles: dict[str, RawArticle] = {}
        cutoff = time.time() - max_age_hours * 3600

        for source in self.enabled_sources:
            try:
                articles = self._fetch_source(source, tickers, cutoff)
                for a in articles:
                    if a.id not in all_articles:
                        all_articles[a.id] = a
                logger.debug("Source %s: %d articles", source, len(articles))
            except Exception as exc:
                logger.warning("News source %s failed: %s", source, exc)

        result = sorted(all_articles.values(), key=lambda a: a.published_at, reverse=True)
        logger.info("NewsCollector: %d articles (sources=%s)", len(result), self.enabled_sources)
        return result

    # ------------------------------------------------------------------ #
    # Sources individuelles
    # ------------------------------------------------------------------ #

    def _fetch_source(
        self,
        source: str,
        tickers: list[str] | None,
        cutoff: float,
    ) -> list[RawArticle]:
        if source == "finnhub":
            return self._fetch_finnhub(tickers, cutoff)
        if source == "newsapi":
            return self._fetch_newsapi(tickers, cutoff)
        if source == "alpha_vantage":
            return self._fetch_alpha_vantage(tickers, cutoff)
        if source == "rss":
            return self._fetch_rss(tickers, cutoff)
        return []

    def _fetch_finnhub(
        self, tickers: list[str] | None, cutoff: float
    ) -> list[RawArticle]:
        api_key = os.environ.get("FINNHUB_API_KEY", self.cfg.get("finnhub_api_key", ""))
        if not api_key:
            logger.debug("FINNHUB_API_KEY not set — skipping Finnhub.")
            return []

        try:
            import requests
            articles = []
            date_from = time.strftime("%Y-%m-%d", time.gmtime(cutoff))
            date_to   = time.strftime("%Y-%m-%d", time.gmtime())

            # 1. News générales de marché (toujours disponible, plan gratuit)
            resp = requests.get(
                "https://finnhub.io/api/v1/news",
                params={"category": "general", "token": api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                for item in resp.json():
                    pub = float(item.get("datetime", 0))
                    if pub < cutoff:
                        continue
                    # Détecter les tickers mentionnés dans le titre
                    related = _detect_tickers(item.get("headline", ""), tickers or [])
                    articles.append(RawArticle.from_headline(
                        headline=item.get("headline", ""),
                        summary=item.get("summary", ""),
                        source="finnhub",
                        url=item.get("url", ""),
                        published_at=pub,
                        assets=related,
                        raw=item,
                    ))

            # 2. News par ticker (plan gratuit : données récentes limitées)
            for ticker in (tickers or [])[:3]:
                resp = requests.get(
                    "https://finnhub.io/api/v1/company-news",
                    params={"symbol": ticker, "from": date_from, "to": date_to, "token": api_key},
                    timeout=10,
                )
                if resp.status_code == 200:
                    for item in resp.json():
                        articles.append(RawArticle.from_headline(
                            headline=item.get("headline", ""),
                            summary=item.get("summary", ""),
                            source="finnhub",
                            url=item.get("url", ""),
                            published_at=float(item.get("datetime", 0)),
                            assets=[ticker],
                            raw=item,
                        ))

            logger.debug("Finnhub: %d articles fetched", len(articles))
            return articles
        except ImportError:
            logger.debug("requests not installed — cannot use Finnhub.")
            return []

    def _fetch_newsapi(
        self, tickers: list[str] | None, cutoff: float
    ) -> list[RawArticle]:
        api_key = os.environ.get("NEWSAPI_KEY", self.cfg.get("newsapi_key", ""))
        if not api_key:
            return []

        try:
            import requests
            query = " OR ".join(tickers[:5]) if tickers else "stock market finance"
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": api_key,
            }
            resp = requests.get(url, params=params, timeout=10)
            articles = []
            if resp.status_code == 200:
                for item in resp.json().get("articles", []):
                    articles.append(RawArticle.from_headline(
                        headline=item.get("title", ""),
                        summary=item.get("description", ""),
                        source="newsapi",
                        url=item.get("url", ""),
                        published_at=_parse_iso(item.get("publishedAt", "")),
                        assets=_detect_tickers(item.get("title", ""), tickers or []),
                        raw=item,
                    ))
            return articles
        except ImportError:
            return []

    def _fetch_alpha_vantage(
        self, tickers: list[str] | None, cutoff: float
    ) -> list[RawArticle]:
        api_key = os.environ.get("ALPHA_VANTAGE_KEY", self.cfg.get("alpha_vantage_key", ""))
        if not api_key:
            return []

        try:
            import requests
            tickers_str = ",".join((tickers or [])[:5])
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": tickers_str or "AAPL",
                "apikey": api_key,
                "limit": 20,
            }
            resp = requests.get(url, params=params, timeout=10)
            articles = []
            if resp.status_code == 200:
                feed = resp.json().get("feed", [])
                for item in feed:
                    articles.append(RawArticle.from_headline(
                        headline=item.get("title", ""),
                        summary=item.get("summary", ""),
                        source="alpha_vantage",
                        url=item.get("url", ""),
                        published_at=_parse_av_time(item.get("time_published", "")),
                        assets=[t["ticker"] for t in item.get("ticker_sentiment", []) if "ticker" in t],
                        raw=item,
                    ))
            return articles
        except ImportError:
            return []

    def _fetch_rss(
        self, tickers: list[str] | None, cutoff: float
    ) -> list[RawArticle]:
        """RSS : Yahoo Finance par ticker + flux marché généraux (sans API key)."""
        articles = []
        try:
            import requests
            from xml.etree import ElementTree as ET

            def _parse_feed(url: str, source: str, assets: list[str]) -> None:
                try:
                    resp = requests.get(url, timeout=8,
                                        headers={"User-Agent": "Mozilla/5.0"})
                    if resp.status_code != 200:
                        return
                    root = ET.fromstring(resp.text)
                    for item in root.findall(".//item"):
                        title = item.findtext("title") or ""
                        desc  = item.findtext("description") or ""
                        link  = item.findtext("link") or ""
                        pub   = _parse_rfc822(item.findtext("pubDate") or "")
                        if pub < cutoff or not title:
                            continue
                        articles.append(RawArticle.from_headline(
                            headline=title, summary=desc, source=source,
                            url=link, published_at=pub, assets=assets,
                        ))
                except Exception:
                    pass

            # Flux marché généraux — pas besoin d'API key
            _parse_feed(
                "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
                "rss_wsj", [],
            )
            _parse_feed(
                "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
                "rss_reuters", [],
            )
            _parse_feed(
                "https://feeds.content.dowjones.io/public/rss/mw_topstories",
                "rss_marketwatch", [],
            )

            # Flux Yahoo Finance par ticker (jusqu'à 8 tickers)
            ticker_rss = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"
            for ticker in (tickers or [])[:8]:
                _parse_feed(ticker_rss.format(t=ticker), "rss_yahoo", [ticker])

        except ImportError:
            pass

        logger.debug("RSS collector: %d articles bruts", len(articles))
        return articles


# ------------------------------------------------------------------ #
# Helpers de parsing de dates
# ------------------------------------------------------------------ #

def _parse_iso(s: str) -> float:
    """Parse ISO 8601 en timestamp UNIX."""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return time.time()


def _parse_av_time(s: str) -> float:
    """Parse le format Alpha Vantage (YYYYMMDDTHHmmss)."""
    try:
        from datetime import datetime
        dt = datetime.strptime(s, "%Y%m%dT%H%M%S")
        return dt.timestamp()
    except Exception:
        return time.time()


def _parse_rfc822(s: str) -> float:
    """Parse RFC 822 (format RSS pubDate)."""
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s).timestamp()
    except Exception:
        return time.time()


def _detect_tickers(text: str, candidates: list[str]) -> list[str]:
    """Détecte les tickers mentionnés dans un texte."""
    return [t for t in candidates if t.upper() in text.upper()]
