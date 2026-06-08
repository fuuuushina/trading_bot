"""
src/news/classifier.py

News Classifier + Sentiment Scorer.

Pipeline :
  RawArticle → clean → classify topics → score sentiment → NewsImpact

Deux modes de scoring :
  1. Keyword-based (rapide, sans API, bonne baseline)
  2. LLM-based (Claude, optionnel, activé via config)

Sortie par article : NewsImpact avec sentiment, risk_score, topics, impact_label.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.news.collector import RawArticle

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Topics financiers reconnus
# ------------------------------------------------------------------ #

# Mapping yfinance forex tickers → keywords to find in article text
FOREX_TICKER_KEYWORDS: dict[str, list[str]] = {
    "EURUSD=X": ["eur/usd", "eurusd", "euro", "eur usd", "euro dollar", "ecb", "european central bank"],
    "GBPUSD=X": ["gbp/usd", "gbpusd", "pound", "sterling", "bank of england"],
    "USDJPY=X": ["usd/jpy", "usdjpy", "yen", "boj", "bank of japan"],
    "USDCHF=X": ["usd/chf", "usdchf", "franc", "snb"],
    "DX-Y.NYB": ["dollar index", "dxy", "us dollar", "usd index"],
}

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "earnings":        ["earnings", "eps", "revenue", "profit", "loss", "beat", "miss", "guidance"],
    "fed_policy":      ["fed", "federal reserve", "interest rate", "fomc", "powell", "rate hike", "rate cut"],
    "macro":           ["gdp", "inflation", "cpi", "unemployment", "jobs", "recession", "growth"],
    "geopolitical":    ["war", "conflict", "sanctions", "trade war", "tariff", "china", "russia"],
    "ai_tech":         ["ai", "artificial intelligence", "chip", "gpu", "data center", "llm", "nvidia"],
    "chip_export":     ["export", "export control", "ban", "restriction", "semiconductor"],
    "merger_acq":      ["merger", "acquisition", "deal", "takeover", "buyout"],
    "analyst":         ["upgrade", "downgrade", "buy", "sell", "hold", "price target", "rating"],
    "crypto":          ["bitcoin", "ethereum", "crypto", "blockchain", "defi", "nft"],
    "regulatory":      ["sec", "regulation", "lawsuit", "fine", "probe", "investigation"],
    "dividend":        ["dividend", "yield", "payout", "buyback", "share repurchase"],
    "ipo":             ["ipo", "initial public offering", "listing", "debut"],
}

# ------------------------------------------------------------------ #
# Dictionnaires de sentiment
# ------------------------------------------------------------------ #

_POSITIVE_WORDS = {
    "beat", "surpassed", "exceeded", "strong", "growth", "rally", "surge",
    "upgrade", "buy", "outperform", "record", "profit", "gain", "rise",
    "positive", "bullish", "recovery", "momentum", "opportunity", "breakthrough",
    "innovation", "expansion", "upside", "higher", "increase",
}

_NEGATIVE_WORDS = {
    "miss", "missed", "below", "weak", "decline", "fall", "drop", "plunge",
    "downgrade", "sell", "underperform", "loss", "concern", "risk", "warning",
    "negative", "bearish", "recession", "layoff", "cut", "lower", "decrease",
    "probe", "lawsuit", "sanction", "ban", "restriction", "fear",
}

_HIGH_RISK_WORDS = {
    "ban", "sanction", "lawsuit", "investigation", "crash", "halt",
    "bankruptcy", "default", "fraud", "scandal", "recall", "warning",
    "geopolitical", "war", "conflict",
}


# ------------------------------------------------------------------ #
# NewsImpact — sortie du classifier
# ------------------------------------------------------------------ #

@dataclass
class NewsImpact:
    """Impact d'un ensemble de news sur un asset."""
    asset: str
    sentiment: float       # -1.0 (très négatif) à +1.0 (très positif)
    risk_score: float      # 0.0 à 1.0
    topics: list[str]      = field(default_factory=list)
    impact: str            = "neutral"  # positive_high/medium/low | neutral | negative_low/medium/high
    headline_count: int    = 0
    source: str            = "keyword"
    headlines: list[str]   = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "sentiment": round(self.sentiment, 3),
            "risk_score": round(self.risk_score, 3),
            "topics": self.topics,
            "impact": self.impact,
            "headline_count": self.headline_count,
            "source": self.source,
        }


# ------------------------------------------------------------------ #
# Classifier
# ------------------------------------------------------------------ #

class NewsClassifier:
    """
    Classe + score les articles par asset.

    Usage :
        classifier = NewsClassifier(cfg)
        impacts = classifier.classify(articles, universe=["AAPL", "NVDA"])
        # → dict[str, NewsImpact]
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._use_llm = cfg.get("use_llm_classification", False)
        self._ai_cfg = cfg.get("ai", {})

    def classify(
        self,
        articles: list[RawArticle],
        universe: list[str],
    ) -> dict[str, NewsImpact]:
        """
        Groupe les articles par asset et calcule un NewsImpact par asset.

        Returns
        -------
        Dict {ticker: NewsImpact}
        """
        # Associer chaque article aux assets de l'univers
        asset_articles: dict[str, list[RawArticle]] = {t: [] for t in universe}
        for article in articles:
            assigned = set(article.assets) & set(universe)
            # Si aucun ticker connu, essayer de détecter via le texte
            if not assigned:
                text = article.headline + " " + article.summary
                text_lower = text.lower()
                # Standard ticker match (e.g. "AAPL", "SPY")
                assigned = {t for t in universe if t.upper() in text.upper()}
                # Forex tickers need keyword matching (articles never contain "EURUSD=X")
                for t in universe:
                    if t not in assigned and t in FOREX_TICKER_KEYWORDS:
                        if any(kw in text_lower for kw in FOREX_TICKER_KEYWORDS[t]):
                            assigned.add(t)
            for ticker in assigned:
                asset_articles[ticker].append(article)

        impacts: dict[str, NewsImpact] = {}
        for ticker, arts in asset_articles.items():
            if not arts:
                continue
            impacts[ticker] = self._score_asset(ticker, arts)

        logger.info(
            "NewsClassifier: %d assets with news (total articles=%d)",
            len(impacts), len(articles),
        )
        return impacts

    def _score_asset(self, ticker: str, articles: list[RawArticle]) -> NewsImpact:
        """Calcule le score de sentiment et de risque pour un asset."""
        sentiments = []
        risk_scores = []
        all_topics: set[str] = set()
        headlines = []

        for article in articles:
            text = (article.headline + " " + article.summary).lower()
            sent, risk = _keyword_score(text)
            topics = _detect_topics(text)
            sentiments.append(sent)
            risk_scores.append(risk)
            all_topics.update(topics)
            headlines.append(article.headline)

        avg_sentiment = sum(sentiments) / len(sentiments)
        max_risk = max(risk_scores)
        avg_risk = sum(risk_scores) / len(risk_scores)
        final_risk = 0.7 * max_risk + 0.3 * avg_risk

        impact_label = _label_impact(avg_sentiment, final_risk)

        return NewsImpact(
            asset=ticker,
            sentiment=round(avg_sentiment, 3),
            risk_score=round(final_risk, 3),
            topics=sorted(all_topics),
            impact=impact_label,
            headline_count=len(articles),
            source="keyword",
            headlines=headlines[:5],
        )


# ------------------------------------------------------------------ #
# Helpers de scoring
# ------------------------------------------------------------------ #

def _keyword_score(text: str) -> tuple[float, float]:
    """
    Retourne (sentiment -1→+1, risk_score 0→1) via comptage de mots-clés.
    """
    words = set(re.findall(r"\b\w+\b", text.lower()))

    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    risk_hits = len(words & _HIGH_RISK_WORDS)

    total = pos + neg
    sentiment = (pos - neg) / max(total, 1)
    risk = min(risk_hits * 0.20 + (neg / max(total, 1)) * 0.5, 1.0)

    return round(sentiment, 3), round(risk, 3)


def _detect_topics(text: str) -> list[str]:
    """Retourne les topics présents dans le texte."""
    found = []
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(topic)
    return found


def _label_impact(sentiment: float, risk: float) -> str:
    """Produit un label lisible basé sur sentiment et risque."""
    if risk > 0.70:
        return "negative_high"
    if sentiment > 0.50:
        return "positive_high"
    if sentiment > 0.20:
        return "positive_medium"
    if sentiment > 0.05:
        return "positive_low"
    if sentiment < -0.50:
        return "negative_high"
    if sentiment < -0.20:
        return "negative_medium"
    if sentiment < -0.05:
        return "negative_low"
    return "neutral"
