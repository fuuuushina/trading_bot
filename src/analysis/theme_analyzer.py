"""
src/analysis/theme_analyzer.py

Uses Groq LLM to analyze news articles and identify sector/thematic trends.
Produces a ThemeScore per sector (used by ThematicMomentumStrategy and the dashboard).

Refresh interval: 4 hours by default (configurable).
Falls back to keyword-based scoring when Groq is unavailable.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from src.analysis.sector_universe import SECTOR_UNIVERSE

logger = logging.getLogger(__name__)

_REFRESH_INTERVAL = 4 * 3600  # 4 hours


@dataclass
class ThemeScore:
    sector: str
    label: str
    score: float          # -1.0 to +1.0
    momentum: str         # "rising" | "falling" | "neutral"
    reason: str
    top_picks: list[str]  = field(default_factory=list)
    article_count: int    = 0

    def to_dict(self) -> dict:
        return {
            "sector":        self.sector,
            "label":         self.label,
            "score":         round(self.score, 3),
            "momentum":      self.momentum,
            "reason":        self.reason,
            "top_picks":     self.top_picks,
            "article_count": self.article_count,
        }


class ThemeAnalyzer:
    """
    Analyzes news articles to produce sector trend scores.

    Primary: Groq LLaMA (free tier).
    Fallback: keyword frequency scoring.
    """

    def __init__(self, groq_api_key: str = "", model: str = "llama-3.3-70b-versatile") -> None:
        self._api_key = groq_api_key
        self._model = model
        self._last_run: float = 0.0
        self._cached: dict[str, ThemeScore] = {}
        self._narrative: str = ""

    @property
    def narrative(self) -> str:
        return self._narrative

    @property
    def cached_scores(self) -> dict[str, ThemeScore]:
        return self._cached

    def analyze(
        self,
        articles: list,        # list[RawArticle] — or any object with .headline/.summary
        force: bool = False,
    ) -> dict[str, ThemeScore]:
        """
        Return {sector_key: ThemeScore} from recent news.
        Cached for 4 hours unless force=True.
        """
        now = time.time()
        if not force and (now - self._last_run) < _REFRESH_INTERVAL and self._cached:
            logger.debug("ThemeAnalyzer: returning cached scores (age=%.0fmin)",
                         (now - self._last_run) / 60)
            return self._cached

        if not articles:
            scores = self._neutral_scores()
            self._cached = scores
            return scores

        if self._api_key:
            try:
                scores = self._analyze_with_groq(articles)
                if scores:
                    self._cached = scores
                    self._last_run = now
                    logger.info(
                        "ThemeAnalyzer: Groq analysis done — top sector: %s",
                        max(scores, key=lambda k: scores[k].score)
                    )
                    return scores
            except Exception as exc:
                logger.warning("ThemeAnalyzer Groq failed: %s — using keyword fallback", exc)

        scores = self._keyword_scores(articles)
        self._cached = scores
        self._last_run = now
        return scores

    # ------------------------------------------------------------------ #
    # Groq analysis
    # ------------------------------------------------------------------ #

    def _analyze_with_groq(self, articles: list) -> dict[str, ThemeScore]:
        from groq import Groq  # lazy import

        news_lines = [f"- {a.headline}" for a in articles[:60]]
        news_text = "\n".join(news_lines) if news_lines else "No articles available."

        sector_list = "\n".join(
            f"- {k}: {v['description']}" for k, v in SECTOR_UNIVERSE.items()
        )
        available_picks = {k: v["tickers"][:5] for k, v in SECTOR_UNIVERSE.items()}

        prompt = f"""Tu es un analyste financier senior spécialisé en analyse sectorielle.

Analyse les actualités financières suivantes et évalue la tendance attendue pour chaque secteur sur les 1 à 4 prochaines semaines.

SECTEURS À ANALYSER:
{sector_list}

ACTUALITÉS RÉCENTES:
{news_text}

TICKERS DISPONIBLES PAR SECTEUR (utilise uniquement ceux-là pour top_picks):
{json.dumps(available_picks, indent=2)}

Règles de scoring (-1.0 à +1.0):
  +0.6 à +1.0 : tendance haussière forte et confirmée
  +0.3 à +0.6 : légèrement positif, momentum naissant
  -0.1 à +0.3 : neutre, pas de signal clair
  -0.6 à -0.1 : légèrement négatif
  -1.0 à -0.6 : tendance baissière confirmée

Réponds UNIQUEMENT en JSON valide (sans texte avant ou après):
{{
  "sectors": {{
    "pharma_biotech":    {{"score": 0.0, "momentum": "neutral", "reason": "...", "top_picks": []}},
    "ai_software":       {{"score": 0.0, "momentum": "neutral", "reason": "...", "top_picks": []}},
    "semiconductors":    {{"score": 0.0, "momentum": "neutral", "reason": "...", "top_picks": []}},
    "energy":            {{"score": 0.0, "momentum": "neutral", "reason": "...", "top_picks": []}},
    "fintech_banking":   {{"score": 0.0, "momentum": "neutral", "reason": "...", "top_picks": []}},
    "ev_clean_energy":   {{"score": 0.0, "momentum": "neutral", "reason": "...", "top_picks": []}},
    "defense_aerospace": {{"score": 0.0, "momentum": "neutral", "reason": "...", "top_picks": []}},
    "retail_consumer":   {{"score": 0.0, "momentum": "neutral", "reason": "...", "top_picks": []}}
  }},
  "narrative": "Résumé global des grandes tendances sectorielles en 2-3 phrases."
}}"""

        client = Groq(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            max_tokens=1400,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                if part.startswith("json"):
                    raw = part[4:].strip()
                    break
                if part.strip().startswith("{"):
                    raw = part.strip()
                    break

        data = json.loads(raw)
        self._narrative = data.get("narrative", "")

        result: dict[str, ThemeScore] = {}
        sectors_data = data.get("sectors", {})
        n_articles = len(articles)

        for sector_key, info in SECTOR_UNIVERSE.items():
            sd = sectors_data.get(sector_key, {})
            raw_score = float(sd.get("score", 0.0))
            momentum   = sd.get("momentum", "neutral")
            reason     = sd.get("reason", "Pas d'information significative.")
            raw_picks  = sd.get("top_picks", [])
            picks = [t for t in raw_picks if t in info["tickers"]]
            if not picks:
                picks = info["tickers"][:3]

            result[sector_key] = ThemeScore(
                sector=sector_key,
                label=info["label"],
                score=max(-1.0, min(1.0, raw_score)),
                momentum=momentum,
                reason=reason,
                top_picks=picks,
                article_count=n_articles,
            )

        return result

    # ------------------------------------------------------------------ #
    # Keyword fallback
    # ------------------------------------------------------------------ #

    def _keyword_scores(self, articles: list) -> dict[str, ThemeScore]:
        """Count keyword hits per sector as a rough sentiment proxy."""
        hits: dict[str, int] = {k: 0 for k in SECTOR_UNIVERSE}
        total = len(articles)

        for article in articles:
            text = (getattr(article, "headline", "") + " " +
                    getattr(article, "summary", "")).lower()
            for sector_key, info in SECTOR_UNIVERSE.items():
                if any(kw in text for kw in info["keywords"]):
                    hits[sector_key] += 1

        result: dict[str, ThemeScore] = {}
        for sector_key, count in hits.items():
            info = SECTOR_UNIVERSE[sector_key]
            # More coverage → more attention (cap at 0.55 — keyword method is uncertain)
            raw = count / max(total * 0.25, 1)
            score = round(min(raw, 0.55), 3)
            result[sector_key] = ThemeScore(
                sector=sector_key,
                label=info["label"],
                score=score,
                momentum="rising" if score > 0.30 else "neutral",
                reason=f"{count} articles avec des mots-clés {sector_key.replace('_', ' ')}.",
                top_picks=info["tickers"][:3],
                article_count=total,
            )
        return result

    def _neutral_scores(self) -> dict[str, ThemeScore]:
        return {
            k: ThemeScore(
                sector=k,
                label=v["label"],
                score=0.0,
                momentum="neutral",
                reason="Aucune donnée news disponible.",
                top_picks=v["tickers"][:3],
            )
            for k, v in SECTOR_UNIVERSE.items()
        }
