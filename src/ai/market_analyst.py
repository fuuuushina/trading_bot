"""
src/ai/market_analyst.py

LLM Market Analyst — analyse de marché par LLM.

Configurable : Claude (Anthropic, défaut) ou Qwen (via API compatible OpenAI).

Trois modes d'utilisation selon la fréquence :
  1. NEWS_QUICK   (toutes les 30-60 min) : résumé news + sentiment par asset
  2. MARKET_SCAN  (toutes les 4-6h)      : scan régime + risques macro
  3. FULL_STRATEGY (1x/jour)             : analyse complète + recommandations

L'analyste est ADVISORY ONLY — il ne passe jamais d'ordres.

Sortie : MarketAnalysis structurée
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AnalysisMode(str, Enum):
    NEWS_QUICK    = "news_quick"
    MARKET_SCAN   = "market_scan"
    FULL_STRATEGY = "full_strategy"


@dataclass
class MarketAnalysis:
    """Analyse produite par le LLM Market Analyst."""
    mode: str
    market: str                          # "US_EQUITIES" | "FOREX" | "MIXED"
    regime: str                          # label régime
    risk_level: str                      # "low" | "medium" | "high" | "extreme"
    trend: str                           # "positive" | "negative" | "neutral"
    recommended_exposure: float          # 0.0 – 1.0
    vix_assessment: str = ""            # commentaire sur VIX
    key_risks: list[str] = field(default_factory=list)
    opportunities: list[str] = field(default_factory=list)
    asset_notes: dict[str, str] = field(default_factory=dict)  # {ticker: note}
    summary: str = ""
    computed_at: float = field(default_factory=time.time)
    provider: str = "rules"             # "claude" | "qwen" | "rules"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "market": self.market,
            "regime": self.regime,
            "risk_level": self.risk_level,
            "trend": self.trend,
            "recommended_exposure": round(self.recommended_exposure, 3),
            "vix_assessment": self.vix_assessment,
            "key_risks": self.key_risks,
            "opportunities": self.opportunities,
            "asset_notes": self.asset_notes,
            "summary": self.summary,
            "provider": self.provider,
        }


# ------------------------------------------------------------------ #
# Schéma JSON attendu du LLM
# ------------------------------------------------------------------ #

_OUTPUT_SCHEMA = {
    "market": "US_EQUITIES | FOREX | CRYPTO | MIXED",
    "regime": "bull_trend | bear_trend | range | high_volatility | low_volatility | panic | euphoric | compression",
    "risk_level": "low | medium | high | extreme",
    "trend": "positive | negative | neutral",
    "recommended_exposure": "float 0.0-1.0",
    "vix_assessment": "string",
    "key_risks": ["string"],
    "opportunities": ["string"],
    "asset_notes": {"ticker": "note string"},
    "summary": "string max 200 chars",
}

_SYSTEM_PROMPT = """
Tu es un analyste de marché algorithmique. Tu fournis des analyses structurées
pour un système de trading automatique. Tu es ADVISORY ONLY — tu ne passes jamais d'ordres.

Sois concis, factuel, et conservateur par défaut.
Ta réponse DOIT être un JSON valide correspondant exactement à ce schéma :
{schema}

N'ajoute aucun texte en dehors du JSON.
""".strip()

_NEWS_PROMPT = """
Analyse rapide (30 min) basée sur les données suivantes :

Régime détecté (règles ML) : {regime}
VIX : {vix}
Assets surveillés : {tickers}

News récentes (résumé) :
{news_summary}

Features clés :
{features_summary}

Produis une analyse rapide du risque et du sentiment courant.
""".strip()

_MARKET_SCAN_PROMPT = """
Scan de marché complet.

Profil utilisateur : {profile_label} | Capital : {capital} | Drawdown max : {max_dd}
Régime actuel (ML) : {regime} (conf={confidence:.0%})
VIX : {vix} ({vix_trend})

Features par asset :
{features_table}

News récentes (sentiment) :
{news_items}

Donne une analyse du régime macro, des risques principaux, et une recommandation d'exposition.
""".strip()

_FULL_STRATEGY_PROMPT = """
Analyse stratégique quotidienne complète.

=== PROFIL ===
{profile_json}

=== MARCHÉ ===
Régime : {regime} | VIX : {vix} | Tendance SPY 20j : {spy_change_20d:+.1%}
SPY vs MA200 : {spy_vs_ma200:+.1%} | RSI14 : {spy_rsi:.1f} | ADX : {spy_adx:.1f}

=== FEATURES PAR ASSET ===
{features_table}

=== NEWS (dernières 24h) ===
{news_summary}

=== PORTEFEUILLE ACTUEL ===
Capital : ${capital:,.0f} | Exposition : {exposure:.0%} | Drawdown : {drawdown:+.1%}
Positions ouvertes : {open_positions}

Fournis :
1. Évaluation du régime et de la tendance
2. Principaux risques (2-3 maximum)
3. Opportunités identifiées
4. Recommandation d'exposition (0.0 - 1.0)
5. Notes par asset si pertinentes
6. Résumé en 1 phrase
""".strip()


class MarketAnalyst:
    """
    Analyste LLM. Supporte Claude (Anthropic) et Qwen (API OpenAI-compatible).

    Configuration dans settings.yaml (section market_analyst) :
      provider: "claude" | "qwen"
      model: "claude-sonnet-4-6" | "qwen-plus" | etc.
      qwen_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

    Fréquences recommandées :
      NEWS_QUICK   → toutes les 30-60 min (coût minimal)
      MARKET_SCAN  → toutes les 4-6h
      FULL_STRATEGY→ 1 fois par jour
    """

    def __init__(self, cfg: dict) -> None:
        self.enabled = cfg.get("enabled", False)
        self.provider = cfg.get("provider", "claude")
        self.model = cfg.get("model", "claude-sonnet-4-6")
        self.qwen_base_url = cfg.get("qwen_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.max_tokens = cfg.get("max_tokens", 800)
        self.temperature = cfg.get("temperature", 0.2)
        self.timeout = cfg.get("timeout_seconds", 20)

        # Cache pour éviter les appels répétés
        self._last_analysis: Optional[MarketAnalysis] = None
        self._last_analysis_ts: float = 0.0

    # ------------------------------------------------------------------ #
    # Interface publique
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        mode: AnalysisMode,
        regime: str,
        features_snapshot,          # MarketSnapshot
        profile_dict: dict,
        portfolio_state: dict,
        news_summary: str = "",
        force: bool = False,
    ) -> MarketAnalysis:
        """
        Lance une analyse LLM selon le mode.

        Si le provider est indisponible, produit une analyse rule-based de fallback.
        """
        if not self.enabled or not self._has_api_key():
            return self._rule_based_analysis(regime, features_snapshot, profile_dict, portfolio_state)

        try:
            prompt = self._build_prompt(
                mode, regime, features_snapshot, profile_dict, portfolio_state, news_summary
            )
            raw = self._call_llm(prompt)
            if raw:
                analysis = self._parse_response(raw, mode)
                if analysis:
                    self._last_analysis = analysis
                    self._last_analysis_ts = time.time()
                    logger.info(
                        "MarketAnalyst (%s/%s): regime=%s risk=%s exposure=%.0f%%",
                        self.provider, mode.value,
                        analysis.regime, analysis.risk_level,
                        analysis.recommended_exposure * 100,
                    )
                    return analysis
        except Exception as exc:
            logger.warning("MarketAnalyst LLM failed: %s — using rule-based fallback", exc)

        return self._rule_based_analysis(regime, features_snapshot, profile_dict, portfolio_state)

    @property
    def last_analysis(self) -> Optional[MarketAnalysis]:
        return self._last_analysis

    @property
    def last_analysis_age_minutes(self) -> float:
        if self._last_analysis_ts == 0:
            return float("inf")
        return (time.time() - self._last_analysis_ts) / 60.0

    # ------------------------------------------------------------------ #
    # Construction du prompt
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        mode: AnalysisMode,
        regime: str,
        snapshot,
        profile_dict: dict,
        portfolio_state: dict,
        news_summary: str,
    ) -> str:
        bench = snapshot.benchmark_features() if snapshot else None

        features_table = self._features_table(snapshot)

        if mode == AnalysisMode.NEWS_QUICK:
            return _NEWS_PROMPT.format(
                regime=regime,
                vix=f"{snapshot.vix_level:.1f}" if snapshot and snapshot.vix_level else "N/A",
                tickers=", ".join(list(snapshot.assets.keys())[:8]) if snapshot else "",
                news_summary=news_summary or "(pas de news disponibles)",
                features_summary=features_table[:500],
            )

        if mode == AnalysisMode.MARKET_SCAN:
            return _MARKET_SCAN_PROMPT.format(
                profile_label=profile_dict.get("risk_tolerance", "moderate"),
                capital=portfolio_state.get("total_capital", 0),
                max_dd=profile_dict.get("max_drawdown_tolerance", 0.20),
                regime=regime,
                confidence=0.75,
                vix=f"{snapshot.vix_level:.1f}" if snapshot and snapshot.vix_level else "N/A",
                vix_trend=bench.vix_trend or "stable" if bench else "stable",
                features_table=features_table,
                news_items=news_summary or "(pas de news)",
            )

        # FULL_STRATEGY
        import math as _m

        def _fv(v, default=0.0):
            """Safe float: NaN/None/Inf → default."""
            try:
                v = float(v)
                return v if _m.isfinite(v) else default
            except Exception:
                return default

        spy = snapshot.get("SPY") if snapshot else None
        return _FULL_STRATEGY_PROMPT.format(
            profile_json=json.dumps(profile_dict, indent=2, ensure_ascii=False)[:600],
            regime=regime,
            vix=f"{snapshot.vix_level:.1f}" if snapshot and snapshot.vix_level else "N/A",
            spy_change_20d=_fv(spy.change_20d) if spy else 0.0,
            spy_vs_ma200=_fv(spy.price_vs_ma200_pct) if spy else 0.0,
            spy_rsi=_fv(spy.rsi14, 50.0) if spy else 50.0,
            spy_adx=_fv(spy.adx14, 20.0) if spy else 20.0,
            features_table=features_table,
            news_summary=news_summary or "(pas de news)",
            capital=_fv(portfolio_state.get("total_capital", 0)),
            exposure=_fv(portfolio_state.get("total_exposure", 0)) / max(_fv(portfolio_state.get("total_capital", 1), 1), 1),
            drawdown=_fv(portfolio_state.get("drawdown_pct", 0.0)),
            open_positions=portfolio_state.get("open_positions", 0),
        )

    # ------------------------------------------------------------------ #
    # Appels LLM
    # ------------------------------------------------------------------ #

    def _call_llm(self, user_prompt: str) -> Optional[str]:
        system = _SYSTEM_PROMPT.format(schema=json.dumps(_OUTPUT_SCHEMA, indent=2, ensure_ascii=False))

        if self.provider == "claude":
            return self._call_claude(system, user_prompt)
        if self.provider == "qwen":
            return self._call_qwen(system, user_prompt)
        if self.provider == "groq":
            return self._call_groq(system, user_prompt)
        return None

    def _call_claude(self, system: str, user: str) -> Optional[str]:
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
        except Exception as exc:
            logger.debug("Claude call failed: %s", exc)
            return None

    def _call_qwen(self, system: str, user: str) -> Optional[str]:
        """Qwen via API compatible OpenAI (Alibaba DashScope)."""
        try:
            import requests
            api_key = os.environ.get("QWEN_API_KEY", os.environ.get("DASHSCOPE_API_KEY", ""))
            if not api_key:
                return None
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            resp = requests.post(
                f"{self.qwen_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            logger.debug("Qwen API error %d: %s", resp.status_code, resp.text[:200])
            return None
        except Exception as exc:
            logger.debug("Qwen call failed: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Parse + fallback
    # ------------------------------------------------------------------ #

    def _parse_response(self, raw: str, mode: AnalysisMode) -> Optional[MarketAnalysis]:
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)

            exposure = max(0.0, min(1.0, float(data.get("recommended_exposure", 0.5))))
            risk = data.get("risk_level", "medium")
            if risk not in ("low", "medium", "high", "extreme"):
                risk = "medium"

            return MarketAnalysis(
                mode=mode.value,
                market=str(data.get("market", "US_EQUITIES")),
                regime=str(data.get("regime", "unknown")),
                risk_level=risk,
                trend=str(data.get("trend", "neutral")),
                recommended_exposure=exposure,
                vix_assessment=str(data.get("vix_assessment", "")),
                key_risks=list(data.get("key_risks", [])),
                opportunities=list(data.get("opportunities", [])),
                asset_notes=dict(data.get("asset_notes", {})),
                summary=str(data.get("summary", ""))[:300],
                provider=self.provider,
            )
        except Exception as exc:
            logger.warning("Failed to parse LLM response: %s | raw=%s", exc, raw[:200])
            return None

    def _rule_based_analysis(
        self,
        regime: str,
        snapshot,
        profile_dict: dict,
        portfolio_state: dict,
    ) -> MarketAnalysis:
        """Analyse deterministe de fallback (sans LLM)."""
        bench = snapshot.benchmark_features() if snapshot else None
        vix = snapshot.vix_level if snapshot else None

        # Niveau de risque basé sur VIX + régime
        risk_level = "medium"
        if vix is not None:
            if vix > 30:
                risk_level = "high"
            elif vix > 40:
                risk_level = "extreme"
            elif vix < 15:
                risk_level = "low"

        if regime in ("panic", "high_volatility"):
            risk_level = "high" if risk_level != "extreme" else "extreme"
        elif regime in ("bull_trend", "low_volatility"):
            risk_level = "low" if vix and vix < 18 else "medium"

        # Exposition recommandée
        exposure_map = {
            "low": 0.80, "medium": 0.65, "high": 0.40, "extreme": 0.20
        }
        exposure = exposure_map.get(risk_level, 0.60)

        trend = "neutral"
        if bench:
            if bench.price_vs_ma200_pct > 0.03 and bench.change_20d > 0:
                trend = "positive"
            elif bench.price_vs_ma200_pct < -0.03 or bench.change_20d < -0.05:
                trend = "negative"

        return MarketAnalysis(
            mode="rule_based",
            market="US_EQUITIES",
            regime=regime,
            risk_level=risk_level,
            trend=trend,
            recommended_exposure=exposure,
            vix_assessment=f"VIX={vix:.1f}" if vix else "VIX N/A",
            key_risks=[],
            opportunities=[],
            summary=f"Analyse rule-based : {regime}, risque {risk_level}",
            provider="rules",
        )

    def _call_groq(self, system: str, user: str) -> Optional[str]:
        """Groq via API compatible OpenAI — gratuit, ultra-rapide."""
        try:
            import requests
            api_key = os.environ.get("GROQ_API_KEY", "")
            if not api_key:
                return None
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            logger.debug("Groq API error %d: %s", resp.status_code, resp.text[:200])
            return None
        except Exception as exc:
            logger.debug("Groq call failed: %s", exc)
            return None

    def _has_api_key(self) -> bool:
        if self.provider == "claude":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.provider == "qwen":
            return bool(os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))
        if self.provider == "groq":
            return bool(os.environ.get("GROQ_API_KEY"))
        return False

    @staticmethod
    def _features_table(snapshot) -> str:
        if snapshot is None:
            return "(no data)"
        import math as _m

        def _sf(v, fmt: str) -> str:
            try:
                v = float(v)
                return format(v if _m.isfinite(v) else 0.0, fmt)
            except Exception:
                return "?"

        rows = []
        for ticker, feat in list(snapshot.assets.items())[:8]:
            try:
                rows.append(
                    f"{ticker:<8} close={_sf(feat.close, '.2f')} "
                    f"1d={_sf(feat.change_1d, '+.1%')} 20d={_sf(feat.change_20d, '+.1%')} "
                    f"RSI={_sf(feat.rsi14, '.0f')} ADX={_sf(feat.adx14, '.0f')} "
                    f"vs_MA200={_sf(feat.price_vs_ma200_pct, '+.1%')}"
                )
            except Exception:
                rows.append(f"{ticker:<8} (données indisponibles)")
        return "\n".join(rows)
