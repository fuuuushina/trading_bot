"""
src/engine/signal_aggregator.py

Signal Aggregator — reçoit tous les signaux bruts des stratégies/bots,
les groupe par asset, résout les conflits et produit un signal consolidé
par asset (et par direction) avant transmission au Risk Manager.

Logique d'agrégation :
  - Plusieurs BUY sur le même asset  → confidence compound (boost accord)
  - BUY + SELL en conflit            → direction dominante, confiance réduite
  - N bots en accord                 → boost de +10% par bot supplémentaire (max +25%)
  - Dissension                       → pénalité de -20% par bot opposé
  - Score final < min_confidence     → signal écarté (NO_TRADE)

Le hook `apply_news_override()` permet à un futur News Layer d'injecter
un score de risque externe qui capera la confidence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.strategies.base import Horizon, Signal, SignalType, no_trade

logger = logging.getLogger(__name__)

# Boost de confidence par bot supplémentaire en accord
_AGREEMENT_BOOST_PER_BOT = 0.10
_MAX_AGREEMENT_BOOST = 0.25

# Pénalité par bot en désaccord
_DISSENT_PENALTY_PER_BOT = 0.20

# Confidence minimale pour qu'un signal agrégé soit transmis
_DEFAULT_MIN_CONFIDENCE = 0.45


@dataclass
class BotContribution:
    """Trace la contribution d'un bot/stratégie individuel."""
    strategy_name: str
    signal_type: str          # BUY / SELL / EXIT
    confidence: float
    reason: str
    stop_loss: Optional[float]
    take_profit: Optional[float]
    entry_price: Optional[float]
    risk_reward: Optional[float]


@dataclass
class AggregatedSignal:
    """
    Signal consolidé issu de l'agrégation de plusieurs signaux.
    Contient les métadonnées de tous les bots contributeurs.
    """
    asset: str
    action: SignalType
    confidence: float          # 0.0 – 1.0, après agrégation
    horizon: Horizon
    timeframe: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward: Optional[float]
    contributors: list[BotContribution] = field(default_factory=list)
    n_agreeing: int = 0
    n_dissenting: int = 0
    reasons: list[str] = field(default_factory=list)
    news_risk_override: Optional[float] = None  # Injecté par le News Layer

    @property
    def combined_reason(self) -> str:
        bots = ", ".join(c.strategy_name for c in self.contributors)
        base = f"[{self.n_agreeing} bots en accord: {bots}]"
        if self.n_dissenting:
            base += f" [{self.n_dissenting} en désaccord]"
        if self.reasons:
            base += " | " + " / ".join(self.reasons[:3])
        return base

    def to_signal(self) -> Signal:
        """Convertit en Signal standard pour le pipeline Rules→AI→Risk."""
        # Nom de stratégie : si 1 contributeur → son nom, sinon "strat1+strat2"
        if len(self.contributors) == 1:
            strategy_name = self.contributors[0].strategy_name
        else:
            names = [c.strategy_name for c in self.contributors[:3]]
            strategy_name = "+".join(names)

        meta: dict = {
            "aggregated": True,
            "n_agreeing": self.n_agreeing,
            "n_dissenting": self.n_dissenting,
            "contributors": [
                {
                    "strategy": c.strategy_name,
                    "signal": c.signal_type,
                    "confidence": round(c.confidence, 3),
                }
                for c in self.contributors
            ],
        }
        if self.news_risk_override is not None:
            meta["news_risk_override"] = self.news_risk_override

        return Signal(
            strategy_name=strategy_name,
            asset=self.asset,
            timeframe=self.timeframe,
            signal=self.action,
            confidence=round(self.confidence, 4),
            entry_price=self.entry_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            risk_reward=self.risk_reward,
            horizon=self.horizon,
            reason=self.combined_reason,
            metadata=meta,
        )


class SignalAggregator:
    """
    Agrège une liste de signaux bruts en signaux consolidés par asset.

    Usage :
        aggregator = SignalAggregator()
        aggregated_signals = aggregator.aggregate(raw_signals)
        # → list[Signal] prêts pour le Decision Engine
    """

    def __init__(self, min_confidence: float = _DEFAULT_MIN_CONFIDENCE) -> None:
        self.min_confidence = min_confidence
        # Slot pour injection News Layer : {asset: risk_score 0.0-1.0}
        self._news_risk_scores: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Interface publique
    # ------------------------------------------------------------------ #

    def aggregate(self, signals: list[Signal]) -> list[Signal]:
        """
        Agrège les signaux bruts et retourne des Signal standards.

        Parameters
        ----------
        signals : Signaux bruts de toutes les stratégies (NO_TRADE exclus).

        Returns
        -------
        Liste de Signal agrégés (un par asset × direction).
        """
        if not signals:
            return []

        # Grouper par (asset, horizon)
        groups: dict[tuple[str, Horizon], list[Signal]] = {}
        for sig in signals:
            if sig.signal == SignalType.NO_TRADE:
                continue
            key = (sig.asset, sig.horizon)
            groups.setdefault(key, []).append(sig)

        results: list[Signal] = []
        for (asset, horizon), group in groups.items():
            agg = self._aggregate_group(asset, horizon, group)
            if agg is not None:
                results.append(agg.to_signal())
                logger.info(
                    "Aggregated: %s %s conf=%.2f (%d agreeing, %d dissenting)",
                    asset, agg.action.value, agg.confidence,
                    agg.n_agreeing, agg.n_dissenting,
                )

        logger.debug(
            "SignalAggregator: %d raw → %d aggregated (min_conf=%.2f)",
            len(signals), len(results), self.min_confidence,
        )
        return results

    def apply_news_override(self, risk_scores: dict[str, float]) -> None:
        """
        Injecte des scores de risque depuis le News Layer.
        risk_scores : {ticker: 0.0 (aucun risque) … 1.0 (risque max)}
        Appelé par le News Manager avant chaque cycle.
        """
        self._news_risk_scores = {k: max(0.0, min(1.0, v)) for k, v in risk_scores.items()}
        logger.debug("News risk scores updated: %s", self._news_risk_scores)

    # ------------------------------------------------------------------ #
    # Logique d'agrégation interne
    # ------------------------------------------------------------------ #

    def _aggregate_group(
        self,
        asset: str,
        horizon: Horizon,
        signals: list[Signal],
    ) -> Optional[AggregatedSignal]:
        """Agrège un groupe de signaux pour un même (asset, horizon)."""

        # Séparer BUY/EXIT vs SELL
        buy_signals  = [s for s in signals if s.signal in (SignalType.BUY, SignalType.HOLD)]
        sell_signals = [s for s in signals if s.signal in (SignalType.SELL, SignalType.EXIT)]

        # Déterminer direction dominante
        buy_weight  = sum(s.confidence for s in buy_signals)
        sell_weight = sum(s.confidence for s in sell_signals)

        if buy_weight >= sell_weight:
            dominant_signals  = buy_signals
            dissent_signals   = sell_signals
            dominant_action   = SignalType.BUY
        else:
            dominant_signals  = sell_signals
            dissent_signals   = buy_signals
            dominant_action   = SignalType.SELL

        if not dominant_signals:
            return None

        n_agreeing   = len(dominant_signals)
        n_dissenting = len(dissent_signals)

        # Confidence de base : moyenne pondérée des signaux dominants
        base_conf = sum(s.confidence for s in dominant_signals) / n_agreeing

        # Boost accord
        if n_agreeing > 1:
            boost = min(_AGREEMENT_BOOST_PER_BOT * (n_agreeing - 1), _MAX_AGREEMENT_BOOST)
            base_conf = base_conf * (1.0 + boost)

        # Pénalité dissidence
        if n_dissenting > 0:
            penalty = min(_DISSENT_PENALTY_PER_BOT * n_dissenting, 0.50)
            base_conf = base_conf * (1.0 - penalty)

        # Override news risk (cappliqué si score > 0.5)
        news_risk = self._news_risk_scores.get(asset)
        if news_risk is not None and news_risk > 0.50:
            # Cap la confidence proportionnellement au risque news
            cap = 1.0 - (news_risk - 0.50) * 1.6   # à risk=1.0 → cap=0.20
            base_conf = min(base_conf, max(cap, 0.10))
            logger.debug("%s: news risk %.2f caps confidence to %.2f", asset, news_risk, base_conf)

        # Clamp 0-1
        final_conf = max(0.0, min(1.0, base_conf))

        if final_conf < self.min_confidence:
            logger.debug(
                "Discarded %s %s: confidence %.2f < min %.2f",
                asset, dominant_action.value, final_conf, self.min_confidence,
            )
            return None

        # Meilleurs paramètres prix : entry=moyenne, SL=le plus large (conservateur), TP=le plus ambitieux
        entry_prices = [s.entry_price for s in dominant_signals if s.entry_price]
        stop_losses  = [s.stop_loss  for s in dominant_signals if s.stop_loss]
        take_profits = [s.take_profit for s in dominant_signals if s.take_profit]

        entry = sum(entry_prices) / len(entry_prices) if entry_prices else None
        # BUY (long)  : SL est sous l'entrée → le plus large = le plus bas  → min()
        #               TP est au-dessus      → le plus ambitieux = le plus haut → max()
        # SELL (short): SL est au-dessus      → le plus large = le plus haut → max()
        #               TP est dessous        → le plus ambitieux = le plus bas  → min()
        if dominant_action == SignalType.BUY:
            sl = min(stop_losses)  if stop_losses  else None
            tp = max(take_profits) if take_profits else None
        else:
            sl = max(stop_losses)  if stop_losses  else None
            tp = min(take_profits) if take_profits else None

        rr = None
        if entry and sl and tp:
            risk   = abs(entry - sl)
            reward = abs(tp - entry)
            rr = round(reward / risk, 2) if risk > 0 else None

        # Rejet si R:R aggregé < 2.0
        if rr is not None and rr < 2.0:
            logger.debug("Discarded %s: R:R %.2f < 2.0 minimum", asset, rr)
            return None

        # Timeframe dominant (le plus fréquent)
        timeframes = [s.timeframe for s in dominant_signals]
        timeframe = max(set(timeframes), key=timeframes.count)

        contributors = [
            BotContribution(
                strategy_name=s.strategy_name,
                signal_type=s.signal.value,
                confidence=s.confidence,
                reason=s.reason,
                stop_loss=s.stop_loss,
                take_profit=s.take_profit,
                entry_price=s.entry_price,
                risk_reward=s.risk_reward,
            )
            for s in dominant_signals
        ]
        reasons = [s.reason for s in dominant_signals if s.reason]

        return AggregatedSignal(
            asset=asset,
            action=dominant_action,
            confidence=final_conf,
            horizon=horizon,
            timeframe=timeframe,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            risk_reward=rr,
            contributors=contributors,
            n_agreeing=n_agreeing,
            n_dissenting=n_dissenting,
            reasons=reasons,
            news_risk_override=news_risk,
        )
