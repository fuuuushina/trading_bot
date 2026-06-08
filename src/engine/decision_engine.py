"""
src/engine/decision_engine.py

The Decision Engine is the central orchestrator.

Pipeline (per cycle) :
  1. Fetch features from FeatureEngine
  2. Detect market regime
  3. Run all applicable strategies → raw Signals
  4. SignalAggregator : consolide les signaux par asset (résolution conflits + boost accord)
  5. Run Rules Engine on each aggregated non-NO_TRADE signal
  6. Query AI advisory layer (optional)
  7. Run Risk Manager on each rules-approved signal
  8. Return a list of RiskVerdict-stamped Signals for execution

The Decision Engine NEVER submits orders. It returns decisions.
The Execution Engine handles submission.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from src.engine.signal_aggregator import SignalAggregator
from src.features.regime_detector import MarketRegimeDetector, RegimeResult
from src.risk.risk_manager import KillSwitchTriggered, RiskDecision, RiskManager, RiskVerdict
from src.rules.rules_engine import RulesEngine, RulesVerdict
from src.strategies.base import Horizon, Signal, SignalType
from src.strategies.breakout import BreakoutStrategy
from src.strategies.intraday_bollinger_rsi import IntradayBollingerRSIStrategy
from src.strategies.intraday_ema_cross import IntradayEMACrossStrategy
from src.strategies.intraday_session_breakout import IntradaySessionBreakoutStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.rsi_dip_buyer import RSIDipBuyerStrategy
from src.strategies.tactical_dca import TacticalDCAStrategy
from src.strategies.true_dca import TrueDCAStrategy
from src.strategies.trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """Full audit trail for a single signal decision."""
    asset: str
    signal: Signal
    regime: RegimeResult
    rules_verdict: RulesVerdict
    risk_verdict: RiskVerdict
    ai_verdict: Optional[dict]
    final_action: str           # EXECUTE | BLOCK | NO_TRADE
    explanation: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "signal": self.signal.to_dict(),
            "regime": self.regime.to_dict(),
            "rules_verdict": self.rules_verdict.summary,
            "rules_blocking": [r.to_dict() for r in self.rules_verdict.blocking_rules],
            "rules_warnings": [r.to_dict() for r in self.rules_verdict.warnings],
            "risk_verdict": self.risk_verdict.to_dict(),
            "ai_verdict": self.ai_verdict,
            "final_action": self.final_action,
            "explanation": self.explanation,
        }


class DecisionEngine:
    """
    Wires together: RegimeDetector → Strategies → Rules → AI → RiskManager.

    Usage:
        engine = DecisionEngine(risk_cfg, settings_cfg, strategy_cfg)
        decisions = engine.run_cycle(data_map, portfolio_state, open_positions, trade_history)
    """

    def __init__(
        self,
        risk_cfg: dict,
        settings_cfg: dict,
        strategy_cfg: dict,
        ai_layer: Any = None,           # Optional AIAdvisoryLayer instance
    ) -> None:
        self.settings = settings_cfg
        self.strategy_cfg = strategy_cfg

        self.regime_detector = MarketRegimeDetector()
        self.rules_engine = RulesEngine(risk_cfg, settings_cfg, strategy_cfg)
        self.risk_manager = RiskManager(risk_cfg)
        self.ai_layer = ai_layer
        self.signal_aggregator = SignalAggregator(
            min_confidence=settings_cfg.get("signal_aggregator", {}).get("min_confidence", 0.45)
        )
        self.last_no_trade: list[dict] = []

        # Build strategy library
        scfg = strategy_cfg.get("strategies", {})
        self._enabled_strategies = {
            name for name, cfg in scfg.items() if cfg.get("enabled", False)
        }
        self._strategies = {
            "trend_following":          TrendFollowingStrategy(scfg.get("trend_following", {})),
            "mean_reversion":           MeanReversionStrategy(scfg.get("mean_reversion", {})),
            "breakout":                 BreakoutStrategy(scfg.get("breakout", {})),
            "rsi_dip_buyer":            RSIDipBuyerStrategy(scfg.get("rsi_dip_buyer", {})),
            "tactical_dca":             TacticalDCAStrategy(scfg.get("tactical_dca", {})),
            "true_dca":                 TrueDCAStrategy(scfg.get("true_dca", {})),
            # Intraday forex strategies
            "intraday_ema_cross":       IntradayEMACrossStrategy(scfg.get("intraday_ema_cross", {})),
            "intraday_bollinger_rsi":   IntradayBollingerRSIStrategy(scfg.get("intraday_bollinger_rsi", {})),
            "intraday_session_breakout": IntradaySessionBreakoutStrategy(scfg.get("intraday_session_breakout", {})),
        }

    # ------------------------------------------------------------------ #
    # Main cycle
    # ------------------------------------------------------------------ #

    def run_cycle(
        self,
        data_map: dict[str, pd.DataFrame],
        portfolio_state: dict,
        open_positions: list[dict],
        trade_history: list[dict],
        vix_series: Optional[pd.Series] = None,
        intraday_data_map: Optional[dict[str, pd.DataFrame]] = None,
    ) -> list[DecisionRecord]:
        """
        Run one full decision cycle across all assets.

        Parameters
        ----------
        data_map        : {ticker: feature-enriched OHLCV DataFrame}
        portfolio_state : Current portfolio snapshot dict
        open_positions  : Open positions list
        trade_history   : Historical trade list
        vix_series      : Optional VIX price series

        Returns
        -------
        List of DecisionRecord — one per actionable or notable signal
        """
        if self.risk_manager.is_halted:
            logger.critical("Decision Engine skipped — Kill Switch active.")
            return []

        decisions: list[DecisionRecord] = []
        self.last_no_trade: list[dict] = []  # reset each cycle

        # Benchmark pour la détection de régime
        benchmark = "SPY"
        if benchmark in data_map:
            regime_result = self.regime_detector.detect(data_map[benchmark], vix_series)
        elif data_map:
            first_asset = next(iter(data_map))
            regime_result = self.regime_detector.detect(data_map[first_asset], vix_series)
        else:
            from src.features.regime_detector import MarketRegime, RegimeResult
            regime_result = RegimeResult(regime=MarketRegime.RANGE, confidence=0.2,
                                         explanation="No benchmark data — defaulting to range.")

        regime_str = regime_result.regime.value
        logger.info("Market regime: %s (confidence=%.2f)", regime_str, regime_result.confidence)

        # --- Phase 1 : collecte des signaux bruts de toutes les stratégies ---
        regime_map = self.strategy_cfg.get("regime_strategy_map", {})
        active_by_horizon: dict[str, list[str]] = regime_map.get(regime_str, {})

        raw_signals: list[Signal] = []
        df_map: dict[str, pd.DataFrame] = {}  # garde le df par asset pour _evaluate_signal

        # Daily / swing / long_term strategies — run on daily data
        daily_horizons = {k: v for k, v in active_by_horizon.items() if k != "intraday"}
        for asset, df in data_map.items():
            if df is None or df.empty:
                continue
            df_map[asset] = df

            for horizon_name, strategy_names in daily_horizons.items():
                for strategy_name in strategy_names:
                    if strategy_name not in self._strategies:
                        continue
                    if strategy_name not in self._enabled_strategies:
                        logger.debug("Strategy disabled: %s", strategy_name)
                        continue

                    strategy = self._strategies[strategy_name]
                    try:
                        raw_signal = strategy.generate_signal(df, asset, regime_str)
                    except Exception as exc:
                        logger.error(
                            "Strategy %s failed for %s: %s",
                            strategy_name, asset, exc, exc_info=True
                        )
                        continue

                    if raw_signal.signal == SignalType.NO_TRADE:
                        logger.debug("NO_TRADE: %s / %s — %s", strategy_name, asset, raw_signal.reason)
                        continue

                    raw_signals.append(raw_signal)

        # Intraday strategies — run on 5min intraday data (EURUSD=X etc.)
        intraday_strategy_names = active_by_horizon.get("intraday", [])
        for asset, df in (intraday_data_map or {}).items():
            if df is None or df.empty:
                continue
            df_map[asset] = df  # used later for _evaluate_signal

            for strategy_name in intraday_strategy_names:
                if strategy_name not in self._strategies:
                    continue
                if strategy_name not in self._enabled_strategies:
                    logger.debug("Strategy disabled: %s", strategy_name)
                    continue

                strategy = self._strategies[strategy_name]
                try:
                    raw_signal = strategy.generate_signal(df, asset, regime_str)
                except Exception as exc:
                    logger.error(
                        "Intraday strategy %s failed for %s: %s",
                        strategy_name, asset, exc, exc_info=True
                    )
                    continue

                if raw_signal.signal == SignalType.NO_TRADE:
                    logger.info("NO_TRADE: %s / %s — %s", strategy_name, asset, raw_signal.reason)
                    self.last_no_trade.append({
                        "asset": asset,
                        "strategy": strategy_name,
                        "reason": raw_signal.reason or "",
                    })
                    continue

                raw_signals.append(raw_signal)

        logger.info(
            "Cycle: %d raw signals from %d assets",
            len(raw_signals), len(df_map),
        )

        # --- Phase 2 : agrégation des signaux ---
        aggregated_signals = self.signal_aggregator.aggregate(raw_signals)

        logger.info(
            "After aggregation: %d consolidated signals",
            len(aggregated_signals),
        )

        # --- Phase 3 : évaluation Rules → AI → Risk sur chaque signal agrégé ---
        for agg_signal in aggregated_signals:
            df = df_map.get(agg_signal.asset)
            if df is None:
                continue

            record = self._evaluate_signal(
                agg_signal, df, regime_result, portfolio_state,
                open_positions, trade_history, regime_str
            )
            decisions.append(record)

            if record.final_action == "EXECUTE":
                logger.info(
                    "DECISION → EXECUTE: %s %s $%.2f (conf=%.2f, %d bots)",
                    agg_signal.asset,
                    agg_signal.signal.value,
                    record.risk_verdict.approved_size_usd,
                    agg_signal.confidence,
                    agg_signal.metadata.get("n_agreeing", 1),
                )
            else:
                blocking = "; ".join(r.reason for r in record.rules_verdict.blocking_rules) if record.rules_verdict.blocking_rules else record.risk_verdict.reason
                logger.info(
                    "DECISION → %s: %s %s (conf=%.2f) — %s",
                    record.final_action,
                    agg_signal.asset,
                    agg_signal.signal.value,
                    agg_signal.confidence,
                    blocking,
                )

        return decisions

    # ------------------------------------------------------------------ #
    # Single signal evaluation pipeline
    # ------------------------------------------------------------------ #

    def _evaluate_signal(
        self,
        signal: Signal,
        df: pd.DataFrame,
        regime_result: RegimeResult,
        portfolio_state: dict,
        open_positions: list[dict],
        trade_history: list[dict],
        regime_str: str,
    ) -> DecisionRecord:
        # ---- Step 1: Rules Engine ----
        rules_verdict = self.rules_engine.evaluate(
            signal=signal,
            df=df,
            regime=regime_str,
            portfolio_state=portfolio_state,
            open_positions=open_positions,
            trade_history=trade_history,
        )

        # ---- Step 2: AI Advisory (optional, non-blocking by default) ----
        ai_verdict: Optional[dict] = None
        if self.ai_layer and self.settings.get("ai", {}).get("enabled", False):
            try:
                ai_verdict = self.ai_layer.consult(signal, regime_result, portfolio_state)
            except Exception as exc:
                logger.warning("AI advisory failed: %s", exc)
                ai_verdict = None

        # Re-run rules with AI verdict if available
        if ai_verdict:
            rules_verdict = self.rules_engine.evaluate(
                signal=signal,
                df=df,
                regime=regime_str,
                portfolio_state=portfolio_state,
                open_positions=open_positions,
                trade_history=trade_history,
                ai_verdict=ai_verdict,
            )

        # ---- Step 3: Risk Manager ----
        try:
            risk_verdict = self.risk_manager.evaluate(
                signal=signal,
                portfolio_state=portfolio_state,
                open_positions=open_positions,
                rules_approved=rules_verdict.approved,
            )
        except KillSwitchTriggered as e:
            risk_verdict = RiskVerdict(
                RiskDecision.KILL, 0.0, 0.0, str(e)
            )

        # ---- Step 4: Final determination ----
        if risk_verdict.decision == RiskDecision.KILL:
            final_action = "KILL"
        elif risk_verdict.is_executable:
            final_action = "EXECUTE"
        else:
            final_action = "BLOCK"

        # Stamp the signal
        signal.approved = final_action == "EXECUTE"
        signal.approved_size_usd = risk_verdict.approved_size_usd
        signal.rejection_reason = risk_verdict.reason if not signal.approved else ""

        explanation = self._build_explanation(
            signal, rules_verdict, risk_verdict, ai_verdict, regime_str
        )

        return DecisionRecord(
            asset=signal.asset,
            signal=signal,
            regime=regime_result,
            rules_verdict=rules_verdict,
            risk_verdict=risk_verdict,
            ai_verdict=ai_verdict,
            final_action=final_action,
            explanation=explanation,
        )

    @staticmethod
    def _build_explanation(
        signal: Signal,
        rules: RulesVerdict,
        risk: RiskVerdict,
        ai: Optional[dict],
        regime: str,
    ) -> str:
        lines = [
            f"Strategy: {signal.strategy_name} | Asset: {signal.asset} | Regime: {regime}",
            f"Signal: {signal.signal.value} | Confidence: {signal.confidence:.2f} | R:R: {signal.risk_reward}",
            f"Signal reason: {signal.reason}",
            f"Rules: {rules.summary}",
        ]
        if rules.blocking_rules:
            lines.append("Blocking rules: " + "; ".join(r.reason for r in rules.blocking_rules))
        if ai:
            lines.append(
                f"AI: {ai.get('recommended_action')} | risk={ai.get('risk_score'):.2f} | "
                f"opp={ai.get('opportunity_score'):.2f} | '{ai.get('summary', '')}'"
            )
        lines.append(f"Risk: {risk.decision.value} | Size: ${risk.approved_size_usd:.2f} | {risk.reason}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Pass-through to Risk Manager for event registration
    # ------------------------------------------------------------------ #

    def apply_news_risk(self, risk_scores: dict[str, float]) -> None:
        """Injecte des scores de risque news dans l'agrégateur avant le prochain cycle."""
        self.signal_aggregator.apply_news_override(risk_scores)

    def register_trade_result(
        self, pnl: float, horizon: Horizon, total_capital: float
    ) -> None:
        self.risk_manager.register_trade_result(pnl, horizon, total_capital)

    def reset_daily(self) -> None:
        self.risk_manager.reset_daily()

    def reset_weekly(self) -> None:
        self.risk_manager.reset_weekly()

    def reset_monthly(self) -> None:
        self.risk_manager.reset_monthly()
