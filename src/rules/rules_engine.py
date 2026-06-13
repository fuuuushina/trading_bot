"""
src/rules/rules_engine.py

Deterministic filter layer.
All rules return a RuleResult (pass/fail + reason).
The RulesEngine aggregates them and returns a final verdict.

Rules are HARD GATES — not suggestions.
A single blocking rule stops the trade regardless of strategy confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone, timedelta
from typing import Any

import pandas as pd

from src.features.indicators import atr, volume_ratio, z_score
from src.features.regime_detector import MarketRegime
from src.strategies.base import Horizon, Signal, SignalType


def _strategy_names_for_signal(signal: Signal) -> list[str]:
    """Return the real contributing strategy names for normal or aggregated signals."""
    names: list[str] = []
    for contributor in signal.metadata.get("contributors", []):
        name = contributor.get("strategy") if isinstance(contributor, dict) else None
        if name and name not in names:
            names.append(name)
    if signal.strategy_name and signal.strategy_name not in names:
        names.append(signal.strategy_name)
    return names


@dataclass
class RuleResult:
    rule_name: str
    passed: bool
    reason: str
    severity: str = "block"     # block | warn

    def to_dict(self) -> dict:
        return {
            "rule": self.rule_name,
            "passed": self.passed,
            "reason": self.reason,
            "severity": self.severity,
        }


@dataclass
class RulesVerdict:
    approved: bool
    blocking_rules: list[RuleResult] = field(default_factory=list)
    passed_rules: list[RuleResult] = field(default_factory=list)
    warnings: list[RuleResult] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.approved:
            return f"APPROVED ({len(self.passed_rules)} rules passed)"
        blocked = [r.rule_name for r in self.blocking_rules]
        return f"BLOCKED by: {', '.join(blocked)}"


class StatisticalRules:
    """Pure data-quality and statistical filter rules."""

    def __init__(self, risk_cfg: dict, settings_cfg: dict) -> None:
        # Unwrap: accept both the full YAML dict and the inner "risk:" section
        self.risk = risk_cfg.get("risk", risk_cfg)
        self.settings = settings_cfg

    def check_minimum_volume(
        self, df: pd.DataFrame, signal: Signal
    ) -> RuleResult:
        name = "min_volume"
        if "volume" not in df.columns:
            return RuleResult(name, True, "Volume column absent — skipped.", "warn")

        # Forex assets (yfinance ticker ends with =X) report tick volume,
        # not contract volume — skip absolute threshold, use ratio check only.
        if signal.asset.endswith("=X"):
            avg_vol = float(df["volume"].rolling(20).mean().iloc[-1])
            if avg_vol == 0:
                return RuleResult(name, True, "Forex: zero volume (normal for some feeds).", "warn")
            return RuleResult(name, True, f"Forex tick volume {avg_vol:,.0f} — threshold skipped.")

        min_vol = (
            self.risk.get("intraday", {}).get("min_volume_intraday", 1_000_000)
            if signal.horizon == Horizon.INTRADAY
            else self.settings.get("data", {}).get("min_volume_threshold", 100_000)
        )
        avg_vol = float(df["volume"].rolling(20).mean().iloc[-1])
        passed = avg_vol >= min_vol
        return RuleResult(
            name, passed,
            f"Avg volume {avg_vol:,.0f} {'≥' if passed else '<'} min {min_vol:,.0f}"
        )

    def check_spread(
        self, df: pd.DataFrame, signal: Signal, current_spread_pct: float = 0.0
    ) -> RuleResult:
        name = "max_spread"
        max_spread = self.risk.get("intraday", {}).get("max_spread_pct", 0.002)
        if signal.horizon != Horizon.INTRADAY:
            return RuleResult(name, True, "Spread check only for intraday.", "warn")
        passed = current_spread_pct <= max_spread
        return RuleResult(
            name, passed,
            f"Spread {current_spread_pct:.4%} {'≤' if passed else '>'} max {max_spread:.4%}"
        )

    def check_volatility_acceptable(
        self, df: pd.DataFrame, signal: Signal
    ) -> RuleResult:
        import math as _m
        name = "volatility_acceptable"
        atr_val = atr(df, 14).iloc[-1]
        price = float(df["close"].iloc[-1])

        # Guard NaN/inf (yfinance returns NaN for incomplete bars or pre-market)
        if not _m.isfinite(float(atr_val)) or not _m.isfinite(price) or price <= 0:
            return RuleResult(name, True, "ATR data unavailable — volatility check skipped",
                              severity="block")

        atr_pct = float(atr_val) / price

        # Block if ATR > 8% of price (extreme volatility for equity)
        max_atr_pct = 0.08
        passed = atr_pct <= max_atr_pct
        return RuleResult(
            name, passed,
            f"ATR/Price {atr_pct:.2%} {'≤' if passed else '>'} max {max_atr_pct:.2%}",
            severity="block" if not passed else "block",
        )

    def check_zscore_extreme(
        self, df: pd.DataFrame, signal: Signal
    ) -> RuleResult:
        """Block trend entries when z-score is extreme (better for mean reversion)."""
        name = "zscore_not_extreme_for_trend"
        strategy_names = set(_strategy_names_for_signal(signal))
        if not strategy_names.intersection({"trend_following", "breakout", "momentum"}):
            return RuleResult(name, True, "Z-score extreme check not applicable.", "warn")

        z = float(z_score(df["close"], 20).iloc[-1])
        # If z > 3 on a buy, or z < -3 on a sell → overextended
        if signal.signal == SignalType.BUY and z > 3.0:
            return RuleResult(name, False, f"Z-score={z:.2f} extremely overbought for trend entry.")
        if signal.signal == SignalType.SELL and z < -3.0:
            return RuleResult(name, False, f"Z-score={z:.2f} extremely oversold for trend short.")
        return RuleResult(name, True, f"Z-score={z:.2f} acceptable.")

    def check_correlation_risk(
        self,
        signal: Signal,
        open_positions: list[dict],
        correlation_matrix: pd.DataFrame | None,
    ) -> RuleResult:
        name = "correlation_risk"
        if correlation_matrix is None or signal.asset not in correlation_matrix.columns:
            return RuleResult(name, True, "Correlation data unavailable — skipped.", "warn")

        max_corr = self.risk.get("max_correlation_exposure", 0.70)
        open_assets = [p["asset"] for p in open_positions if p["asset"] != signal.asset]

        for asset in open_assets:
            if asset in correlation_matrix.columns:
                corr = float(correlation_matrix.loc[signal.asset, asset])
                if abs(corr) > max_corr:
                    return RuleResult(
                        name, False,
                        f"Correlation with existing position {asset}: {corr:.2f} > {max_corr}"
                    )
        return RuleResult(name, True, "Correlation exposure within limits.")

    def check_liquidity(self, df: pd.DataFrame, signal: Signal) -> RuleResult:
        name = "sufficient_liquidity"
        if "volume" not in df.columns:
            return RuleResult(name, True, "No volume data — skipped.", "warn")
        # Skip for forex (=X), crypto (-USD), and futures (=F) — volume scales differ
        asset = signal.asset
        if asset.endswith("=X") or asset.endswith("-USD") or asset.endswith("=F"):
            return RuleResult(name, True, "Non-equity asset — liquidity check skipped.", "warn")
        vol_r = float(volume_ratio(df, 20).iloc[-1])
        # Liquidity must be at least 50% of average
        passed = vol_r >= 0.5
        return RuleResult(name, passed, f"Volume ratio={vol_r:.2f} ({'OK' if passed else 'LOW'})")

    def check_regime_compatible(
        self, signal: Signal, regime: str, regime_strategy_map: dict
    ) -> RuleResult:
        name = "regime_compatible"
        allowed = regime_strategy_map.get(regime, {}).get(signal.horizon.value, [])
        if not allowed:
            return RuleResult(
                name, False,
                f"Strategy '{signal.strategy_name}' not allowed in regime '{regime}' for horizon '{signal.horizon.value}'."
            )

        strategy_names = _strategy_names_for_signal(signal)
        matching = [strategy for strategy in strategy_names if strategy in allowed]
        if not matching:
            return RuleResult(
                name, False,
                f"{strategy_names} not in allowed strategies {allowed} for regime '{regime}'."
            )
        return RuleResult(
            name, True,
            f"Strategy allowed in {regime} regime via {', '.join(matching)}."
        )


class StrategicRules:
    """Higher-level strategic / operational rules."""

    def __init__(self, risk_cfg: dict, settings_cfg: dict) -> None:
        self.risk = risk_cfg.get("risk", risk_cfg)
        self.settings = settings_cfg

    def check_market_hours(self, signal: Signal) -> RuleResult:
        name = "market_hours"
        if signal.horizon != Horizon.INTRADAY:
            return RuleResult(name, True, "Market hours check only for intraday.")

        now_utc = datetime.utcnow().time()
        mh = self.settings.get("market_hours", {})
        buffer = mh.get("intraday_cutoff_minutes_before_close", 30)

        is_forex = signal.asset.endswith("=X")
        if is_forex:
            open_str = mh.get("forex_open", "07:00")
            close_str = mh.get("forex_close", "21:00")
        else:
            open_str = mh.get("us_open", "13:30")
            close_str = mh.get("us_close", "20:00")

        open_t = time(*map(int, open_str.split(":")))
        close_t = time(*map(int, close_str.split(":")))
        cutoff_h = (close_t.hour * 60 + close_t.minute - buffer) // 60
        cutoff_m = (close_t.hour * 60 + close_t.minute - buffer) % 60
        cutoff_t = time(cutoff_h, cutoff_m)

        in_hours = open_t <= now_utc < cutoff_t
        return RuleResult(
            name, in_hours,
            f"Market {'open' if in_hours else 'closed or too close to close'} "
            f"(now={now_utc}, window={open_str}-{close_str} minus {buffer}min)"
        )

    def check_no_counter_trend(
        self, signal: Signal, regime: str
    ) -> RuleResult:
        name = "no_counter_trend"
        # Seules ces stratégies peuvent aller à contre-tendance du régime global
        exempt = ("mean_reversion", "intraday_mean_reversion", "thematic_momentum",
                  "rsi_dip_buyer", "ema_cross_swing", "momentum_burst",
                  "intraday_bollinger_rsi", "intraday_local_rebound")
        strategy_names = set(_strategy_names_for_signal(signal))
        exempt_matches = strategy_names.intersection(exempt)

        bull_regimes = {"bull_trend", "breakout_expansion", "euphoric"}
        bear_regimes = {"bear_trend", "panic"}

        # Régimes forts : aucune exception (même les stratégies exemptées respectent la tendance)
        if regime in bull_regimes:
            if signal.signal == SignalType.SELL:
                if not exempt_matches:
                    return RuleResult(name, False,
                                      f"SELL interdit en régime {regime} (marché haussier fort).")
                # Stratégies exemptées : acceptées uniquement si RSI suracheté (> 70)
                rsi_val = signal.metadata.get("rsi", 50)
                try:
                    rsi_val = float(rsi_val)
                except (TypeError, ValueError):
                    rsi_val = 50.0
                if rsi_val < 68:
                    return RuleResult(name, False,
                                      f"SELL en {regime}: RSI={rsi_val:.1f} < 68, pas suracheté.")
            return RuleResult(name, True, f"BUY en {regime}: aligné.")

        if regime in bear_regimes:
            if signal.signal == SignalType.BUY:
                if not exempt_matches:
                    return RuleResult(name, False,
                                      f"BUY interdit en régime {regime} (marché baissier fort).")
                rsi_val = signal.metadata.get("rsi", 50)
                try:
                    rsi_val = float(rsi_val)
                except (TypeError, ValueError):
                    rsi_val = 50.0
                if rsi_val > 32:
                    return RuleResult(name, False,
                                      f"BUY en {regime}: RSI={rsi_val:.1f} > 32, pas survendu.")
            return RuleResult(name, True, f"SELL en {regime}: aligné.")

        # Régimes neutres (range, compression, high_volatility…) : pas de filtre directionnel
        return RuleResult(name, True, f"Régime neutre ({regime}): les deux directions acceptées.")

    def check_no_position_averaging_down(
        self, signal: Signal, open_positions: list[dict]
    ) -> RuleResult:
        name = "no_averaging_down"
        # Check if there's an open losing position in same asset
        for pos in open_positions:
            if pos["asset"] == signal.asset and pos.get("unrealized_pnl_pct", 0) < -0.02:
                if signal.signal == SignalType.BUY and pos["side"] == "long":
                    return RuleResult(
                        name, False,
                        f"Refusing to add to losing long position in {signal.asset} "
                        f"(P&L: {pos['unrealized_pnl_pct']:.1%})"
                    )
        return RuleResult(name, True, "No averaging down detected.")

    def check_concentration(
        self, signal: Signal, portfolio_state: dict
    ) -> RuleResult:
        name = "concentration_limit"
        max_asset_pct = self.risk.get("max_exposure_per_asset_pct", 0.10)
        total_capital = portfolio_state.get("total_capital", 1.0)
        current_exposure = portfolio_state.get("asset_exposure", {}).get(signal.asset, 0.0)
        current_pct = current_exposure / total_capital

        if current_pct >= max_asset_pct:
            return RuleResult(
                name, False,
                f"Asset {signal.asset} already at {current_pct:.1%} exposure (max {max_asset_pct:.1%})"
            )
        return RuleResult(name, True, f"Asset exposure {current_pct:.1%} within limit.")

    def check_minimum_confidence(self, signal: Signal) -> RuleResult:
        name = "minimum_confidence"
        min_conf = self.settings.get("signal_aggregator", {}).get("min_confidence", 0.35)
        passed = signal.confidence >= min_conf
        return RuleResult(
            name, passed,
            f"Signal confidence {signal.confidence:.2f} {'≥' if passed else '<'} {min_conf}"
        )

    def check_consecutive_losses(
        self, signal: Signal, trade_history: list[dict]
    ) -> RuleResult:
        name = "consecutive_losses"
        if signal.horizon == Horizon.LONG_TERM:
            return RuleResult(name, True, "Consecutive loss check not applied to long-term.")

        max_losses = (
            self.risk.get("intraday", {}).get("max_consecutive_losses", 3)
            if signal.horizon == Horizon.INTRADAY
            else self.risk.get("swing", {}).get("max_consecutive_losses", 4)
        )

        recent = [t for t in trade_history[-max_losses:] if t.get("horizon") == signal.horizon.value]
        if len(recent) >= max_losses and all(t.get("pnl", 0) < 0 for t in recent):
            return RuleResult(
                name, False,
                f"Last {max_losses} {signal.horizon.value} trades all losses — pausing."
            )
        return RuleResult(name, True, f"Consecutive losses within limit.")

    def check_no_duplicate_position(
        self, signal: Signal, open_positions: list[dict]
    ) -> RuleResult:
        """Bloque l'ouverture d'une position si une position identique (même asset, même sens)
        est déjà ouverte — évite la multiplication de positions sur le même asset."""
        name = "no_duplicate_position"
        if signal.horizon != Horizon.INTRADAY:
            return RuleResult(name, True, "Duplicate check not strict for swing/long-term.")

        signal_side = "long" if signal.signal == SignalType.BUY else "short"
        for pos in open_positions:
            if (pos.get("asset") == signal.asset
                    and pos.get("horizon", "intraday") == "intraday"
                    and pos.get("side") == signal_side):
                return RuleResult(
                    name, False,
                    f"Déjà une position {signal_side} ouverte sur {signal.asset} en intraday."
                )
        return RuleResult(name, True, "Pas de position dupliquée.")

    def check_intraday_cooldown(
        self, signal: Signal, trade_history: list[dict]
    ) -> RuleResult:
        """Impose un cooldown minimal entre deux trades intraday sur le même asset.
        Évite l'over-trading : réentrée immédiate après une sortie perdante."""
        name = "intraday_cooldown"
        if signal.horizon != Horizon.INTRADAY:
            return RuleResult(name, True, "Cooldown uniquement pour intraday.")

        cooldown_minutes = self.risk.get("intraday", {}).get("cooldown_minutes", 15)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=cooldown_minutes)

        for trade in reversed(trade_history):
            if trade.get("asset") != signal.asset:
                continue
            if trade.get("horizon", "intraday") != "intraday":
                continue
            closed_raw = trade.get("closed_at", "")
            if not closed_raw:
                continue
            try:
                closed_at = datetime.fromisoformat(closed_raw.replace("Z", "+00:00"))
                if closed_at.tzinfo is None:
                    closed_at = closed_at.replace(tzinfo=timezone.utc)
                if closed_at >= cutoff:
                    minutes_ago = int((now - closed_at).total_seconds() / 60)
                    return RuleResult(
                        name, False,
                        f"Cooldown actif sur {signal.asset}: dernier trade fermé il y a "
                        f"{minutes_ago}min (cooldown={cooldown_minutes}min)."
                    )
            except (ValueError, TypeError):
                continue
        return RuleResult(name, True, f"Cooldown OK (>{cooldown_minutes}min depuis dernier trade).")

    def check_ai_disagreement(
        self,
        signal: Signal,
        ai_verdict: dict | None,
        threshold: float = 0.4,
    ) -> RuleResult:
        name = "ai_disagreement"
        if ai_verdict is None:
            return RuleResult(name, True, "No AI verdict available — skipped.", "warn")

        recommended = ai_verdict.get("recommended_action", "APPROVE")
        risk_score = ai_verdict.get("risk_score", 0.5)
        agreement = ai_verdict.get("agreement", True)

        if recommended == "REJECT" and risk_score > 0.8:
            return RuleResult(
                name, False,
                f"AI strongly rejects trade (risk_score={risk_score:.2f}, action={recommended})."
            )
        if not agreement and risk_score > threshold + 0.2:
            return RuleResult(
                name, False,
                f"Strategic and Live AI strongly disagree (risk={risk_score:.2f})."
            )
        return RuleResult(name, True, f"AI verdict: {recommended}, risk={risk_score:.2f}")


class RulesEngine:
    """
    Aggregates all rules and returns a single RulesVerdict.
    Instantiated once, called per signal.
    """

    def __init__(self, risk_cfg: dict, settings_cfg: dict, strategy_cfg: dict) -> None:
        self.stat = StatisticalRules(risk_cfg, settings_cfg)
        self.strat = StrategicRules(risk_cfg, settings_cfg)
        self.strategy_cfg = strategy_cfg
        self.risk_cfg = risk_cfg

    def evaluate(
        self,
        signal: Signal,
        df: pd.DataFrame,
        regime: str,
        portfolio_state: dict,
        open_positions: list[dict],
        trade_history: list[dict],
        ai_verdict: dict | None = None,
        correlation_matrix: pd.DataFrame | None = None,
        current_spread_pct: float = 0.0,
    ) -> RulesVerdict:
        """
        Run all applicable rules and return verdict.
        Any blocking rule failure → approved=False.
        """
        regime_map = self.strategy_cfg.get("regime_strategy_map", {})

        all_results: list[RuleResult] = []

        # Statistical
        all_results += [
            self.stat.check_minimum_volume(df, signal),
            self.stat.check_spread(df, signal, current_spread_pct),
            self.stat.check_volatility_acceptable(df, signal),
            self.stat.check_zscore_extreme(df, signal),
            self.stat.check_correlation_risk(signal, open_positions, correlation_matrix),
            self.stat.check_liquidity(df, signal),
            self.stat.check_regime_compatible(signal, regime, regime_map),
        ]

        # Strategic
        all_results += [
            self.strat.check_market_hours(signal),
            self.strat.check_no_counter_trend(signal, regime),
            self.strat.check_no_position_averaging_down(signal, open_positions),
            self.strat.check_no_duplicate_position(signal, open_positions),
            self.strat.check_intraday_cooldown(signal, trade_history),
            self.strat.check_concentration(signal, portfolio_state),
            self.strat.check_minimum_confidence(signal),
            self.strat.check_consecutive_losses(signal, trade_history),
            self.strat.check_ai_disagreement(signal, ai_verdict),
        ]

        # Skip NO_TRADE signals entirely
        if signal.signal == SignalType.NO_TRADE:
            return RulesVerdict(
                approved=False,
                blocking_rules=[RuleResult("no_trade_signal", False, "Signal is NO_TRADE.")],
            )

        blocking = [r for r in all_results if not r.passed and r.severity == "block"]
        warnings = [r for r in all_results if not r.passed and r.severity == "warn"]
        passed = [r for r in all_results if r.passed]

        return RulesVerdict(
            approved=len(blocking) == 0,
            blocking_rules=blocking,
            passed_rules=passed,
            warnings=warnings,
        )
