"""
src/ai/advisory.py

AI Consultative Layer — ADVISORY ONLY.

The AI layer queries the Anthropic API to get a second opinion on signals.
It can NEVER execute trades. It returns a structured advisory verdict.

Two modes:
  - StrategicAI: low temperature, evaluates risk/consistency/macro context
  - LiveDataAI : higher temperature, evaluates catalysts/sentiment/news

Both are aggregated into a consensus by AIAdvisoryLayer.consult().
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared output schema (AI must conform to this)
# ---------------------------------------------------------------------------

AI_OUTPUT_SCHEMA = {
    "opportunity_score": "float 0.0-1.0",
    "risk_score": "float 0.0-1.0",
    "agreement": "bool",
    "warnings": "list[str]",
    "summary": "str",
    "recommended_action": "APPROVE | WATCH | REJECT | REDUCE_SIZE",
}

SYSTEM_PROMPT_TEMPLATE = """
You are a {role} assistant for an algorithmic trading system.
You are ADVISORY ONLY. You CANNOT execute trades.
You provide structured risk/opportunity assessment to human-overseen automated systems.

RULES:
- Be conservative and risk-aware by default.
- Never recommend trades you cannot justify with data.
- Never be overconfident.
- If data is insufficient, say so clearly.
- Your output MUST be valid JSON matching exactly this schema:
  {schema}

Do not add any text outside the JSON object.
""".strip()

USER_PROMPT_TEMPLATE = """
Evaluate this potential trade signal:

Asset: {asset}
Strategy: {strategy}
Signal: {signal}
Confidence: {confidence}
Entry: {entry}
Stop Loss: {stop_loss}
Take Profit: {take_profit}
Risk/Reward: {risk_reward}
Market Regime: {regime}
Reason: {reason}

Portfolio context:
- Total capital: ${total_capital:,.0f}
- Proposed size: ${proposed_size:,.0f}
- Open positions: {open_positions}
- Current drawdown: {drawdown_pct:.1%}

Signal metadata: {metadata}

Assess this trade and return a JSON advisory verdict.
Be cautious. Capital preservation is the top priority.
""".strip()


def _call_anthropic(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int = 512,
    timeout: int = 15,
) -> Optional[str]:
    """
    Call the Anthropic API and return the raw text response.
    Returns None on failure (advisory layer is non-blocking).
    """
    try:
        import anthropic  # type: ignore
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — AI advisory disabled.")
            return None

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text
    except Exception as exc:
        logger.warning("AI advisory call failed: %s", exc)
        return None


def _parse_ai_response(raw: str) -> Optional[dict]:
    """Parse and validate AI JSON response."""
    try:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)

        # Validate and clamp numeric fields
        data["opportunity_score"] = max(0.0, min(1.0, float(data.get("opportunity_score", 0.5))))
        data["risk_score"] = max(0.0, min(1.0, float(data.get("risk_score", 0.5))))
        data["agreement"] = bool(data.get("agreement", True))
        data["warnings"] = list(data.get("warnings", []))
        data["summary"] = str(data.get("summary", ""))
        data["recommended_action"] = str(data.get("recommended_action", "WATCH"))

        valid_actions = {"APPROVE", "WATCH", "REJECT", "REDUCE_SIZE"}
        if data["recommended_action"] not in valid_actions:
            data["recommended_action"] = "WATCH"

        return data
    except Exception as exc:
        logger.warning("AI response parse failed: %s | raw=%s", exc, raw[:200])
        return None


class StrategicAI:
    """
    Low temperature. Evaluates coherence, risk, allocation, macro context.
    Called with signal + portfolio context.
    """

    def __init__(self, model: str, temperature: float = 0.2) -> None:
        self.model = model
        self.temperature = temperature

    def evaluate(
        self,
        signal_dict: dict,
        regime: str,
        portfolio_state: dict,
    ) -> Optional[dict]:
        system = SYSTEM_PROMPT_TEMPLATE.format(
            role="strategic risk-management",
            schema=json.dumps(AI_OUTPUT_SCHEMA, indent=2),
        )
        user = USER_PROMPT_TEMPLATE.format(
            asset=signal_dict["asset"],
            strategy=signal_dict["strategy_name"],
            signal=signal_dict["signal"],
            confidence=signal_dict["confidence"],
            entry=signal_dict.get("entry_price", "?"),
            stop_loss=signal_dict.get("stop_loss", "?"),
            take_profit=signal_dict.get("take_profit", "?"),
            risk_reward=signal_dict.get("risk_reward", "?"),
            regime=regime,
            reason=signal_dict.get("reason", ""),
            total_capital=portfolio_state.get("total_capital", 0),
            proposed_size=portfolio_state.get("proposed_size", 0),
            open_positions=portfolio_state.get("open_positions", 0),
            drawdown_pct=portfolio_state.get("drawdown_pct", 0.0),
            metadata=json.dumps(signal_dict.get("metadata", {})),
        )
        raw = _call_anthropic(system, user, self.model, self.temperature)
        return _parse_ai_response(raw) if raw else None


class LiveDataAI:
    """
    Higher temperature. Evaluates recent catalysts, news sentiment, emerging risks.
    """

    def __init__(self, model: str, temperature: float = 0.6) -> None:
        self.model = model
        self.temperature = temperature

    def evaluate(
        self,
        signal_dict: dict,
        regime: str,
        portfolio_state: dict,
    ) -> Optional[dict]:
        system = SYSTEM_PROMPT_TEMPLATE.format(
            role="live market sentiment and catalyst analysis",
            schema=json.dumps(AI_OUTPUT_SCHEMA, indent=2),
        ) + "\n\nPay special attention to: recent news, earnings surprises, macro events, sector rotation."

        user = USER_PROMPT_TEMPLATE.format(
            asset=signal_dict["asset"],
            strategy=signal_dict["strategy_name"],
            signal=signal_dict["signal"],
            confidence=signal_dict["confidence"],
            entry=signal_dict.get("entry_price", "?"),
            stop_loss=signal_dict.get("stop_loss", "?"),
            take_profit=signal_dict.get("take_profit", "?"),
            risk_reward=signal_dict.get("risk_reward", "?"),
            regime=regime,
            reason=signal_dict.get("reason", ""),
            total_capital=portfolio_state.get("total_capital", 0),
            proposed_size=portfolio_state.get("proposed_size", 0),
            open_positions=portfolio_state.get("open_positions", 0),
            drawdown_pct=portfolio_state.get("drawdown_pct", 0.0),
            metadata=json.dumps(signal_dict.get("metadata", {})),
        )
        raw = _call_anthropic(system, user, self.model, self.temperature)
        return _parse_ai_response(raw) if raw else None


class AIAdvisoryLayer:
    """
    Aggregates StrategicAI and LiveDataAI into a single advisory verdict.
    Non-blocking — returns None gracefully on any failure.

    IMPORTANT: This layer is ADVISORY ONLY.
    It has no write access to orders, positions, or execution systems.
    """

    def __init__(self, ai_cfg: dict) -> None:
        self.enabled = ai_cfg.get("enabled", False)
        self.strategic = StrategicAI(
            model=ai_cfg.get("strategic_model", "claude-opus-4-6"),
            temperature=ai_cfg.get("strategic_temperature", 0.2),
        )
        self.live = LiveDataAI(
            model=ai_cfg.get("live_model", "claude-sonnet-4-6"),
            temperature=ai_cfg.get("live_temperature", 0.6),
        )

    def consult(
        self,
        signal: Any,          # Signal object
        regime_result: Any,   # RegimeResult object
        portfolio_state: dict,
    ) -> Optional[dict]:
        """
        Get advisory verdict. Returns None if AI is disabled or fails.
        Result must NOT be used to execute trades directly.
        """
        if not self.enabled:
            return None

        signal_dict = signal.to_dict()
        regime_str = regime_result.regime.value

        strategic_verdict = self.strategic.evaluate(signal_dict, regime_str, portfolio_state)
        live_verdict = self.live.evaluate(signal_dict, regime_str, portfolio_state)

        return self._merge(strategic_verdict, live_verdict)

    @staticmethod
    def _merge(
        strategic: Optional[dict],
        live: Optional[dict],
    ) -> Optional[dict]:
        """Merge two AI verdicts into one consensus."""
        if not strategic and not live:
            return None

        # Use available verdicts only
        verdicts = [v for v in [strategic, live] if v]

        avg_opp = sum(v["opportunity_score"] for v in verdicts) / len(verdicts)
        avg_risk = sum(v["risk_score"] for v in verdicts) / len(verdicts)
        both_agree = all(v["agreement"] for v in verdicts)

        # Aggregated warnings
        all_warnings = []
        for v in verdicts:
            all_warnings.extend(v.get("warnings", []))

        # Conservative consensus for action
        actions = [v["recommended_action"] for v in verdicts]
        action = _conservative_consensus(actions)

        # Disagreement flag
        agreement = len(set(actions)) == 1

        summaries = " | ".join(v["summary"] for v in verdicts if v.get("summary"))

        return {
            "opportunity_score": round(avg_opp, 3),
            "risk_score": round(avg_risk, 3),
            "agreement": agreement,
            "warnings": list(set(all_warnings)),
            "summary": summaries,
            "recommended_action": action,
            "strategic_action": strategic["recommended_action"] if strategic else None,
            "live_action": live["recommended_action"] if live else None,
        }


def _conservative_consensus(actions: list[str]) -> str:
    """Most conservative action wins."""
    priority = {"REJECT": 0, "REDUCE_SIZE": 1, "WATCH": 2, "APPROVE": 3}
    return min(actions, key=lambda a: priority.get(a, 2))
