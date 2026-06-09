"""
src/risk/risk_manager.py

Global Risk Manager — the last line of defence before execution.

Responsibilities:
  1. Enforce all exposure and loss limits
  2. Size positions using fixed-fractional / Kelly
  3. Approve, reduce, or block trades
  4. Trigger Kill Switch when conditions are critical
  5. Switch to Defensive Mode on portfolio deterioration

The Risk Manager has ABSOLUTE VETO POWER over every trade.
No strategy, AI, or operator instruction can override a block.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.strategies.base import Horizon, Signal, SignalType

logger = logging.getLogger(__name__)


class RiskDecision(str, Enum):
    APPROVE = "APPROVE"
    REDUCE_SIZE = "REDUCE_SIZE"
    BLOCK = "BLOCK"
    FORCE_EXIT = "FORCE_EXIT"
    KILL = "KILL"


@dataclass
class RiskVerdict:
    decision: RiskDecision
    approved_size_usd: float
    approved_shares: float
    reason: str
    details: dict = field(default_factory=dict)

    @property
    def is_executable(self) -> bool:
        return self.decision in (RiskDecision.APPROVE, RiskDecision.REDUCE_SIZE)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "approved_size_usd": round(self.approved_size_usd, 2),
            "approved_shares": round(self.approved_shares, 4),
            "reason": self.reason,
            "details": self.details,
        }


class KillSwitchTriggered(Exception):
    """Raised when the Kill Switch fires. Bot must halt immediately."""
    pass


class RiskManager:
    """
    Stateful risk manager. Must be instantiated once and passed the
    live portfolio state on each evaluation call.
    """

    def __init__(self, risk_cfg: dict) -> None:
        self.cfg = risk_cfg
        self.r = risk_cfg.get("risk", {})
        self.ks = risk_cfg.get("kill_switch", {})
        self.dm = risk_cfg.get("defensive_mode", {})
        self.ps = risk_cfg.get("position_sizing", {})

        self._kill_switch_active = False
        self._defensive_mode = False
        self._intraday_killed = False

        # Running counters (reset via reset_daily / reset_weekly / reset_monthly)
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._monthly_pnl: float = 0.0
        self._consecutive_blocked: int = 0
        self._api_errors_hour: int = 0
        self._intraday_consecutive_losses: int = 0
        self._swing_consecutive_losses: int = 0

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        signal: Signal,
        portfolio_state: dict,
        open_positions: list[dict],
        rules_approved: bool,
    ) -> RiskVerdict:
        """
        Main entry point. Call after Rules Engine has approved.

        Parameters
        ----------
        signal          : The candidate signal.
        portfolio_state : Current portfolio snapshot.
        open_positions  : List of open position dicts.
        rules_approved  : Whether the Rules Engine approved.

        Returns
        -------
        RiskVerdict
        """
        if self._kill_switch_active:
            raise KillSwitchTriggered("Kill switch is active. Bot is halted.")

        if not rules_approved:
            self._consecutive_blocked += 1
            return RiskVerdict(
                RiskDecision.BLOCK, 0.0, 0.0,
                "Rules Engine rejected — Risk Manager not evaluating.",
                {"consecutive_blocked": self._consecutive_blocked},
            )

        if signal.signal == SignalType.NO_TRADE:
            return RiskVerdict(RiskDecision.BLOCK, 0.0, 0.0, "NO_TRADE signal.")

        total_capital = portfolio_state.get("total_capital", 0.0)

        # ---- Global loss limits ----
        verdict = self._check_loss_limits(total_capital)
        if verdict:
            return verdict

        # ---- Defensive mode ----
        self._update_defensive_mode(portfolio_state)
        if self._defensive_mode:
            verdict = self._apply_defensive_constraints(signal)
            if verdict:
                return verdict

        # ---- Intraday kill ----
        if self._intraday_killed and signal.horizon == Horizon.INTRADAY:
            return RiskVerdict(
                RiskDecision.BLOCK, 0.0, 0.0,
                "Intraday trading halted for today.",
            )

        # ---- Exposure checks ----
        verdict = self._check_exposure(signal, portfolio_state, open_positions)
        if verdict:
            return verdict

        # ---- Position count ----
        max_pos = self.r.get("max_open_positions", 15)
        current_pos_count = portfolio_state.get("open_positions", len(open_positions))
        if current_pos_count >= max_pos:
            return RiskVerdict(
                RiskDecision.BLOCK, 0.0, 0.0,
                f"Max open positions reached ({max_pos}).",
            )

        # ---- Size the position ----
        size_usd = self._calculate_position_size(signal, total_capital)

        # Hard cap from config
        max_pos_usd = self.ps.get("max_position_usd", 10_000.0)
        min_pos_usd = self.ps.get("min_position_usd", 100.0)

        if size_usd < min_pos_usd:
            return RiskVerdict(
                RiskDecision.BLOCK, 0.0, 0.0,
                f"Calculated size ${size_usd:.2f} below minimum ${min_pos_usd:.2f}.",
            )

        reduced = False
        if size_usd > max_pos_usd:
            size_usd = max_pos_usd
            reduced = True

        # Apply horizon allocation caps
        size_usd, reduced_alloc = self._apply_allocation_cap(
            signal, size_usd, portfolio_state
        )
        reduced = reduced or reduced_alloc

        if size_usd <= 0:
            self._consecutive_blocked += 1
            return RiskVerdict(
                RiskDecision.BLOCK, 0.0, 0.0,
                "Allocation cap leaves no available budget for this horizon.",
                {"consecutive_blocked": self._consecutive_blocked},
            )

        if size_usd < min_pos_usd:
            self._consecutive_blocked += 1
            return RiskVerdict(
                RiskDecision.BLOCK, 0.0, 0.0,
                f"Allocation-adjusted size ${size_usd:.2f} below minimum ${min_pos_usd:.2f}.",
                {"consecutive_blocked": self._consecutive_blocked},
            )

        shares = (
            size_usd / signal.entry_price
            if signal.entry_price and signal.entry_price > 0
            else 0.0
        )

        if shares <= 0:
            self._consecutive_blocked += 1
            return RiskVerdict(
                RiskDecision.BLOCK, 0.0, 0.0,
                "Cannot size trade because entry price is missing or invalid.",
                {"consecutive_blocked": self._consecutive_blocked},
            )

        self._consecutive_blocked = 0
        decision = RiskDecision.REDUCE_SIZE if reduced else RiskDecision.APPROVE

        logger.info(
            "RiskManager APPROVED: %s %s size=$%.2f shares=%.4f",
            signal.asset, signal.signal.value, size_usd, shares,
        )

        return RiskVerdict(
            decision=decision,
            approved_size_usd=round(size_usd, 2),
            approved_shares=round(shares, 4),
            reason=f"{'Reduced: ' if reduced else ''}Approved — R:R={signal.risk_reward}, conf={signal.confidence:.2f}",
            details={
                "total_capital": total_capital,
                "daily_pnl_pct": round(self._daily_pnl / (total_capital + 1e-10), 4),
                "defensive_mode": self._defensive_mode,
            },
        )

    def register_trade_result(
        self, pnl: float, horizon: Horizon, total_capital: float
    ) -> None:
        """Call after every completed trade to update running P&L."""
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._monthly_pnl += pnl

        if pnl < 0:
            if horizon == Horizon.INTRADAY:
                self._intraday_consecutive_losses += 1
            elif horizon == Horizon.SWING:
                self._swing_consecutive_losses += 1
        else:
            if horizon == Horizon.INTRADAY:
                self._intraday_consecutive_losses = 0
            elif horizon == Horizon.SWING:
                self._swing_consecutive_losses = 0

        # Check intraday daily loss
        intraday_daily_loss_pct = self.r.get("intraday", {}).get(
            "stop_after_daily_loss_pct", 0.005
        )
        if (
            horizon == Horizon.INTRADAY
            and self._daily_pnl / (total_capital + 1e-10) < -intraday_daily_loss_pct
        ):
            logger.warning("Intraday daily loss limit hit — killing intraday for today.")
            self._intraday_killed = True

        # Global daily loss
        if self._daily_pnl / (total_capital + 1e-10) < -self.ks.get("daily_loss_pct", 0.01):
            self.trigger_kill_switch("Daily loss limit exceeded.")

    def register_api_error(self, total_capital: float) -> None:
        self._api_errors_hour += 1
        if self._api_errors_hour >= self.ks.get("max_api_errors_per_hour", 10):
            self.trigger_kill_switch("Too many API errors in the past hour.")

    def trigger_kill_switch(self, reason: str) -> None:
        logger.critical("KILL SWITCH TRIGGERED: %s", reason)
        self._kill_switch_active = True
        raise KillSwitchTriggered(reason)

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0
        self._intraday_killed = False
        self._intraday_consecutive_losses = 0
        self._api_errors_hour = 0
        logger.info("Daily risk counters reset.")

    def reset_weekly(self) -> None:
        self._weekly_pnl = 0.0
        self._swing_consecutive_losses = 0

    def reset_monthly(self) -> None:
        self._monthly_pnl = 0.0

    @property
    def is_halted(self) -> bool:
        return self._kill_switch_active

    @property
    def is_defensive(self) -> bool:
        return self._defensive_mode

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _check_loss_limits(self, total_capital: float) -> Optional[RiskVerdict]:
        if total_capital <= 0:
            return RiskVerdict(RiskDecision.BLOCK, 0.0, 0.0, "Zero capital.")

        daily_pct = self._daily_pnl / total_capital
        weekly_pct = self._weekly_pnl / total_capital
        monthly_pct = self._monthly_pnl / total_capital

        if daily_pct < -self.r.get("max_daily_loss_pct", 0.01):
            self.trigger_kill_switch(f"Daily loss {daily_pct:.2%} exceeded limit.")

        if weekly_pct < -self.r.get("max_weekly_loss_pct", 0.03):
            self.trigger_kill_switch(f"Weekly loss {weekly_pct:.2%} exceeded limit.")

        if monthly_pct < -self.r.get("max_monthly_drawdown_pct", 0.06):
            self.trigger_kill_switch(f"Monthly drawdown {monthly_pct:.2%} exceeded limit.")

        return None

    def _update_defensive_mode(self, portfolio_state: dict) -> None:
        dd = portfolio_state.get("drawdown_pct", 0.0)
        trigger = self.dm.get("trigger_drawdown_pct", 0.05)
        if dd < -trigger and not self._defensive_mode:
            logger.warning(
                "Entering DEFENSIVE MODE — drawdown %.2f%% exceeds trigger %.2f%%",
                dd * 100, trigger * 100
            )
            self._defensive_mode = True
        elif dd > -trigger * 0.5 and self._defensive_mode:
            logger.info("Exiting defensive mode — drawdown recovered.")
            self._defensive_mode = False

    def _apply_defensive_constraints(
        self, signal: Signal
    ) -> Optional[RiskVerdict]:
        if signal.horizon == Horizon.INTRADAY and not self.dm.get("intraday_enabled", False):
            return RiskVerdict(
                RiskDecision.BLOCK, 0.0, 0.0,
                "Defensive mode: intraday trading disabled.",
            )
        return None

    def _check_exposure(
        self,
        signal: Signal,
        portfolio_state: dict,
        open_positions: list[dict],
    ) -> Optional[RiskVerdict]:
        total_capital = portfolio_state.get("total_capital", 1.0)
        total_exposure = portfolio_state.get("total_exposure", 0.0)
        asset_exposure = portfolio_state.get("asset_exposure", {})
        horizon_exposure = portfolio_state.get("horizon_exposure", {})

        # Intraday signals use their own dedicated budget — don't block them
        # because of swing/long-term positions filling the global cap.
        if signal.horizon == Horizon.INTRADAY:
            intraday_exp = horizon_exposure.get("intraday", 0.0)
            max_intraday = self.r.get("max_intraday_allocation_pct", 0.30)
            if intraday_exp / (total_capital + 1e-10) >= max_intraday:
                return RiskVerdict(
                    RiskDecision.BLOCK, 0.0, 0.0,
                    f"Intraday allocation {intraday_exp/total_capital:.1%} at cap {max_intraday:.1%}.",
                )
        else:
            # Total exposure cap for swing / long-term
            max_exp = self.r.get("max_total_exposure_pct", 0.80)
            if total_exposure / total_capital >= max_exp:
                return RiskVerdict(
                    RiskDecision.BLOCK, 0.0, 0.0,
                    f"Total exposure {total_exposure/total_capital:.1%} at max {max_exp:.1%}.",
                )

        # Asset cap
        max_asset = self.r.get("max_exposure_per_asset_pct", 0.10)
        current = asset_exposure.get(signal.asset, 0.0)
        if current / total_capital >= max_asset:
            return RiskVerdict(
                RiskDecision.BLOCK, 0.0, 0.0,
                f"Asset {signal.asset} already at {current/total_capital:.1%} (max {max_asset:.1%}).",
            )

        return None

    def _calculate_position_size(
        self, signal: Signal, total_capital: float
    ) -> float:
        """
        Fixed-fractional sizing with optional strategy guidance.
        If a strategy provides a requested size, use it. Otherwise apply
        a risk multiplier to the fixed risk sizing.
        """
        requested_size_usd = signal.metadata.get("requested_size_usd")
        if requested_size_usd is not None:
            try:
                requested = float(requested_size_usd)
                if requested > 0:
                    return requested
            except (TypeError, ValueError):
                pass

        requested_size_pct = signal.metadata.get("requested_size_pct")
        if requested_size_pct is not None:
            try:
                pct = float(requested_size_pct)
                if 0.0 < pct <= 1.0:
                    return total_capital * pct
            except (TypeError, ValueError):
                pass

        risk_multiplier = 1.0
        if "risk_multiplier" in signal.metadata:
            try:
                risk_multiplier = float(signal.metadata.get("risk_multiplier", 1.0))
            except (TypeError, ValueError):
                risk_multiplier = 1.0

        risk_pct = self.r.get("max_risk_per_trade_pct", 0.005) * risk_multiplier
        risk_usd = total_capital * risk_pct

        entry = signal.entry_price or 0.0
        sl = signal.stop_loss

        if sl and entry > 0:
            stop_distance = abs(entry - sl)
            if stop_distance > 0:
                shares = risk_usd / stop_distance
                return shares * entry

        size_multiplier = 1.0
        if "size_multiplier" in signal.metadata:
            try:
                size_multiplier = float(signal.metadata.get("size_multiplier", 1.0))
            except (TypeError, ValueError):
                size_multiplier = 1.0

        # Fallback: 1% of capital scaled by DCA/regime size.
        return total_capital * 0.01 * size_multiplier

    def _apply_allocation_cap(
        self, signal: Signal, size_usd: float, portfolio_state: dict
    ) -> tuple[float, bool]:
        total_capital = portfolio_state.get("total_capital", 1.0)
        horizon_exposure = portfolio_state.get("horizon_exposure", {})
        reduced = False

        caps = {
            Horizon.INTRADAY: self.r.get("max_intraday_allocation_pct", 0.30),
            Horizon.SWING: self.r.get("max_swing_allocation_pct", 0.50),
            Horizon.LONG_TERM: self.r.get("max_long_term_allocation_pct", 0.90),
        }

        cap_pct = caps.get(signal.horizon, 0.20)
        cap_usd = total_capital * cap_pct
        current_in_horizon = horizon_exposure.get(signal.horizon.value, 0.0)
        available = cap_usd - current_in_horizon

        if available <= 0:
            return 0.0, True

        if size_usd > available:
            size_usd = available
            reduced = True

        return size_usd, reduced

    def _check_consecutive_blocked(self) -> None:
        max_blocked = self.ks.get("max_consecutive_blocked_trades", 10)
        if self._consecutive_blocked >= max_blocked:
            logger.critical(
                "Kill switch: %d consecutive blocked trades.", self._consecutive_blocked
            )
            self.trigger_kill_switch(
                f"{self._consecutive_blocked} consecutive blocked trades — possible system fault."
            )
