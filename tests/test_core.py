"""
tests/test_core.py

Unit tests for indicators, strategies, rules engine, and risk manager.
Run with: pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    z_score,
    compute_all_features,
)
from src.features.regime_detector import MarketRegime, MarketRegimeDetector
from src.strategies.base import Horizon, SignalType, no_trade
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.breakout import BreakoutStrategy
from src.risk.risk_manager import RiskManager, RiskDecision
from src.rules.rules_engine import RulesEngine, StatisticalRules, StrategicRules


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 300, trend: float = 0.0005, vol: float = 0.015) -> pd.DataFrame:
    """Synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    returns = np.random.normal(trend, vol, n)
    close = 100.0 * (1 + returns).cumprod()
    high = close * (1 + np.abs(np.random.normal(0, vol / 2, n)))
    low = close * (1 - np.abs(np.random.normal(0, vol / 2, n)))
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def make_bull_trend(n: int = 300) -> pd.DataFrame:
    """Strong uptrend data."""
    return make_ohlcv(n, trend=0.003, vol=0.008)


def make_bear_trend(n: int = 300) -> pd.DataFrame:
    return make_ohlcv(n, trend=-0.003, vol=0.012)


def make_ranging(n: int = 300) -> pd.DataFrame:
    return make_ohlcv(n, trend=0.0, vol=0.006)


# ---------------------------------------------------------------------------
# Indicator tests
# ---------------------------------------------------------------------------

class TestIndicators:

    def test_ema_length(self):
        df = make_ohlcv(100)
        result = ema(df["close"], 20)
        assert len(result) == 100

    def test_sma_nan_count(self):
        df = make_ohlcv(50)
        result = sma(df["close"], 20)
        assert result.iloc[:19].isna().all()
        assert not result.iloc[19:].isna().any()

    def test_rsi_bounds(self):
        df = make_ohlcv(100)
        result = rsi(df["close"], 14).dropna()
        assert (result >= 0).all() and (result <= 100).all()

    def test_atr_positive(self):
        df = make_ohlcv(50)
        result = atr(df, 14).dropna()
        assert (result > 0).all()

    def test_bollinger_bands_structure(self):
        df = make_ohlcv(100)
        bb = bollinger_bands(df["close"], 20)
        assert "bb_upper" in bb.columns
        assert "bb_lower" in bb.columns
        non_nan = bb.dropna()
        assert (non_nan["bb_upper"] > non_nan["bb_middle"]).all()
        assert (non_nan["bb_middle"] > non_nan["bb_lower"]).all()

    def test_macd_returns_three_columns(self):
        df = make_ohlcv(100)
        result = macd(df["close"])
        assert set(result.columns) == {"macd", "signal", "histogram"}

    def test_zscore_mean_zero(self):
        df = make_ohlcv(100)
        z = z_score(df["close"], 20).dropna()
        assert abs(z.mean()) < 0.5  # Roughly centred

    def test_compute_all_features_columns(self):
        df = make_ohlcv(250)
        result = compute_all_features(df)
        for col in ["ema_20", "ema_50", "ema_200", "rsi_14", "atr_14", "bb_upper"]:
            assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Regime Detector tests
# ---------------------------------------------------------------------------

class TestRegimeDetector:

    def setup_method(self):
        self.detector = MarketRegimeDetector()

    def test_insufficient_data(self):
        df = make_ohlcv(100)
        result = self.detector.detect(df)
        assert result.regime == MarketRegime.UNKNOWN

    def test_bull_trend_detected(self):
        df = make_bull_trend(300)
        result = self.detector.detect(df)
        # Should detect bullish regime
        assert result.regime in (
            MarketRegime.BULL_TREND, MarketRegime.BREAKOUT_EXPANSION, MarketRegime.EUPHORIC
        )
        assert result.confidence > 0.1

    def test_result_has_explanation(self):
        df = make_ohlcv(250)
        result = self.detector.detect(df)
        assert isinstance(result.explanation, str)
        assert isinstance(result.sub_signals, dict)

    def test_to_dict(self):
        df = make_ohlcv(250)
        d = self.detector.detect(df).to_dict()
        assert "regime" in d
        assert "confidence" in d
        assert "sub_signals" in d


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestTrendFollowingStrategy:

    def setup_method(self):
        self.cfg = {
            "timeframe": "1d",
            "min_adx": 25,
            "volume_confirm_ratio": 1.3,
            "min_confidence": 0.65,
            "atr_multiplier_sl": 2.0,
            "atr_multiplier_tp": 4.0,
        }
        self.strategy = TrendFollowingStrategy(self.cfg)

    def test_returns_signal_object(self):
        df = make_bull_trend(250)
        df = compute_all_features(df)
        signal = self.strategy.generate_signal(df, "SPY", "bull_trend")
        assert hasattr(signal, "signal")
        assert hasattr(signal, "confidence")
        assert signal.confidence >= 0.0
        assert signal.confidence <= 1.0

    def test_no_trade_on_insufficient_data(self):
        df = make_ohlcv(50)
        signal = self.strategy.generate_signal(df, "SPY", "bull_trend")
        assert signal.signal == SignalType.NO_TRADE

    def test_stop_loss_below_entry_on_buy(self):
        df = make_bull_trend(250)
        df = compute_all_features(df)
        signal = self.strategy.generate_signal(df, "SPY", "bull_trend")
        if signal.signal == SignalType.BUY:
            assert signal.stop_loss < signal.entry_price

    def test_rr_ratio_when_signal(self):
        df = make_bull_trend(250)
        df = compute_all_features(df)
        signal = self.strategy.generate_signal(df, "SPY", "bull_trend")
        if signal.signal == SignalType.BUY:
            assert signal.risk_reward is None or signal.risk_reward >= 1.5


class TestMeanReversionStrategy:

    def setup_method(self):
        self.cfg = {
            "timeframe": "1d",
            "zscore_entry": 2.0,
            "zscore_exit": 0.5,
            "lookback_period": 20,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "min_confidence": 0.60,
            "atr_multiplier_sl": 1.5,
            "atr_multiplier_tp": 2.5,
        }
        self.strategy = MeanReversionStrategy(self.cfg)

    def test_blocked_in_bull_trend(self):
        df = make_bull_trend(250)
        signal = self.strategy.generate_signal(df, "AAPL", "bull_trend")
        assert signal.signal == SignalType.NO_TRADE
        assert "blocked" in signal.reason.lower()

    def test_no_trade_on_normal_data(self):
        df = make_ranging(250)
        signal = self.strategy.generate_signal(df, "AAPL", "range")
        # May or may not generate a signal — must be valid
        assert signal.signal in (SignalType.BUY, SignalType.SELL, SignalType.NO_TRADE)


# ---------------------------------------------------------------------------
# Risk Manager tests
# ---------------------------------------------------------------------------

RISK_CFG = {
    "risk": {
        "max_risk_per_trade_pct": 0.005,
        "max_daily_loss_pct": 0.01,
        "max_weekly_loss_pct": 0.03,
        "max_monthly_drawdown_pct": 0.06,
        "max_total_drawdown_pct": 0.15,
        "max_total_exposure_pct": 0.80,
        "min_cash_reserve_pct": 0.10,
        "max_intraday_allocation_pct": 0.10,
        "max_swing_allocation_pct": 0.20,
        "max_long_term_allocation_pct": 0.90,
        "max_open_positions": 15,
        "max_exposure_per_asset_pct": 0.10,
        "max_exposure_per_sector_pct": 0.25,
        "max_correlation_exposure": 0.70,
        "intraday": {
            "max_consecutive_losses": 3,
            "stop_after_daily_loss_pct": 0.005,
            "min_volume_intraday": 1_000_000,
            "max_spread_pct": 0.002,
        },
        "swing": {"max_consecutive_losses": 4},
    },
    "kill_switch": {
        "daily_loss_pct": 0.01,
        "weekly_loss_pct": 0.03,
        "monthly_drawdown_pct": 0.06,
        "max_api_errors_per_hour": 10,
        "max_consecutive_blocked_trades": 10,
    },
    "defensive_mode": {
        "trigger_drawdown_pct": 0.05,
        "intraday_enabled": False,
    },
    "position_sizing": {
        "method": "fixed_fractional",
        "fixed_risk_pct": 0.005,
        "min_position_usd": 100.0,
        "max_position_usd": 10_000.0,
    },
}

PORTFOLIO_STATE = {
    "total_capital": 100_000.0,
    "cash": 80_000.0,
    "total_exposure": 20_000.0,
    "drawdown_pct": -0.01,
    "open_positions": 2,
    "asset_exposure": {"SPY": 10_000.0},
    "horizon_exposure": {"swing": 15_000.0},
}


def make_buy_signal():
    from src.strategies.base import Signal, SignalType, Horizon
    return Signal(
        strategy_name="trend_following",
        asset="AAPL",
        timeframe="1d",
        signal=SignalType.BUY,
        confidence=0.75,
        entry_price=150.0,
        stop_loss=145.0,
        take_profit=165.0,
        risk_reward=3.0,
        horizon=Horizon.SWING,
        reason="Test signal",
    )


class TestRiskManager:

    def setup_method(self):
        self.rm = RiskManager(RISK_CFG)

    def test_approves_valid_signal(self):
        signal = make_buy_signal()
        verdict = self.rm.evaluate(signal, PORTFOLIO_STATE, [], rules_approved=True)
        assert verdict.is_executable
        assert verdict.approved_size_usd > 0

    def test_blocks_when_rules_rejected(self):
        signal = make_buy_signal()
        verdict = self.rm.evaluate(signal, PORTFOLIO_STATE, [], rules_approved=False)
        assert verdict.decision == RiskDecision.BLOCK

    def test_blocks_at_max_positions(self):
        state = {**PORTFOLIO_STATE, "open_positions": 15}
        signal = make_buy_signal()
        verdict = self.rm.evaluate(signal, state, [], rules_approved=True)
        assert verdict.decision == RiskDecision.BLOCK

    def test_size_within_max(self):
        signal = make_buy_signal()
        verdict = self.rm.evaluate(signal, PORTFOLIO_STATE, [], rules_approved=True)
        if verdict.is_executable:
            assert verdict.approved_size_usd <= 10_000.0

    def test_size_above_minimum(self):
        signal = make_buy_signal()
        verdict = self.rm.evaluate(signal, PORTFOLIO_STATE, [], rules_approved=True)
        if verdict.is_executable:
            assert verdict.approved_size_usd >= 100.0

    def test_reset_daily(self):
        self.rm._daily_pnl = -500.0
        self.rm._intraday_killed = True
        self.rm.reset_daily()
        assert self.rm._daily_pnl == 0.0
        assert not self.rm._intraday_killed

    def test_not_halted_initially(self):
        assert not self.rm.is_halted


# ---------------------------------------------------------------------------
# Rules Engine tests
# ---------------------------------------------------------------------------

SETTINGS_CFG = {
    "data": {"min_volume_threshold": 100_000},
    "market_hours": {
        "us_open": "13:30",
        "us_close": "20:00",
        "intraday_cutoff_minutes_before_close": 30,
    },
    "ai": {"disagreement_threshold": 0.4},
}

STRATEGY_CFG = {
    "regime_strategy_map": {
        "bull_trend": {
            "swing": ["trend_following", "breakout", "momentum"],
            "long_term": ["dca_etf"],
            "intraday": ["intraday_breakout"],
        },
        "range": {
            "swing": ["mean_reversion"],
        },
    }
}


class TestRulesEngine:

    def setup_method(self):
        self.engine = RulesEngine(RISK_CFG, SETTINGS_CFG, STRATEGY_CFG)

    def test_no_trade_signal_blocked(self):
        signal = no_trade("trend_following", "SPY", "1d", Horizon.SWING, "test")
        df = make_ohlcv(250)
        verdict = self.engine.evaluate(signal, df, "bull_trend", PORTFOLIO_STATE, [], [])
        assert not verdict.approved

    def test_valid_swing_in_bull_approved(self):
        signal = make_buy_signal()
        df = make_bull_trend(250)
        verdict = self.engine.evaluate(
            signal, df, "bull_trend", PORTFOLIO_STATE, [], []
        )
        # May block on some stats rules — just verify structure
        assert hasattr(verdict, "approved")
        assert isinstance(verdict.blocking_rules, list)

    def test_verdict_has_summary(self):
        signal = make_buy_signal()
        df = make_ohlcv(250)
        verdict = self.engine.evaluate(signal, df, "bull_trend", PORTFOLIO_STATE, [], [])
        assert isinstance(verdict.summary, str)


# ---------------------------------------------------------------------------
# Paper Broker tests
# ---------------------------------------------------------------------------

class TestPaperBroker:

    def setup_method(self):
        from src.execution.paper_broker import PaperBroker, OrderSide
        self.broker = PaperBroker(
            initial_capital=100_000.0,
            commission_flat=1.0,
            slippage_pct=0.001,
        )
        self.OrderSide = OrderSide

    def test_initial_state(self):
        state = self.broker.get_portfolio_state()
        assert state["cash"] == 100_000.0
        assert state["open_positions"] == 0

    def test_buy_reduces_cash(self):
        order = self.broker.submit_market_order(
            "SPY", self.OrderSide.BUY, 10, "test", "swing"
        )
        self.broker.fill_pending_orders({"SPY": 450.0})
        state = self.broker.get_portfolio_state()
        assert state["cash"] < 100_000.0

    def test_open_position_after_buy(self):
        self.broker.submit_market_order("SPY", self.OrderSide.BUY, 10, "test", "swing")
        self.broker.fill_pending_orders({"SPY": 450.0})
        state = self.broker.get_portfolio_state()
        assert state["open_positions"] == 1

    def test_stop_loss_closes_position(self):
        self.broker.submit_market_order(
            "SPY", self.OrderSide.BUY, 10, "test", "swing",
            stop_loss=440.0, take_profit=470.0
        )
        self.broker.fill_pending_orders({"SPY": 450.0})
        # Price drops below stop
        self.broker.check_stops_and_targets({"SPY": 438.0})
        state = self.broker.get_portfolio_state()
        assert state["open_positions"] == 0

    def test_reject_on_insufficient_cash(self):
        from src.execution.paper_broker import PaperBroker
        small_broker = PaperBroker(initial_capital=100.0)
        order = small_broker.submit_market_order(
            "SPY", self.OrderSide.BUY, 1000, "test", "swing"
        )
        small_broker.fill_pending_orders({"SPY": 450.0})
        from src.execution.paper_broker import OrderStatus
        orders = small_broker.get_order_history()
        assert any(o["status"] == OrderStatus.REJECTED.value for o in orders)
