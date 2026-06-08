"""
src/main.py

Bot entry point. Loads config, initialises all modules, runs the main loop.

Usage:
    python src/main.py                   # Paper mode (default)
    python src/main.py --backtest SPY    # Run backtest on SPY
    python src/main.py --mode paper      # Explicit paper mode

LIVE MODE IS DISABLED BY DEFAULT.
To enable live trading:
  1. Complete full backtest validation
  2. Complete >= 30 days paper trading
  3. Set live_enabled: true in settings.yaml  AND  mode: live
  4. Verify checklist in docs/LIVE_CHECKLIST.md
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import get_profile_config, get_risk_config, get_settings, get_strategy_config, is_live_enabled
from src.monitoring.logger import AuditLogger, AlertDispatcher, DailyReporter, configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Hybrid Trading Bot")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--backtest", metavar="TICKER", help="Run backtest on ticker and exit")
    parser.add_argument("--strategy", default="trend_following", help="Strategy for backtest")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def run_backtest(ticker: str, strategy_name: str) -> None:
    """Run a quick backtest and print summary."""
    import yfinance as yf  # type: ignore

    from src.backtesting.backtester import Backtester
    from src.features.indicators import compute_all_features
    from src.data.yfinance_helpers import configure_yfinance_cache, normalize_yfinance_columns
    from src.strategies.breakout import BreakoutStrategy
    from src.strategies.rsi_dip_buyer import RSIDipBuyerStrategy
    from src.strategies.trend_following import TrendFollowingStrategy
    from src.strategies.true_dca import TrueDCAStrategy

    strategy_cfg = get_strategy_config()
    scfg = strategy_cfg.get("strategies", {})

    strategies = {
        "true_dca":        TrueDCAStrategy(scfg.get("true_dca", {})),
        "trend_following": TrendFollowingStrategy(scfg.get("trend_following", {})),
        "breakout":        BreakoutStrategy(scfg.get("breakout", {})),
        "rsi_dip_buyer":   RSIDipBuyerStrategy(scfg.get("rsi_dip_buyer", {})),
    }

    if strategy_name not in strategies:
        print(f"Unknown strategy: {strategy_name}. Choose from {list(strategies.keys())}")
        sys.exit(1)

    configure_yfinance_cache()
    print(f"\nDownloading {ticker} data...")
    df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)

    if df.empty:
        print(f"No data for {ticker}")
        sys.exit(1)

    df = normalize_yfinance_columns(df)

    print(f"Running backtest: {strategy_name} on {ticker} ({len(df)} bars)...")
    settings = get_settings()
    risk_cfg = get_risk_config()
    broker_cfg = settings.get("broker", {}).get("paper", {})
    sizing_cfg = risk_cfg.get("position_sizing", {})

    bt = Backtester(
        initial_capital=broker_cfg.get("initial_capital", 500.0),
        commission_flat=broker_cfg.get("commission_per_trade", 0.0),
        slippage_pct=broker_cfg.get("slippage_pct", 0.001),
        risk_pct_per_trade=sizing_cfg.get("fixed_risk_pct", 0.005),
        min_position_usd=sizing_cfg.get("min_position_usd", 0.0),
        max_position_usd=sizing_cfg.get("max_position_usd"),
    )
    result = bt.run(strategies[strategy_name], df, asset=ticker)
    print(result.summary())

    # Walk-forward
    print("\nRunning walk-forward analysis (5 folds)...")
    wf_results = bt.walk_forward(strategies[strategy_name], df, asset=ticker, n_splits=5)
    for r in wf_results:
        print(f"  {r.strategy_name}: return={r.metrics.get('total_return_pct', 0):.2f}% "
              f"| sharpe={r.metrics.get('sharpe_ratio', 0):.3f} "
              f"| maxDD={r.metrics.get('max_drawdown_pct', 0):.2f}%")


def run_paper_loop() -> None:
    """Main paper trading loop — piloté par le Market Watcher."""
    from src.ai.advisory import AIAdvisoryLayer
    from src.ai.market_analyst import MarketAnalyst
    from src.ai.strategic_planner import StrategicPlanner
    from src.alerts.alert_manager import AlertLevel, AlertManager, AlertType
    from src.data.yfinance_helpers import configure_yfinance_cache
    from src.engine.decision_engine import DecisionEngine
    from src.execution.paper_broker import PaperBroker
    from src.features.feature_engine import FeatureEngine
    from src.ml.regime_model import RegimeModel
    from src.monitoring.logger import AuditLogger, DailyReporter
    from src.news.news_manager import NewsManager
    from src.portfolio.allocation_engine import AllocationEngine
    from src.profile.client_profile import ClientProfile
    from src.watchers.market_watcher import MarketWatcher
    from src.watchers.portfolio_watcher import PortfolioWatcher
    from src.watchers.universe_builder import MarketUniverseBuilder

    settings = get_settings()
    risk_cfg = get_risk_config()
    strategy_cfg = get_strategy_config()
    profile_cfg = get_profile_config()

    broker_cfg     = settings.get("broker", {}).get("paper", {})
    ai_cfg         = settings.get("ai", {})
    alerts_cfg     = settings.get("alerts", {})
    news_cfg       = settings.get("news", {})
    analyst_cfg    = settings.get("market_analyst", {})

    log = logging.getLogger("main")

    # --- Profil client + plan stratégique ---
    profile = ClientProfile.from_dict(profile_cfg.get("client", {}))
    planner = StrategicPlanner(ai_cfg)
    plan    = planner.build_plan(profile)
    AllocationEngine(risk_cfg, strategy_plan=plan)

    log.info("Profile: %s | target=%.0f%% | %s", profile.risk_profile_label,
             plan.target_annual_return * 100, plan.allocation)

    # --- Universe ---
    universe = MarketUniverseBuilder().build(profile)
    log.info("Universe: %s", universe.to_dict())

    # --- Broker ---
    broker = PaperBroker(
        initial_capital=broker_cfg.get("initial_capital", 100_000.0),
        commission_flat=broker_cfg.get("commission_per_trade", 0.0),
        slippage_pct=broker_cfg.get("slippage_pct", 0.001),
    )
    # Restore previous paper session if available
    broker.load_state()

    # --- Composants ---
    ai_layer    = AIAdvisoryLayer(ai_cfg) if ai_cfg.get("enabled") else None
    engine      = DecisionEngine(risk_cfg, settings, strategy_cfg, ai_layer=ai_layer)

    alert_mgr   = AlertManager(alerts_cfg)
    portfolio_w = PortfolioWatcher(risk_cfg, alert_mgr)
    feature_eng = FeatureEngine()
    regime_mdl  = RegimeModel()
    analyst     = MarketAnalyst(analyst_cfg)

    news_mgr    = NewsManager(news_cfg, universe=universe.primary_tickers)
    news_mgr.start()

    configure_yfinance_cache()
    audit    = AuditLogger()
    reporter = DailyReporter()

    # --- Market Watcher (cœur) ---
    watcher = MarketWatcher(
        cfg=settings,
        broker=broker,
        decision_engine=engine,
        alert_manager=alert_mgr,
        portfolio_watcher=portfolio_w,
        feature_engine=feature_eng,
        regime_model=regime_mdl,
        market_analyst=analyst,
        news_manager=news_mgr,
        client_profile=profile,
        universe=universe,
    )
    watcher.start()

    alert_mgr.send(AlertLevel.INFO, AlertType.GENERAL, "Bot started in PAPER MODE.")
    log.info("Market Watcher running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(60)
            # Rapport périodique
            state = watcher.current_state
            if state:
                log.info(
                    "Status: regime=%s risk=%s vix=%s cycles=%d",
                    state.regime, state.risk_level,
                    f"{state.vix:.1f}" if state.vix else "N/A",
                    watcher.cycle_count,
                )
            # Rapport journalier + sauvegarde de l'état
            ps = broker.get_portfolio_state()
            daily = reporter.generate(ps, broker.get_trade_history(), [], state.regime if state else "unknown")
            audit.log_decision({"type": "heartbeat", "portfolio": ps, "regime": state.to_dict() if state else {}})
            broker.save_state()  # persist positions every minute

    except KeyboardInterrupt:
        log.info("Bot stopped by user.")
        alert_mgr.send(AlertLevel.WARNING, AlertType.GENERAL, "Bot stopped by user.")
        watcher.stop()
        news_mgr.stop()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if args.backtest:
        run_backtest(args.backtest, args.strategy)
        return

    # Safety check for live mode
    if args.mode == "live":
        if not is_live_enabled():
            print(
                "\n❌  LIVE MODE BLOCKED.\n"
                "    live_enabled is false in settings.yaml.\n"
                "    Complete full validation before enabling live trading.\n"
            )
            sys.exit(1)
        print("⚠️  LIVE MODE — real capital at risk. Starting in 5 seconds...")
        time.sleep(5)
        # live_broker would be imported here
        # run_live_loop()
    else:
        run_paper_loop()


if __name__ == "__main__":
    main()
