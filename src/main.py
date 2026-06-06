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
    """Main paper trading loop."""
    import yfinance as yf  # type: ignore

    from src.ai.advisory import AIAdvisoryLayer
    from src.ai.strategic_planner import StrategicPlanner
    from src.data.yfinance_helpers import configure_yfinance_cache, normalize_yfinance_columns
    from src.engine.decision_engine import DecisionEngine
    from src.execution.paper_broker import PaperBroker, OrderSide
    from src.features.indicators import compute_all_features
    from src.news.news_manager import NewsManager
    from src.portfolio.allocation_engine import AllocationEngine
    from src.profile.client_profile import ClientProfile

    settings = get_settings()
    risk_cfg = get_risk_config()
    strategy_cfg = get_strategy_config()
    profile_cfg = get_profile_config()

    broker_cfg = settings.get("broker", {}).get("paper", {})
    ai_cfg = settings.get("ai", {})
    alerts_cfg = settings.get("alerts", {})
    news_cfg = settings.get("news", {})

    # --- Profil client + plan stratégique ---
    profile = ClientProfile.from_dict(profile_cfg.get("client", {}))
    planner = StrategicPlanner(ai_cfg)
    plan = planner.build_plan(profile)
    allocation_engine = AllocationEngine(risk_cfg, strategy_plan=plan)

    logger = logging.getLogger("main")
    logger.info(
        "Client profile: %s | target=%.0f%% | allocation=%s",
        profile.risk_profile_label,
        plan.target_annual_return * 100,
        plan.allocation,
    )
    logger.info("Strategy reasoning: %s", plan.reasoning)

    broker = PaperBroker(
        initial_capital=broker_cfg.get("initial_capital", 100_000.0),
        commission_flat=broker_cfg.get("commission_per_trade", 1.0),
        slippage_pct=broker_cfg.get("slippage_pct", 0.001),
    )
    ai_layer = AIAdvisoryLayer(ai_cfg) if ai_cfg.get("enabled") else None
    engine = DecisionEngine(risk_cfg, settings, strategy_cfg, ai_layer=ai_layer)

    # --- News Manager ---
    news_manager = NewsManager(news_cfg, universe=universe)
    news_manager.start()
    audit = AuditLogger()
    alerts = AlertDispatcher(alerts_cfg)
    reporter = DailyReporter()
    # logger déjà défini plus haut (après import)

    universe = (
        settings.get("universe", {}).get("swing", []) +
        settings.get("universe", {}).get("long_term", [])
    )[:10]  # Cap at 10 for now

    configure_yfinance_cache()
    logger.info("Paper trading loop started. Universe: %s", universe)
    alerts.send("Bot started in PAPER MODE.", "INFO")

    cycle = 0
    decisions_today: list[dict] = []
    last_regime = "unknown"

    while True:
        try:
            cycle += 1
            logger.info("--- Cycle %d ---", cycle)

            # Download latest daily bars
            data_map: dict = {}
            for ticker in universe:
                try:
                    df_raw = yf.download(
                        ticker, period="2y", interval="1d",
                        auto_adjust=True, progress=False
                    )
                    if df_raw.empty or len(df_raw) < 50:
                        continue
                    df_raw = normalize_yfinance_columns(df_raw)
                    data_map[ticker] = compute_all_features(df_raw)
                except Exception as exc:
                    logger.warning("Data download failed for %s: %s", ticker, exc)

            if not data_map:
                logger.error("No data available. Sleeping 60s.")
                time.sleep(60)
                continue

            portfolio_state = broker.get_portfolio_state()
            open_positions = broker.get_open_positions()
            trade_history = broker.get_trade_history()

            # Injecter les scores news dans l'agrégateur avant le cycle
            if news_manager.enabled:
                engine.apply_news_risk(news_manager.get_risk_scores())

            # Run decision cycle
            decisions = engine.run_cycle(
                data_map=data_map,
                portfolio_state=portfolio_state,
                open_positions=open_positions,
                trade_history=trade_history,
            )

            # Process executable decisions
            current_prices = {t: float(df["close"].iloc[-1]) for t, df in data_map.items()}
            broker.fill_pending_orders(current_prices)
            broker.check_stops_and_targets(current_prices)

            for dec in decisions:
                audit.log_decision(dec.to_dict())
                decisions_today.append(dec.to_dict())

                if dec.final_action == "EXECUTE":
                    signal = dec.signal
                    side = OrderSide.BUY if signal.signal.value == "BUY" else OrderSide.SELL
                    shares = dec.risk_verdict.approved_shares
                    if shares > 0:
                        broker.submit_market_order(
                            asset=signal.asset,
                            side=side,
                            quantity=shares,
                            strategy_name=signal.strategy_name,
                            horizon=signal.horizon.value,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                        )
                        broker.fill_pending_orders(current_prices)
                        alerts.send(
                            f"PAPER TRADE: {side.value} {signal.asset} "
                            f"x{shares:.2f} @ {signal.entry_price:.2f} | "
                            f"{signal.strategy_name} | conf={signal.confidence:.2f}",
                            "INFO",
                        )

            # Update regime for reporting
            if decisions:
                last_regime = decisions[0].regime.regime.value

            # Daily report at cycle end
            daily_report = reporter.generate(
                broker.get_portfolio_state(),
                broker.get_trade_history(),
                decisions_today,
                last_regime,
            )
            logger.info("Daily report: PnL=%.2f drawdown=%.2f%%",
                        daily_report["daily_pnl"],
                        daily_report["drawdown_pct"] * 100)

            # Sleep until next cycle (daily bar = 24h in production)
            # In development, set to shorter interval
            sleep_seconds = 3600  # 1 hour polling for demonstration
            logger.info("Sleeping %ds until next cycle.", sleep_seconds)
            time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            alerts.send("Bot stopped by user (KeyboardInterrupt).", "WARNING")
            news_manager.stop()
            break
        except Exception as exc:
            logger.critical("Unhandled error in main loop: %s", exc, exc_info=True)
            alerts.send(f"CRITICAL ERROR: {exc}", "CRITICAL")
            time.sleep(30)


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
