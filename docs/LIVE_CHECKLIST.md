# ============================================================
# LIVE TRADING CHECKLIST
# ============================================================
# Complete EVERY item before switching mode: live
# This document is the final gate before real capital is at risk.
# ============================================================

## 1. BACKTESTING VALIDATION
- [ ] Each strategy backtested independently on ≥ 2 years of data
- [ ] All backtests include realistic commissions ($1 flat + slippage 0.1%)
- [ ] All backtests include slippage simulation
- [ ] Walk-forward analysis completed (5+ folds, OOS only)
- [ ] Stress tests run on crisis periods: 2008, 2020, 2022
- [ ] Sharpe ratio ≥ 0.8 on out-of-sample data
- [ ] Max drawdown ≤ 15% in all tested periods
- [ ] Win rate ≥ 40% and profit factor ≥ 1.3
- [ ] No overfitting: OOS performance ≥ 70% of IS performance
- [ ] Buy-and-hold comparison computed for each strategy

## 2. PAPER TRADING VALIDATION
- [ ] Paper trading run for ≥ 30 calendar days
- [ ] All strategies active during paper trading period
- [ ] No critical errors or crashes during paper period
- [ ] All risk limits respected (no overrides)
- [ ] Kill switch tested and confirmed working
- [ ] Defensive mode triggered and confirmed working
- [ ] Daily and weekly reports generated correctly
- [ ] Alert system (Telegram/Discord) confirmed delivering messages
- [ ] Audit trail verified for every decision
- [ ] Paper trading P&L within expected range of backtest

## 3. RISK MANAGER VERIFICATION
- [ ] Daily loss limit tested: confirms halt at -1%
- [ ] Weekly loss limit tested: confirms halt at -3%
- [ ] Monthly drawdown limit tested: confirms halt at -6%
- [ ] Max position count limit enforced
- [ ] Max exposure per asset enforced
- [ ] Correlation limit enforced
- [ ] Intraday kill switch works after daily loss
- [ ] Consecutive blocked trades trigger verified
- [ ] All kill switch conditions individually verified

## 4. TECHNICAL INFRASTRUCTURE
- [ ] No API keys hardcoded anywhere in the codebase
- [ ] All secrets in environment variables or vault
- [ ] .env file in .gitignore (never committed)
- [ ] API permissions verified: trading only, NO withdrawal rights
- [ ] Separate paper and live API keys
- [ ] Broker API connection tested (authentication, rate limits)
- [ ] Fallback to paper mode on broker API failure confirmed
- [ ] Log rotation configured (logs don't fill disk)
- [ ] Data cache TTL appropriate for trading frequency
- [ ] System time/timezone correctly set and verified

## 5. MONITORING & ALERTS
- [ ] Alert system delivers to at least one channel (Telegram/Discord/email)
- [ ] Kill switch sends alert before halting
- [ ] Daily report verified (correct P&L, positions, decisions)
- [ ] Weekly report verified
- [ ] Log files readable and structured correctly
- [ ] Audit trail covers every decision with full context
- [ ] Dashboard (if enabled) shows live portfolio state

## 6. OPERATIONAL SECURITY
- [ ] Server/VPS has firewall configured
- [ ] Bot process runs as non-root user
- [ ] Auto-restart configured (systemd / pm2 / supervisor)
- [ ] Backup process: config and state backed up daily
- [ ] Disaster recovery plan documented
- [ ] Manual override procedure documented
- [ ] Kill switch accessible via external command (not just process kill)
- [ ] Emergency contacts and shutdown procedure tested

## 7. LIVE TRADING LIMITS (START CONSERVATIVELY)
- [ ] Start with ≤ 10% of intended capital for first 30 days
- [ ] All risk limits at CONSERVATIVE settings for first 30 days:
      max_risk_per_trade: 0.25%  (not 0.5%)
      max_daily_loss: 0.5%       (not 1%)
      max_exposure: 50%           (not 80%)
- [ ] Increase limits only after 30+ days of validated live performance
- [ ] Never trade intraday strategies in first live month

## 8. LEGAL & COMPLIANCE
- [ ] Confirm trading activity complies with local regulations
- [ ] Tax implications understood (frequent trading may have tax consequences)
- [ ] Broker terms of service reviewed (algorithmic trading permitted)
- [ ] Understand margin rules if applicable

## 9. FINAL SIGN-OFF
- [ ] settings.yaml reviewed line by line
- [ ] risk.yaml reviewed line by line
- [ ] strategies.yaml reviewed — only tested strategies enabled
- [ ] Set `live_enabled: true` in settings.yaml
- [ ] Set `mode: live` in settings.yaml  OR  use --mode live flag
- [ ] Start bot in a monitored session (tmux/screen/systemd)
- [ ] Monitor for first 2 hours manually

## ⚠️ REMEMBER
- The bot can lose real money. Capital preservation is the first priority.
- Never disable the Risk Manager or Kill Switch.
- Never manually override a block without full analysis.
- The bot is a tool — not a guarantee of profit.
- Past performance in backtesting does not guarantee future results.
- Markets change. Strategies that work today may not work tomorrow.
- Review and re-validate quarterly.
