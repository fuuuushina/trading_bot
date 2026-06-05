"""
src/monitoring/logger.py

Structured JSON logging + alert dispatcher.
Every decision is logged with full audit trail.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Any

LOG_DIR = Path(os.environ.get("LOG_DIR", "data/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with both console and JSON file handlers."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # JSON file handler
    log_file = LOG_DIR / f"bot_{date.today().isoformat()}.jsonl"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(_JSONFormatter())

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.addHandler(file_handler)
    root.addHandler(console)


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


class AuditLogger:
    """
    Writes structured decision records to a separate audit JSONL file.
    Every trade decision (approve/block/kill) is recorded here.
    """

    def __init__(self) -> None:
        self._path = LOG_DIR / f"audit_{date.today().isoformat()}.jsonl"

    def log_decision(self, record: dict[str, Any]) -> None:
        entry = {"ts": datetime.utcnow().isoformat(), **record}
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def log_trade_open(self, order_dict: dict) -> None:
        self.log_decision({"event": "TRADE_OPEN", **order_dict})

    def log_trade_close(self, trade_dict: dict) -> None:
        self.log_decision({"event": "TRADE_CLOSE", **trade_dict})

    def log_kill_switch(self, reason: str) -> None:
        self.log_decision({"event": "KILL_SWITCH", "reason": reason})

    def log_regime_change(self, old: str, new: str, confidence: float) -> None:
        self.log_decision({
            "event": "REGIME_CHANGE",
            "from": old, "to": new, "confidence": confidence
        })


class AlertDispatcher:
    """
    Sends alerts via Telegram / Discord / Email.
    Non-blocking — failures are logged but don't stop the bot.
    """

    def __init__(self, alerts_cfg: dict) -> None:
        self.cfg = alerts_cfg
        self.min_severity = alerts_cfg.get("min_severity", "WARNING")
        self._severity_order = ["DEBUG", "INFO", "WARNING", "CRITICAL"]

    def send(self, message: str, severity: str = "INFO") -> None:
        if not self._should_send(severity):
            return
        formatted = f"[{severity}] {datetime.utcnow().strftime('%H:%M:%S')} | {message}"
        self._send_telegram(formatted)
        self._send_discord(formatted)

    def _should_send(self, severity: str) -> bool:
        try:
            return (
                self._severity_order.index(severity)
                >= self._severity_order.index(self.min_severity)
            )
        except ValueError:
            return True

    def _send_telegram(self, text: str) -> None:
        cfg = self.cfg.get("telegram", {})
        if not cfg.get("enabled", False):
            return
        try:
            import requests
            token = os.environ.get(cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN"), "")
            chat_id = os.environ.get(cfg.get("chat_id_env", "TELEGRAM_CHAT_ID"), "")
            if not token or not chat_id:
                return
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text[:4000]},
                timeout=5,
            )
        except Exception as exc:
            logging.getLogger(__name__).debug("Telegram alert failed: %s", exc)

    def _send_discord(self, text: str) -> None:
        cfg = self.cfg.get("discord", {})
        if not cfg.get("enabled", False):
            return
        try:
            import requests
            webhook = os.environ.get(cfg.get("webhook_env", "DISCORD_WEBHOOK_URL"), "")
            if not webhook:
                return
            requests.post(webhook, json={"content": text[:2000]}, timeout=5)
        except Exception as exc:
            logging.getLogger(__name__).debug("Discord alert failed: %s", exc)


class DailyReporter:
    """
    Generates a daily performance summary.
    """

    REPORT_DIR = Path("data/reports")

    def __init__(self) -> None:
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        portfolio_state: dict,
        trade_history: list[dict],
        decisions_today: list[dict],
        regime: str,
    ) -> dict:
        today_trades = [
            t for t in trade_history
            if t.get("closed_at", "").startswith(date.today().isoformat())
        ]
        today_pnl = sum(t.get("pnl", 0) for t in today_trades)
        total_capital = portfolio_state.get("total_capital", 0)

        report = {
            "date": date.today().isoformat(),
            "regime": regime,
            "total_capital": total_capital,
            "daily_pnl": round(today_pnl, 2),
            "daily_pnl_pct": round(today_pnl / (total_capital + 1e-10) * 100, 4),
            "open_positions": portfolio_state.get("open_positions", 0),
            "cash": portfolio_state.get("cash", 0),
            "drawdown_pct": portfolio_state.get("drawdown_pct", 0),
            "trades_today": len(today_trades),
            "decisions_evaluated": len(decisions_today),
            "decisions_executed": sum(1 for d in decisions_today if d.get("final_action") == "EXECUTE"),
            "decisions_blocked": sum(1 for d in decisions_today if d.get("final_action") == "BLOCK"),
        }

        # Write to file
        report_path = self.REPORT_DIR / f"daily_{date.today().isoformat()}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        return report
