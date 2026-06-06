"""
src/alerts/alert_manager.py

Alert Manager — dispatch unifié des alertes vers tous les canaux.

Niveaux d'alerte :
  DEBUG    → log seulement
  INFO     → log + stockage
  WARNING  → log + canaux activés (Telegram, Discord, Email)
  CRITICAL → tous les canaux + flag kill switch

Types d'alertes :
  TRADE_EXECUTED     → ordre exécuté
  TRADE_BLOCKED      → ordre bloqué par Risk Manager
  RISK_LIMIT         → limite de risque atteinte
  KILL_SWITCH        → kill switch déclenché
  REGIME_CHANGE      → changement de régime détecté
  DRAWDOWN_ALERT     → drawdown dépassant un seuil
  NEWS_RISK          → risque news élevé détecté
  MARKET_OPPORTUNITY → opportunité détectée
  SYSTEM_ERROR       → erreur système
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    DEBUG    = "DEBUG"
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    TRADE_EXECUTED     = "TRADE_EXECUTED"
    TRADE_BLOCKED      = "TRADE_BLOCKED"
    RISK_LIMIT         = "RISK_LIMIT"
    KILL_SWITCH        = "KILL_SWITCH"
    REGIME_CHANGE      = "REGIME_CHANGE"
    DRAWDOWN_ALERT     = "DRAWDOWN_ALERT"
    NEWS_RISK          = "NEWS_RISK"
    MARKET_OPPORTUNITY = "MARKET_OPPORTUNITY"
    SYSTEM_ERROR       = "SYSTEM_ERROR"
    GENERAL            = "GENERAL"


@dataclass
class Alert:
    level: AlertLevel
    type: AlertType
    message: str
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "type": self.type.value,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def format_message(self) -> str:
        prefix = {
            AlertLevel.DEBUG:    "[DBG]",
            AlertLevel.INFO:     "[INF]",
            AlertLevel.WARNING:  "[WRN]",
            AlertLevel.CRITICAL: "[CRT]",
        }.get(self.level, "[???]")
        return f"{prefix} {self.type.value} | {self.message}"


class AlertManager:
    """
    Dispatch des alertes vers les canaux configurés.

    Remplace/wraps l'AlertDispatcher existant avec un système plus riche.

    Usage :
        manager = AlertManager(cfg)
        manager.send(AlertLevel.WARNING, AlertType.REGIME_CHANGE,
                     "Régime passé de bull_trend à bear_trend",
                     data={"from": "bull_trend", "to": "bear_trend"})
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.min_level = AlertLevel(cfg.get("min_severity", "WARNING"))
        self._history: list[Alert] = []
        self._max_history = 500

        # Canaux
        tg = cfg.get("telegram", {})
        dc = cfg.get("discord", {})
        em = cfg.get("email", {})

        self._telegram_enabled  = tg.get("enabled", False)
        self._discord_enabled   = dc.get("enabled", False)
        self._email_enabled     = em.get("enabled", False)
        self._telegram_token    = tg.get("bot_token_env", "TELEGRAM_BOT_TOKEN")
        self._telegram_chat_id  = tg.get("chat_id_env", "TELEGRAM_CHAT_ID")
        self._discord_webhook   = dc.get("webhook_env", "DISCORD_WEBHOOK_URL")

    # ------------------------------------------------------------------ #
    # Interface principale
    # ------------------------------------------------------------------ #

    def send(
        self,
        level: AlertLevel | str,
        alert_type: AlertType | str = AlertType.GENERAL,
        message: str = "",
        data: dict | None = None,
    ) -> None:
        """Envoie une alerte."""
        if isinstance(level, str):
            level = AlertLevel(level)
        if isinstance(alert_type, str):
            try:
                alert_type = AlertType(alert_type)
            except ValueError:
                alert_type = AlertType.GENERAL

        alert = Alert(
            level=level,
            type=alert_type,
            message=message,
            data=data or {},
        )

        # Toujours logger
        log_level = {
            AlertLevel.DEBUG:    logging.DEBUG,
            AlertLevel.INFO:     logging.INFO,
            AlertLevel.WARNING:  logging.WARNING,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }.get(level, logging.INFO)
        logger.log(log_level, "ALERT %s | %s", alert_type.value, message)

        # Historique
        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Dispatch externe si niveau suffisant
        if self._should_dispatch(level):
            formatted = alert.format_message()
            self._dispatch_telegram(formatted, level)
            self._dispatch_discord(formatted, level)
            if level == AlertLevel.CRITICAL:
                self._dispatch_email(formatted, alert)

    # ---- Raccourcis ----

    def trade_executed(self, asset: str, side: str, size: float, price: float, strategy: str) -> None:
        self.send(
            AlertLevel.INFO, AlertType.TRADE_EXECUTED,
            f"TRADE {side} {asset} x{size:.2f} @ ${price:.2f} [{strategy}]",
            {"asset": asset, "side": side, "size": size, "price": price, "strategy": strategy},
        )

    def trade_blocked(self, asset: str, reason: str) -> None:
        self.send(
            AlertLevel.INFO, AlertType.TRADE_BLOCKED,
            f"BLOCKED {asset}: {reason}",
        )

    def regime_change(self, old: str, new: str, confidence: float) -> None:
        self.send(
            AlertLevel.WARNING, AlertType.REGIME_CHANGE,
            f"Regime: {old} → {new} (conf={confidence:.0%})",
            {"from": old, "to": new, "confidence": confidence},
        )

    def drawdown_alert(self, drawdown_pct: float, threshold: float) -> None:
        self.send(
            AlertLevel.WARNING, AlertType.DRAWDOWN_ALERT,
            f"Drawdown {drawdown_pct:.1%} dépasse le seuil {threshold:.1%}",
            {"drawdown_pct": drawdown_pct, "threshold": threshold},
        )

    def kill_switch(self, reason: str) -> None:
        self.send(
            AlertLevel.CRITICAL, AlertType.KILL_SWITCH,
            f"KILL SWITCH DÉCLENCHÉ: {reason}",
        )

    def news_risk(self, asset: str, risk_score: float, topics: list[str]) -> None:
        self.send(
            AlertLevel.WARNING, AlertType.NEWS_RISK,
            f"News risk {asset}: score={risk_score:.2f} topics={topics}",
            {"asset": asset, "risk_score": risk_score, "topics": topics},
        )

    def opportunity(self, asset: str, signal: str, confidence: float, reason: str) -> None:
        self.send(
            AlertLevel.INFO, AlertType.MARKET_OPPORTUNITY,
            f"Opportunité {asset} {signal} conf={confidence:.0%}: {reason}",
            {"asset": asset, "signal": signal, "confidence": confidence},
        )

    # ---- Historique ----

    def get_recent(self, n: int = 50, level: Optional[AlertLevel] = None) -> list[dict]:
        alerts = self._history[-n:]
        if level:
            alerts = [a for a in alerts if a.level == level]
        return [a.to_dict() for a in alerts]

    # ------------------------------------------------------------------ #
    # Dispatch canaux
    # ------------------------------------------------------------------ #

    def _should_dispatch(self, level: AlertLevel) -> bool:
        order = [AlertLevel.DEBUG, AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.CRITICAL]
        return order.index(level) >= order.index(self.min_level)

    def _dispatch_telegram(self, message: str, level: AlertLevel) -> None:
        if not self._telegram_enabled:
            return
        try:
            import os, requests
            token = os.environ.get(self._telegram_token, "")
            chat_id = os.environ.get(self._telegram_chat_id, "")
            if not token or not chat_id:
                return
            icon = {"DEBUG": "🔍", "INFO": "📊", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(level.value, "")
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": f"{icon} {message}"},
                timeout=5,
            )
        except Exception as exc:
            logger.debug("Telegram dispatch failed: %s", exc)

    def _dispatch_discord(self, message: str, level: AlertLevel) -> None:
        if not self._discord_enabled:
            return
        try:
            import os, requests
            webhook = os.environ.get(self._discord_webhook, "")
            if not webhook:
                return
            color = {"DEBUG": 0x888888, "INFO": 0x3498db, "WARNING": 0xf39c12, "CRITICAL": 0xe74c3c}.get(level.value, 0x888888)
            requests.post(
                webhook,
                json={"embeds": [{"description": message, "color": color}]},
                timeout=5,
            )
        except Exception as exc:
            logger.debug("Discord dispatch failed: %s", exc)

    def _dispatch_email(self, message: str, alert: Alert) -> None:
        if not self._email_enabled:
            return
        try:
            import os, smtplib
            from email.mime.text import MIMEText
            smtp_host = self.cfg.get("email", {}).get("smtp_host", "smtp.gmail.com")
            smtp_port = self.cfg.get("email", {}).get("smtp_port", 587)
            user = os.environ.get(self.cfg.get("email", {}).get("user_env", "EMAIL_USER"), "")
            pwd  = os.environ.get(self.cfg.get("email", {}).get("password_env", "EMAIL_PASSWORD"), "")
            rcpt = os.environ.get(self.cfg.get("email", {}).get("recipient_env", "EMAIL_RECIPIENT"), "")
            if not all([user, pwd, rcpt]):
                return
            msg = MIMEText(f"{message}\n\n{json.dumps(alert.data, indent=2)}")
            msg["Subject"] = f"[TradingBot] {alert.type.value}"
            msg["From"] = user
            msg["To"] = rcpt
            with smtplib.SMTP(smtp_host, smtp_port) as s:
                s.starttls()
                s.login(user, pwd)
                s.sendmail(user, [rcpt], msg.as_string())
        except Exception as exc:
            logger.debug("Email dispatch failed: %s", exc)
