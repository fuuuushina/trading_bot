"""
src/ml/performance_tracker.py

Tracker de performance des signaux.

Lie chaque signal envoyé à l'exécution à son résultat (P&L) quand la
position se ferme. Alimente SignalQualityFilter avec des exemples labelisés.

Architecture :
  1. À l'ouverture d'un trade : enregistre (asset, entry_time, features)
  2. À la fermeture : retrouve le trade, calcule le label, appelle filter.record()
  3. Sauvegarde un historique JSON pour analyse et reporting

Stats produites (rolling window) :
  - win_rate par asset / stratégie / session
  - avg_pnl, avg_rr_réalisé
  - sharpe_ratio (rolling 20 trades)
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TRACKER_PATH = Path("data/models/performance_tracker.json")
MAX_HISTORY  = 5000   # nombre max d'entrées en mémoire


class PerformanceTracker:
    """
    Suit les performances signal par signal.

    Usage :
        tracker = PerformanceTracker(signal_filter)

        # À l'ouverture :
        tracker.on_open(asset, strategy, features, entry_price)

        # À la fermeture :
        tracker.on_close(asset, exit_price, pnl, side)
    """

    def __init__(self, signal_filter=None) -> None:
        self._filter = signal_filter
        self._open: dict[str, dict] = {}     # asset → pending entry
        self._history: list[dict] = []
        self._load()

    # ------------------------------------------------------------------ #
    # Interface publique
    # ------------------------------------------------------------------ #

    def on_open(
        self,
        asset: str,
        strategy: str,
        features: dict,
        entry_price: float,
        side: str = "long",
    ) -> None:
        """Enregistre l'ouverture d'un trade."""
        self._open[asset] = {
            "asset":        asset,
            "strategy":     strategy,
            "features":     features.copy(),
            "entry_price":  entry_price,
            "side":         side,
            "opened_at":    datetime.now(timezone.utc).isoformat(),
        }

    def on_close(
        self,
        asset: str,
        exit_price: float,
        pnl: float,
        horizon: str = "intraday",
    ) -> None:
        """
        Enregistre la fermeture d'un trade.
        Si un signal en attente existe pour cet asset, calcule le résultat
        et alimente le filtre ML.
        """
        pending = self._open.pop(asset, None)
        if pending is None:
            return

        entry = pending["entry_price"]
        side  = pending["side"]
        ret_pct = ((exit_price - entry) / entry * 100) if entry > 0 else 0.0
        if side == "short":
            ret_pct = -ret_pct

        record = {
            **pending,
            "exit_price":  exit_price,
            "pnl":         round(pnl, 4),
            "return_pct":  round(ret_pct, 4),
            "win":         pnl > 0,
            "closed_at":   datetime.now(timezone.utc).isoformat(),
            "horizon":     horizon,
        }
        self._history.append(record)
        if len(self._history) > MAX_HISTORY:
            self._history = self._history[-MAX_HISTORY:]

        # Alimenter le filtre ML
        if self._filter is not None:
            self._filter.record(pending["features"], pnl)

        # Log
        status = "WIN" if pnl > 0 else "LOSS"
        logger.info(
            "PerformanceTracker [%s] %s %s pnl=%.2f (%.3f%%) | %d trades total",
            status, asset, pending["strategy"], pnl, ret_pct, len(self._history),
        )
        self._save()

    # ------------------------------------------------------------------ #
    # Statistiques
    # ------------------------------------------------------------------ #

    def stats(
        self,
        asset: Optional[str] = None,
        strategy: Optional[str] = None,
        last_n: int = 50,
    ) -> dict:
        """Retourne les statistiques de performance."""
        rows = [
            r for r in self._history
            if (asset is None or r["asset"] == asset)
            and (strategy is None or r["strategy"] == strategy)
        ][-last_n:]

        if not rows:
            return {"n": 0, "win_rate": 0.5, "avg_pnl": 0.0, "sharpe": 0.0}

        pnls  = [r["pnl"] for r in rows]
        wins  = [r for r in rows if r["win"]]
        n     = len(rows)
        win_r = len(wins) / n if n else 0.5
        avg   = sum(pnls) / n if n else 0.0
        std   = float((sum((p - avg) ** 2 for p in pnls) / max(n - 1, 1)) ** 0.5)
        sharpe = (avg / std * (252 ** 0.5)) if std > 0 else 0.0

        # Par asset
        by_asset: dict[str, dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
        for r in rows:
            ba = by_asset[r["asset"]]
            ba["n"]   += 1
            ba["wins"] += int(r["win"])
            ba["pnl"]  += r["pnl"]

        return {
            "n":          n,
            "win_rate":   round(win_r, 3),
            "avg_pnl":    round(avg, 3),
            "total_pnl":  round(sum(pnls), 2),
            "max_win":    round(max(pnls), 2) if pnls else 0,
            "max_loss":   round(min(pnls), 2) if pnls else 0,
            "sharpe":     round(sharpe, 3),
            "by_asset":   {
                a: {
                    "n": v["n"],
                    "win_rate": round(v["wins"] / v["n"], 3),
                    "total_pnl": round(v["pnl"], 2),
                }
                for a, v in by_asset.items()
            },
        }

    def kelly_params(self, asset: Optional[str] = None, last_n: int = 40) -> dict:
        """
        Retourne win_rate, avg_win, avg_loss pour le sizing Kelly.
        Si pas assez de données, retourne des valeurs prudentes.
        """
        rows = [
            r for r in self._history
            if (asset is None or r["asset"] == asset)
        ][-last_n:]

        if len(rows) < 10:
            return {"win_rate": 0.45, "avg_win": 1.0, "avg_loss": 1.0, "n": len(rows)}

        wins  = [r["pnl"] for r in rows if r["pnl"] > 0]
        losses = [abs(r["pnl"]) for r in rows if r["pnl"] <= 0]

        win_rate = len(wins) / len(rows)
        avg_win  = sum(wins) / len(wins) if wins else 1.0
        avg_loss = sum(losses) / len(losses) if losses else 1.0

        return {
            "win_rate": round(win_rate, 3),
            "avg_win":  round(avg_win, 3),
            "avg_loss": round(avg_loss, 3),
            "n":        len(rows),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        try:
            TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TRACKER_PATH, "w") as fh:
                json.dump(self._history[-MAX_HISTORY:], fh, indent=2)
        except Exception as exc:
            logger.debug("PerformanceTracker save failed: %s", exc)

    def _load(self) -> None:
        if not TRACKER_PATH.exists():
            return
        try:
            with open(TRACKER_PATH) as fh:
                self._history = json.load(fh)
            logger.info(
                "PerformanceTracker: %d trades historiques chargés.",
                len(self._history),
            )
        except Exception as exc:
            logger.warning("PerformanceTracker load failed: %s", exc)
