"""
src/ml/signal_filter.py

Filtre de qualité ML sur les signaux intraday.

Deux phases :
  Phase 1 (< MIN_SAMPLES trades) : score heuristique basé sur features techniques
  Phase 2 (>= MIN_SAMPLES trades) : GradientBoosting entraîné sur l'historique réel

Features utilisées :
  - ema_spread_ratio   : séparation EMA / ATR (qualité du trend)
  - rsi_dist_center    : |RSI - 50| / 50 (proximité de la zone neutre, inversé)
  - bb_position        : position du prix dans les bandes (0=bas, 1=haut)
  - atr_normalized     : ATR / prix (volatilité relative)
  - session_score      : London=1.0, NY=0.9, Overlap=1.0, Asian=0.4, Off=0.1
  - regime_aligned     : signal aligné avec le régime (1/0)
  - rr_ratio           : rapport risque/rendement normalisé
  - confidence_raw     : score de confiance de la stratégie source
  - hour_sin/cos       : heure du jour (encodage cyclique)

Sortie : float [0, 1] — probabilité que le trade soit profitable.
Un score < threshold bloque l'exécution.
"""
from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path("data/models/signal_filter.pkl")
MIN_SAMPLES = 60       # trades minimum pour activer le ML
RETRAIN_EVERY = 30     # retraining toutes les N nouvelles observations
THRESHOLD = 0.45       # score minimum pour exécuter (bloque si < threshold)


class SignalQualityFilter:
    """
    Filtre de qualité ML appliqué sur chaque signal avant exécution.

    Usage :
        filt = SignalQualityFilter()
        score = filt.score(features_dict)
        if score < filt.threshold:
            # bloquer le signal
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        threshold: float = THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._clf = None
        self._X: list[list[float]] = []
        self._y: list[int] = []          # 1 = trade gagnant, 0 = perdant
        self._since_last_train = 0
        self._load()

    # ------------------------------------------------------------------ #
    # Interface publique
    # ------------------------------------------------------------------ #

    def score(self, features: dict) -> float:
        """
        Retourne la probabilité de profit [0, 1].
        En Phase 1 : score heuristique.
        En Phase 2 : GradientBoosting.
        """
        vec = self._to_vector(features)
        if vec is None:
            return 0.5  # neutre si features incomplètes

        if self._clf is not None:
            try:
                proba = self._clf.predict_proba([vec])[0]
                # classe 1 = profitable
                return float(proba[1]) if len(proba) > 1 else float(proba[0])
            except Exception as exc:
                logger.debug("ML score failed: %s", exc)

        # Phase 1 : score heuristique
        return self._heuristic_score(features)

    def record(self, features: dict, pnl: float) -> None:
        """
        Enregistre un résultat de trade pour l'entraînement.
        Appeler après fermeture d'une position.
        """
        vec = self._to_vector(features)
        if vec is None:
            return
        self._X.append(vec)
        self._y.append(1 if pnl > 0 else 0)
        self._since_last_train += 1

        # Auto-retraining
        if len(self._X) >= MIN_SAMPLES and self._since_last_train >= RETRAIN_EVERY:
            self._train()

        self._save()

    def maybe_retrain(self) -> bool:
        """Force un retraining si assez de données."""
        if len(self._X) >= MIN_SAMPLES:
            return self._train()
        return False

    @property
    def n_samples(self) -> int:
        return len(self._X)

    @property
    def is_ml_active(self) -> bool:
        return self._clf is not None

    @property
    def win_rate(self) -> float:
        if not self._y:
            return 0.5
        return sum(self._y) / len(self._y)

    # ------------------------------------------------------------------ #
    # Score heuristique (Phase 1)
    # ------------------------------------------------------------------ #

    def _heuristic_score(self, f: dict) -> float:
        """
        Score de qualité basé sur des règles expertes.
        Sert de bootstrap avant que le ML ne soit entraîné.
        """
        score = 0.50

        # 1. Qualité du spread EMA (signal de trend fiable)
        sep_ratio = float(f.get("ema_spread_ratio", 0))
        if sep_ratio > 0.5:   score += 0.08
        if sep_ratio > 1.0:   score += 0.07
        if sep_ratio > 1.5:   score += 0.05

        # 2. RSI dans la zone idéale (40-60 = trend propre, ni surach ni surv)
        rsi = float(f.get("rsi", 50))
        if 42 <= rsi <= 58:   score += 0.08
        elif 35 <= rsi <= 65: score += 0.04
        else:                 score -= 0.08  # RSI extrême = risqué

        # 3. R:R ratio (plus c'est haut, mieux c'est)
        rr = float(f.get("rr_ratio", 1.5))
        if rr >= 2.0:   score += 0.07
        elif rr >= 1.5: score += 0.03
        elif rr < 1.2:  score -= 0.10

        # 4. Session de trading (volatilité et liquidité)
        session = float(f.get("session_score", 0.5))
        score += (session - 0.5) * 0.10

        # 5. Alignement régime (signal dans le sens du marché)
        if f.get("regime_aligned", False):
            score += 0.06
        else:
            score -= 0.04

        # 6. ATR correct (assez volatile pour un trade mais pas panic)
        atr_norm = float(f.get("atr_normalized", 0.001))
        if 0.0005 <= atr_norm <= 0.003:  score += 0.04
        elif atr_norm > 0.005:           score -= 0.06  # trop volatile

        # 7. Confiance brute de la stratégie
        conf = float(f.get("confidence_raw", 0.5))
        score += (conf - 0.5) * 0.12

        return round(max(0.0, min(1.0, score)), 4)

    # ------------------------------------------------------------------ #
    # ML interne
    # ------------------------------------------------------------------ #

    def _train(self) -> bool:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import cross_val_score
        except ImportError:
            logger.warning("scikit-learn non installé — ML signal filter désactivé.")
            return False

        X = np.array(self._X)
        y = np.array(self._y)

        # Besoin des deux classes
        if len(np.unique(y)) < 2:
            logger.info("Signal filter: une seule classe (%d samples) — ML différé.", len(y))
            return False

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("gb", GradientBoostingClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            )),
        ])

        # Cross-validation rapide sur 3 folds si assez de données
        if len(X) >= 90:
            try:
                scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
                logger.info(
                    "Signal filter CV AUC: %.3f ± %.3f (%d samples)",
                    scores.mean(), scores.std(), len(X),
                )
            except Exception:
                pass

        clf.fit(X, y)
        self._clf = clf
        self._since_last_train = 0

        # Feature importances
        try:
            gb = clf.named_steps["gb"]
            feat_names = self._feature_names()
            importances = sorted(
                zip(feat_names, gb.feature_importances_),
                key=lambda x: -x[1],
            )
            logger.info(
                "Signal filter retrained on %d samples. Top features: %s",
                len(X),
                ", ".join(f"{n}={v:.3f}" for n, v in importances[:4]),
            )
        except Exception:
            logger.info("Signal filter retrained on %d samples.", len(X))

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as fh:
            pickle.dump({"clf": self._clf, "X": self._X, "y": self._y}, fh)

        return True

    # ------------------------------------------------------------------ #
    # Vecteur de features
    # ------------------------------------------------------------------ #

    @staticmethod
    def _feature_names() -> list[str]:
        return [
            "ema_spread_ratio", "rsi", "rsi_dist_center",
            "bb_position", "atr_normalized",
            "session_score", "regime_aligned", "rr_ratio",
            "confidence_raw", "hour_sin", "hour_cos",
        ]

    @staticmethod
    def _to_vector(f: dict) -> Optional[list[float]]:
        """Convertit un dict features en vecteur numérique."""
        try:
            rsi = float(f.get("rsi", 50))
            hour = int(f.get("hour", 12))
            hour_rad = hour * 2 * np.pi / 24
            return [
                float(f.get("ema_spread_ratio", 0)),
                rsi,
                abs(rsi - 50) / 50,                    # distance au centre
                float(f.get("bb_position", 0.5)),
                float(f.get("atr_normalized", 0.001)),
                float(f.get("session_score", 0.5)),
                float(bool(f.get("regime_aligned", False))),
                float(f.get("rr_ratio", 1.5)),
                float(f.get("confidence_raw", 0.5)),
                float(np.sin(hour_rad)),
                float(np.cos(hour_rad)),
            ]
        except Exception as exc:
            logger.debug("Feature vector error: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "wb") as fh:
                pickle.dump({
                    "clf": self._clf,
                    "X": self._X[-2000:],   # garder max 2000 samples
                    "y": self._y[-2000:],
                }, fh)
        except Exception as exc:
            logger.debug("Signal filter save failed: %s", exc)

    def _load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            with open(self.model_path, "rb") as fh:
                data = pickle.load(fh)
            self._clf = data.get("clf")
            self._X   = data.get("X", [])
            self._y   = data.get("y", [])
            status = "ML actif" if self._clf else "heuristique"
            logger.info(
                "Signal filter chargé: %d samples, mode=%s",
                len(self._X), status,
            )
        except Exception as exc:
            logger.warning("Signal filter load failed: %s", exc)


# ------------------------------------------------------------------ #
# Helper : construit features depuis un signal + contexte
# ------------------------------------------------------------------ #

def build_signal_features(
    signal,          # Signal object
    regime: str,
    df=None,         # DataFrame 5m pour calculer BB position
    hour: Optional[int] = None,
) -> dict:
    """
    Construit le dict de features pour SignalQualityFilter depuis un signal.
    Compatible avec le Signal dataclass du projet.
    """
    meta = getattr(signal, "metadata", {}) or {}
    entry = float(getattr(signal, "entry_price", 0) or 0)
    sl    = float(getattr(signal, "stop_loss", 0) or 0)
    tp    = float(getattr(signal, "take_profit", 0) or 0)
    conf  = float(getattr(signal, "confidence", 0.5) or 0.5)
    asset = str(getattr(signal, "asset", ""))

    # R:R
    rr = float(getattr(signal, "risk_reward", 0) or 0)
    if rr <= 0 and entry > 0 and sl > 0 and tp > 0:
        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        rr = (tp_dist / sl_dist) if sl_dist > 0 else 1.5

    # ATR normalisé
    atr = float(meta.get("atr", 0))
    atr_norm = (atr / entry) if entry > 0 else 0.001

    # EMA spread ratio
    ema_fast = float(meta.get("ema_fast", entry) or entry)
    ema_slow = float(meta.get("ema_slow", entry) or entry)
    ema_sep  = abs(ema_fast - ema_slow)
    sep_ratio = (ema_sep / atr) if atr > 0 else 0

    # RSI
    rsi_v = float(meta.get("rsi", 50) or 50)

    # BB position (0 = near lower, 1 = near upper)
    bb_pos = 0.5
    if df is not None and len(df) >= 20:
        try:
            import pandas as pd
            close = df["close"]
            sma = close.rolling(20).mean().iloc[-1]
            std = close.rolling(20).std(ddof=0).iloc[-1]
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std
            band_width = bb_upper - bb_lower
            if band_width > 0:
                bb_pos = float((entry - bb_lower) / band_width)
                bb_pos = max(0.0, min(1.0, bb_pos))
        except Exception:
            pass

    # Session score
    now_hour = hour if hour is not None else datetime.now(timezone.utc).hour
    session_score = _session_quality(now_hour, asset)

    # Régime aligné
    sig_val = str(getattr(signal, "signal", "")).upper()
    regime_aligned = _check_regime_alignment(sig_val, regime)

    return {
        "ema_spread_ratio": round(sep_ratio, 4),
        "rsi": round(rsi_v, 2),
        "rsi_dist_center": round(abs(rsi_v - 50) / 50, 4),
        "bb_position": round(bb_pos, 4),
        "atr_normalized": round(atr_norm, 6),
        "session_score": session_score,
        "regime_aligned": regime_aligned,
        "rr_ratio": round(rr, 3),
        "confidence_raw": round(conf, 4),
        "hour": now_hour,
        "hour_sin": float(np.sin(now_hour * 2 * np.pi / 24)),
        "hour_cos": float(np.cos(now_hour * 2 * np.pi / 24)),
        # Extra context (non-vectorisé, pour logging)
        "_asset": asset,
        "_regime": regime,
        "_signal_type": sig_val,
    }


def _session_quality(hour_utc: int, asset: str) -> float:
    """Score de qualité de la session pour cet asset à cette heure UTC."""
    # Crypto : 24h, légèrement favoriser heures actives
    if asset.endswith("-USD") and asset not in {"GC=F"}:
        if 12 <= hour_utc <= 22:
            return 0.85
        return 0.65

    # Forex / commodities
    london_open  = 7
    london_close = 16
    ny_open      = 13
    ny_close     = 21

    in_london = london_open <= hour_utc < london_close
    in_ny     = ny_open <= hour_utc < ny_close
    overlap   = in_london and in_ny   # 13:00-16:00 UTC = meilleure liquidité

    if overlap:    return 1.00
    if in_london:  return 0.90
    if in_ny:      return 0.85
    # Asian session
    if 0 <= hour_utc < 7:   return 0.45
    # After NY close
    return 0.20


def _check_regime_alignment(signal_type: str, regime: str) -> bool:
    """True si le signal est aligné avec le régime de marché."""
    bull_regimes = {"bull_trend", "breakout_expansion", "euphoric"}
    bear_regimes = {"bear_trend", "panic"}
    if signal_type in ("BUY", "LONG"):
        return regime not in bear_regimes
    if signal_type in ("SELL", "SHORT"):
        return regime not in bull_regimes
    return True
