"""
src/ml/regime_model.py

ML Regime Model — classifie le régime de marché via apprentissage automatique.

Deux couches :
  1. Rule-based fallback  : toujours disponible, déterministe (wraps RegimeDetector existant)
  2. ML model (sklearn)   : RandomForest entraîné sur features historiques
                            → entraîné auto dès que 200 barres SPY dispo
                            → sauvegardé sur disque pour reload rapide

Régimes prédits :
  bull_trend | bear_trend | range | high_volatility | low_volatility |
  panic | euphoric | compression | breakout_expansion

Sortie : MarketRegimePrediction avec label + confidence + source
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.features.feature_engine import AssetFeatures, MarketSnapshot
from src.features.regime_detector import MarketRegimeDetector, RegimeResult

logger = logging.getLogger(__name__)

MODEL_PATH = Path("data/models/regime_model.pkl")

_REGIME_LABELS = [
    "bull_trend", "bear_trend", "range", "high_volatility", "low_volatility",
    "panic", "euphoric", "compression", "breakout_expansion",
]

# Mapping règle → label (pour générer des labels supervisés depuis RegimeDetector)
_RULE_TO_INT = {label: i for i, label in enumerate(_REGIME_LABELS)}


@dataclass
class MarketRegimePrediction:
    label: str          # Régime prédit
    confidence: float   # 0.0 – 1.0
    source: str         # "ml" | "rules"
    probabilities: dict[str, float] = None  # type: ignore

    def to_dict(self) -> dict:
        return {
            "regime": self.label,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "probabilities": {k: round(v, 3) for k, v in (self.probabilities or {}).items()},
        }


class RegimeModel:
    """
    Prédit le régime de marché.

    Usage :
        model = RegimeModel()
        prediction = model.predict(snapshot, spy_df)
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self._clf = None              # sklearn classifier
        self._rule_detector = MarketRegimeDetector()
        self._load_model()

    # ------------------------------------------------------------------ #
    # Interface publique
    # ------------------------------------------------------------------ #

    def predict(
        self,
        snapshot: MarketSnapshot,
        spy_df: Optional[pd.DataFrame] = None,
        vix_series: Optional[pd.Series] = None,
    ) -> MarketRegimePrediction:
        """
        Prédit le régime.
        Essaie le ML model d'abord, fall back sur rules si non disponible.
        """
        # Essai ML
        if self._clf is not None:
            bench = snapshot.benchmark_features()
            if bench is not None:
                try:
                    return self._predict_ml(bench)
                except Exception as exc:
                    logger.warning("ML prediction failed, using rules: %s", exc)

        # Fallback règles
        if spy_df is not None and not spy_df.empty:
            rule_result = self._rule_detector.detect(spy_df, vix_series)
            return MarketRegimePrediction(
                label=rule_result.regime.value,
                confidence=rule_result.confidence,
                source="rules",
            )

        return MarketRegimePrediction(label="unknown", confidence=0.0, source="rules")

    def train_from_history(
        self,
        spy_df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
        min_samples: int = 200,
    ) -> bool:
        """
        Entraîne le modèle sur l'historique SPY.

        Génère les labels via le RegimeDetector (pseudo-supervised).
        Sauvegarde le modèle si l'entraînement réussit.

        Returns True si entraînement réussi.
        """
        if len(spy_df) < min_samples:
            logger.info("Not enough data for training (%d < %d)", len(spy_df), min_samples)
            return False

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
        except ImportError:
            logger.warning("scikit-learn not installed — ML regime model disabled. Run: pip install scikit-learn")
            return False

        logger.info("Training regime model on %d bars...", len(spy_df))

        X, y = self._build_training_data(spy_df, vix_series)
        if len(X) < min_samples // 2:
            logger.warning("Not enough valid training samples: %d", len(X))
            return False

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        clf.fit(X, y)
        self._clf = clf

        # Sauvegarde
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(clf, f)

        # Score simple sur données d'entraînement
        train_score = clf.score(X, y)
        logger.info("Regime model trained: accuracy=%.2f%% samples=%d", train_score * 100, len(X))
        return True

    # ------------------------------------------------------------------ #
    # ML interne
    # ------------------------------------------------------------------ #

    def _predict_ml(self, features: AssetFeatures) -> MarketRegimePrediction:
        vec = np.array(features.to_ml_vector()).reshape(1, -1)
        pred_int = self._clf.predict(vec)[0]
        probas = self._clf.predict_proba(vec)[0]
        label = _REGIME_LABELS[int(pred_int)] if int(pred_int) < len(_REGIME_LABELS) else "unknown"
        confidence = float(probas[int(pred_int)])

        proba_dict = {
            _REGIME_LABELS[i]: float(p)
            for i, p in enumerate(probas)
            if i < len(_REGIME_LABELS)
        }

        return MarketRegimePrediction(
            label=label,
            confidence=confidence,
            source="ml",
            probabilities=proba_dict,
        )

    def _build_training_data(
        self,
        df: pd.DataFrame,
        vix_series: Optional[pd.Series],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Génère (X, y) en appliquant RegimeDetector sur des fenêtres glissantes.
        Chaque ligne = features calculées sur les 200 dernières barres.
        Label = régime détecté par les règles.
        """
        from src.features.feature_engine import FeatureEngine
        fe = FeatureEngine()
        X_rows, y_rows = [], []

        step = 5  # Sous-échantillonnage pour la vitesse
        window = 200

        for i in range(window, len(df), step):
            window_df = df.iloc[i - window: i].copy()
            window_vix = (
                vix_series.iloc[max(0, i - window): i] if vix_series is not None else None
            )

            # Features
            snap = fe.compute({"SPY": window_df}, window_vix)
            bench = snap.get("SPY")
            if bench is None:
                continue

            # Label via règles
            rule_result = self._rule_detector.detect(window_df, window_vix)
            label_str = rule_result.regime.value
            label_int = _RULE_TO_INT.get(label_str, 2)  # default: "range"

            X_rows.append(bench.to_ml_vector())
            y_rows.append(label_int)

        if not X_rows:
            return np.array([]), np.array([])

        return np.array(X_rows), np.array(y_rows)

    def _load_model(self) -> None:
        if not self.model_path.exists():
            return
        try:
            with open(self.model_path, "rb") as f:
                self._clf = pickle.load(f)
            logger.info("Regime ML model loaded from %s", self.model_path)
        except Exception as exc:
            logger.warning("Could not load regime model: %s", exc)
            self._clf = None

    @property
    def is_trained(self) -> bool:
        return self._clf is not None
