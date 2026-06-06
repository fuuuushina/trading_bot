"""
src/watchers/universe_builder.py

Market Universe Builder.

Reçoit un ClientProfile et construit automatiquement la watchlist complète :
  - actifs demandés par l'utilisateur
  - instruments corrélés nécessaires (VIX, TLT, GLD…)
  - calendrier macro (CPI, NFP, Fed)

Chaque actif est assigné à une fréquence d'analyse :
  DAILY    → régime, rotation ETF, macro
  HOURLY   → actifs principaux, VIX, positions ouvertes
  INTRADAY → 5 min — forex, bots court terme, spikes volatilité
  REALTIME → exécution, stops, limites de risque (pas de fetch données)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.profile.client_profile import ClientProfile, Objective, RiskTolerance


class WatchFrequency(str, Enum):
    DAILY    = "daily"
    HOURLY   = "hourly"
    INTRADAY = "intraday"   # 5 min
    REALTIME = "realtime"   # pas de fetch — surveillance pure


@dataclass
class WatchedAsset:
    ticker: str
    frequency: WatchFrequency
    asset_class: str          # equity | etf | forex | crypto | volatility | bond | commodity
    is_primary: bool          # demandé explicitement par l'utilisateur
    is_benchmark: bool = False
    notes: str = ""


@dataclass
class MarketUniverse:
    """
    Watchlist complète pour un utilisateur.
    Les actifs sont regroupés par fréquence pour optimiser les appels données.
    """
    profile_label: str
    assets: list[WatchedAsset] = field(default_factory=list)

    # ---- Vues pré-calculées ----
    @property
    def by_frequency(self) -> dict[WatchFrequency, list[str]]:
        result: dict[WatchFrequency, list[str]] = {f: [] for f in WatchFrequency}
        for a in self.assets:
            result[a.frequency].append(a.ticker)
        return result

    @property
    def all_tickers(self) -> list[str]:
        return [a.ticker for a in self.assets]

    @property
    def primary_tickers(self) -> list[str]:
        return [a.ticker for a in self.assets if a.is_primary]

    @property
    def benchmarks(self) -> list[str]:
        return [a.ticker for a in self.assets if a.is_benchmark]

    def tickers_for(self, freq: WatchFrequency) -> list[str]:
        return [a.ticker for a in self.assets if a.frequency == freq]

    def to_dict(self) -> dict:
        return {
            "profile_label": self.profile_label,
            "total_assets": len(self.assets),
            "by_frequency": {
                f.value: tickers
                for f, tickers in self.by_frequency.items()
                if tickers
            },
        }


# ------------------------------------------------------------------ #
# Règles d'enrichissement automatique
# ------------------------------------------------------------------ #

# Actifs corrélés ajoutés automatiquement selon le profil
_EQUITY_CORE = [
    WatchedAsset("SPY",  WatchFrequency.HOURLY,  "etf",        False, is_benchmark=True, notes="US equity benchmark"),
    WatchedAsset("QQQ",  WatchFrequency.HOURLY,  "etf",        False, is_benchmark=True, notes="Tech benchmark"),
    WatchedAsset("^VIX", WatchFrequency.HOURLY,  "volatility", False, notes="Fear index"),
    WatchedAsset("TLT",  WatchFrequency.DAILY,   "bond",       False, notes="Long-term rates"),
]

_DEFENSIVE_ADDITIONS = [
    WatchedAsset("GLD",  WatchFrequency.DAILY,   "commodity",  False, notes="Gold — safe haven"),
    WatchedAsset("^VIX", WatchFrequency.HOURLY,  "volatility", False, notes="Fear index"),
]

_FOREX_CORE = [
    WatchedAsset("EURUSD=X", WatchFrequency.INTRADAY, "forex", False, notes="EUR/USD major pair"),
    WatchedAsset("DX-Y.NYB", WatchFrequency.HOURLY,   "forex", False, notes="DXY — dollar index"),
]

_CRYPTO_CORE = [
    WatchedAsset("BTC-USD", WatchFrequency.HOURLY,   "crypto", False, notes="Bitcoin"),
    WatchedAsset("ETH-USD", WatchFrequency.HOURLY,   "crypto", False, notes="Ethereum"),
]

# Actifs à analyser en intraday selon préférence
_INTRADAY_CANDIDATES = {"SPY", "QQQ", "AAPL", "MSFT", "NVDA"}

# Instruments macro suivis (non-tradables, pour contexte seulement)
_MACRO_INSTRUMENTS = [
    WatchedAsset("^TNX",  WatchFrequency.DAILY, "bond",      False, notes="10Y US Treasury yield"),
    WatchedAsset("^GSPC", WatchFrequency.DAILY, "etf",       False, is_benchmark=True, notes="S&P 500"),
]


# ------------------------------------------------------------------ #
# Builder
# ------------------------------------------------------------------ #

class MarketUniverseBuilder:
    """
    Construit une MarketUniverse à partir d'un ClientProfile.

    Règles MVP :
      - Max 5 actifs primaires (scalable plus tard)
      - Ajoute VIX, TLT, SPY automatiquement si l'utilisateur a des équités
      - Ajoute GLD si profil conservateur ou high_volatility
      - Ajoute DXY si l'utilisateur a du forex
    """

    MAX_PRIMARY = 5  # MVP : max 5 actifs primaires

    def build(
        self,
        profile: ClientProfile,
        custom_tickers: list[str] | None = None,
    ) -> MarketUniverse:
        """
        Construit la watchlist complète pour un profil.

        Parameters
        ----------
        profile        : Profil client
        custom_tickers : Tickers supplémentaires demandés (override settings.yaml)
        """
        seen: set[str] = set()
        assets: list[WatchedAsset] = []

        def add(wa: WatchedAsset) -> None:
            if wa.ticker not in seen:
                seen.add(wa.ticker)
                assets.append(wa)

        # 1. Actifs primaires (depuis settings universe ou custom)
        primary_tickers = (custom_tickers or self._default_tickers(profile))[: self.MAX_PRIMARY]
        for ticker in primary_tickers:
            freq = self._ticker_frequency(ticker, profile)
            asset_class = self._ticker_class(ticker)
            add(WatchedAsset(ticker, freq, asset_class, is_primary=True))

        # 2. Benchmarks US equities (toujours présents si préf equity ou ETF)
        if profile.preferences.etf or profile.preferences.stocks:
            for wa in _EQUITY_CORE:
                add(wa)
            for wa in _MACRO_INSTRUMENTS:
                add(wa)

        # 3. Forex core si préf forex activée ou ticker forex dans les primaires
        has_forex = (
            getattr(profile.preferences, "forex", False)
            or any(self._ticker_class(t) == "forex" for t in primary_tickers)
        )
        if has_forex:
            for wa in _FOREX_CORE:
                add(wa)

        # 4. Crypto core si préférence crypto
        if profile.preferences.crypto:
            for wa in _CRYPTO_CORE:
                add(wa)

        # 5. Défense supplémentaire si conservateur ou risque max bas
        if (
            profile.risk_tolerance in (RiskTolerance.CONSERVATIVE, RiskTolerance.MODERATE)
            or profile.max_drawdown_tolerance < 0.15
        ):
            for wa in _DEFENSIVE_ADDITIONS:
                add(wa)

        # 6. Intraday tickers si préférence intraday
        if profile.preferences.intraday:
            for ticker in primary_tickers:
                if ticker in _INTRADAY_CANDIDATES:
                    # Ajouter version intraday en plus de la version daily/hourly
                    add(WatchedAsset(
                        ticker + "_5M", WatchFrequency.INTRADAY, "equity", False,
                        notes=f"{ticker} intraday 5min"
                    ))

        return MarketUniverse(
            profile_label=profile.risk_profile_label,
            assets=assets,
        )

    # ------------------------------------------------------------------ #

    def _default_tickers(self, profile: ClientProfile) -> list[str]:
        """Tickers par défaut selon objectif et préférences."""
        # Forex-only mode: return major FX pairs instead of equities
        if (
            getattr(profile.preferences, "forex", False)
            and not profile.preferences.etf
            and not profile.preferences.stocks
        ):
            return ["EURUSD=X"]

        if profile.objective == Objective.GROWTH:
            return ["SPY", "QQQ", "NVDA", "MSFT", "AAPL"]
        if profile.objective == Objective.INCOME:
            return ["SPY", "TLT", "GLD", "VT", "IWM"]
        if profile.objective == Objective.WEALTH_PRESERVATION:
            return ["SPY", "TLT", "GLD", "VT", "BND"]
        return ["SPY", "QQQ", "IWM", "GLD", "TLT"]

    def _ticker_frequency(self, ticker: str, profile: ClientProfile) -> WatchFrequency:
        ticker_up = ticker.upper()
        if "USD" in ticker_up or "EUR" in ticker_up or "=X" in ticker_up:
            return WatchFrequency.INTRADAY if profile.preferences.intraday else WatchFrequency.HOURLY
        if "BTC" in ticker_up or "ETH" in ticker_up:
            return WatchFrequency.HOURLY
        if ticker_up in ("SPY", "QQQ", "NVDA", "AAPL", "MSFT"):
            return WatchFrequency.HOURLY
        return WatchFrequency.DAILY

    @staticmethod
    def _ticker_class(ticker: str) -> str:
        t = ticker.upper()
        if any(x in t for x in ("USD", "EUR", "GBP", "JPY", "=X")):
            return "forex"
        if any(x in t for x in ("BTC", "ETH", "SOL", "DOGE")):
            return "crypto"
        if t in ("^VIX", "^VXN"):
            return "volatility"
        if t in ("TLT", "IEF", "SHY", "^TNX", "BND"):
            return "bond"
        if t in ("GLD", "SLV", "USO", "DBC"):
            return "commodity"
        if t in ("SPY", "QQQ", "IWM", "VT", "EFA", "EEM", "XLE", "XLF"):
            return "etf"
        return "equity"
