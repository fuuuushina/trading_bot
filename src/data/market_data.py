"""
src/data/market_data.py

Market Data Engine.
Handles data fetching, caching, normalization, and quality checks.
Supports daily and intraday data. Provider-agnostic interface.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class DataQualityError(Exception):
    """Raised when data fails quality checks."""
    pass


class MarketDataEngine:
    """
    Fetches, caches, validates, and normalises OHLCV data.

    Columns normalised to lowercase: open, high, low, close, volume.
    Index is a DatetimeIndex.
    """

    def __init__(self, settings: dict) -> None:
        self.cfg = settings.get("data", {})
        self.provider = self.cfg.get("provider", "yfinance")
        self.cache_ttl = self.cfg.get("cache_ttl_seconds", 300)
        self.max_missing_pct = self.cfg.get("max_missing_pct", 0.02)
        self.min_volume = self.cfg.get("min_volume_threshold", 100_000)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_daily(
        self,
        ticker: str,
        lookback_days: int = 730,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data."""
        return self._fetch(ticker, interval="1d", lookback_days=lookback_days,
                           use_cache=use_cache)

    def get_intraday(
        self,
        ticker: str,
        interval: str = "5m",
        lookback_days: int = 5,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch intraday OHLCV data."""
        return self._fetch(ticker, interval=interval, lookback_days=lookback_days,
                           use_cache=use_cache)

    def get_multi(
        self,
        tickers: list[str],
        interval: str = "1d",
        lookback_days: int = 730,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers. Returns dict of valid DataFrames."""
        result: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                df = self._fetch(ticker, interval=interval, lookback_days=lookback_days)
                result[ticker] = df
            except Exception as exc:
                logger.warning("Skipping %s: %s", ticker, exc)
        return result

    # ------------------------------------------------------------------ #
    # Core fetch → cache → validate → normalise pipeline
    # ------------------------------------------------------------------ #

    def _fetch(
        self,
        ticker: str,
        interval: str,
        lookback_days: int,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_key = f"{ticker}_{interval}_{lookback_days}"

        # Try cache first
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.debug("Cache hit: %s", cache_key)
                return cached

        # Fetch from provider
        df = self._provider_fetch(ticker, interval, lookback_days)

        # Validate
        df = self._validate(df, ticker)

        # Normalise
        df = self._normalise(df)

        # Cache
        if use_cache:
            self._save_cache(cache_key, df)

        return df

    def _provider_fetch(
        self, ticker: str, interval: str, lookback_days: int
    ) -> pd.DataFrame:
        if self.provider == "yfinance":
            return self._fetch_yfinance(ticker, interval, lookback_days)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def _fetch_yfinance(
        self, ticker: str, interval: str, lookback_days: int
    ) -> pd.DataFrame:
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            raise ImportError("yfinance is required. Install with: pip install yfinance")

        period_map = {
            "1m": "7d", "5m": "60d", "15m": "60d",
            "30m": "60d", "1h": "730d", "1d": "max",
        }
        period = period_map.get(interval, "max")

        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            raise DataQualityError(f"No data returned for {ticker}")

        # Slice to requested lookback
        if lookback_days and interval in ("1d",):
            df = df.iloc[-lookback_days:]

        return df

    def _validate(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Run quality checks. Raises DataQualityError on hard failures."""
        if df.empty:
            raise DataQualityError(f"{ticker}: empty DataFrame.")

        # Normalise columns for checks
        df.columns = [c.lower() if isinstance(c, str) else str(c).lower()
                      for c in df.columns]

        required = {"open", "high", "low", "close"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise DataQualityError(f"{ticker}: missing columns {missing_cols}")

        # Missing values
        missing_pct = df["close"].isna().mean()
        if missing_pct > self.max_missing_pct:
            raise DataQualityError(
                f"{ticker}: {missing_pct:.1%} missing close prices (max {self.max_missing_pct:.1%})"
            )

        # Forward-fill small gaps (up to 3 bars)
        df = df.ffill(limit=3)

        # Price sanity: no negative prices
        if (df["close"] <= 0).any():
            logger.warning("%s: Non-positive close prices found — dropping rows.", ticker)
            df = df[df["close"] > 0]

        # OHLC sanity
        bad_bars = (df["high"] < df["low"]).sum()
        if bad_bars > 0:
            logger.warning("%s: %d bars where high < low — dropping.", ticker, bad_bars)
            df = df[df["high"] >= df["low"]]

        # Volume check (warn only)
        if "volume" in df.columns:
            avg_vol = df["volume"].mean()
            if avg_vol < self.min_volume:
                logger.warning(
                    "%s: Average volume %.0f below minimum %.0f.",
                    ticker, avg_vol, self.min_volume
                )

        # Spread anomaly detection (warn only)
        spread_pct = ((df["high"] - df["low"]) / df["close"]).mean()
        if spread_pct > 0.10:
            logger.warning(
                "%s: Average spread %.1f%% is unusually high.", ticker, spread_pct * 100
            )

        return df

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure lowercase columns, sorted datetime index."""
        df.columns = [c.lower() if isinstance(c, str) else str(c).lower()
                      for c in df.columns]
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    # ------------------------------------------------------------------ #
    # Cache helpers (simple parquet cache)
    # ------------------------------------------------------------------ #

    def _cache_path(self, key: str) -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()
        return CACHE_DIR / f"{safe}.parquet"

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.cache_ttl:
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        try:
            path = self._cache_path(key)
            df.to_parquet(path)
        except Exception as exc:
            logger.debug("Cache write failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Anomaly helpers (callable from the main loop)
    # ------------------------------------------------------------------ #

    @staticmethod
    def detect_data_gaps(df: pd.DataFrame, max_gap_bars: int = 5) -> list[dict]:
        """Return a list of gap events where consecutive bars are far apart."""
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            return []

        gaps = []
        diffs = df.index.to_series().diff().dt.days.dropna()
        for i, gap in enumerate(diffs):
            if gap > max_gap_bars:
                gaps.append({
                    "start": str(df.index[i]),
                    "end": str(df.index[i + 1]),
                    "gap_bars": int(gap),
                })
        return gaps
