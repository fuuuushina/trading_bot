"""
src/features/indicators.py

Pure-function technical indicators. All functions take a pd.DataFrame with
OHLCV columns (open, high, low, close, volume — lowercase) and return either
a pd.Series or the DataFrame with a new column added.

No side effects, no I/O. Fully testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")


# ---------------------------------------------------------------------------
# Trend indicators
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    _validate(df, "high", "low", "close")
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_vals = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_vals)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_vals)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx_vals = dx.rolling(period).mean()
    return adx_vals


# ---------------------------------------------------------------------------
# Momentum indicators
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)


def momentum_return(series: pd.Series, period: int) -> pd.Series:
    """Simple percentage return over N periods."""
    return series.pct_change(period)


def rate_of_change(series: pd.Series, period: int = 14) -> pd.Series:
    return (series - series.shift(period)) / (series.shift(period) + 1e-10) * 100


# ---------------------------------------------------------------------------
# Volatility indicators
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    _validate(df, "high", "low", "close")
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands: upper, middle (SMA), lower, bandwidth, %B."""
    middle = sma(series, period)
    std = series.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bandwidth = (upper - lower) / (middle + 1e-10)
    pct_b = (series - lower) / (upper - lower + 1e-10)
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "bb_bandwidth": bandwidth,
            "bb_pct_b": pct_b,
        }
    )


def keltner_channels(
    df: pd.DataFrame, period: int = 20, atr_mult: float = 2.0
) -> pd.DataFrame:
    """Keltner Channels for squeeze detection."""
    middle = ema(df["close"], period)
    atr_vals = atr(df, period)
    upper = middle + atr_mult * atr_vals
    lower = middle - atr_mult * atr_vals
    return pd.DataFrame(
        {"kc_upper": upper, "kc_middle": middle, "kc_lower": lower}
    )


def historical_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Annualised historical volatility (log returns std)."""
    log_returns = np.log(series / series.shift(1))
    return log_returns.rolling(period).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# Volume indicators
# ---------------------------------------------------------------------------

def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume relative to rolling average."""
    _validate(df, "volume")
    avg_vol = df["volume"].rolling(period).mean()
    return df["volume"] / (avg_vol + 1e-10)


def vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP — typically reset each trading day. Works on intraday data."""
    _validate(df, "high", "low", "close", "volume")
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / (cum_vol + 1e-10)


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    _validate(df, "close", "volume")
    direction = np.sign(df["close"].diff())
    return (direction * df["volume"]).cumsum()


# ---------------------------------------------------------------------------
# Statistical indicators
# ---------------------------------------------------------------------------

def z_score(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling z-score."""
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - mean) / (std + 1e-10)


def rolling_correlation(
    s1: pd.Series, s2: pd.Series, period: int = 20
) -> pd.Series:
    """Rolling Pearson correlation between two series."""
    return s1.rolling(period).corr(s2)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def drawdown(series: pd.Series) -> pd.Series:
    """Running drawdown from peak."""
    running_max = series.cummax()
    return (series - running_max) / (running_max + 1e-10)


def max_drawdown(series: pd.Series) -> float:
    """Maximum drawdown over the entire series."""
    return float(drawdown(series).min())


# ---------------------------------------------------------------------------
# Market breadth (requires multiple assets)
# ---------------------------------------------------------------------------

def advance_decline_ratio(returns: pd.DataFrame) -> pd.Series:
    """
    Given a DataFrame where each column is daily returns for an asset,
    returns the ratio of advancing assets to declining assets.
    """
    advances = (returns > 0).sum(axis=1)
    declines = (returns < 0).sum(axis=1)
    return advances / (declines + 1e-10)


def pct_above_ema(prices: pd.DataFrame, period: int = 200) -> pd.Series:
    """Percentage of assets trading above their N-day EMA."""
    above = prices.apply(lambda s: s > ema(s, period), axis=0)
    return above.mean(axis=1)


# ---------------------------------------------------------------------------
# Feature bundle — compute all indicators for a DataFrame in one call
# ---------------------------------------------------------------------------

def compute_all_features(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Compute the full feature set for a single-asset OHLCV DataFrame.
    Returns the original DataFrame augmented with new columns.
    """
    cfg = config or {}
    out = df.copy()

    close = out["close"]

    # Trend
    out["ema_20"] = ema(close, 20)
    out["ema_50"] = ema(close, 50)
    out["ema_200"] = ema(close, 200)
    out["sma_20"] = sma(close, 20)
    out["sma_50"] = sma(close, 50)

    # MACD
    macd_df = macd(close)
    out = pd.concat([out, macd_df], axis=1)

    # Momentum
    out["rsi_14"] = rsi(close, 14)
    out["roc_14"] = rate_of_change(close, 14)
    out["momentum_5"] = momentum_return(close, 5)
    out["momentum_20"] = momentum_return(close, 20)
    out["momentum_60"] = momentum_return(close, 60)

    # Volatility
    out["atr_14"] = atr(out, 14)
    out["hist_vol_20"] = historical_volatility(close, 20)
    out["z_score_20"] = z_score(close, 20)

    # Bollinger Bands
    bb_df = bollinger_bands(close, 20, 2.0)
    out = pd.concat([out, bb_df], axis=1)

    # Volume
    if "volume" in out.columns:
        out["volume_ratio_20"] = volume_ratio(out, 20)
        out["obv"] = on_balance_volume(out)

    # Drawdown
    out["drawdown"] = drawdown(close)

    # ADX
    out["adx_14"] = adx(out, 14)

    return out
