"""Small helpers for yfinance compatibility."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def configure_yfinance_cache() -> None:
    """Keep yfinance's sqlite cache inside the writable project tree."""
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        return

    cache_dir = Path("data/yfinance_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(cache_dir))


def normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance columns, including newer MultiIndex outputs."""
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex) or isinstance(df.columns[0], tuple):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    return df
