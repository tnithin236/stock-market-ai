"""Download, validate and cache OHLCV market data."""
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config import resolve_path

logger = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
_TICKER_RE = re.compile(r"^[A-Z0-9.\-\^=]{1,15}$")


def normalize_ticker(ticker: str) -> str:
    """Uppercase/strip and reject suspicious input (matters once this sits behind an API)."""
    t = ticker.strip().upper()
    if not _TICKER_RE.match(t):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    return t


def download_stock_data(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance.

    Prices are split/dividend adjusted (auto_adjust=True), so Close is the
    adjusted close. This avoids fake 'crashes' on split dates.
    """
    ticker = normalize_ticker(ticker)
    df = yf.download(
        ticker, start=start, end=end, interval=interval,
        auto_adjust=True, progress=False,
    )
    if df is None or df.empty:
        raise ValueError(
            f"No data returned for {ticker!r}. Check the symbol, date range, "
            "or your internet connection (Yahoo may also be rate-limiting you)."
        )

    # Recent yfinance versions return MultiIndex columns like ('Close', 'AAPL')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")

    df = df[OHLCV].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity-check and clean OHLCV data. Returns a cleaned copy; logs every issue."""
    df = df.copy()
    n0 = len(df)

    n_nan = int(df[OHLCV].isna().any(axis=1).sum())
    if n_nan:
        logger.warning("Dropping %d rows with NaN values", n_nan)
        df = df.dropna(subset=OHLCV)

    bad_price = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    if bad_price.any():
        logger.warning("Dropping %d rows with non-positive prices", int(bad_price.sum()))
        df = df[~bad_price]

    bad_range = df["High"] < df["Low"]
    if bad_range.any():
        logger.warning("Dropping %d rows where High < Low", int(bad_range.sum()))
        df = df[~bad_range]

    zero_vol = int((df["Volume"] == 0).sum())
    if zero_vol:
        logger.warning("%d rows have zero volume (kept)", zero_vol)

    if len(df) < 250:
        raise ValueError(
            f"Only {len(df)} rows after cleaning; need at least ~1 year of data to model."
        )
    logger.info("Validation complete: %d -> %d rows", n0, len(df))
    return df


def _cache_path(raw_dir: Path, ticker: str, start: str, end: Optional[str], interval: str) -> Path:
    return raw_dir / f"{ticker}_{interval}_{start}_{end or 'latest'}.csv"


def load_stock_data(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    raw_dir: str = "data/raw",
    max_age_hours: float = 24,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return validated OHLCV data, using a local CSV cache when it is fresh enough."""
    ticker = normalize_ticker(ticker)
    raw_path = resolve_path(raw_dir)
    cache = _cache_path(raw_path, ticker, start, end, interval)

    if cache.exists() and not force_refresh:
        age_h = (time.time() - cache.stat().st_mtime) / 3600
        # A fixed end date never goes stale; an open-ended one does.
        if end is not None or age_h <= max_age_hours:
            logger.info("Loading %s from cache (%s)", ticker, cache.name)
            df = pd.read_csv(cache, index_col="Date", parse_dates=True)
            return validate_data(df)

    logger.info("Downloading %s from Yahoo Finance", ticker)
    df = validate_data(download_stock_data(ticker, start, end, interval))
    df.to_csv(cache)
    logger.info("Saved to %s", cache)
    return df
