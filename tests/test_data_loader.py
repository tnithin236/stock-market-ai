import numpy as np
import pytest

yf = pytest.importorskip("yfinance")

from src.data.loader import normalize_ticker, validate_data  # noqa: E402
from tests.helpers import make_ohlcv  # noqa: E402


def test_normalize_ticker_accepts_valid():
    assert normalize_ticker(" aapl ") == "AAPL"
    assert normalize_ticker("BRK-B") == "BRK-B"
    assert normalize_ticker("^GSPC") == "^GSPC"


@pytest.mark.parametrize("bad", ["", "AA PL", "AAPL;DROP", "../etc", "X" * 30])
def test_normalize_ticker_rejects_invalid(bad):
    with pytest.raises(ValueError):
        normalize_ticker(bad)


def test_validate_drops_bad_rows():
    df = make_ohlcv(400)
    df.iloc[10, df.columns.get_loc("Close")] = np.nan
    df.iloc[20, df.columns.get_loc("Open")] = -1.0
    cleaned = validate_data(df)
    assert len(cleaned) == 398
    assert not cleaned.isna().any().any()


def test_validate_requires_enough_rows():
    with pytest.raises(ValueError):
        validate_data(make_ohlcv(100))
