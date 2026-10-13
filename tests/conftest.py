import pytest

from tests.helpers import make_ohlcv


@pytest.fixture
def ohlcv():
    return make_ohlcv()
