"""aurum: AI quant research and trading toolkit for gold (XAUUSD)."""

from .backtest import BacktestConfig, run_backtest
from .data import MarketData, load_market, synthetic_market
from .features import build_features
from .instrument import GC_FUTURE, MGC_FUTURE, XAUUSD, GoldInstrument
from .regime import detect_regimes
from .strategies import STRATEGIES

__version__ = "0.1.0"

__all__ = [
    "BacktestConfig", "run_backtest", "MarketData", "load_market", "synthetic_market",
    "build_features", "GoldInstrument", "XAUUSD", "GC_FUTURE", "MGC_FUTURE",
    "detect_regimes", "STRATEGIES",
]
