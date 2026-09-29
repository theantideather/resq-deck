"""What every broker adapter provides, so the runner never cares which one it has."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import pandas as pd

from ..instrument import XAUUSD, GoldInstrument


@dataclass
class Position:
    side: int                 # +1 long, -1 short
    lots: float
    entry_price: float
    stop: float | None = None
    target: float | None = None
    opened_utc: str = ""
    broker_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Account:
    equity: float
    balance: float
    margin_used: float = 0.0
    currency: str = "USD"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OrderResult:
    ok: bool
    action: str               # open / close
    side: int
    lots: float
    price: float | None = None
    broker_id: str = ""
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Broker(ABC):
    """One XAUUSD position at a time, market orders with a protective stop."""

    name: str = "broker"
    live: bool = False        # True when orders reach a real account with real money
    instrument: GoldInstrument = XAUUSD

    @abstractmethod
    def candles(self, count: int = 3000, granularity: str = "H1") -> pd.DataFrame:
        """Completed OHLCV bars, UTC index, oldest first. The forming bar is excluded."""

    @abstractmethod
    def account(self) -> Account: ...

    @abstractmethod
    def position(self) -> Position | None: ...

    @abstractmethod
    def open(self, side: int, lots: float, stop: float | None, target: float | None,
             comment: str = "") -> OrderResult: ...

    @abstractmethod
    def close(self, comment: str = "") -> OrderResult: ...

    def sync(self, bars: pd.DataFrame) -> list[dict]:
        """Let simulated brokers catch up on bars (stops, swap). Real brokers do this themselves."""
        return []
