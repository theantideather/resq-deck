"""Order execution: a paper broker plus OANDA and MetaTrader 5 adapters behind one interface."""

from .base import Account, Broker, OrderResult, Position
from .paper import PaperBroker, ReplayFeed

__all__ = ["Account", "Broker", "OrderResult", "Position", "PaperBroker", "ReplayFeed", "make_broker"]


def make_broker(name: str, feed=None, **kw) -> Broker:
    """paper | oanda | mt5. The paper broker needs a feed (callable count -> OHLCV)."""
    if name == "paper":
        if feed is None:
            raise ValueError("paper broker needs a feed")
        return PaperBroker(feed, **kw)
    if name == "oanda":
        from .oanda import OandaBroker
        return OandaBroker(**kw)
    if name == "mt5":
        from .mt5 import MT5Broker
        return MT5Broker(**kw)
    raise ValueError(f"unknown broker {name!r}")
