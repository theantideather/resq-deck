"""Paper broker: simulated XAUUSD account with the backtester's cost model.

Fills pay half the spread plus slippage and half the round turn commission,
stops and targets trigger on bar highs and lows (gaps fill at the open),
and swap is charged at every 17:00 New York rollover with the Wednesday
triple. State lives in a JSON file, so the account survives restarts and
the dashboard, the MCP server and the runner all see the same book.

Bars come from a feed: any callable returning OHLCV (Yahoo, a CSV, OANDA
candles, or a ReplayFeed that walks through history one bar per cycle).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable

import pandas as pd

from .. import calendar as cal
from ..instrument import XAUUSD, GoldInstrument
from .base import Account, Broker, OrderResult, Position

Feed = Callable[[int], pd.DataFrame]
DEFAULT_STATE = Path(os.environ.get("AURUM_HOME", Path.home() / ".aurum")) / "paper_account.json"


class PaperBroker(Broker):
    name = "paper"
    live = False

    def __init__(self, feed: Feed, state_path: str | Path = DEFAULT_STATE,
                 initial_equity: float = 100_000.0, instrument: GoldInstrument = XAUUSD):
        self.feed = feed
        self.instrument = instrument
        self.path = Path(state_path)
        self._lock = threading.Lock()
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"balance": initial_equity, "initial": initial_equity, "position": None,
                          "trades": [], "last_bar": None, "last_price": None}
            self._save()

    # -- persistence ----------------------------------------------------------
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str))
        tmp.replace(self.path)

    def reset(self, initial_equity: float | None = None) -> None:
        eq = initial_equity or self.state.get("initial", 100_000.0)
        self.state = {"balance": eq, "initial": eq, "position": None, "trades": [],
                      "last_bar": None, "last_price": None}
        self._save()

    # -- market data ------------------------------------------------------------
    def candles(self, count: int = 3000, granularity: str = "H1") -> pd.DataFrame:
        return self.feed(count)

    def _cost(self, lots: float, event: bool = False, rollover: bool = False) -> float:
        ins = self.instrument
        spread = ins.spread_at(event_risk=event, rollover=rollover)
        return lots * ins.contract_size_oz * (spread / 2 + ins.slippage_usd) + lots * ins.commission_per_lot / 2

    # -- account ------------------------------------------------------------------
    def account(self) -> Account:
        pos = self.state["position"]
        open_pnl = 0.0
        margin = 0.0
        price = self.state.get("last_price")
        if pos and price:
            open_pnl = pos["side"] * (price - pos["entry_price"]) * pos["lots"] * self.instrument.contract_size_oz
            margin = self.instrument.margin_required(pos["lots"], price)
        return Account(equity=self.state["balance"] + open_pnl, balance=self.state["balance"], margin_used=margin)

    def position(self) -> Position | None:
        p = self.state["position"]
        if not p:
            return None
        return Position(p["side"], p["lots"], p["entry_price"], p.get("stop"), p.get("target"),
                        p.get("opened_utc", ""), p.get("id", ""))

    def trades(self) -> list[dict]:
        return list(self.state["trades"])

    # -- orders -------------------------------------------------------------------
    def open(self, side: int, lots: float, stop: float | None, target: float | None,
             comment: str = "") -> OrderResult:
        with self._lock:
            if self.state["position"]:
                return OrderResult(False, "open", side, lots, message="position already open; close it first")
            price = self.state.get("last_price")
            if price is None:
                return OrderResult(False, "open", side, lots, message="no price yet; sync bars first")
            lots = self.instrument.round_lots(lots)
            if lots <= 0:
                return OrderResult(False, "open", side, lots, message="size rounds to zero lots")
            cost = self._cost(lots)
            self.state["balance"] -= cost
            tid = f"P{len(self.state['trades']) + 1:05d}"
            self.state["position"] = {
                "id": tid, "side": int(side), "lots": lots, "entry_price": float(price),
                "stop": stop, "target": target, "opened_utc": self.state.get("last_bar") or "",
                "costs": cost, "swap": 0.0, "comment": comment,
            }
            self._save()
            return OrderResult(True, "open", side, lots, price, tid, comment)

    def _close_at(self, price: float, when: str, reason: str) -> OrderResult:
        p = self.state["position"]
        cost = self._cost(p["lots"])
        gross = p["side"] * (price - p["entry_price"]) * p["lots"] * self.instrument.contract_size_oz
        self.state["balance"] += gross - cost
        rec = {**p, "exit_price": float(price), "closed_utc": when, "reason": reason,
               "gross_pnl": gross, "costs": p["costs"] + cost,
               "net_pnl": gross - p["costs"] - cost + p["swap"]}
        self.state["trades"].append(rec)
        self.state["position"] = None
        self._save()
        return OrderResult(True, "close", p["side"], p["lots"], price, p["id"], reason)

    def close(self, comment: str = "signal") -> OrderResult:
        with self._lock:
            if not self.state["position"]:
                return OrderResult(False, "close", 0, 0.0, message="no open position")
            return self._close_at(self.state["last_price"], self.state.get("last_bar") or "", comment)

    def set_price(self, price: float, when: str = "") -> None:
        """Mark the book at an external price (e.g. a TradingView alert) without bars."""
        with self._lock:
            self.state["last_price"] = float(price)
            if when:
                self.state["last_bar"] = when
            self._save()

    # -- simulation -----------------------------------------------------------------
    def sync(self, bars: pd.DataFrame) -> list[dict]:
        """Walk bars newer than the last one seen: rollover swap, stops and targets, last price."""
        events: list[dict] = []
        if bars is None or bars.empty:
            return events
        idx = cal.ensure_utc(pd.DatetimeIndex(bars.index))
        bars = bars.set_axis(idx)
        last = self.state.get("last_bar")
        new = bars[bars.index > pd.Timestamp(last)] if last else bars.tail(1)
        if new.empty:
            return events
        days = cal.trading_day(new.index)
        prev_day = cal.trading_day(pd.DatetimeIndex([pd.Timestamp(last)]))[0] if last else days[0]
        ins = self.instrument
        with self._lock:
            for (ts, bar), day in zip(new.iterrows(), days):
                p = self.state["position"]
                if p and day != prev_day:
                    nights = 3 if pd.Timestamp(prev_day).weekday() == ins.triple_swap_weekday else 1
                    rate = ins.swap_long_per_lot if p["side"] > 0 else ins.swap_short_per_lot
                    sw = rate * p["lots"] * nights
                    p["swap"] += sw
                    self.state["balance"] += sw
                    events.append({"time": ts.isoformat(), "event": "swap", "usd": sw})
                prev_day = day
                if p:
                    s = p["side"]
                    stop, tgt = p.get("stop"), p.get("target")
                    if stop is not None and ((s > 0 and bar["low"] <= stop) or (s < 0 and bar["high"] >= stop)):
                        gapped = (s > 0 and bar["open"] < stop) or (s < 0 and bar["open"] > stop)
                        r = self._close_at(bar["open"] if gapped else stop, ts.isoformat(), "stop")
                        events.append({"time": ts.isoformat(), "event": "stop", "price": r.price})
                    elif tgt is not None and ((s > 0 and bar["high"] >= tgt) or (s < 0 and bar["low"] <= tgt)):
                        gapped = (s > 0 and bar["open"] > tgt) or (s < 0 and bar["open"] < tgt)
                        r = self._close_at(bar["open"] if gapped else tgt, ts.isoformat(), "target")
                        events.append({"time": ts.isoformat(), "event": "target", "price": r.price})
                self.state["last_bar"] = ts.isoformat()
                self.state["last_price"] = float(bar["close"])
            self._save()
        return events


class ReplayFeed:
    """Walks through a historical frame one bar per `advance()`, for dry runs of the live loop."""

    def __init__(self, bars: pd.DataFrame, start: int):
        self.bars = bars
        self.pos = start

    def advance(self, n: int = 1) -> bool:
        self.pos = min(self.pos + n, len(self.bars))
        return self.pos < len(self.bars)

    def __call__(self, count: int) -> pd.DataFrame:
        return self.bars.iloc[max(0, self.pos - count): self.pos]
