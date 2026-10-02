"""OANDA v20 REST adapter for XAU_USD (standard library only).

    OANDA_TOKEN        personal access token (My Account > Manage API Access)
    OANDA_ACCOUNT_ID   e.g. 101-004-1234567-001
    OANDA_ENV          practice (default) or live

On OANDA, XAU_USD units are troy ounces, so 1 lot = 100 units. Practice
accounts are free and are where this adapter should run until the paper and
practice results agree with the backtest.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable

import pandas as pd

from .base import Account, Broker, OrderResult, Position

HOSTS = {"practice": "https://api-fxpractice.oanda.com", "live": "https://api-fxtrade.oanda.com"}
GRANULARITY = {"M15": "M15", "H1": "H1", "H4": "H4", "D": "D", "1h": "H1", "4h": "H4", "15m": "M15"}

Transport = Callable[[str, str, dict | None], dict]


class OandaBroker(Broker):
    name = "oanda"

    def __init__(self, token: str | None = None, account_id: str | None = None,
                 env: str | None = None, instrument_code: str = "XAU_USD",
                 transport: Transport | None = None):
        self.token = token or os.environ.get("OANDA_TOKEN", "")
        self.account_id = account_id or os.environ.get("OANDA_ACCOUNT_ID", "")
        self.env = (env or os.environ.get("OANDA_ENV", "practice")).lower()
        if self.env not in HOSTS:
            raise ValueError("OANDA_ENV must be practice or live")
        if not (self.token and self.account_id) and transport is None:
            raise ValueError("set OANDA_TOKEN and OANDA_ACCOUNT_ID")
        self.live = self.env == "live"
        self.code = instrument_code
        self.host = HOSTS[self.env]
        self._transport = transport or self._http

    # -- HTTP -----------------------------------------------------------------
    def _http(self, method: str, path: str, body: dict | None) -> dict:
        req = urllib.request.Request(
            self.host + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
                     "Accept-Datetime-Format": "RFC3339"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            raise RuntimeError(f"OANDA {method} {path} -> {e.code}: {detail}") from None

    def _acct(self, suffix: str) -> str:
        return f"/v3/accounts/{self.account_id}{suffix}"

    # -- data ----------------------------------------------------------------------
    def candles(self, count: int = 3000, granularity: str = "H1") -> pd.DataFrame:
        g = GRANULARITY.get(granularity, granularity)
        out = []
        remaining = min(count, 60000)
        to = None
        # OANDA caps a request at 5000 candles; page backwards for more.
        while remaining > 0:
            n = min(remaining, 5000)
            q = f"?granularity={g}&price=M&count={n}" + (f"&to={to}" if to else "")
            data = self._transport("GET", f"/v3/instruments/{self.code}/candles{q}", None)
            rows = [c for c in data.get("candles", []) if c.get("complete", True)]
            if not rows:
                break
            out = rows + out
            remaining -= len(rows)
            to = rows[0]["time"]
            if len(data.get("candles", [])) < n:
                break
        df = pd.DataFrame({
            "open": [float(c["mid"]["o"]) for c in out],
            "high": [float(c["mid"]["h"]) for c in out],
            "low": [float(c["mid"]["l"]) for c in out],
            "close": [float(c["mid"]["c"]) for c in out],
            "volume": [float(c.get("volume", 0)) for c in out],
        }, index=pd.DatetimeIndex(pd.to_datetime([c["time"] for c in out], utc=True)))
        return df[~df.index.duplicated()].sort_index()

    # -- account ---------------------------------------------------------------------
    def account(self) -> Account:
        a = self._transport("GET", self._acct("/summary"), None)["account"]
        return Account(float(a["NAV"]), float(a["balance"]), float(a.get("marginUsed", 0)), a.get("currency", "USD"))

    def position(self) -> Position | None:
        trades = [t for t in self._transport("GET", self._acct("/openTrades"), None).get("trades", [])
                  if t.get("instrument") == self.code]
        if not trades:
            return None
        units = sum(float(t["currentUnits"]) for t in trades)
        if units == 0:
            return None
        avg = sum(float(t["price"]) * abs(float(t["currentUnits"])) for t in trades) / sum(abs(float(t["currentUnits"])) for t in trades)
        t0 = trades[0]
        stop = float(t0["stopLossOrder"]["price"]) if t0.get("stopLossOrder") else None
        tgt = float(t0["takeProfitOrder"]["price"]) if t0.get("takeProfitOrder") else None
        return Position(1 if units > 0 else -1, abs(units) / self.instrument.contract_size_oz, avg, stop, tgt,
                        t0.get("openTime", ""), ",".join(t["id"] for t in trades))

    # -- orders ------------------------------------------------------------------------
    def open(self, side: int, lots: float, stop: float | None, target: float | None,
             comment: str = "") -> OrderResult:
        units = int(round(lots * self.instrument.contract_size_oz)) * (1 if side > 0 else -1)
        if units == 0:
            return OrderResult(False, "open", side, lots, message="size rounds to zero units")
        order = {"type": "MARKET", "instrument": self.code, "units": str(units),
                 "timeInForce": "FOK", "positionFill": "DEFAULT"}
        if stop is not None:
            order["stopLossOnFill"] = {"price": f"{stop:.2f}"}
        if target is not None:
            order["takeProfitOnFill"] = {"price": f"{target:.2f}"}
        if comment:
            order["clientExtensions"] = {"comment": comment[:120], "tag": "aurum"}
        r = self._transport("POST", self._acct("/orders"), {"order": order})
        fill = r.get("orderFillTransaction")
        if not fill:
            reason = (r.get("orderCancelTransaction") or {}).get("reason", "not filled")
            return OrderResult(False, "open", side, lots, message=reason)
        tid = (fill.get("tradeOpened") or {}).get("tradeID", fill.get("id", ""))
        return OrderResult(True, "open", side, abs(units) / self.instrument.contract_size_oz,
                           float(fill["price"]), str(tid), comment)

    def modify_stop(self, stop: float, comment: str = "") -> OrderResult:
        pos = self.position()
        if pos is None:
            return OrderResult(False, "modify", 0, 0.0, message="no open position")
        last = None
        for tid in filter(None, pos.broker_id.split(",")):
            last = self._transport("PUT", self._acct(f"/trades/{tid}/orders"),
                                   {"stopLoss": {"price": f"{stop:.2f}", "timeInForce": "GTC"}})
        ok = bool(last) and "stopLossOrderTransaction" in last
        return OrderResult(ok, "modify", pos.side, pos.lots, float(stop), pos.broker_id,
                           comment if ok else json.dumps(last)[:300])

    def close(self, comment: str = "") -> OrderResult:
        pos = self.position()
        if pos is None:
            return OrderResult(False, "close", 0, 0.0, message="no open position")
        body = {"longUnits": "ALL"} if pos.side > 0 else {"shortUnits": "ALL"}
        r = self._transport("PUT", self._acct(f"/positions/{self.code}/close"), body)
        fill = r.get("longOrderFillTransaction") or r.get("shortOrderFillTransaction")
        if not fill:
            return OrderResult(False, "close", pos.side, pos.lots, message=json.dumps(r)[:300])
        return OrderResult(True, "close", pos.side, pos.lots, float(fill["price"]), str(fill.get("id", "")), comment)
