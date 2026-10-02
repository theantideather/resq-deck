"""MetaTrader 5 adapter (Windows, needs the MetaTrader5 package and a running terminal).

    pip install MetaTrader5
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER    optional; otherwise the logged in terminal is used
    MT5_SYMBOL                             default XAUUSD (some brokers use XAUUSD.a, GOLD ...)
    MT5_SERVER_UTC_OFFSET                  hours; MT5 bar times are broker server time,
                                           typically UTC+2 in winter and UTC+3 in summer

Whether the account is demo or real is read from the terminal, and `live`
is set from it, so the runner's live guard applies to real MT5 accounts.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from .base import Account, Broker, OrderResult, Position

MAGIC = 20260929
TIMEFRAMES = {"M15": "TIMEFRAME_M15", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D": "TIMEFRAME_D1",
              "15m": "TIMEFRAME_M15", "1h": "TIMEFRAME_H1", "4h": "TIMEFRAME_H4"}


class MT5Broker(Broker):
    name = "mt5"

    def __init__(self, symbol: str | None = None, server_utc_offset: float | None = None, module: Any = None):
        if module is None:
            try:
                import MetaTrader5 as module  # type: ignore
            except ImportError as e:
                raise ImportError("pip install MetaTrader5 (Windows only) and start the MT5 terminal") from e
        self.mt5 = module
        self.symbol = symbol or os.environ.get("MT5_SYMBOL", "XAUUSD")
        self.offset = float(server_utc_offset if server_utc_offset is not None else os.environ.get("MT5_SERVER_UTC_OFFSET", 0))
        kw = {}
        if os.environ.get("MT5_LOGIN"):
            kw = {"login": int(os.environ["MT5_LOGIN"]), "password": os.environ.get("MT5_PASSWORD", ""),
                  "server": os.environ.get("MT5_SERVER", "")}
        if not self.mt5.initialize(**kw):
            raise RuntimeError(f"MT5 initialize failed: {self.mt5.last_error()}")
        if not self.mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"symbol {self.symbol} not available in this terminal")
        info = self.mt5.account_info()
        # ACCOUNT_TRADE_MODE_DEMO == 0, CONTEST == 1, REAL == 2
        self.live = getattr(info, "trade_mode", 2) == 2

    def candles(self, count: int = 3000, granularity: str = "H1") -> pd.DataFrame:
        tf = getattr(self.mt5, TIMEFRAMES.get(granularity, "TIMEFRAME_H1"))
        # Position 1 skips the bar still forming.
        rates = self.mt5.copy_rates_from_pos(self.symbol, tf, 1, count)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"no rates for {self.symbol}: {self.mt5.last_error()}")
        df = pd.DataFrame(rates)
        idx = pd.to_datetime(df["time"], unit="s", utc=True) - pd.Timedelta(hours=self.offset)
        out = df[["open", "high", "low", "close"]].astype(float)
        out["volume"] = df.get("tick_volume", 0).astype(float)
        out.index = pd.DatetimeIndex(idx)
        return out.sort_index()

    def account(self) -> Account:
        a = self.mt5.account_info()
        return Account(float(a.equity), float(a.balance), float(a.margin), getattr(a, "currency", "USD"))

    def _positions(self):
        return [p for p in (self.mt5.positions_get(symbol=self.symbol) or []) if p.magic == MAGIC]

    def position(self) -> Position | None:
        ps = self._positions()
        if not ps:
            return None
        p = ps[0]
        side = 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1
        return Position(side, float(p.volume), float(p.price_open), float(p.sl) or None, float(p.tp) or None,
                        pd.Timestamp(p.time, unit="s", tz="UTC").isoformat(), str(p.ticket))

    def _filling(self) -> int:
        info = self.mt5.symbol_info(self.symbol)
        mode = getattr(info, "filling_mode", 0)
        # Bit flags: 1 = FOK allowed, 2 = IOC allowed.
        if mode & 2:
            return self.mt5.ORDER_FILLING_IOC
        if mode & 1:
            return self.mt5.ORDER_FILLING_FOK
        return self.mt5.ORDER_FILLING_RETURN

    def _send(self, req: dict) -> Any:
        res = self.mt5.order_send(req)
        if res is None:
            raise RuntimeError(f"order_send failed: {self.mt5.last_error()}")
        return res

    def open(self, side: int, lots: float, stop: float | None, target: float | None,
             comment: str = "") -> OrderResult:
        tick = self.mt5.symbol_info_tick(self.symbol)
        price = tick.ask if side > 0 else tick.bid
        req = {
            "action": self.mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": float(lots),
            "type": self.mt5.ORDER_TYPE_BUY if side > 0 else self.mt5.ORDER_TYPE_SELL,
            "price": price, "sl": float(stop or 0.0), "tp": float(target or 0.0),
            "deviation": 30, "magic": MAGIC, "comment": (comment or "aurum")[:31],
            "type_time": self.mt5.ORDER_TIME_GTC, "type_filling": self._filling(),
        }
        res = self._send(req)
        ok = res.retcode == self.mt5.TRADE_RETCODE_DONE
        return OrderResult(ok, "open", side, lots, float(getattr(res, "price", price)),
                           str(getattr(res, "order", "")), "" if ok else f"retcode {res.retcode}: {res.comment}")

    def modify_stop(self, stop: float, comment: str = "") -> OrderResult:
        ps = self._positions()
        if not ps:
            return OrderResult(False, "modify", 0, 0.0, message="no open position")
        p = ps[0]
        side = 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1
        res = self._send({"action": self.mt5.TRADE_ACTION_SLTP, "symbol": self.symbol, "position": p.ticket,
                          "sl": float(stop), "tp": float(p.tp or 0.0), "magic": MAGIC})
        ok = res.retcode == self.mt5.TRADE_RETCODE_DONE
        return OrderResult(ok, "modify", side, float(p.volume), float(stop), str(p.ticket),
                           comment if ok else f"retcode {res.retcode}: {res.comment}")

    def close(self, comment: str = "") -> OrderResult:
        ps = self._positions()
        if not ps:
            return OrderResult(False, "close", 0, 0.0, message="no open position")
        p = ps[0]
        side = 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1
        tick = self.mt5.symbol_info_tick(self.symbol)
        req = {
            "action": self.mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": float(p.volume),
            "type": self.mt5.ORDER_TYPE_SELL if side > 0 else self.mt5.ORDER_TYPE_BUY,
            "position": p.ticket, "price": tick.bid if side > 0 else tick.ask,
            "deviation": 30, "magic": MAGIC, "comment": (comment or "aurum close")[:31],
            "type_time": self.mt5.ORDER_TIME_GTC, "type_filling": self._filling(),
        }
        res = self._send(req)
        ok = res.retcode == self.mt5.TRADE_RETCODE_DONE
        return OrderResult(ok, "close", side, float(p.volume), float(getattr(res, "price", 0.0)),
                           str(p.ticket), "" if ok else f"retcode {res.retcode}: {res.comment}")
