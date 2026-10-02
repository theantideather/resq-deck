"""The trading loop: at each bar close, decide and reconcile one XAUUSD position.

    python -m aurum run --broker paper --feed yahoo            # paper, loops forever
    python -m aurum run --broker oanda --once                  # OANDA practice, one cycle (cron friendly)

One cycle:
    1. pull completed bars from the broker (or the paper feed) and let a
       simulated broker catch up on stops and swap
    2. skip if the data is stale (market closed, feed down)
    3. build features exactly as the backtest does and compute the strategy signal
    4. optionally run the analyst desk; the risk manager can only shrink or veto
    5. apply account guards: daily loss limit, drawdown kill switch, max lots,
       re-entry lock after a stop, time stop
    6. close and/or open so the broker position matches the decision
    7. append everything to the journal and notify on orders and guard trips

Live money is blocked unless the runner is started with allow_live and the
environment variable AURUM_LIVE_ACK holds the exact acknowledgement string.
OANDA practice and MT5 demo accounts are not live and need neither.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import calendar as cal
from .backtest import BacktestConfig, initial_stop_distance, risk_lots
from .data import market_from_bars
from .execution.base import Broker
from .features import atr as _atr
from .features import build_features
from .notify import Notifier
from .strategies import STRATEGIES

AURUM_HOME = Path(os.environ.get("AURUM_HOME", Path.home() / ".aurum"))
LIVE_ACK = "I accept the risk of trading real money"
BAR_SECONDS = {"M15": 900, "15m": 900, "H1": 3600, "1h": 3600, "H4": 14400, "4h": 14400, "D": 86400}


def completed_market(md, timeframe: str, now: pd.Timestamp):
    """Strategy timeframe candles that have fully closed by `now`."""
    from .data import MarketData
    from .swing import bars_for, to_daily

    if timeframe in ("D", "1D"):
        b = to_daily(md.bars, now=now)
    else:
        b = bars_for(md.bars, timeframe)
        if len(b) and b.index[-1] + pd.Timedelta(timeframe) > now:
            b = b.iloc[:-1]
    return MarketData(b, pd.DataFrame(index=b.index), md.events, md.source)


class LiveTradingBlocked(RuntimeError):
    pass


@dataclass
class RunnerConfig:
    strategy: str = "ensemble"
    granularity: str = "H1"
    bars: int = 3000
    with_macro: bool = True
    use_desk: bool = True
    use_llm: bool = False
    max_lots: float = 2.0
    risk_per_trade: float | None = None
    daily_loss_limit: float = 0.03
    max_drawdown_kill: float = 0.20
    stale_after_bars: float = 3.0
    # Swing strategies (daily, 4h) need years of history for the EMA 200 and the
    # path dependent HalfTrend state to match the backtest.
    swing_history_days: int = 1500
    allow_live: bool = False
    journal_path: Path = field(default_factory=lambda: AURUM_HOME / "journal.jsonl")
    state_path: Path = field(default_factory=lambda: AURUM_HOME / "runner_state.json")


class Runner:
    def __init__(self, broker: Broker, cfg: RunnerConfig | None = None,
                 notifier: Notifier | None = None, now: Callable[[], pd.Timestamp] | None = None,
                 headlines: Callable[[], list[str]] | None = None):
        self.broker = broker
        self.cfg = cfg or RunnerConfig()
        if self.cfg.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.cfg.strategy!r}")
        self.notifier = notifier or Notifier()
        self.now = now or (lambda: pd.Timestamp.now(tz="UTC"))
        self.headlines = headlines
        self._check_live()
        self.state = self._load_state()

    # -- safety -------------------------------------------------------------------
    def _check_live(self) -> None:
        if self.broker.live and not (self.cfg.allow_live and os.environ.get("AURUM_LIVE_ACK") == LIVE_ACK):
            raise LiveTradingBlocked(
                f"{self.broker.name} is a real money account. Start with --allow-live and set "
                f'AURUM_LIVE_ACK="{LIVE_ACK}" to trade it. Use a practice or demo account first.')

    def _load_state(self) -> dict:
        p = Path(self.cfg.state_path)
        if p.exists():
            return json.loads(p.read_text())
        return {"day": None, "day_start_equity": None, "peak_equity": None, "halted_day": None,
                "killed": False, "locked_side": 0, "last_side": 0, "entry_bar": None, "closed_by_us": False}

    def _save_state(self) -> None:
        p = Path(self.cfg.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.state, indent=1, default=str))

    def _journal(self, rec: dict) -> None:
        p = Path(self.cfg.journal_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # -- one cycle -------------------------------------------------------------------
    def cycle(self) -> dict:
        cfg = self.cfg
        now = self.now()
        rec: dict = {"time_utc": now.isoformat(), "broker": self.broker.name, "live": self.broker.live,
                     "strategy": cfg.strategy, "orders": [], "notes": []}
        strat = STRATEGIES[cfg.strategy]
        need = cfg.bars if strat.timeframe == "H1" else max(cfg.bars, cfg.swing_history_days * 24)
        bars = self.broker.candles(need, cfg.granularity)
        if bars is None or len(bars) < 300:
            rec["action"] = "skip"
            rec["notes"].append(f"only {0 if bars is None else len(bars)} bars; need 300")
            self._journal(rec)
            return rec
        rec["sim_events"] = self.broker.sync(bars)
        last_bar = pd.Timestamp(bars.index[-1])
        last_bar = last_bar.tz_localize("UTC") if last_bar.tz is None else last_bar.tz_convert("UTC")
        rec["last_bar"] = last_bar.isoformat()
        rec["price"] = float(bars["close"].iloc[-1])
        bar_s = BAR_SECONDS.get(cfg.granularity, 3600)
        age = (now - last_bar).total_seconds()
        if age > bar_s * (1 + cfg.stale_after_bars):
            rec["action"] = "skip"
            rec["notes"].append(f"last bar is {age / 3600:.1f} h old; market closed or feed down")
            self._journal(rec)
            return rec

        md = market_from_bars(bars, with_macro=cfg.with_macro, source=self.broker.name, quiet=True)
        # Hourly features are only needed by hourly strategies and by the desk.
        need_feats = strat.timeframe == "H1" or cfg.use_desk
        feats = build_features(md, fair_window=min(2000, max(200, len(bars) // 3))) if need_feats \
            else pd.DataFrame(index=md.bars.index)
        # Swing strategies run on their own candles, completed ones only.
        m = md if strat.timeframe == "H1" else completed_market(md, strat.timeframe, now)
        sfeats = feats if m is md else pd.DataFrame(index=m.bars.index)
        signal = strat.signal(m, sfeats)
        line = strat.stop_line(m) if strat.stop_line else None
        stops = strat.entry_stops(m) if strat.entry_stops else None
        rec["timeframe"] = strat.timeframe
        if line is not None and len(line) and np.isfinite(line.iloc[-1]):
            rec["trail_level"] = float(line.iloc[-1])
        quant = float(np.clip(signal.iloc[-1], -1, 1)) if np.isfinite(signal.iloc[-1]) else 0.0
        rec["quant_signal"] = quant
        final = quant
        if cfg.use_desk:
            from .agents import run_desk
            heads = None
            if cfg.use_llm and self.headlines:
                try:
                    heads = self.headlines()
                except Exception as e:
                    rec["notes"].append(f"headlines unavailable: {e}")
            _, decision = run_desk(md, feats, quant, headlines=heads, use_llm=cfg.use_llm)
            final = decision.final_signal
            rec["desk_notes"] = decision.notes
            rec["views"] = [{"analyst": v.analyst, "bias": round(v.bias, 3), "confidence": round(v.confidence, 3)}
                            for v in decision.views]
        rec["desk_signal"] = final

        # Account guards.
        acct = self.broker.account()
        rec["equity"] = acct.equity
        st = self.state
        day = str(cal.trading_day(pd.DatetimeIndex([now]))[0])
        if st["day"] != day:
            st["day"], st["day_start_equity"] = day, acct.equity
        st["peak_equity"] = max(st["peak_equity"] or acct.equity, acct.equity)
        if acct.equity <= st["day_start_equity"] * (1 - cfg.daily_loss_limit) and st["halted_day"] != day:
            st["halted_day"] = day
            self._alert(f"aurum: daily loss limit hit ({acct.equity:,.0f} vs {st['day_start_equity']:,.0f}); flat for the day")
        if not st["killed"] and acct.equity <= st["peak_equity"] * (1 - cfg.max_drawdown_kill):
            st["killed"] = True
            self._alert(f"aurum: drawdown kill switch ({acct.equity:,.0f} vs peak {st['peak_equity']:,.0f}); trading stopped. "
                        f"Delete {cfg.state_path} to re-arm after review.")
        if st["killed"]:
            final = 0.0
            rec["notes"].append("kill switch active")
        elif st["halted_day"] == day:
            final = 0.0
            rec["notes"].append("daily loss limit active")

        pos = self.broker.position()
        # A position that vanished without us closing it was stopped or hit its target.
        if st["last_side"] and pos is None and not st["closed_by_us"]:
            st["locked_side"] = st["last_side"]
            rec["notes"].append(f"position closed by stop/target; side {st['last_side']:+d} locked until the signal changes")
        st["closed_by_us"] = False
        want = int(np.sign(final))
        if st["locked_side"] and want != st["locked_side"]:
            st["locked_side"] = 0
        if want and want == st["locked_side"]:
            rec["notes"].append("re-entry lock")
            want = 0 if pos is None else want

        bt_cfg = strat.config
        if cfg.risk_per_trade:
            bt_cfg = BacktestConfig(**{**bt_cfg.__dict__, "risk_per_trade": cfg.risk_per_trade})
        # Time stop.
        time_stop = False
        if pos and bt_cfg.max_bars_in_trade and st["entry_bar"]:
            held = (last_bar - pd.Timestamp(st["entry_bar"])).total_seconds() / bar_s
            if held >= bt_cfg.max_bars_in_trade:
                rec["notes"].append(f"time stop after {held:.0f} bars")
                want = 0
                time_stop = True

        if pos and want != pos.side:
            r = self.broker.close("time" if time_stop else ("signal" if want == 0 else "reverse"))
            rec["orders"].append(r.to_dict())
            if r.ok:
                st["closed_by_us"] = True
                pos = None
        if want and pos is None:
            a = float(_atr(m.bars).iloc[-1])
            line_dist = want * (rec["price"] - float(line.iloc[-1])) if line is not None else None
            if stops is not None:
                sd = want * (rec["price"] - float(stops["long" if want > 0 else "short"].iloc[-1]))
                dist = sd if np.isfinite(sd) and sd > 0 else 2.0 * a
            else:
                dist = initial_stop_distance(bt_cfg, a, rec["price"], line_dist)
            lots, stop_dist = risk_lots(acct.equity, abs(final), a, rec["price"], bt_cfg, dist)
            lots = min(lots, cfg.max_lots)
            if lots > 0:
                has_stop = bool(bt_cfg.stop_pct or bt_cfg.stop_atr or line is not None or stops is not None)
                stop = rec["price"] - want * stop_dist if has_stop else None
                if bt_cfg.target_r:
                    target = rec["price"] + want * bt_cfg.target_r * stop_dist
                else:
                    target = rec["price"] + want * bt_cfg.target_atr * a if bt_cfg.target_atr else None
                r = self.broker.open(want, lots, stop, target, comment=f"aurum {cfg.strategy}")
                rec["orders"].append(r.to_dict())
                if r.ok:
                    st["entry_bar"] = last_bar.isoformat()
                    st["entry_price"] = r.price or rec["price"]
                    st["init_dist"] = stop_dist
                    st["best"] = st["entry_price"]
            else:
                rec["notes"].append("size rounds to zero; account too small for this stop")
        elif pos and pos.side == want and pos.stop is not None:
            # Trailing and breakeven, with the same rules as the backtester.
            st["best"] = max(st.get("best") or pos.entry_price, rec["price"]) if pos.side > 0 else \
                min(st.get("best") or pos.entry_price, rec["price"])
            new = pos.stop
            if line is not None and np.isfinite(line.iloc[-1]):
                lv = float(line.iloc[-1])
                new = max(new, lv) if pos.side > 0 else min(new, lv)
            if bt_cfg.breakeven_r and st.get("init_dist") and \
                    pos.side * (st["best"] - pos.entry_price) >= bt_cfg.breakeven_r * st["init_dist"]:
                new = max(new, pos.entry_price) if pos.side > 0 else min(new, pos.entry_price)
            if abs(new - pos.stop) > 1e-6 and pos.side * (rec["price"] - new) > 0:
                r = self.broker.modify_stop(new, "trail")
                rec["orders"].append(r.to_dict())
        new_pos = self.broker.position()
        st["last_side"] = new_pos.side if new_pos else 0
        rec["position"] = new_pos.to_dict() if new_pos else None
        rec["action"] = "trade" if any(o["action"] != "modify" for o in rec["orders"]) else (
            "trail" if rec["orders"] else "hold")
        self._save_state()
        self._journal(rec)
        for o in rec["orders"]:
            verb = {"open": "OPEN", "close": "CLOSE", "modify": "MOVE STOP"}[o["action"]]
            side = "LONG" if o["side"] > 0 else "SHORT"
            status = "ok" if o["ok"] else f"FAILED ({o['message']})"
            px = f" @ {o['price']:.2f}" if o.get("price") else ""
            self._alert(f"aurum [{self.broker.name}] {verb} {side} {o['lots']} lots XAUUSD{px} {status} | "
                        f"{cfg.strategy} signal {quant:+.2f} -> {final:+.2f}")
        return rec

    def _alert(self, text: str) -> None:
        print(text)
        self.notifier.send(text)

    # -- loop ------------------------------------------------------------------------------
    def loop(self, delay_s: float = 20.0, max_cycles: int | None = None,
             sleep: Callable[[float], None] = time.sleep) -> None:
        bar_s = BAR_SECONDS.get(self.cfg.granularity, 3600)
        n = 0
        self._alert(f"aurum runner started: {self.broker.name}, {self.cfg.strategy}, {self.cfg.granularity}")
        while max_cycles is None or n < max_cycles:
            try:
                rec = self.cycle()
                print(f"[{rec['time_utc'][:19]}] {rec['action']} price {rec.get('price', 0):.2f} "
                      f"signal {rec.get('quant_signal', 0):+.2f} -> {rec.get('desk_signal', 0):+.2f} "
                      f"{'; '.join(rec['notes'])}")
            except LiveTradingBlocked:
                raise
            except Exception as e:
                traceback.print_exc()
                self._alert(f"aurum runner error: {e}")
            n += 1
            if max_cycles is not None and n >= max_cycles:
                break
            now = self.now().timestamp()
            wait = bar_s - (now % bar_s) + delay_s
            sleep(wait)


def read_journal(path: Path | None = None, limit: int = 200) -> list[dict]:
    p = Path(path or AURUM_HOME / "journal.jsonl")
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
