"""Gold's clock: trading sessions, the LBMA auctions and US macro events.

Gold trades nearly 24 hours, but not evenly. Physical buying in Shanghai and
Mumbai sets the tone in Asia, London is where the benchmark price is set and
most OTC volume clears, and New York (COMEX plus US data at 08:30 ET) is where
the biggest moves happen. Everything here works on a UTC DatetimeIndex and
handles daylight saving through zoneinfo, so a London open is always 08:00 in
London whatever the month.
"""

from __future__ import annotations

from datetime import date, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")
TOKYO = ZoneInfo("Asia/Tokyo")

# LBMA Gold Price auctions, London local time.
LBMA_AM_FIX = time(10, 30)
LBMA_PM_FIX = time(15, 0)

# FOMC statement days (14:00 ET). Published schedule; check
# federalreserve.gov/monetarypolicy/fomccalendars.htm and extend each year.
FOMC_DATES = [
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    "2026-09-16", "2026-10-28", "2026-12-09",
]


def _local_hour(index: pd.DatetimeIndex, tz: ZoneInfo) -> np.ndarray:
    local = index.tz_convert(tz)
    return local.hour.to_numpy() + local.minute.to_numpy() / 60.0


def ensure_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC, nanosecond resolution, so integer comparisons across indexes are safe."""
    index = pd.DatetimeIndex(index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return index.as_unit("ns")


def _ns(values) -> np.ndarray:
    return ensure_utc(pd.DatetimeIndex(pd.to_datetime(values, utc=True))).asi8


def session_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean columns for each session a bar's open falls in.

    asia       Tokyo 09:00-15:00 local, which also covers the Shanghai day
    london     08:00-16:30 London
    new_york   08:20-13:30 New York, the COMEX floor hours when liquidity peaks
    overlap    London and New York both open, the most volatile window
    """
    index = ensure_utc(index)
    tk = _local_hour(index, TOKYO)
    ld = _local_hour(index, LONDON)
    ny = _local_hour(index, NEW_YORK)
    out = pd.DataFrame(index=index)
    out["asia"] = (tk >= 9) & (tk < 15)
    out["london"] = (ld >= 8) & (ld < 16.5)
    out["new_york"] = (ny >= 8 + 20 / 60) & (ny < 13.5)
    out["overlap"] = out["london"] & out["new_york"]
    return out


def near_lbma_fix(index: pd.DatetimeIndex, window_minutes: int = 30) -> pd.Series:
    """True for bars whose open is within `window_minutes` before an LBMA auction."""
    index = ensure_utc(index)
    ld = _local_hour(index, LONDON)
    am = LBMA_AM_FIX.hour + LBMA_AM_FIX.minute / 60
    pm = LBMA_PM_FIX.hour + LBMA_PM_FIX.minute / 60
    w = window_minutes / 60
    flag = ((ld > am - w - 1e-9) & (ld <= am)) | ((ld > pm - w - 1e-9) & (ld <= pm))
    return pd.Series(flag, index=index, name="near_fix")


def trading_day(index: pd.DatetimeIndex) -> pd.Index:
    """The gold trading day a bar belongs to. It rolls at 17:00 New York."""
    index = ensure_utc(index)
    ny = index.tz_convert(NEW_YORK)
    shifted = ny + pd.Timedelta(hours=7)
    return pd.Index(shifted.date, name="trading_day")


def nfp_dates(start: int, end: int) -> list[date]:
    """First Friday of each month, the usual Nonfarm Payrolls release day.

    The BLS occasionally moves it (a Friday on the 1st sometimes slips a week,
    shutdowns delay it). Pass real dates to `event_calendar` when you have them.
    """
    out = []
    for y in range(start, end + 1):
        for m in range(1, 13):
            d = date(y, m, 1)
            offset = (4 - d.weekday()) % 7
            out.append(date(y, m, 1 + offset))
    return out


def event_calendar(
    start: int,
    end: int,
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """US events that move gold, as UTC timestamps.

    Columns: timestamp (UTC), event. `extra` takes the same columns so you can
    add CPI, PCE, Powell speeches or a scraped ForexFactory calendar.
    """
    rows = []
    for d in nfp_dates(start, end):
        ts = pd.Timestamp.combine(d, time(8, 30)).tz_localize(NEW_YORK).tz_convert("UTC")
        rows.append((ts, "NFP"))
    for s in FOMC_DATES:
        d = pd.Timestamp(s)
        if start <= d.year <= end:
            ts = pd.Timestamp.combine(d.date(), time(14, 0)).tz_localize(NEW_YORK).tz_convert("UTC")
            rows.append((ts, "FOMC"))
    cal = pd.DataFrame(rows, columns=["timestamp", "event"])
    if extra is not None and len(extra):
        extra = extra[["timestamp", "event"]].copy()
        extra["timestamp"] = pd.to_datetime(extra["timestamp"], utc=True)
        cal = pd.concat([cal, extra], ignore_index=True)
    return cal.sort_values("timestamp").reset_index(drop=True)


def event_risk(
    index: pd.DatetimeIndex,
    calendar: pd.DataFrame,
    before: str = "1h",
    after: str = "1h",
) -> pd.Series:
    """True for bars inside [event - before, event + after]."""
    index = ensure_utc(index)
    flag = np.zeros(len(index), dtype=bool)
    if len(calendar) == 0:
        return pd.Series(flag, index=index, name="event_risk")
    b, a = pd.Timedelta(before).value, pd.Timedelta(after).value
    values = index.asi8
    for ts in _ns(calendar["timestamp"]):
        lo = np.searchsorted(values, ts - b, side="left")
        hi = np.searchsorted(values, ts + a, side="right")
        flag[lo:hi] = True
    return pd.Series(flag, index=index, name="event_risk")


def hours_to_next_event(index: pd.DatetimeIndex, calendar: pd.DataFrame, cap: float = 240.0) -> pd.Series:
    """Hours until the next scheduled event, capped. Known in advance, so no lookahead."""
    index = ensure_utc(index)
    if len(calendar) == 0:
        return pd.Series(cap, index=index, name="hours_to_event")
    ev = np.sort(_ns(calendar["timestamp"]))
    values = index.asi8
    pos = np.searchsorted(ev, values, side="left")
    out = np.full(len(index), cap)
    ok = pos < len(ev)
    delta = (ev[pos[ok]] - values[ok]) / pd.Timedelta(hours=1).value
    out[ok] = np.minimum(delta, cap)
    return pd.Series(out, index=index, name="hours_to_event")


def is_rollover(index: pd.DatetimeIndex) -> pd.Series:
    """True for the bar that opens at 17:00 New York, when brokers roll and spreads blow out."""
    index = ensure_utc(index)
    ny = index.tz_convert(NEW_YORK)
    return pd.Series((ny.hour == 17) & (ny.minute == 0), index=index, name="rollover")
