"""Build an hourly XAUUSD dataset (mid OHLC + average spread) from bid/ask minute CSVs.

Source used during development: the public GitHub repo Dypoi/XAUUSD_Dataset
(yearly files XAUUSD_M1_YYYY0901_YYYY0901.csv, Sept 2016 to Sept 2026, bid and
ask OHLC per minute). Check that repo's terms before redistributing the data;
this script only converts it locally.

    git clone --filter=blob:none --no-checkout https://github.com/Dypoi/XAUUSD_Dataset
    python scripts/build_xau_dataset.py XAUUSD_Dataset data/XAUUSD_H1.csv

Any folder of minute CSVs with columns timestamp, open_bid, high_bid, low_bid,
close_bid, open_ask, high_ask, low_ask, close_ask works. Timestamps are taken
as UTC. Raw files are checked out one at a time and deleted after conversion
when the folder is a git checkout, so disk use stays small.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

COLS = ["timestamp", "open_bid", "high_bid", "low_bid", "close_bid", "open_ask", "high_ask", "low_ask", "close_ask"]


def minute_to_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=COLS)
    idx = pd.to_datetime(df["timestamp"], utc=True)
    mid = pd.DataFrame({
        "open": (df["open_bid"] + df["open_ask"]) / 2,
        "high": (df["high_bid"] + df["high_ask"]) / 2,
        "low": (df["low_bid"] + df["low_ask"]) / 2,
        "close": (df["close_bid"] + df["close_ask"]) / 2,
        "spread": df["close_ask"] - df["close_bid"],
        "volume": 1.0,  # minute count as an activity proxy
    })
    mid.index = idx
    h = mid.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                "spread": "mean", "volume": "sum"})
    return h.dropna(subset=["open"])


def main(src: str, out: str) -> None:
    src_dir = Path(src)
    is_git = (src_dir / ".git").exists()
    if is_git:
        names = subprocess.run(["git", "-C", str(src_dir), "ls-tree", "--name-only", "HEAD"],
                               capture_output=True, text=True, check=True).stdout.split()
        files = sorted(n for n in names if n.endswith(".csv") and "_M1_" in n)
    else:
        files = sorted(p.name for p in src_dir.glob("*.csv"))
    parts = []
    for name in files:
        path = src_dir / name
        if is_git and not path.exists():
            subprocess.run(["git", "-C", str(src_dir), "checkout", "-q", "HEAD", "--", name], check=True)
        h = minute_to_hourly(path)
        print(f"{name}: {len(h):,} hourly bars, {h.index[0]:%Y-%m-%d} to {h.index[-1]:%Y-%m-%d}, "
              f"median spread {h['spread'].median():.3f}")
        parts.append(h)
        if is_git:
            path.unlink()
    data = pd.concat(parts).sort_index()
    data = data[~data.index.duplicated(keep="last")]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out, index_label="timestamp", float_format="%.3f")
    print(f"wrote {out}: {len(data):,} bars")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
