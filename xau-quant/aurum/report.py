"""Self contained HTML tear sheet: equity, drawdown, stats, validation, trades."""

from __future__ import annotations

import html
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .validation import Scorecard


def _usd(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def _path(values: np.ndarray, w: int, h: int, lo: float, hi: float) -> str:
    if len(values) < 2 or hi == lo:
        return ""
    x = np.linspace(0, w, len(values))
    y = h - (values - lo) / (hi - lo) * h
    return "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))


def _chart(series: pd.Series, w: int = 900, h: int = 220, fill: bool = False) -> str:
    s = series.resample("1D").last().dropna()
    v = s.to_numpy(float)
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad
    d = _path(v, w, h, lo, hi)
    area = f'<path d="{d} L{w},{h} L0,{h} Z" class="area"/>' if fill and d else ""
    labels = (
        f'<text x="4" y="12">{hi:,.0f}</text><text x="4" y="{h - 4}">{lo:,.0f}</text>'
        if not fill else f'<text x="4" y="{h - 4}">{lo:.0%}</text>'
    )
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img">{area}'
            f'<path d="{d}" class="line"/>{labels}</svg>')


def render(name: str, result: BacktestResult, card: Scorecard | None = None,
           source: str = "", notes: str = "") -> str:
    s = result.stats
    eq = result.equity
    dd = eq / eq.cummax() - 1
    rows = [
        ("Net return", f"{s['total_return']:.1%}"), ("CAGR", f"{s['cagr']:.1%}"),
        ("Sharpe", f"{s['sharpe']:.2f}"), ("Sortino", f"{s['sortino']:.2f}"),
        ("Max drawdown", f"{s['max_drawdown']:.1%}"), ("Calmar", f"{s['calmar']:.2f}"),
        ("Trades", f"{s['trades']}"), ("Win rate", f"{s['win_rate']:.1%}"),
        ("Profit factor", f"{s['profit_factor']:.2f}"), ("Avg R", f"{s['avg_r']:.2f}"),
        ("Time in market", f"{s['exposure']:.1%}"),
        ("Spread, slippage, commission", _usd(-s["costs"])), ("Swap", _usd(s["swap"])),
    ]
    stats_html = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    card_html = f"<pre>{html.escape(card.text())}</pre>" if card else ""
    t = result.trades.tail(40).copy()
    if len(t):
        t["entry_time"] = t["entry_time"].astype(str).str[:16]
        t["exit_time"] = t["exit_time"].astype(str).str[:16]
        trades_html = t[["entry_time", "exit_time", "side", "lots", "entry_price", "exit_price",
                         "reason", "net_pnl", "r_multiple"]].to_html(index=False, float_format=lambda x: f"{x:,.2f}", border=0)
    else:
        trades_html = "<p>No trades.</p>"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(name)} tear sheet</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1b16;--muted:#6b665c;--line:#a8791c;--area:#c0392b22;--rule:#e4dfd3}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15140f;--fg:#ece7da;--muted:#a29c8e;--line:#e0b252;--area:#e74c3c33;--rule:#2d2a22}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:0 auto;padding:24px 16px}}
h1{{font-size:22px;margin:0}} h2{{font-size:15px;margin:28px 0 8px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}}
.meta{{color:var(--muted);font-size:13px}} svg{{width:100%;height:auto;border-bottom:1px solid var(--rule)}}
svg .line{{fill:none;stroke:var(--line);stroke-width:1.5;vector-effect:non-scaling-stroke}}
svg .area{{fill:var(--area)}} svg text{{fill:var(--muted);font-size:11px}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px;width:100%}}
th,td{{text-align:left;padding:4px 8px;border-bottom:1px solid var(--rule)}} .stats{{max-width:420px}}
pre{{background:transparent;border:1px solid var(--rule);padding:12px;overflow-x:auto;font-size:13px}}
.wrap{{overflow-x:auto}}
</style></head><body>
<h1>{html.escape(name)}</h1>
<div class="meta">XAUUSD · data: {html.escape(source)} · {eq.index[0]:%Y-%m-%d} to {eq.index[-1]:%Y-%m-%d} · generated {stamp}</div>
{f'<p>{html.escape(notes)}</p>' if notes else ''}
<h2>Equity</h2>{_chart(eq)}
<h2>Drawdown</h2>{_chart(dd, h=120, fill=True)}
<h2>Statistics</h2><table class="stats">{stats_html}</table>
{('<h2>Validation</h2>' + card_html) if card_html else ''}
<h2>Last 40 trades</h2><div class="wrap">{trades_html}</div>
<p class="meta">Research output, not investment advice. Past and simulated results do not predict future returns.</p>
</body></html>"""
