"""Post-trade review: what the closed trades say, and lessons worth remembering.

Works on the paper broker's trade log (or any list of trade dicts with side,
lots, entry_price, exit_price, reason, net_pnl, opened_utc). Rule based
statistics always run. With ANTHROPIC_API_KEY set, Claude reads them and
writes a few concrete lessons, which are stored in ~/.aurum/lessons.json
(last 20 kept) and shown to the strategist as advisory context. Lessons never
change parameters by themselves: a person decides whether to act on them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import calendar as cal

AURUM_HOME = Path(os.environ.get("AURUM_HOME", Path.home() / ".aurum"))
LESSONS_PATH = AURUM_HOME / "lessons.json"


def trade_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0}
    df = pd.DataFrame(trades)
    df["opened"] = pd.to_datetime(df.get("opened_utc"), utc=True, errors="coerce")
    pnl = df["net_pnl"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    sess = cal.session_flags(pd.DatetimeIndex(df["opened"].fillna(pd.Timestamp(0, tz="UTC"))))
    df["session"] = np.select([sess["overlap"], sess["new_york"], sess["london"], sess["asia"]],
                              ["overlap", "new_york", "london", "asia"], "off_hours")
    by = lambda col: {str(k): {"trades": int(len(g)), "net_pnl": round(float(g["net_pnl"].sum()), 2),
                               "win_rate": round(float((g["net_pnl"] > 0).mean()), 3)}
                      for k, g in df.groupby(col)}
    gross = float(df.get("gross_pnl", pnl).astype(float).sum())
    return {
        "trades": int(len(df)),
        "net_pnl": round(float(pnl.sum()), 2),
        "win_rate": round(float(len(wins) / len(df)), 3),
        "profit_factor": round(float(wins.sum() / -losses.sum()), 3) if losses.sum() < 0 else None,
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "costs": round(float(df.get("costs", pd.Series(0.0)).astype(float).sum()), 2),
        "swap": round(float(df.get("swap", pd.Series(0.0)).astype(float).sum()), 2),
        "gross_pnl": round(gross, 2),
        "by_reason": by("reason"),
        "by_side": by("side"),
        "by_session": by("session"),
        "worst": df.nsmallest(3, "net_pnl")[["opened_utc", "side", "reason", "net_pnl"]].to_dict("records"),
        "best": df.nlargest(3, "net_pnl")[["opened_utc", "side", "reason", "net_pnl"]].to_dict("records"),
    }


def observations(s: dict) -> list[str]:
    """Plain rule based findings from trade_stats."""
    if s.get("trades", 0) < 5:
        return ["fewer than 5 closed trades; too early to read anything"]
    out = []
    reasons = s["by_reason"]
    stops = reasons.get("stop", {}).get("trades", 0)
    if stops / s["trades"] > 0.6:
        out.append(f"{stops} of {s['trades']} exits were stops; stops may be too tight for the holding period")
    friction = -(s["costs"]) + min(s["swap"], 0)
    if s["gross_pnl"] > 0 and -friction > 0.5 * s["gross_pnl"]:
        out.append(f"costs and swap ({-friction:,.0f}) ate over half of gross profit ({s['gross_pnl']:,.0f})")
    if s["swap"] < 0 and abs(s["swap"]) > 0.2 * abs(s["net_pnl"] or 1):
        out.append(f"swap paid {s['swap']:,.0f}; holding longs overnight is expensive")
    for name, g in s["by_session"].items():
        if g["trades"] >= 5 and g["net_pnl"] < 0 and g["win_rate"] < 0.35:
            out.append(f"{name} session: {g['trades']} trades, win rate {g['win_rate']:.0%}, net {g['net_pnl']:,.0f}")
    for side, g in s["by_side"].items():
        if g["trades"] >= 5 and g["net_pnl"] < 0:
            label = "longs" if str(side) in ("1", "1.0") else "shorts"
            out.append(f"{label} lost {g['net_pnl']:,.0f} over {g['trades']} trades")
    return out or ["nothing stands out; keep collecting trades"]


LESSONS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "lessons": {"type": "array", "items": {
            "type": "object",
            "properties": {"lesson": {"type": "string"}, "evidence": {"type": "string"},
                           "suggested_change": {"type": "string"}},
            "required": ["lesson", "evidence", "suggested_change"], "additionalProperties": False}},
    },
    "required": ["summary", "lessons"],
    "additionalProperties": False,
}

REVIEW_SYSTEM = """You review the closed trades of a systematic XAUUSD strategy.
You get trade statistics and rule based observations. Write at most five
lessons that a careful quant would take from them. Each lesson must cite the
numbers it rests on and suggest one concrete, testable change (a parameter,
a filter, a session to avoid) that should be backtested before use. If the
sample is too small to support a lesson, say so instead of inventing one."""


def claude_lessons(stats: dict, obs: list[str], client=None) -> dict | None:
    try:
        import anthropic
    except ImportError:
        return None
    if client is None:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return None
        client = anthropic.Anthropic()
    try:
        resp = client.beta.messages.create(
            model=os.environ.get("AURUM_MODEL", "claude-opus-5"),
            max_tokens=16000,
            system=REVIEW_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": LESSONS_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": json.dumps({"stats": stats, "observations": obs}, default=str)}],
        )
    except anthropic.APIError as e:
        print(f"[aurum] review model unavailable: {e}")
        return None
    if resp.stop_reason == "refusal":
        return None
    text = next((b.text for b in resp.content if b.type == "text"), None)
    return json.loads(text) if text else None


def load_lessons(limit: int = 5, path: Path | None = None) -> list[dict]:
    p = Path(path or LESSONS_PATH)
    if not p.exists():
        return []
    return json.loads(p.read_text())[-limit:]


def save_lessons(new: list[dict], path: Path | None = None) -> None:
    p = Path(path or LESSONS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(p.read_text()) if p.exists() else []
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    p.write_text(json.dumps((old + [{**l, "saved_utc": stamp} for l in new])[-20:], indent=1))


def review(trades: list[dict], use_llm: bool = True, client=None, lessons_path: Path | None = None) -> dict:
    s = trade_stats(trades)
    obs = observations(s)
    out = {"stats": s, "observations": obs, "claude": None}
    if use_llm and s.get("trades", 0) >= 5:
        res = claude_lessons(s, obs, client=client)
        if res:
            out["claude"] = res
            save_lessons(res.get("lessons", []), lessons_path)
    return out
