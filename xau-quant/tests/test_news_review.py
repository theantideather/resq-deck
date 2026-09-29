import json
from types import SimpleNamespace

import pandas as pd
import pytest

from aurum.news import fetch_headlines, headline_texts, parse_feed, relevant
from aurum.review import load_lessons, observations, review, trade_stats

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>
<item><title>Gold hits record as Fed signals rate cut</title><link>https://a/1</link>
<pubDate>Mon, 28 Sep 2026 10:00:00 GMT</pubDate></item>
<item><title>Gold hits record as  Fed signals rate cut</title><link>https://b/1</link>
<pubDate>Mon, 28 Sep 2026 10:05:00 GMT</pubDate></item>
<item><title>Local team wins cup</title><link>https://a/2</link><pubDate>Mon, 28 Sep 2026 09:00:00 GMT</pubDate></item>
<item><title>Treasury yields climb after CPI</title><link>https://a/3</link><pubDate>Fri, 01 Aug 2025 09:00:00 GMT</pubDate></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom"><title>f</title>
<entry><title>PBoC adds gold to reserves for 23rd month</title><link href="https://c/1"/>
<updated>2026-09-28T08:00:00Z</updated></entry></feed>"""


def test_parse_rss_and_atom():
    r = parse_feed(RSS, "a")
    assert len(r) == 4 and r[0]["published"] == pd.Timestamp("2026-09-28 10:00", tz="UTC")
    a = parse_feed(ATOM, "c")
    assert a[0]["title"].startswith("PBoC") and a[0]["link"] == "https://c/1"


def test_fetch_filters_dedupes_and_ages():
    feeds = {"https://a/rss": RSS, "https://c/atom": ATOM}
    out = fetch_headlines(list(feeds), now=pd.Timestamp("2026-09-28 12:00", tz="UTC"), fetch=lambda u: feeds[u])
    titles = [o["title"] for o in out]
    assert titles == ["Gold hits record as Fed signals rate cut", "PBoC adds gold to reserves for 23rd month"]
    assert "UTC" in headline_texts(out)[0]
    assert relevant("Dollar slides") and not relevant("Local team wins cup")


def test_fetch_survives_a_broken_feed():
    out = fetch_headlines(["https://bad/x"], fetch=lambda u: (_ for _ in ()).throw(OSError("down")))
    assert out == []


def trades(n=12):
    rows = []
    for i in range(n):
        win = i % 3 == 0
        rows.append({"side": 1 if i % 2 else -1, "lots": 0.5, "entry_price": 2000, "exit_price": 2010 if win else 1995,
                     "reason": "target" if win else "stop", "net_pnl": 400.0 if win else -260.0, "gross_pnl": 500.0 if win else -250.0,
                     "costs": 10.0, "swap": -5.0, "opened_utc": f"2026-09-{1 + i:02d}T08:00:00+00:00"})
    return rows


def test_trade_stats_and_observations():
    s = trade_stats(trades())
    assert s["trades"] == 12 and s["by_reason"]["stop"]["trades"] == 8
    obs = observations(s)
    assert any("stops" in o for o in obs)
    assert observations(trade_stats(trades(3))) == ["fewer than 5 closed trades; too early to read anything"]


def test_review_with_fake_claude_saves_lessons(tmp_path):
    pytest.importorskip("anthropic")
    payload = {"summary": "stops too tight", "lessons": [
        {"lesson": "Widen stops to 2 ATR", "evidence": "8 of 12 exits were stops", "suggested_change": "stop_atr 1.5 -> 2.0"}]}
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(payload))])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    path = tmp_path / "lessons.json"
    out = review(trades(), client=client, lessons_path=path)
    assert out["claude"]["summary"] == "stops too tight"
    assert calls[0]["output_config"]["format"]["type"] == "json_schema"
    saved = load_lessons(path=path)
    assert saved[0]["lesson"] == "Widen stops to 2 ATR" and "saved_utc" in saved[0]


def test_strategist_receives_lessons():
    pytest.importorskip("anthropic")
    from aurum.agents import claude_strategist

    seen = {}

    def create(**kw):
        seen.update(kw)
        body = {"bias": "neutral", "confidence": 0.3, "thesis": "t", "key_drivers": [], "risks": [], "veto_reason": ""}
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(body))])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    claude_strategist({"price": 1}, client=client, lessons=[{"lesson": "Avoid Asia longs", "evidence": "win 20%"}])
    assert "Avoid Asia longs" in seen["messages"][0]["content"] and "advisory" in seen["messages"][0]["content"]
