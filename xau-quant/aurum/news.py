"""Gold relevant headlines from RSS and Atom feeds, for the Claude strategist.

    AURUM_NEWS_FEEDS   comma separated feed URLs; defaults below

Headlines are untrusted text. They only ever reach the strategist, whose
prompt tells it to treat them as information and ignore instructions inside
them, and the strategist can only shrink or veto a position.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd

DEFAULT_FEEDS = [
    # Federal Reserve press releases (FOMC statements, minutes, speeches).
    "https://www.federalreserve.gov/feeds/press_all.xml",
    # Google News search for gold market coverage.
    "https://news.google.com/rss/search?q=gold+price+OR+XAUUSD+OR+bullion&hl=en-US&gl=US&ceid=US:en",
]

KEYWORDS = [
    "gold", "xau", "bullion", "precious metal", "fomc", "federal reserve", "fed ", "powell", "rate cut",
    "rate hike", "inflation", "cpi", "pce", "payroll", "jobs report", "treasury", "yield", "dollar", "dxy",
    "central bank", "pboc", "reserve bank", "tariff", "sanction", "geopolit", "war", "safe haven", "etf",
]
MAX_BYTES = 2_000_000


def _text(el, *names) -> str:
    for n in names:
        found = el.find(n)
        if found is not None and (found.text or "").strip():
            return found.text.strip()
    return ""


def _when(raw: str):
    if not raw:
        return None
    try:
        return pd.Timestamp(parsedate_to_datetime(raw)).tz_convert("UTC")
    except (TypeError, ValueError):
        try:
            return pd.Timestamp(raw).tz_convert("UTC") if pd.Timestamp(raw).tz else pd.Timestamp(raw).tz_localize("UTC")
        except (TypeError, ValueError):
            return None


def parse_feed(xml_bytes: bytes, source: str = "") -> list[dict]:
    """Items from an RSS 2.0 or Atom document."""
    root = ET.fromstring(xml_bytes)
    atom = "{http://www.w3.org/2005/Atom}"
    items = []
    for it in root.iter("item"):
        items.append({"title": _text(it, "title"), "link": _text(it, "link"),
                      "published": _when(_text(it, "pubDate", "{http://purl.org/dc/elements/1.1/}date")),
                      "source": source})
    for it in root.iter(f"{atom}entry"):
        link = it.find(f"{atom}link")
        items.append({"title": _text(it, f"{atom}title"), "link": link.get("href", "") if link is not None else "",
                      "published": _when(_text(it, f"{atom}updated", f"{atom}published")), "source": source})
    for i in items:
        i["title"] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", i["title"]))[:300]
    return [i for i in items if i["title"]]


def relevant(title: str, keywords: list[str] = KEYWORDS) -> bool:
    t = f" {title.lower()} "
    return any(k in t for k in keywords)


def fetch_headlines(feeds: list[str] | None = None, max_age_hours: float = 36, limit: int = 40,
                    now: pd.Timestamp | None = None, fetch=None) -> list[dict]:
    """Recent, relevant, de-duplicated headlines, newest first."""
    if feeds is None:
        env = os.environ.get("AURUM_NEWS_FEEDS", "")
        feeds = [f.strip() for f in env.split(",") if f.strip()] or DEFAULT_FEEDS
    now = now or pd.Timestamp.now(tz="UTC")

    def _get(url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "aurum/0.3 (+research)"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read(MAX_BYTES)

    fetch = fetch or _get
    seen, out = set(), []
    for url in feeds:
        try:
            items = parse_feed(fetch(url), source=url.split("/")[2] if "//" in url else url)
        except Exception as e:
            print(f"[aurum] feed failed {url}: {e}")
            continue
        for it in items:
            key = hashlib.sha1(re.sub(r"\W+", "", it["title"].lower()).encode()).hexdigest()
            if key in seen or not relevant(it["title"]):
                continue
            if it["published"] is not None and (now - it["published"]).total_seconds() > max_age_hours * 3600:
                continue
            seen.add(key)
            out.append(it)
    out.sort(key=lambda i: i["published"] or pd.Timestamp(0, tz="UTC"), reverse=True)
    return out[:limit]


def headline_texts(items: list[dict]) -> list[str]:
    return [f"{i['published']:%Y-%m-%d %H:%M} UTC | {i['source']} | {i['title']}" if i["published"] is not None
            else f"{i['source']} | {i['title']}" for i in items]
