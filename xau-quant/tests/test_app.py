import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest

from aurum import calendar as cal
from aurum import service

MKW = {"source": "synthetic", "start": "2022-01-01", "end": "2022-12-31", "seed": 5}


def tv_rows(n=900):
    idx = pd.date_range("2026-06-01", periods=n, freq="1h", tz="UTC")
    p = 3300 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.002, n)))
    return [{"time": int(t.timestamp()) * 1000, "open": x, "high": x + 2, "low": x - 2, "close": x + 0.3}
            for t, x in zip(idx, p)]


def test_backtest_payload_is_json_safe():
    r = service.backtest("macro_reversion", n_trials=3, **MKW)
    text = json.dumps(r, allow_nan=False)  # raises on NaN or inf
    assert r["validation"]["n_trials"] == 3
    assert r["candles"] and r["equity"] and r["drawdown"]
    assert all(c["low"] <= c["high"] for c in r["candles"])
    assert len(text) < 5_000_000


def test_market_from_tradingview_rows_ms_timestamps():
    md, feats = service.market_from_bars(tv_rows())
    assert len(md.bars) == 900
    assert md.bars.index.tz is not None and md.bars.index[0].year == 2026
    out = service.analyze_bars(tv_rows())
    assert out["snapshot"]["price"] > 0
    assert -1 <= out["decision"]["final_signal"] <= 1


def test_analyze_rejects_short_history():
    with pytest.raises(ValueError):
        service.analyze_bars(tv_rows(100))


def test_pine_script_matches_python_calendar():
    src = service.pine_source()
    assert src.count("//@version=6") == 1
    assert "strategy(" in src and "request.security" in src
    assert "lookahead = barmerge.lookahead_on" in src and "close[1]" in src  # non repainting pattern
    pine_fomc = set(re.findall(r"\b(202[5-9]\d{4})\b", src.split("array.from(")[1].split(")")[0]))
    py_fomc = {d.replace("-", "") for d in cal.FOMC_DATES if d >= "2025"}
    assert pine_fomc == py_fomc
    assert src.count("(") == src.count(")") and src.count("[") == src.count("]")
    assert "\t" not in src  # Pine indentation must be spaces here


@pytest.fixture()
def server(tmp_path, monkeypatch):
    from aurum.web.server import Handler

    monkeypatch.setattr(service, "ALERT_LOG", tmp_path / "alerts.jsonl")
    monkeypatch.setenv("AURUM_WEBHOOK_TOKEN", "s3cret")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.status, r.read()


def _post(url, body: bytes):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_web_pages_and_api(server):
    code, html = _get(server + "/")
    assert code == 200 and b"aurum" in html
    code, js = _get(server + "/static/lightweight-charts.standalone.production.js")
    assert code == 200 and b"Lightweight Charts" in js
    code, body = _get(server + "/api/strategies")
    assert {s["name"] for s in json.loads(body)} >= {"trend", "ensemble"}
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(server + "/static/../server.py")
    assert e.value.code == 404


def test_webhook_requires_token_and_logs(server):
    alert = json.dumps({"action": "buy", "qty_oz": 10, "price": 2400.0}).encode()
    code, _ = _post(server + "/webhook/tradingview", alert)
    assert code == 401
    code, _ = _post(server + "/webhook/tradingview?token=wrong", alert)
    assert code == 401
    code, body = _post(server + "/webhook/tradingview?token=s3cret", alert)
    assert code == 200 and body["alert"]["action"] == "buy"
    code, body = _post(server + "/webhook/tradingview?token=s3cret", b"plain text alert")
    assert code == 200 and body["alert"]["message"] == "plain text alert"
    code, raw = _get(server + "/api/alerts")
    alerts = json.loads(raw)
    assert [a.get("action") for a in alerts] == ["buy", None]


def test_mcp_server_registers_tools():
    pytest.importorskip("mcp")
    import anyio

    from aurum.mcp_server import mcp

    names = {t.name for t in anyio.run(mcp.list_tools)}
    assert {"backtest_strategy", "analyze_tradingview_bars", "get_pine_strategy", "desk_brief"} <= names


def _post_json(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_paper_api_needs_header_and_runs_cycles(server, tmp_path, monkeypatch):
    monkeypatch.setenv("AURUM_HOME", str(tmp_path / "home"))
    service.paper_reset()
    code, _ = _post_json(server + "/api/paper/cycle", {"feed": "replay"})
    assert code == 403  # no X-Aurum header: a cross site post is refused
    for _ in range(3):
        code, rec = _post_json(server + "/api/paper/cycle", {"feed": "replay", "strategy": "london_breakout"},
                               {"X-Aurum": "1"})
        assert code == 200 and rec["action"] in ("hold", "trade", "skip")
    code, raw = _get(server + "/api/paper")
    st = json.loads(raw)
    assert st["exists"] and len(st["journal"]) == 3
    code, body = _post_json(server + "/api/review", {"llm": False}, {"X-Aurum": "1"})
    assert code == 200 and "observations" in body


def test_webhook_can_trade_paper_when_enabled(server, tmp_path, monkeypatch):
    monkeypatch.setenv("AURUM_HOME", str(tmp_path / "home2"))
    monkeypatch.setenv("AURUM_WEBHOOK_EXECUTE", "paper")
    url = server + "/webhook/tradingview?token=s3cret"
    code, body = _post(url, json.dumps({"action": "buy", "qty_oz": 50, "price": 2400.0, "mode": "Ensemble"}).encode())
    assert code == 200 and body["execution"]["orders"][0]["ok"]
    code, body = _post(url, json.dumps({"action": "sell", "qty_oz": 30, "price": 2410.0}).encode())
    kinds = [(o["action"], o["side"]) for o in body["execution"]["orders"]]
    assert kinds == [("close", 1), ("open", -1)]
    st = service.paper_status()
    assert st["position"]["side"] == -1 and st["position"]["lots"] == 0.3
    assert st["trades"][0]["exit_price"] == 2410.0


def test_webhook_does_not_trade_by_default(server, tmp_path, monkeypatch):
    monkeypatch.setenv("AURUM_HOME", str(tmp_path / "home3"))
    monkeypatch.delenv("AURUM_WEBHOOK_EXECUTE", raising=False)
    code, body = _post(server + "/webhook/tradingview?token=s3cret",
                       json.dumps({"action": "buy", "qty_oz": 50, "price": 2400.0}).encode())
    assert code == 200 and body["execution"] is None
