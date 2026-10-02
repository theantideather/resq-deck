"""aurum web dashboard: python -m aurum.web [--port 8765] [--host 127.0.0.1]

Standard library only. Serves the single page UI and a small JSON API:

    GET  /api/strategies
    GET  /api/compare?source=&start=&end=&seed=
    GET  /api/backtest?strategy=&trials=&source=&start=&end=&seed=
    GET  /api/brief?llm=0|1&source=...
    GET  /api/pine                      Pine Script source (text/plain)
    GET  /api/alerts?limit=50
    GET  /api/paper                     paper account, position, trades, journal
    GET  /api/news                      gold headlines
    GET  /api/swing?strategy=           swing desk chart: candles, trailing line, EMAs, flips,
                                        trades with stop/target, timeframe table, PO3, plan
    GET  /api/optimize?strategy=&train=&test=   walk forward optimisation report
    POST /api/paper/cycle               {"strategy", "feed", "desk", "llm"}: one runner cycle
    POST /api/paper/reset
    POST /api/review                    {"llm": bool}: post-trade review of the paper account
    POST /api/analyze                   {"bars": [...], "headlines": [...]}

POST /api/* calls need the header X-Aurum: 1 (the dashboard sends it).
    POST /webhook/tradingview?token=    TradingView alert webhook

If AURUM_WEBHOOK_TOKEN is set, webhooks must carry it as ?token= or a
"token" field in the JSON body. Set it whenever the server is reachable from
the internet.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import service

INDEX = Path(__file__).with_name("index.html")
STATIC = Path(__file__).with_name("static")
STATIC_FILES = {"lightweight-charts.standalone.production.js": "application/javascript"}
MAX_BODY = 5_000_000


def _market_kwargs(q: dict) -> dict:
    src = q.get("source", "synthetic")
    if src not in ("synthetic", "yahoo", "csv"):
        raise ValueError("source must be synthetic, yahoo or csv")
    return {
        "source": src,
        "start": q.get("start", "2019-01-01"),
        "end": q.get("end", "2024-12-31"),
        "seed": int(q.get("seed", 7)),
        "csv_path": q.get("csv") or None,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "aurum/0.2"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/" in (args[0] if args else "") or "webhook" in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _send(self, code: int, body, ctype: str = "application/json") -> None:
        data = body if isinstance(body, bytes) else (
            body.encode() if isinstance(body, str) else json.dumps(body, default=str).encode())
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _query(self) -> dict:
        return {k: v[-1] for k, v in parse_qs(urlparse(self.path).query).items()}

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            # TradingView sends the alert message as plain text if it is not valid JSON.
            return {"message": raw.decode(errors="replace")}

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        q = self._query()
        try:
            if path in ("/", "/index.html"):
                self._send(200, INDEX.read_bytes(), "text/html")
            elif path.startswith("/static/") and path[8:] in STATIC_FILES:
                # Fixed allowlist, so no path traversal.
                self._send(200, (STATIC / path[8:]).read_bytes(), STATIC_FILES[path[8:]])
            elif path == "/api/strategies":
                self._send(200, service.strategies())
            elif path == "/api/compare":
                self._send(200, service.compare(**_market_kwargs(q)))
            elif path == "/api/backtest":
                self._send(200, service.backtest(q.get("strategy", "ensemble"), n_trials=int(q.get("trials", 1)),
                                                 **_market_kwargs(q)))
            elif path == "/api/brief":
                self._send(200, service.brief(use_llm=q.get("llm") == "1", **_market_kwargs(q)))
            elif path == "/api/pine":
                self._send(200, service.pine_source(), "text/plain")
            elif path == "/api/alerts":
                self._send(200, service.recent_alerts(int(q.get("limit", 50))))
            elif path == "/api/paper":
                self._send(200, service.paper_status())
            elif path == "/api/swing":
                self._send(200, service.swing_chart(q.get("strategy", "swing_halftrend_structure"), **_market_kwargs(q)))
            elif path == "/api/news":
                self._send(200, service.news(int(q.get("limit", 30))))
            elif path == "/api/optimize":
                self._send(200, service.optimize(q.get("strategy", "london_breakout"),
                                                 train_days=int(q.get("train", 365)), test_days=int(q.get("test", 91)),
                                                 **_market_kwargs(q)))
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            self._send(400, {"error": str(e)})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/webhook/tradingview":
                expected = os.environ.get("AURUM_WEBHOOK_TOKEN", "")
                given = self._query().get("token") or str(body.pop("token", "") if isinstance(body, dict) else "")
                if expected and not hmac.compare_digest(given, expected):
                    self._send(401, {"error": "bad token"})
                    return
                alert = service.record_alert(body)
                self._send(200, {"ok": True, "alert": alert, "execution": service.execute_alert(body)})
            elif path.startswith("/api/") and self.headers.get("X-Aurum") != "1":
                # Custom header: a cross site form post cannot set it, so other
                # pages open in the browser cannot trade the paper account.
                self._send(403, {"error": "missing X-Aurum header"})
            elif path == "/api/paper/cycle":
                self._send(200, service.paper_cycle(body.get("strategy", "ensemble"), body.get("feed", "replay"),
                                                    body.get("csv") or None, bool(body.get("desk", True)),
                                                    bool(body.get("llm", False))))
            elif path == "/api/paper/reset":
                self._send(200, service.paper_reset())
            elif path == "/api/review":
                self._send(200, service.review_paper(use_llm=bool(body.get("llm", False))))
            elif path == "/api/analyze":
                self._send(200, service.analyze_bars(body.get("bars", []), use_llm=bool(body.get("llm")),
                                                     headlines=body.get("headlines")))
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            self._send(400, {"error": str(e)})


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aurum.web")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args(argv)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"aurum dashboard on http://{a.host}:{a.port}")
    if a.host not in ("127.0.0.1", "localhost") and not os.environ.get("AURUM_WEBHOOK_TOKEN"):
        print("warning: listening beyond localhost without AURUM_WEBHOOK_TOKEN; anyone can post alerts")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
