"""aurum as an MCP server, to run next to a TradingView MCP in Claude Code.

    python -m aurum.mcp_server            # stdio, for Claude Code / Claude Desktop

With both servers connected, Claude can pull XAUUSD bars from your
TradingView chart, run the aurum desk on them, push the aurum Pine strategy
into TradingView's editor, compile it, and read the Strategy Tester. The
Python backtester stays the reference: TradingView's tester does not charge
overnight swap or widen spreads around news.
"""

from __future__ import annotations

from typing import Any

from . import service

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # 1.x
    from mcp.server.fastmcp import FastMCP as _Server

INSTRUCTIONS = """aurum is a quant research toolkit for gold (XAUUSD).

Typical workflows when a TradingView MCP (tradesdontlie/tradingview-mcp) is also connected:

1. Run the aurum strategy on TradingView:
   get_pine_strategy -> tradingview pine_new (strategy) -> pine_set_source ->
   pine_smart_compile -> pine_get_errors (fix and repeat) -> chart_set_symbol OANDA:XAUUSD ->
   chart_set_timeframe 60 -> ui_open_panel strategy tester -> capture_screenshot.
   Then compare with backtest_strategy here, which also charges swap and news spreads.

2. Desk read on the live chart:
   tradingview data_get_ohlcv (XAUUSD, 60 minute, at least 1000 bars) ->
   analyze_tradingview_bars with those rows -> explain the decision.

3. Research: compare_strategies, then backtest_strategy with n_trials set to the
   number of variants tried, and report the deflated Sharpe and the verdict honestly.

Never place real orders from these results without the user's explicit instruction."""

mcp = _Server("aurum", instructions=INSTRUCTIONS)


def _mkw(source: str, start: str, end: str, seed: int, csv_path: str | None) -> dict:
    return {"source": source, "start": start, "end": end, "seed": seed, "csv_path": csv_path}


@mcp.tool()
def list_strategies() -> list[dict]:
    """The aurum gold strategies, with their stop, target and risk settings."""
    return service.strategies()


@mcp.tool()
def compare_strategies(source: str = "synthetic", start: str = "2019-01-01", end: str = "2024-12-31",
                       seed: int = 7, csv_path: str | None = None) -> dict:
    """Backtest every strategy on XAUUSD and return a league table.

    source: synthetic (offline), yahoo (GC=F hourly plus DXY, FRED real yields, CFTC COT) or csv (MT5 export at csv_path).
    """
    return service.compare(**_mkw(source, start, end, seed, csv_path))


@mcp.tool()
def backtest_strategy(strategy: str = "ensemble", n_trials: int = 1, source: str = "synthetic",
                      start: str = "2019-01-01", end: str = "2024-12-31", seed: int = 7,
                      csv_path: str | None = None, risk_per_trade: float | None = None) -> dict:
    """Backtest one strategy with realistic XAUUSD costs and validate it.

    Returns stats, the validation scorecard (probabilistic and deflated Sharpe, bootstrap
    drawdowns, cost stress, verdict) and the last 25 trades. n_trials is how many
    variants were tried before this one; it drives the deflated Sharpe.
    """
    r = service.backtest(strategy, n_trials=n_trials, risk_per_trade=risk_per_trade,
                         **_mkw(source, start, end, seed, csv_path))
    return {k: r[k] for k in ("strategy", "description", "source", "stats", "validation")} | {"last_trades": r["trades"][-25:]}


@mcp.tool()
def desk_brief(use_llm: bool = True, headlines: list[str] | None = None, source: str = "synthetic",
               start: str = "2019-01-01", end: str = "2024-12-31", seed: int = 7,
               csv_path: str | None = None) -> dict:
    """The analyst desk on the latest bar: snapshot, analyst views, optional Claude strategist, final size."""
    return service.brief(use_llm=use_llm, headlines=headlines, **_mkw(source, start, end, seed, csv_path))


@mcp.tool()
def analyze_tradingview_bars(bars: list[dict[str, Any]], headlines: list[str] | None = None,
                             use_llm: bool = False) -> dict:
    """Run the desk on OHLCV bars from TradingView (e.g. data_get_ohlcv output).

    Each bar: time (unix s or ms, or ISO), open, high, low, close, optional volume. Needs 300+
    bars; 1000+ hourly bars is better. Macro features are unavailable from bars alone.
    """
    return service.analyze_bars(bars, use_llm=use_llm, headlines=headlines)


@mcp.tool()
def get_pine_strategy() -> str:
    """Pine Script v6 source of the aurum gold strategy, ready for TradingView's pine_set_source."""
    return service.pine_source()


@mcp.tool()
def recent_tradingview_alerts(limit: int = 20) -> list[dict]:
    """Latest TradingView webhook alerts received by the aurum web server."""
    return service.recent_alerts(limit)


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
