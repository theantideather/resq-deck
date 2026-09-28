# Running aurum on TradingView

There are three ways to connect them, and you can use all three together.

| Path | What you get | Needs |
|---|---|---|
| **A. Pine Script** | aurum's strategies running natively on your TradingView chart, in the Strategy Tester, with alerts | Any TradingView account (Desktop or web) |
| **B. MCP** | Claude Code drives TradingView Desktop and aurum together: loads and compiles the Pine strategy, reads the tester, pulls bars into the aurum desk | TradingView Desktop, Node 18+, Claude Code |
| **C. Webhooks** | TradingView alerts from the strategy land in the aurum dashboard's alert log | A public HTTPS URL (tunnel) to the dashboard |

## A. Pine Script by hand

1. Open `tradingview/aurum_gold.pine`, or press **Copy Pine Script** in the dashboard.
2. In TradingView, go to Pine Editor → New → Strategy, paste the code, save it, then **Add to chart**.
3. Set the chart to `OANDA:XAUUSD` (any XAUUSD CFD works), 1 hour.
4. In the settings, pick **Strategy**: London breakout, Trend, Macro reversion or Ensemble.
5. Open the Strategy Tester.

How the Pine port differs from the Python engine:

- **Regime.** The HMM becomes a volatility percentile (calm under the 33rd, stressed over the 66th).
- **Costs.** Commission is set to $0.135 per ounce per side (half a $0.20 spread plus $7 per lot round turn), with 5 ticks of slippage. TradingView doesn't charge overnight swap or widen spreads on news. The desk table shows a swap estimate so you can subtract it yourself.
- **Macro data.** DXY is `TVC:DXY` (previous day's close). The real yield is `FRED:DFII10` (two days back, matching FRED's publication lag). If your data plan lacks either symbol, macro reversion falls back to a DXY-only fit.
- **Reference results.** The Python backtester is the reference. If the two disagree by more than costs and swap, treat that as a bug and report it.

## B. With a TradingView MCP in Claude Code

This uses [tradesdontlie/tradingview-mcp](https://github.com/tradesdontlie/tradingview-mcp) (MIT). It talks to TradingView Desktop through the Chrome DevTools Protocol, using undocumented internals that may break when TradingView updates. It's for personal workflow automation only; check that your use fits TradingView's terms.

```bash
# 1. TradingView MCP, cloned next to this repo
git clone https://github.com/tradesdontlie/tradingview-mcp.git
cd tradingview-mcp && npm install && cd -

# 2. aurum with the MCP extra
cd resq-deck/xau-quant
pip install -e ".[mcp,llm,data]"

# 3. Start TradingView Desktop with the debug port
#    macOS:  /Applications/TradingView.app/Contents/MacOS/TradingView --remote-debugging-port=9222
#    Windows: "%LOCALAPPDATA%\TradingView\TradingView.exe" --remote-debugging-port=9222
#    (or use the launch scripts in tradingview-mcp/scripts)

# 4. Open Claude Code in xau-quant/. The .mcp.json here registers both servers.
#    If tradingview-mcp isn't at ../../tradingview-mcp, point to it:
export TRADINGVIEW_MCP_DIR=/path/to/tradingview-mcp
claude
```

Approve both servers when Claude Code asks, then try:

- *"Check TradingView is connected, then load the aurum Pine strategy onto OANDA:XAUUSD 1h, compile it, fix any errors, and screenshot the Strategy Tester."*
  Claude calls `get_pine_strategy` (aurum), then `pine_new`, `pine_set_source`, `pine_smart_compile`, `pine_get_errors`, `chart_set_symbol`, `chart_set_timeframe`, `ui_open_panel` and `capture_screenshot` (TradingView).
- *"Pull 1500 hourly bars from my chart and run the aurum desk on them."*
  Claude calls `data_get_ohlcv`, then `analyze_tradingview_bars`.
- *"Switch the strategy to Macro reversion, then compare TradingView's result with aurum's own backtest on the same dates."*
  Claude reads the tester screenshot, then calls `backtest_strategy`.

### aurum MCP tools

| Tool | What it does |
|---|---|
| `list_strategies` | Strategy names, stops, targets, risk |
| `compare_strategies` | League table (synthetic, yahoo or csv data) |
| `backtest_strategy` | Stats, validation scorecard (deflated Sharpe, bootstrap, cost stress) and last 25 trades |
| `desk_brief` | Analyst desk on the latest bar, with the Claude strategist if `ANTHROPIC_API_KEY` is set |
| `analyze_tradingview_bars` | The desk on bars you pass in (from `data_get_ohlcv`) |
| `get_pine_strategy` | The Pine v6 source |
| `recent_tradingview_alerts` | The webhook alert log |

## C. Alerts into the dashboard

```bash
export AURUM_WEBHOOK_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
python -m aurum.web                    # http://127.0.0.1:8765
cloudflared tunnel --url http://127.0.0.1:8765  # prints https://<random>.trycloudflare.com
```

TradingView only sends webhooks to ports 80 and 443, which is why the dashboard goes behind a tunnel. In TradingView, create an alert on the strategy with the condition **alert() function calls only**. Set the webhook URL to `https://<tunnel>/webhook/tradingview?token=<AURUM_WEBHOOK_TOKEN>`. Each entry, exit and reversal posts JSON like this:

```json
{"source":"aurum-pine","symbol":"OANDA:XAUUSD","mode":"Ensemble","action":"buy","side":1,
 "qty_oz":42,"conviction":0.61,"price":3321.4,"regime":"calm","fv_gap_z":-1.92,"time":1790000000000}
```

Alerts are only logged, never executed. Wiring them to a broker is a separate, deliberate step: paper trade first.
