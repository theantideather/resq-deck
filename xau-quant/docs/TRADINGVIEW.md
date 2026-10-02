# Running aurum on TradingView

There are three ways to connect them, and you can use all three together.

| Path | What you get | Needs |
|---|---|---|
| **A. Pine Script** | aurum's strategies running natively on your TradingView chart, in the Strategy Tester, with alerts | Any TradingView account (Desktop or web) |
| **B. MCP** | Claude Code drives TradingView Desktop and aurum together: loads and compiles the Pine strategy, reads the tester, pulls bars into the aurum desk | TradingView Desktop, Node 18+, Claude Code |
| **C. Webhooks** | TradingView alerts from the strategy land in the aurum dashboard's alert log | A public HTTPS URL (tunnel) to the dashboard |

## The swing desk indicator (his chart, for gold)

`tradingview/aurum_swing_indicator.pine` rebuilds the friend's BTC chart for XAUUSD. It shows:
- the HalfTrend ribbon with ▲/▼ labels
- the black EMA 200, plus blue and purple EMAs
- his Time Frame / Signal table, with Aligned and Long/Short WR rows
- the D candle PO3 box
- an entry / stop / target box with labels for every signal, drawn like the position tool

To add it:
1. TradingView → open `OANDA:XAUUSD` (or `TVC:GOLD`), daily timeframe.
2. Pine Editor (bottom panel) → **Open → New blank indicator**. Select all, delete, and paste the whole file.
3. **Save**, then **Add to chart**. Set the stop mode, engine and risk in the indicator's settings (gear icon).

The ribbon is HalfTrend's ATR channel, and the ▲/▼ squares sit on its outer edge. The thin stepped line is the open trade's stop. Only the latest trade keeps its Stop / Open PnL / Target labels. Trades and win rates count from the date set in **Trades and win rates from** (default 1 Apr 2025). On BINANCE:BTCUSDT.P 1D, that default gives his 4 longs and 5 shorts; the win rates differ until his exit rule is known.

To test it in the Strategy Tester, paste `tradingview/aurum_swing.pine` into a new **strategy** instead. See `docs/SWING.md` for how the chart was identified and the gold backtest.

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

### One command

You need TradingView Desktop, Node 18+, Python 3.10+, git and Claude Code (`npm install -g @anthropic-ai/claude-code`).

```bash
cd resq-deck/xau-quant
./scripts/setup_tradingview.sh                 # macOS / Linux
# Windows: powershell -ExecutionPolicy Bypass -File scripts\setup_tradingview.ps1
```

The script:

1. Clones tradingview-mcp into `~/tradingview-mcp` (or `$TRADINGVIEW_MCP_DIR`) and installs it.
2. Creates `.venv` with aurum and its MCP, LLM and data extras.
3. Compiles `aurum_gold.pine` on TradingView's server.
4. Registers both MCP servers with Claude Code for this folder.
5. Starts the dashboard.
6. Restarts TradingView Desktop with the debug port, after asking, because it quits TradingView first.
7. Opens Claude Code with the run playbook, [`.claude/commands/tradingview-run.md`](../.claude/commands/tradingview-run.md).

That playbook:

- compiles the strategy and fixes any errors in the repo
- loads it on OANDA:XAUUSD 1h and reads the Strategy Tester for each mode
- compares the results with aurum's own backtest
- runs the desk on your chart's bars
- writes `reports/tradingview_run.md`

It never places orders or creates alerts unless you ask. Run it again any time with `/tradingview-run` inside Claude Code.

Options: `--yes` (don't ask before restarting TradingView), `--remote` (start `claude remote-control` so you can drive it from the Claude app on your phone), `--no-launch`, `--no-claude`, `--no-web`.

### By hand

```bash
git clone https://github.com/tradesdontlie/tradingview-mcp.git ~/tradingview-mcp
(cd ~/tradingview-mcp && npm install)
python3 -m venv .venv && .venv/bin/pip install -e ".[mcp,llm,data]"
claude mcp add -s local aurum -- "$PWD/.venv/bin/python" -m aurum.mcp_server
claude mcp add -s local tradingview -- node ~/tradingview-mcp/src/server.js
~/tradingview-mcp/scripts/launch_tv_debug_mac.sh     # or _linux.sh / launch_tv_debug.bat
claude
```

Other things to ask Claude once both servers are connected:

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
