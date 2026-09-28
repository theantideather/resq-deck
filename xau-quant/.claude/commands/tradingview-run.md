Run the aurum gold strategy on my TradingView Desktop and report back. Two MCP servers are connected: `tradingview` (drives my TradingView Desktop over CDP) and `aurum` (the quant toolkit in this repo).

Ground rules:
- Never place, modify or cancel orders, never use replay_trade, and never create or delete TradingView alerts unless I ask in this conversation.
- Don't change my TradingView account settings or saved layouts. Opening a new chart tab or a new Pine script is fine.
- If a step fails, say which one and why, try the documented fallback once, then carry on with the rest.

Steps:

1. **Connection.** Call `tv_health_check`. If TradingView isn't reachable, call `tv_launch` once and check again. If it still fails, stop and tell me to run `scripts/setup_tradingview.sh` (or the `.ps1` on Windows).

2. **Compile.** Get the source with aurum `get_pine_strategy`. Validate it with tradingview `pine_check`. If there are errors, fix them in `tradingview/aurum_gold.pine` in this repo, keeping the strategy logic the same. Re-check until it compiles clean, and tell me what you changed.

3. **Load it on the chart.** Run `chart_set_symbol` OANDA:XAUUSD (if unavailable, use FX:XAUUSD or TVC:GOLD and say so), then `chart_set_timeframe` 60. Then `pine_new` as a strategy, `pine_set_source`, `pine_smart_compile`, and `pine_get_errors` to confirm it's clean. Save it with `pine_save`.

4. **Read the Strategy Tester** for each mode: Ensemble first, then Macro reversion, Trend and London breakout. The mode is the "Strategy" input. Use `indicator_set_inputs` on the aurum strategy if possible; otherwise leave it on Ensemble and say so. For each mode:
   - `ui_open_panel` strategy tester
   - `data_get_strategy_results` for net profit, max drawdown, trades, win rate and profit factor
   - `capture_screenshot` of the strategy tester

5. **Compare with aurum.** Call aurum `backtest_strategy` for the same strategies, with `n_trials` 4, on `source` "yahoo" if it works, otherwise "synthetic" (say which). Explain the differences: TradingView doesn't charge overnight swap (the script's desk table shows an estimate) or widen spreads on news, and it has different data and history length.

6. **Desk read on the live chart.** Call `data_get_ohlcv` with 1500 bars on the 60-minute chart, then aurum `analyze_tradingview_bars` with those bars. Explain the desk's decision in plain words.

7. **Write it up** in `reports/tradingview_run.md`: a results table per mode (TradingView vs aurum), screenshot file paths, compile fixes, the desk read and next steps. End with this line: "Research only, not investment advice." Then give me a short summary here.
