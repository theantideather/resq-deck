# aurum: AI quant for gold (XAUUSD)

A research and trading toolkit built for one market: gold. It has gold's own drivers (dollar, real yields, positioning, sessions, the LBMA fix, US data days), costs charged the way a broker charges them, and validation that tells you when a backtest is luck. An AI analyst desk explains the market and can cut risk, but never add it.

The market research behind it, covering products, open source bots, drivers and data sources, is in [`docs/RESEARCH.md`](docs/RESEARCH.md). To run it on TradingView (Pine Script, MCP with Claude Code, alert webhooks), see [`docs/TRADINGVIEW.md`](docs/TRADINGVIEW.md).

## Dashboard

```bash
pip install -e .
python -m aurum.web          # open http://127.0.0.1:8765
```

The dashboard has a live TradingView chart of OANDA:XAUUSD and the desk brief (tick **Claude** to add the strategist when `ANTHROPIC_API_KEY` is set). It shows:

- each backtest with price and trade markers, equity and drawdown charts, and the validation verdict with the cost stress table
- the strategy league table and the trade list
- **paper trading** (run cycles, see the account, the journal and closed trades)
- the **walk-forward optimiser** with overfitting diagnostics
- **headlines** and the **post-trade review**
- the Pine Script to copy, and the TradingView alert log It uses TradingView's open source Lightweight Charts, bundled so it works offline. The server uses only the standard library.

## TradingView

- `tradingview/aurum_gold.pine`: Pine Script v6 port of the four strategies, with news filter, desk table, swap estimate and JSON alerts.
- `python -m aurum.mcp_server`: aurum as an MCP server, to run next to [tradingview-mcp](https://github.com/tradesdontlie/tradingview-mcp), so Claude Code can load, compile and test the strategy on your TradingView chart and run the desk on your chart's bars.
- **One command sets all of it up on your computer**: `./scripts/setup_tradingview.sh` (macOS/Linux) or `powershell -ExecutionPolicy Bypass -File scripts\setup_tradingview.ps1` (Windows). See [`docs/TRADINGVIEW.md`](docs/TRADINGVIEW.md).

## Quick start

```bash
cd xau-quant
pip install -e ".[dev]"              # numpy, pandas, scikit-learn, pytest
python -m aurum compare              # league table of all strategies
python -m aurum backtest --strategy macro_reversion --trials 4 --out reports
python -m aurum ml --horizon 24 --k-atr 1.5 --out reports
python -m aurum brief                # today's desk view
pytest                               # 65 tests
```

Everything defaults to a **synthetic gold market**, so it runs offline. For real data:

```bash
pip install -e ".[data]"                              # yfinance
python -m aurum compare --source yahoo                # GC=F hourly (about 2 years) + DXY, FRED real yields, CFTC COT
python -m aurum compare --source csv --csv XAUUSD_H1.csv   # MT5 export: File > Export bars
```

To add the Claude strategist to the brief:

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...
python -m aurum brief --source yahoo --headlines headlines.txt
```

## Swing desk: a friend's BTC system, rebuilt for gold

A trend-flip swing system identified from screenshots of a friend's BTC chart: HalfTrend 5 on 1D, EMA 200 regime, a multi-timeframe table, a PO3 candle, and stops at the swing high/low plus 0.5 ATR with a 7R target. It's rebuilt as Python strategies (`swing_*`), a TradingView indicator and strategy, and a dashboard panel, then backtested on ten years of real XAUUSD. Details and results: [`docs/SWING.md`](docs/SWING.md).

## Trading: paper, OANDA, MetaTrader 5

A single runner drives every account type. At each bar close it pulls completed bars, builds the same features as the backtest, and computes the strategy signal. It then runs the desk and applies the guards, reconciles the position, journals the decision and notifies you. A test checks that, on the same bars, the runner makes exactly the same trades as the backtester.

```bash
python -m aurum run --broker paper --feed yahoo                 # paper account on GC=F hourly, loops forever
python -m aurum run --broker paper --feed replay --cycles 200   # offline dry run on synthetic history
python -m aurum run --broker oanda --once                       # OANDA practice account, one cycle (cron it hourly)
python -m aurum run --broker mt5                                # MetaTrader 5 terminal (Windows)
python -m aurum paper                                           # paper account status; `paper reset` to start over
python -m aurum review                                          # post-trade review, Claude lessons if keyed
```

| Guard | What it does |
|---|---|
| Real money lock | A live OANDA or real MT5 account is refused unless you pass `--allow-live` **and** set `AURUM_LIVE_ACK` to the exact sentence in `runner.py`. Practice and demo accounts need neither |
| Daily loss limit | Flat for the rest of the trading day after a 3% loss from the day's start |
| Drawdown kill switch | Stops trading after a 20% fall from peak equity, until you delete the state file on purpose |
| Max lots | A hard cap on position size (default 2 lots) |
| Stale data | Skips the cycle when the last bar is too old (market closed, feed down) |
| Re-entry lock and time stop | Same rules as the backtester |

Brokers: the **paper** broker uses the backtester's cost model (spread, slippage, commission, swap with the Wednesday triple, gap fills) and keeps its state in `~/.aurum`. **OANDA** uses the v20 REST API (`OANDA_TOKEN`, `OANDA_ACCOUNT_ID`, `OANDA_ENV=practice`). **MT5** uses the `MetaTrader5` package and a running terminal. Notifications go to Telegram (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) and/or a Discord or Slack webhook (`AURUM_NOTIFY_WEBHOOK`). With `AURUM_WEBHOOK_EXECUTE=paper`, TradingView alerts trade the paper account. See [`docs/LIVE.md`](docs/LIVE.md) before connecting a real account.

## Research tools

```bash
python -m aurum optimize --strategy london_breakout   # walk-forward optimisation
python -m aurum meta --strategy london_breakout       # meta-labeling
python -m aurum news                                  # gold headlines for the strategist
```

- **Walk-forward optimisation.** Picks parameters on each trailing year and trades them the next quarter. It reports only the out-of-sample result, with the deflated Sharpe computed from the real number of variants tried.
- **Probability of backtest overfitting (CSCV).** How often the in-sample winner lands in the bottom half out of sample. Near 50% means the parameter search is picking noise.
- **White's reality check** (stationary bootstrap). Is the best of N strategies better than zero, once you account for having looked at N?
- **Meta-labeling.** A calibrated second model learns when to believe a strategy's trades. On the synthetic breakout it lifts the hit rate from 48% to 58% and cuts the loss from -21% to -0.2%. That helps, but it still isn't a profitable strategy, and the tools say so.

## Docker

```bash
cp .env.example .env        # fill in the keys you use
docker compose up -d        # dashboard on 127.0.0.1:8765 plus a paper runner on Yahoo prices
```

The image runs as a non-root user and keeps state in a volume. The dashboard is published on localhost only; put a tunnel in front of it for TradingView webhooks and set `AURUM_WEBHOOK_TOKEN`.

## Your own data

- `AURUM_EXTRA_SERIES="cb_buying=/data/cb.csv@30D;gld=/data/gld.csv@1D"` adds any daily series (central bank purchases, ETF holdings) as `x_<name>` macro columns. Each is joined after its publication lag and becomes a z-scored feature.
- `AURUM_EVENTS_CSV=/data/events.csv` (columns `timestamp,event`) adds CPI, PCE or any other releases to the news filter and event features.
- `AURUM_NEWS_FEEDS` sets the RSS or Atom feeds for headlines.

## What's inside

| Module | What it does |
|---|---|
| `instrument.py` | XAUUSD CFD, COMEX GC and MGC specs: 100 oz lots, spread, slippage, commission, long/short swap, Wednesday triple swap, margin |
| `calendar.py` | Asia, London and NY sessions with correct daylight saving, LBMA 10:30/15:00 fixes, NFP and FOMC calendar, rollover at 17:00 NY |
| `data.py` | Yahoo, FRED, CFTC and CSV/MT5 loaders. Each macro series joins intraday bars only **after its publication time**. Includes the synthetic market |
| `features.py` | 42 features: returns, vol, ATR, RSI, trend distance, efficiency ratio, Asian range, fix proximity, event proximity, DXY beta and residual, real yield changes, VIX, gold/silver ratio, COT z-score, walk-forward macro fair value gap |
| `regime.py` | Gaussian HMM in numpy with a **causal** forward filter (calm / normal / stressed) |
| `strategies.py` | London breakout of the Asian range, 1 to 5 month trend, macro fair value reversion, Asian session reversion, London PM fix fade, regime-weighted ensemble. Flat into NFP/FOMC |
| `ml.py` | Triple-barrier labels, purged walk-forward, gradient boosting, out-of-sample permutation importance, meta-labeling with isotonic calibration |
| `research.py` | Walk-forward optimisation, probability of backtest overfitting (CSCV), White's reality check, parameter grids |
| `execution/` | Broker interface, paper broker, OANDA v20 and MT5 adapters |
| `runner.py` | Bar-close trading loop with guards, journal and notifications |
| `news.py`, `review.py`, `notify.py` | Headline ingestion, post-trade review with Claude lessons, Telegram and webhook alerts |
| `backtest.py` | Bar by bar engine: next-bar-open fills, ATR stops and targets, gap fills, news spreads, swap, daily loss limit, drawdown kill switch, lot rounding, margin cap |
| `validation.py` | Probabilistic and deflated Sharpe, trade bootstrap drawdowns, cost stress at 1x/1.5x/2x/3x, and a verdict |
| `agents.py` | Macro, positioning, technical and event analysts, plus a Claude strategist (structured JSON output, adaptive thinking, refusal fallback) and a risk manager that can only shrink or veto |
| `report.py` | Self-contained HTML tear sheet |
| `service.py` | JSON entry points shared by the dashboard and the MCP server |
| `web/` | Dashboard server (standard library) and single-page UI |
| `mcp_server.py` | MCP server with 12 tools: backtests, optimiser, desk, TradingView bars, Pine source, paper trading, review, news, alert log |

## Results on synthetic data, and why they're negative

`python -m aurum compare` on the default synthetic market (2019 to 2024, hourly):

| Strategy | Return | Sharpe | Max DD | Trades |
|---|---|---|---|---|
| london_breakout | -21.0% | -0.61 | -25.2% | 503 |
| trend | -1.4% | -0.70 | -1.8% | 13 |
| macro_reversion | +1.6% | 0.09 | -6.9% | 137 |
| asian_reversion | -7.7% | -1.15 | -7.9% | 202 |
| fix_fade | -14.7% | -0.35 | -25.0% | 1116 |
| ensemble | -11.3% | -0.77 | -13.7% | 1021 |

The synthetic market is there to prove the plumbing works, not to find an edge. It has no breakout, Asian-session or fix structure in it, and its one-week mean reversion works against trend. The validation layer correctly calls every one of these a failure. The tests also prove the ML pipeline finds a real edge when one exists (a planted signal reaches a 65%+ hit rate) and finds nothing in noise (about 50%).

Two things the synthetic runs already show about real gold trading:

- **Hourly gold is cost-dominated.** Spread plus slippage plus commission is about $0.40 per ounce round turn, roughly 10% of a 1.5 ATR hourly move. Short-horizon models need a large edge just to break even.
- **Swap is a silent killer for longs.** At about $45 per lot per night, holding a long for a year costs around $16,000 per lot. Trend systems pay this; many backtests leave it out.

Numbers on real data will differ. Run `--source yahoo` or `--source csv` before believing anything.

## Rules this codebase keeps

1. No feature at bar t uses information from after t. Tests truncate the data and check that nothing changes.
2. No model scores a bar it was trained on, or a bar whose label overlapped its training data.
3. Every fill pays spread and slippage, every lot pays commission, and every night pays swap.
4. Report how many variants you tried (`--trials`). The deflated Sharpe uses it.
5. The LLM can reduce risk and never increase it.
6. Live trading makes the same decisions as the backtest (tested trade for trade), and real money needs two explicit opt-ins.

## Still to do

Run everything on real data (Yahoo, your broker's CSV, OANDA prices), and TradingView (see `docs/TRADINGVIEW.md`). Also open: Dukascopy tick data, Kronos candle embeddings as features, and bringing the two new session strategies into the Pine port.

Research software. Not investment advice.
