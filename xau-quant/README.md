# aurum: AI quant for gold (XAUUSD)

A research and trading toolkit built for one market: gold. It has gold's own drivers (dollar, real yields, positioning, sessions, the LBMA fix, US data days), costs charged the way a broker charges them, and validation that tells you when a backtest is luck. An AI analyst desk explains the market and can cut risk, but never add it.

The market research behind it, covering products, open source bots, drivers and data sources, is in [`docs/RESEARCH.md`](docs/RESEARCH.md).

## Quick start

```bash
cd xau-quant
pip install -e ".[dev]"              # numpy, pandas, scikit-learn, pytest
python -m aurum compare              # league table of all strategies
python -m aurum backtest --strategy macro_reversion --trials 4 --out reports
python -m aurum ml --horizon 24 --k-atr 1.5 --out reports
python -m aurum brief                # today's desk view
pytest                               # 21 tests
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

## What's inside

| Module | What it does |
|---|---|
| `instrument.py` | XAUUSD CFD, COMEX GC and MGC specs: 100 oz lots, spread, slippage, commission, long/short swap, Wednesday triple swap, margin |
| `calendar.py` | Asia, London and NY sessions with correct daylight saving, LBMA 10:30/15:00 fixes, NFP and FOMC calendar, rollover at 17:00 NY |
| `data.py` | Yahoo, FRED, CFTC and CSV/MT5 loaders. Each macro series joins intraday bars only **after its publication time**. Includes the synthetic market |
| `features.py` | 42 features: returns, vol, ATR, RSI, trend distance, efficiency ratio, Asian range, fix proximity, event proximity, DXY beta and residual, real yield changes, VIX, gold/silver ratio, COT z-score, walk-forward macro fair value gap |
| `regime.py` | Gaussian HMM in numpy with a **causal** forward filter (calm / normal / stressed) |
| `strategies.py` | London breakout of the Asian range, 1 to 5 month trend, macro fair value reversion, regime-weighted ensemble. Flat into NFP/FOMC |
| `ml.py` | Triple-barrier labels, purged walk-forward, gradient boosting, out-of-sample permutation importance |
| `backtest.py` | Bar by bar engine: next-bar-open fills, ATR stops and targets, gap fills, news spreads, swap, daily loss limit, drawdown kill switch, lot rounding, margin cap |
| `validation.py` | Probabilistic and deflated Sharpe, trade bootstrap drawdowns, cost stress at 1x/1.5x/2x/3x, and a verdict |
| `agents.py` | Macro, positioning, technical and event analysts, plus a Claude strategist (structured JSON output, adaptive thinking, refusal fallback) and a risk manager that can only shrink or veto |
| `report.py` | Self-contained HTML tear sheet |

## Results on synthetic data, and why they're negative

`python -m aurum compare` on the default synthetic market (2019 to 2024, hourly):

| Strategy | Return | Sharpe | Max DD | Trades |
|---|---|---|---|---|
| london_breakout | -21.0% | -0.61 | -25.2% | 503 |
| trend | -1.4% | -0.70 | -1.8% | 13 |
| macro_reversion | +1.6% | 0.09 | -6.9% | 137 |
| ensemble | -11.3% | -0.77 | -13.7% | 1021 |

The synthetic market is there to prove the plumbing works, not to find an edge. It has no breakout structure in it, and its one-week mean reversion works against trend. The validation layer correctly calls every one of these a failure. The tests also prove the ML pipeline finds a real edge when one exists (a planted signal reaches a 65%+ hit rate) and finds nothing in noise (about 50%).

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

## Roadmap

Live execution (MT5 bridge, OANDA v20), paper-trading mode, Dukascopy tick data, GLD holdings and WGC central bank data, CPI/PCE calendar, Kronos candle embeddings, meta-labeling, combinatorial purged CV, a news ingestion agent, a web dashboard with Telegram alerts. See `docs/RESEARCH.md`, section 4.

Research software. Not investment advice.
