# The friend's BTC swing system, identified and rebuilt for gold

## What his chart shows

From five photos of his screen (BTCUSDT perpetual, Binance, 1D):

| On his chart | Identified as | How sure |
|---|---|---|
| Stepped trailing line, red/teal ribbon, ▲/▼ square markers | A trend flip engine. **HalfTrend, amplitude 5** fits his markers best: 7 of 10 within 3 days | Medium-high |
| Slow black curve | Long moving average, read as EMA 200 | High (type), medium (period) |
| Thin blue and purple lines near price | Fast averages, read as EMA 20 and EMA 50 | Medium |
| Table: Time Frame 5/15/60/240/1D, Bullish/Bearish, "Aligned 3▲/2▼", "Long WR 75% (4)", "Short WR 0% (5)" | Multi-timeframe direction dashboard, with win rates of the indicator's own signals | High |
| "D CANDLE PO3", "Range: 2415.2 USDT" | ICT Power-of-Three view: the current daily candle drawn beside price | High |
| Red/green boxes: "Stop: 4,601.8 (4.057%)", "Target: 32,883.4 (28.991%)", "Risk/reward ratio: 7.15" | TradingView's position tool. Stop ≈ 10-day swing high + 0.46 ATR, target ≈ 7R | Medium |

The indicator legend on his chart is collapsed ("⌄ 6" means six scripts are hidden). A screenshot with it expanded would replace every guess above with the real names.

### How the engine was pinned down

Real BTC daily data (Bitstamp, matching his 28 Oct 2025 candle to within 0.1%) was run through Supertrend, HalfTrend, Chandelier Exit, UT Bot and EMA crosses across their usual settings. Each was scored against his marker dates, read off the screenshots at about 2 px per day:

| Engine | Markers within 3–4 days of his |
|---|---|
| HalfTrend, amplitude 5 | 7 / 10 (five within 1 day: 3 Oct, 14 Oct, 7 Dec 2025, 22 May, 6 Jul 2026) |
| Chandelier Exit 10 / 2.5 | 7 / 10, more extra flips |
| Supertrend 10 / 2 | 7 / 10, many extra flips |
| Supertrend 10 / 3 | 5 exact, misses Dec and Jul |

His position tool on the 28 Oct 2025 short: entry ≈ 113,426, stop 118,028, target ≈ 80,543. The 10-day high was 116,381 and ATR(14) was 3,600, so the stop is the swing high plus 0.46 ATR. The aurum rule "10-bar swing high + 0.5 ATR" puts it at 118,186, 0.13% away.

## Rebuilt

| Piece | File |
|---|---|
| Engines (HalfTrend, Supertrend, Chandelier, UT Bot), MTF table, PO3, win rates | `aurum/indicators.py` |
| Swing system: flips, EMA 200 regime, MTF filter, stop modes, trade plan | `aurum/swing.py` |
| Strategies: `swing_halftrend`, `swing_halftrend_structure`, `swing_halftrend_trail`, `swing_halftrend_4h`, `swing_supertrend` | `aurum/strategies.py` |
| TradingView **indicator** (his chart: ribbon, labels, EMAs, table, PO3, auto entry/SL/TP boxes) | `tradingview/aurum_swing_indicator.pine` |
| TradingView **strategy** (same chart plus the Strategy Tester) | `tradingview/aurum_swing.pine` |
| Dashboard "Swing desk" panel | `python -m aurum.web` |

Stop modes: `swing_r` (his: swing high/low ± 0.5 ATR, 7R target), `pct_r` (4% stop, 7R), `trail` (trail the engine line), `pct_be` (4%, breakeven at 1R, then trail), `atr` (2.5 ATR, trail). All of them work in the backtester, the live runner (stops are moved at the broker) and the Pine scripts.

## Gold backtest: real XAUUSD, Sept 2016 to Sept 2026

Hourly mid prices built from bid/ask minute data (`scripts/build_xau_dataset.py`), resampled to broker trading days. Costs include spread, slippage, commission and overnight swap. Risk is 1% per trade.

| Setup (HalfTrend 5, 1D) | Trades | Return | Sharpe | Max DD | PF | Avg R |
|---|---|---|---|---|---|---|
| EMA 200 filter, 4% stop, 7R | 33 | +22.9% | 0.82 | −5.6% | 2.68 | 0.67 |
| EMA 200 filter, his swing stop, 7R | 33 | +13.5% | 0.49 | −10.6% | 1.92 | 0.45 |
| No filter, 4% stop, 7R | 62 | +19.9% | 0.58 | −5.6% | 1.82 | 0.33 |
| Buy and hold gold | – | +229% | 0.81 | −27% | – | – |

Validation of the first row:
- deflated Sharpe 97.8% (2 trials: the setup was fixed from his chart before any gold data was seen)
- survives 2x costs
- 4.5% chance of losing money in the trade bootstrap
- verdict: **passes, paper trade next**

Out of sample, after picking variants on 2016–2021 and trading 2022–2026:
- **The family works on gold:** 91% of 144 variants made money; median Sharpe 0.58; PBO 0.23.
- **Tuning doesn't carry:** the in-sample rank predicted nothing (rank correlation −0.04).
- **His setup topped the out-of-sample list:** Sharpe 1.50 vs 1.06 for buy and hold, with a −4% drawdown vs −27%.

What this means:
- **The edge is risk-adjusted, not raw return.** Low exposure and a 4% stop sized at 1% risk mean single-digit yearly returns, unless risk per trade goes up (2% gives about +4% a year with an 11% drawdown).
- **It's a small sample.** 33 trades in ten years is too few to trust. The 4h version (`swing_halftrend_4h`, median hold 6.7 days) trades about 20 times a year: +62% over ten years, PF 1.62, max DD −13%. It was chosen from a grid, so treat it as a hypothesis.
- **Swap costs real money.** About a third of gross profit on gold longs went to overnight swap.

## Next

1. His exact SL rule and the expanded indicator legend: plug them into `swing_stops` / `STOP_PRESETS` and re-run `python -m aurum backtest --strategy swing_halftrend_structure --source csv --csv XAUUSD_H1.csv`.
2. Paper trade: `python -m aurum run --broker paper --strategy swing_halftrend` (it checks once an hour and acts only on completed daily candles).
