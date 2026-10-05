# aurum Gold Breakout (daily XAUUSD)

A custom system built from scratch for gold. Each rule family below was tested on real XAUUSD data, with costs and swap. Settings were picked on 2017–2021 data and then checked on 2022–2026.

## Rules

| | |
|---|---|
| Trend | Up: close above EMA 200 and EMA 50 above EMA 200. Down: the mirror. |
| Buy | In an uptrend, the daily close breaks the highest high of the previous 50 days |
| Sell | In a downtrend, the close breaks the lowest low of the previous 50 days |
| Stop | 2.5 ATR(14) from entry, then trails 3 ATR behind the best price since entry |
| Target | 5R |
| Size | 1% of the account at risk per trade (quantity = risk / stop distance) |

## Results, mid-2017 to Sept 2026, 1% risk

- **Trades:** 32 (26 long, 6 short), with 69% winners
- **Profit factor:** 5.2
- **Total:** +25R; average +0.8R per trade
- **Max drawdown:** 4.3%; Sharpe 0.88
- **Time in market:** 26%

| Year | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| Return at 1% risk | +3.0% | +1.7% | +3.9% | −1.9% | −0.1% | −0.7% | +5.2% | +7.7% |

## How it was chosen

| Family | Median Sharpe 2017–21 | Median Sharpe 2022–26 |
|---|---|---|
| Trend breakout (this) | 0.46 | 1.07 |
| Squeeze breakout | 0.09 | 0.75 |
| Donchian (turtle) | −0.14 | 0.50 |
| RSI pullback in trend | 0.12 | −0.27 |

- **Not a lucky setting:** 97% of 96 nearby settings (lookback 30–60, stop 2–3 ATR, trail 2.5–4 ATR, with or without the target) have a full-period Sharpe above 0.5.
- **Daily only:** the same rules on 4-hour bars do worse (Sharpe about 0.3, drawdown about −15%).

## Limits

- **Few trades:** about 3–4 a year, so a single year can be flat.
- **Low raw return at 1% risk:** about 2% a year, against about 9% a year for buy and hold before 2022 and 21% after, with a 27% drawdown. The edge is the small drawdown. At 3% risk, expect roughly 6–7% a year with about 13% drawdowns.
- **Shorts barely matter:** they made about 0R over 6 trades. Gold's long-term uptrend does the work.
- **Python and Pine match:** a line-by-line port of the Pine logic gives the same trades as the research backtest from 2018 on. The only differences are trades in 2017, while the EMAs warm up.

## TradingView

`tradingview/aurum_gold_breakout.pine` is an indicator with:
- BUY/SELL labels showing entry, SL, TP and quantity
- the stop, target and entry lines
- red and green stop/target boxes
- the trend background
- a stats table

Use it on OANDA:XAUUSD, daily.
