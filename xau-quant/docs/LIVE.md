# From backtest to live: the checklist

aurum can place real orders on OANDA and MetaTrader 5. This is the order to do it in. Skipping a step is how most retail gold bots lose money: they go live on a backtest that was lucky, overfitted or missing costs.

## 1. Evidence on real data

- [ ] Run `python -m aurum compare --source yahoo` (or `--source csv` with your broker's MT5 export).
- [ ] Pick one strategy and run `python -m aurum optimize --strategy <name> --source yahoo`. You need:
  - out-of-sample Sharpe above 0.5
  - probability of backtest overfitting under 0.3
  - reality check p-value under 0.1
- [ ] Run `python -m aurum backtest --strategy <name> --source yahoo --trials <every variant you tried>`. The verdict must be **passes**, and the result must survive 1.5x costs.
- [ ] Load `tradingview/aurum_gold.pine` on TradingView (see `TRADINGVIEW.md`) and check the Strategy Tester roughly agrees, allowing for TradingView not charging swap or widening spreads on news.

If any box fails, stop here. A strategy that fails these tests isn't ready for money, however good its equity curve looks.

## 2. Paper trading, at least 4 weeks

```bash
docker compose up -d        # or: python -m aurum run --broker paper --feed yahoo --strategy <name>
```

- [ ] Let it trade on live prices for at least 4 weeks, or 30 closed trades, whichever is longer.
- [ ] Every week, run `python -m aurum review` (add Claude lessons with `ANTHROPIC_API_KEY`). Treat each lesson as a hypothesis to backtest, not a change to make.
- [ ] Compare paper results with a backtest over the same dates: `python -m aurum backtest --source yahoo --start <paper start>`. Differences beyond costs point to a bug. Fix it before moving on.

## 3. Broker practice or demo account, at least 2 weeks

OANDA practice (free):

```bash
export OANDA_TOKEN=...            # My Account > Manage API Access, on the practice site
export OANDA_ACCOUNT_ID=101-...
export OANDA_ENV=practice
python -m aurum run --broker oanda --strategy <name>
```

MT5 demo (Windows, terminal running and logged in to a demo account):

```bash
pip install MetaTrader5
set MT5_SYMBOL=XAUUSD             # check your broker's name for gold
set MT5_SERVER_UTC_OFFSET=2       # broker server time vs UTC (3 in summer for most EU brokers)
python -m aurum run --broker mt5 --strategy <name>
```

- [ ] Check that fills, spreads and swap on the broker match what the paper broker assumed. If they don't, update `instrument.py` with your broker's numbers from the symbol specification and re-run step 1.
- [ ] Set up alerts: `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`, so every order, error and guard trip reaches your phone.
- [ ] Test the kill switch on purpose: stop the runner mid-position, then restart it, and confirm it picks the position up correctly.

## 4. Live, small

```bash
export OANDA_ENV=live   # or log the MT5 terminal into the real account
export AURUM_LIVE_ACK="I accept the risk of trading real money"
python -m aurum run --broker oanda --strategy <name> --allow-live --max-lots 0.05
```

- [ ] Start at the smallest size your broker allows (`--max-lots 0.01` to `0.05`), and lower `risk_per_trade` in the strategy's config.
- [ ] Run it where it can't sleep: a VPS near your broker's servers, with `docker compose` and `restart: unless-stopped`.
- [ ] Only scale up after a month of live results that match paper and practice.

## Guards that are always on

| Guard | Default | Where |
|---|---|---|
| Real money needs `--allow-live` and `AURUM_LIVE_ACK` | on | `runner.py` |
| Daily loss limit | 3% of the day's starting equity | `RunnerConfig.daily_loss_limit` |
| Drawdown kill switch | 20% from peak; re-arm by deleting `~/.aurum/runner_state.json` | `RunnerConfig.max_drawdown_kill` |
| Max position | 2 lots | `--max-lots` |
| Stale data | skip if the last bar is more than 3 bars old | `RunnerConfig.stale_after_bars` |
| Every order carries a stop | the strategy's ATR stop | broker side (OANDA `stopLossOnFill`, MT5 `sl`) |
| The LLM can only shrink or veto | always | `agents.risk_manager` |
| Webhooks never trade real money | only the paper account, and only with `AURUM_WEBHOOK_EXECUTE=paper` | `service.execute_alert` |

Research software. Not investment advice. Trading leveraged gold CFDs can lose more than your deposit.
