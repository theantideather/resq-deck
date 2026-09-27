# AI quant for gold: market research

Compiled 27 September 2026 from public product pages, reviews and GitHub repositories. Performance numbers are the vendors' or authors' own claims, taken as published and not verified.

## 1. What exists today

The market splits into five groups. None of them is a serious, gold-specific quant research product for a single trader or a small desk. That gap is what aurum is built for.

### 1.1 Gold-specific retail tools

| Product | What it is | How it works | Notes |
|---|---|---|---|
| **TradeOS** (tradeos.xyz) | Natural language to AI trading agent | You describe logic in plain words (ATR breakout, Fibonacci, session setups) and it runs as an always-on agent on XAUUSD | Closest to "AI agent for gold"; publishes regime-based XAUUSD playbooks |
| **SnapPChart** | Chart screenshot analysis | Upload a chart and a vision model grades the setup | Discretionary helper, not a system |
| **XAUBOT** (xaubot.com) | MT4/MT5 Expert Advisor platform for gold | Prebuilt gold EAs you run in MetaTrader | Typical EA marketplace model |
| **Gold Trading AI** (Google Play) | Mobile signals app | AI gold signals, Smart Money Concepts, voice alerts | Signals only, no backtests shown |
| **MT5 gold EAs** (Forex Gold Investor, Happy Gold, Dark Gold, Gold Scalper Pro and others) | Paid Expert Advisors | Mostly grid, martingale or scalping logic on M1 to M15 | Reviews report heavy overfitting and blown accounts; the Medium post "I tested 14 specialized bots and lost $6,200" is typical |
| **FXNX** | Content plus agent comparisons | Covers DXY and yield correlation trading for gold | Research and education |

### 1.2 General AI and quant platforms (multi-asset)

| Product | Positioning | Gold relevance | Price |
|---|---|---|---|
| **QuantConnect** | Open source LEAN engine, Python/C#, cloud backtests, live brokers | Can trade XAUUSD CFDs (OANDA) and COMEX futures; you build everything yourself | Free tier; paid compute |
| **StrategyQuant X** | No-code strategy generator plus robustness tests | Popular with MT5 gold traders; generates thousands of rules (an overfitting machine unless you are disciplined) | One-off license |
| **TrendSpider** | Charting with trainable "custom AI indicators" | Pattern search across symbols including gold | Subscription |
| **Trade Ideas** | AI scanner ("Holly") for US stocks | None for spot gold | Up to $2,268/yr |
| **Kavout** | AI stock ranking (Kai Score), InvestGPT, **Commodity Strategist agent** | Commodity agent gives LLM commentary only, no execution | About $16 to $20/mo |
| **Tickeron** | Marketplace of 100+ AI bots, forex and crypto | Has FX and gold bots | Subscription |
| **Composer** | AI-generated portfolios with execution | ETF level (GLD, IAU), not intraday gold | Subscription |
| **Numerai** | Crowdsourced ML hedge fund | Equities only | Free to compete |
| **Alpaca** | Commission-free API broker | Gold via ETFs only | Free API |
| **Autochartist** | Pattern recognition inside broker platforms | Scans XAUUSD | Broker supplied |

### 1.3 Institutional gold analytics

| Tool | Owner | What it does |
|---|---|---|
| **Qaurum** (Gold Valuation Framework) | World Gold Council | Implied gold returns under macro scenarios from supply and demand equilibrium. Long horizon, not a trading signal |
| **GRAM** (Gold Return Attribution Model) | World Gold Council | Multiple regression of monthly gold returns on four driver groups: economic expansion, risk and uncertainty, opportunity cost (rates, dollar), momentum (positioning, flows). Explains the past, doesn't trade |

aurum's `macro_fair_value` is a tradeable, walk-forward version of the GRAM idea.

### 1.4 Open source AI trading frameworks

| Project | What to take from it |
|---|---|
| **TradingAgents** (TauricResearch, LangGraph, v0.2.2 March 2026, supports Anthropic, OpenAI, Google, xAI and others) | Multi-agent desk: fundamentals, sentiment, news and technical analysts, bull/bear researchers, trader, risk team. aurum's desk borrows the role split but gives the LLM no power to add risk |
| **FinRL / FinRL-X** (AI4Finance) | Deep RL environments and agents for trading |
| **Qlib** (Microsoft) | ML quant research platform: data, factor library, model zoo, backtest |
| **Kronos** (shiyu-coder, AAAI 2026) | Foundation model pre-trained on OHLCV candles from 45+ exchanges, weights on Hugging Face. Worth testing as a feature generator for gold |

### 1.5 Open source gold (XAUUSD) bots on GitHub

| Repo | Stack | Claims | License |
|---|---|---|---|
| [buckybonez/xau-ai-trading-bot](https://github.com/buckybonez/xau-ai-trading-bot) | MT5, Polars, XGBoost on 37 features, SMC (order blocks, FVG, BOS), HMM regimes, 14 entry filters, Kelly sizing, Next.js dashboard, Telegram | 654 trades, 63.9% win, PF 2.64, max DD 2.2%, Sharpe 4.83 (Jan 2025 to Feb 2026) | MIT, ~55 stars |
| [zero-was-here/tradingbot](https://github.com/zero-was-here/tradingbot) | PPO and DreamerV3 (Stable-Baselines3), 140+ features including DXY, SPX, yields, VIX, oil, silver, gold ETF, 1,500 economic events, M5 to D1 | "Targeting 80 to 120% annual" | MIT, ~200 stars |
| [Mazonia/xau-traderbot](https://github.com/Mazonia/xau-traderbot) | XGBoost, attention-based multi-timeframe scorer, 61 candle features, Gemini LLM post-mortems on losing trades with ChromaDB memory, FastAPI dashboard | none | MIT |
| [P1Piyush/XAUUSD-Trading-Bots](https://github.com/P1Piyush/XAUUSD-Trading-Bots) | ICT/SMC, HFT scalping, Random Forest inference to MT5 | none | see repo |
| [pressure679/LSTM-PPO…XAUUSD](https://github.com/pressure679/LSTM-PPO-Reinforcement-Learning-Trading-Bot-for-MetaTrader-5-XAUUSD) | LSTM-PPO on M1 data | none | see repo |
| [triqbit/mt5-ai-xauusd-trader](https://github.com/triqbit/mt5-ai-xauusd-trader) | Merge of 25+ repos: PPO, Dreamer, LSTM, Transformers, ensembles | none | see repo |
| [Rithick574/XAUUSD-bot-forex](https://github.com/Rithick574/XAUUSD-bot-forex) | EMA crossover, ADX filter, ATR stops, fixed-fractional sizing | none | see repo |

**How these bots usually go wrong**, and what aurum does about each:

1. **Lookahead.** Daily macro data gets joined to same-day intraday bars, and HMM labels are fit with Viterbi over the full sample. aurum joins each series only after its publication time, and its HMM is a causal forward filter. Both are covered by tests.
2. **Leaky validation.** A random train/test split on overlapping labels leaks future information. aurum uses a purged walk-forward.
3. **Costs.** Most repos ignore spread widening at news, overnight swap (about $45 per lot per night on longs at current rates) and the Wednesday triple swap. aurum charges all of them.
4. **Multiple testing.** A "Sharpe 4.8" after trying hundreds of variants is expected by luck. aurum reports a deflated Sharpe and runs a cost stress test.
5. **Unbounded LLMs.** LLM agents that can open trades on their own are a tail risk. In aurum the LLM can only shrink or veto a position.

## 2. What drives gold (the feature set)

| Driver | Evidence | Feature in aurum |
|---|---|---|
| US real yields (10y TIPS) | Classic inverse relationship, since gold pays no yield. It broke down after 2022 as central bank buying set the marginal price | `ry_chg_24`, `ry_chg_120`, `ry_level` |
| US dollar (DXY) | Tightest intraday coupling; dollar moves pass through to gold within minutes | `dxy_ret_24`, `dxy_ret_120`, `beta_dxy`, `resid_vs_dxy_24` |
| Central bank buying | The dominant driver since 2022 (China, India, Poland, Turkey). Published monthly by the WGC with a lag | Roadmap: WGC and IMF IFS reserve data |
| ETF flows | GLD and IAU holdings, a daily flow proxy | Roadmap: GLD tonnes |
| CFTC positioning | Managed money extremes precede reversals | `cot_mm_z` |
| Risk and uncertainty | VIX spikes bring a haven bid (with liquidation first in a crash) | `vix_z` |
| Macro events | NFP, CPI and FOMC create the biggest intraday moves; spreads widen 3 to 5 times | `hours_to_event`, `event_window`, cost model |
| Session structure | Physical buying in Asia, benchmark and OTC flow in London, informed trading in NY (Iwatsubo, Watkins and Xu, Tokyo vs NY gold futures); weakness around the London PM fix | `sess_*`, `asia_range`, `near_fix` |
| Gold/silver ratio | A risk appetite and relative value gauge | `gsr_z` |
| Momentum | Time series momentum at 1 to 12 months (Moskowitz, Ooi and Pedersen 2012) | `trend` strategy |

## 3. Data sources

| Data | Free source | Frequency | Publication lag used |
|---|---|---|---|
| Gold price | Yahoo `GC=F`, broker MT5 export, Dukascopy ticks | tick to daily | none |
| DXY | Yahoo `DX-Y.NYB` | daily | NY close (22h) |
| 10y real yield | FRED `DFII10` | daily | about 46h |
| 10y breakeven | FRED `T10YIE` | daily | about 46h |
| VIX, silver | Yahoo `^VIX`, `SI=F` | daily | 22h |
| COT gold | CFTC Socrata API, contract 088691 | weekly, as of Tuesday | Friday 15:30 ET (3 days 21h) |
| FOMC dates | federalreserve.gov | scheduled | known in advance |
| NFP | BLS, usually the first Friday | scheduled | known in advance |
| Central bank gold | WGC Goldhub, IMF IFS | monthly | 1 to 2 months |
| ETF holdings | SPDR GLD daily holdings file | daily | next day |

## 4. Product blueprint

What an AI quant product for gold should be, and what is already built in this repo:

| Layer | Built | Next |
|---|---|---|
| Instrument specs (XAUUSD CFD, GC, MGC) | yes | per-broker profiles |
| Data: synthetic, Yahoo, FRED, CFTC, CSV/MT5 | yes | Dukascopy ticks, GLD holdings, WGC central bank data, CPI/PCE calendar |
| Gold feature library (42 features) | yes | Kronos embeddings, options skew (GVZ) |
| Causal HMM regimes | yes | regime-conditional sizing study |
| Strategies: London breakout, trend, macro reversion, regime ensemble | yes | Asian mean reversion, fix fade, NFP straddle |
| ML: triple barrier, purged walk-forward, gradient boosting | yes | meta-labeling, LightGBM/XGBoost, probability calibration |
| Backtester: spread, slippage, commission, swap, gaps, stops, daily limit, kill switch | yes | tick-level fills, partial exits, trailing stops |
| Validation: PSR, DSR, bootstrap, cost stress | yes | combinatorial purged CV (CPCV), White's reality check |
| AI desk: 4 rule-based analysts, Claude strategist, risk manager | yes | news RSS ingestion, post-trade review agent with memory |
| HTML tear sheet, CLI | yes | web dashboard, Telegram alerts |
| Live execution | no | MT5 bridge (Windows), OANDA v20 REST, paper-trading mode first |

## 5. Positioning

The competition is either black-box EAs with unverifiable claims, generic multi-asset platforms where gold is an afterthought, or institutional attribution tools that don't trade. aurum's case: **one asset, done properly.** It has gold's macro drivers built in, costs charged honestly, validation that tells you when an edge is luck, and an AI layer that explains and restrains rather than gambles.

## Sources

- [Best AI for Gold Trading 2026: 5 XAUUSD Tools Ranked (SnapPChart)](https://www.snappchart.app/blog/ai-chart-analysis/best-ai-gold-trading-tools)
- [Best AI for XAUUSD Trading in 2026 (TradeOS)](https://www.tradeos.xyz/post/best-ai-for-xauusd-trading-2026)
- [XAUUSD Trading Strategy 2026: 4 Gold Setups by Market Regime (TradeOS)](https://www.tradeos.xyz/gold-xauusd-trading-ai-strategy-2026)
- [Best Robots For Gold Trading 2026 (AlgoTradingSpace)](https://algotradingspace.com/best-robot-for-gold-trading)
- [XAUBOT AI](https://xaubot.com/product/gold-trading-ea/)
- [Gold Trading AI (Google Play)](https://play.google.com/store/apps/details?id=com.goldtrading.ai&hl=en)
- [AI Gold Trading 2026 (FXNX)](https://fxnx.com/en/blog/ai-gold-trading-2026-best-agents-xauusd-edge)
- [I tested 14 specialized bots and lost $6,200 testing gold (Medium)](https://medium.com/@wise_crimson_lion_733/i-tested-14-specialized-bots-and-lost-6-200-testing-gold-heres-the-only-xauusd-agent-that-b1edc3110d6a)
- [Best Quant Trading Software for Retail Investors 2026 (Quant-Builder)](https://www.quant-builder.ai/articles/best-quant-trading-software-retail-investors)
- [Best Algorithmic Trading Software in 2026 (QuantMonitor)](https://quantmonitor.net/best-algorithmic-trading-software/)
- [Best AI Trading Apps in 2026 (HyScaler)](https://hyscaler.com/insights/top-ai-trading-apps-boost-investment/)
- [Comparing 11 AI trading platforms in 2026 (FXStreet)](https://www.fxstreet.com/press-releases/from-stock-selection-to-execution-comparing-11-ai-trading-platforms-in-2026-202608281437)
- [Kavout review 2026 (Stork.AI)](https://www.stork.ai/en/kavout)
- [Qaurum, World Gold Council](https://www.gold.org/goldhub/tools/gold-valuation-model)
- [Gold Return Attribution Model, World Gold Council](https://www.gold.org/goldhub/tools/gold-return-attribution-model)
- [TradingAgents (GitHub)](https://github.com/tauricresearch/tradingagents) and [paper](https://arxiv.org/pdf/2412.20138)
- [FinRL-X (GitHub)](https://github.com/AI4Finance-Foundation/FinRL-Trading)
- [Kronos (GitHub)](https://github.com/shiyu-coder/Kronos)
- [GitHub topic: xauusd](https://github.com/topics/xauusd?o=asc&s=stars)
- [What are the main drivers of gold price? (KTH thesis)](https://kth.diva-portal.org/smash/get/diva2:1823810/FULLTEXT01.pdf)
- [What moves gold prices (Discovery Alert)](https://discoveryalert.com/what-moves-gold-prices-real-yields-central-bank-demand/)
- [Is it a golden era for gold? (J.P. Morgan Private Bank)](https://privatebank.jpmorgan.com/eur/en/insights/markets-and-investing/is-it-a-golden-era-for-gold)
- [Intraday seasonality in platinum and gold futures in Tokyo and New York (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S2405851318300102)
- [The truth about gold trading sessions (Investing.com)](https://www.investing.com/analysis/the-truth-about-gold-trading-sessions-how-liquidity-changes-from-asia-to-new-york-200684613)
