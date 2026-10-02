"""Command line: python -m aurum <command>

    compare    run every rule based strategy and print a league table
    backtest   one strategy, with validation scorecard and HTML tear sheet
    ml         walk forward gradient boosting model, validated the same way
    brief      today's desk: snapshot, analyst views, Claude strategist (if
               ANTHROPIC_API_KEY is set) and the risk manager's final size
    run        the trading loop on the paper account, OANDA or MT5
    paper      paper account status or reset
    optimize   walk forward optimisation with overfitting diagnostics
    meta       meta-labeling of a strategy's trades
    review     post-trade review of the paper account (Claude lessons optional)
    news       gold relevant headlines
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .data import load_market
from .features import build_features
from .regime import detect_regimes
from .strategies import STRATEGIES, run_strategy


def _market(args):
    kw = {}
    if args.source == "synthetic":
        kw = {"start": args.start, "end": args.end, "seed": args.seed}
    md = load_market(args.source, csv_path=args.csv, interval=args.interval, **kw)
    print(f"[aurum] {len(md.bars):,} bars from {md.source}, {md.bars.index[0]:%Y-%m-%d} to {md.bars.index[-1]:%Y-%m-%d}")
    return md


def _save(args, name, md, signal, result, card):
    if not args.out:
        return
    from .report import render

    os.makedirs(args.out, exist_ok=True)
    result.equity.to_csv(os.path.join(args.out, f"{name}_equity.csv"))
    result.trades.to_csv(os.path.join(args.out, f"{name}_trades.csv"), index=False)
    path = os.path.join(args.out, f"{name}.html")
    with open(path, "w") as f:
        f.write(render(name, result, card, source=md.source))
    print(f"[aurum] wrote {path}")


def cmd_compare(args) -> None:
    md = _market(args)
    feats = build_features(md)
    rows = []
    for name, strat in STRATEGIES.items():
        st = run_strategy(md, strat, feats)[0].stats
        rows.append({"strategy": name, "return": st["total_return"], "cagr": st["cagr"],
                     "sharpe": st["sharpe"], "max_dd": st["max_drawdown"], "trades": st["trades"],
                     "win": st["win_rate"], "pf": st["profit_factor"], "swap": st["swap"]})
    df = pd.DataFrame(rows).set_index("strategy")
    fmt = {"return": "{:.1%}", "cagr": "{:.1%}", "sharpe": "{:.2f}", "max_dd": "{:.1%}",
           "win": "{:.0%}", "pf": "{:.2f}", "swap": "{:,.0f}"}
    print(df.to_string(formatters={k: v.format for k, v in fmt.items()}))
    print(f"\n{len(df)} strategies compared. Pass --trials {len(df)} to backtest when judging the winner.")


def cmd_backtest(args) -> None:
    from .validation import scorecard

    md = _market(args)
    feats = build_features(md)
    strat = STRATEGIES[args.strategy]
    res, sig, m, line = run_strategy(md, strat, feats)
    print(f"\n{strat.name}: {strat.description}\n")
    print(res.summary())
    card = scorecard(m, sig, res, n_trials=args.trials, stop_line=line)
    print("\n" + card.text())
    _save(args, strat.name, m, sig, res, card)


def cmd_ml(args) -> None:
    from .ml import permutation_importance_oos, walk_forward
    from .validation import scorecard

    md = _market(args)
    feats = build_features(md)
    wf = walk_forward(md, feats, horizon=args.horizon, k_atr=args.k_atr)
    print(f"[aurum] {len(wf.folds)} walk forward folds, out of sample hit rate {wf.hit_rate():.1%}")
    sig = wf.signal(args.threshold)
    cfg = BacktestConfig(stop_atr=args.k_atr, target_atr=args.k_atr, max_bars_in_trade=args.horizon)
    res = run_backtest(md, sig, cfg)
    print("\n" + res.summary())
    card = scorecard(md, sig, res, n_trials=args.trials)
    print("\n" + card.text())
    print("\nPermutation importance, last fold (log loss worsening when shuffled):")
    print(permutation_importance_oos(md, feats, wf).head(12).to_string(float_format=lambda x: f"{x:.4f}"))
    _save(args, "ml", md, sig, res, card)


def cmd_brief(args) -> None:
    from .agents import run_desk
    from .strategies import ensemble

    md = _market(args)
    feats = build_features(md)
    regimes = detect_regimes(md.bars["close"])
    comps = {n: STRATEGIES[n].signal(md, feats) for n in ("london_breakout", "trend", "macro_reversion")}
    ens = ensemble(md, feats, regimes, comps)
    headlines = None
    if args.headlines:
        with open(args.headlines) as f:
            headlines = [line.strip() for line in f if line.strip()]
    snap, decision = run_desk(md, feats, float(ens.iloc[-1]), regimes, {**comps, "ensemble": ens},
                              headlines, use_llm=not args.no_llm)
    if args.json:
        print(json.dumps({"snapshot": snap, "decision": decision.to_dict()}, indent=2, default=str))
        return
    print(f"\nXAUUSD desk brief, {snap['time_utc']}\n")
    print(f"Price {snap['price']:,.2f}   regime {snap.get('regime', 'n/a')}   sessions {', '.join(snap['sessions_open']) or 'closed'}")
    print(f"Next events: {', '.join(snap['upcoming_events']) or 'none in calendar'}\n")
    for v in decision.views:
        print(f"{v.analyst:<12} bias {v.bias:+.2f}  confidence {v.confidence:.2f}")
        for r in v.reasons:
            print(f"    - {r}")
    print(f"\nQuant ensemble signal {decision.quant_signal:+.2f}  ->  final {decision.final_signal:+.2f}")
    for n in decision.notes:
        print(f"    - {n}")
    if not any(v.analyst == "strategist" for v in decision.views):
        print("\n(Claude strategist skipped: set ANTHROPIC_API_KEY to add it.)")


def cmd_run(args) -> None:
    from . import service
    from .execution import make_broker
    from .runner import Runner, RunnerConfig

    if args.broker == "paper":
        broker = service.paper_broker(args.feed, args.csv)
    else:
        broker = make_broker(args.broker)
    kw = {"now": broker.feed.now} if args.broker == "paper" and args.feed == "replay" else {}
    headlines = None
    if args.llm:
        from .news import fetch_headlines, headline_texts
        headlines = lambda: headline_texts(fetch_headlines())
    cfg = RunnerConfig(strategy=args.strategy, granularity=args.granularity, use_desk=not args.no_desk,
                       use_llm=args.llm, max_lots=args.max_lots, allow_live=args.allow_live,
                       with_macro=not args.no_macro and not (args.broker == "paper" and args.feed == "replay"))
    runner = Runner(broker, cfg, headlines=headlines, **kw)
    if args.once or args.cycles:
        for _ in range(args.cycles or 1):
            rec = runner.cycle()
            if args.broker == "paper" and args.feed == "replay":
                broker.feed.advance()
            print(json.dumps({k: rec.get(k) for k in ("last_bar", "action", "price", "quant_signal", "desk_signal",
                                                      "orders", "position", "notes")}, default=str))
    else:
        runner.loop()


def cmd_paper(args) -> None:
    from . import service

    if args.action == "reset":
        print(service.paper_reset())
        return
    st = service.paper_status(journal_limit=10)
    if not st["exists"]:
        print("No paper account yet. Start one with: python -m aurum run --broker paper --once")
        return
    pos = st["position"]
    print(f"Equity {st['equity']:,.2f}  balance {st['balance']:,.2f}  closed trades {st['n_trades']}")
    print("Position: " + (f"{'LONG' if pos['side'] > 0 else 'SHORT'} {pos['lots']} lots @ {pos['entry_price']:.2f}, "
                          f"stop {pos['stop']}" if pos else "flat"))
    for t in st["trades"][-10:]:
        print(f"  {str(t['opened_utc'])[:16]}  {'L' if t['side'] > 0 else 'S'} {t['lots']:<5} "
              f"{t['entry_price']:.2f} -> {t['exit_price']:.2f}  {t['reason']:<7} {t['net_pnl']:>10,.2f}")


def cmd_optimize(args) -> None:
    from . import service

    r = service.optimize(args.strategy, train_days=args.train_days, test_days=args.test_days,
                         source=args.source, start=args.start, end=args.end, seed=args.seed, csv_path=args.csv)
    print(f"\nWalk forward optimisation: {r['strategy']}, {r['n_trials']} parameter combinations")
    print(pd.DataFrame(r["windows"]).to_string(index=False))
    fmt = lambda x, f: "n/a" if x is None else format(x, f)
    print(f"\nOut of sample Sharpe   {fmt(r['oos_sharpe'], '.2f')}")
    print(f"Out of sample return   {fmt(r['oos_total_return'], '.1%')}")
    print(f"Deflated Sharpe        {fmt(r['oos_dsr'], '.1%')}  ({r['n_trials']} trials)")
    if r["pbo"]:
        print(f"Overfit probability    {r['pbo']['pbo']:.1%}  (CSCV, {r['pbo']['n_combinations']} splits)")
    if r["reality_check"]:
        print(f"Reality check p-value  {r['reality_check']['p_value']:.3f}  (best of {r['reality_check']['n_strategies']})")


def cmd_meta(args) -> None:
    from .ml import meta_label

    md = _market(args)
    feats = build_features(md)
    strat = STRATEGIES[args.strategy]
    primary = strat.signal(md, feats)
    res = meta_label(md, feats, primary, horizon=args.horizon, k_atr=args.k_atr)
    base, kept = res.precision(args.threshold)
    print(f"[aurum] meta-labeling {args.strategy}: {res.folds} folds")
    print(f"Hit rate of all primary bets   {base:.1%}")
    print(f"Hit rate of bets the model keeps {kept:.1%}  (threshold {args.threshold})")
    for label, sig in (("primary", primary), ("meta filtered", res.signal(primary, args.threshold))):
        st = run_backtest(md, sig, strat.config).stats
        print(f"{label:<14} return {st['total_return']:>7.1%}  sharpe {st['sharpe']:>5.2f}  "
              f"max dd {st['max_drawdown']:>7.1%}  trades {st['trades']}")


def cmd_review(args) -> None:
    from . import service

    r = service.review_paper(use_llm=not args.no_llm)
    s = r["stats"]
    if not s.get("trades"):
        print("No closed paper trades to review yet.")
        return
    print(f"{s['trades']} trades, net {s['net_pnl']:,.2f}, win rate {s['win_rate']:.0%}, costs {s['costs']:,.2f}, swap {s['swap']:,.2f}")
    for o in r["observations"]:
        print(f"  - {o}")
    if r["claude"]:
        print("\nClaude: " + r["claude"]["summary"])
        for l in r["claude"]["lessons"]:
            print(f"  * {l['lesson']}\n    evidence: {l['evidence']}\n    test: {l['suggested_change']}")


def cmd_news(args) -> None:
    from .news import fetch_headlines, headline_texts

    items = fetch_headlines(limit=args.limit)
    print("\n".join(headline_texts(items)) or "No relevant headlines (or feeds unreachable).")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aurum", description="AI quant research for gold (XAUUSD)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--source", default="synthetic", choices=["synthetic", "yahoo", "csv"])
        sp.add_argument("--csv", help="OHLCV CSV or MT5 export, with --source csv")
        sp.add_argument("--interval", default="1h")
        sp.add_argument("--start", default="2019-01-01")
        sp.add_argument("--end", default="2024-12-31")
        sp.add_argument("--seed", type=int, default=7)
        sp.add_argument("--out", help="directory for HTML tear sheet and CSVs")
        sp.add_argument("--trials", type=int, default=1,
                        help="how many strategy variants you tried; feeds the deflated Sharpe")

    sp = sub.add_parser("compare"); common(sp); sp.set_defaults(fn=cmd_compare)
    sp = sub.add_parser("backtest"); common(sp); sp.set_defaults(fn=cmd_backtest)
    sp.add_argument("--strategy", default="ensemble", choices=list(STRATEGIES))
    sp = sub.add_parser("ml"); common(sp); sp.set_defaults(fn=cmd_ml)
    sp.add_argument("--horizon", type=int, default=24)
    sp.add_argument("--k-atr", type=float, default=1.5)
    sp.add_argument("--threshold", type=float, default=0.1)
    sp = sub.add_parser("brief"); common(sp); sp.set_defaults(fn=cmd_brief)
    sp.add_argument("--headlines", help="text file, one headline per line, passed to the strategist")
    sp.add_argument("--no-llm", action="store_true")
    sp.add_argument("--json", action="store_true")

    sp = sub.add_parser("run", help="trading loop on paper, OANDA or MT5")
    sp.set_defaults(fn=cmd_run)
    sp.add_argument("--broker", default="paper", choices=["paper", "oanda", "mt5"])
    sp.add_argument("--feed", default="yahoo", choices=["yahoo", "csv", "oanda", "replay"],
                    help="bars for the paper broker (replay = offline synthetic demo)")
    sp.add_argument("--csv")
    sp.add_argument("--strategy", default="ensemble", choices=list(STRATEGIES))
    sp.add_argument("--granularity", default="H1")
    sp.add_argument("--once", action="store_true", help="one cycle and exit (for cron)")
    sp.add_argument("--cycles", type=int, default=0, help="run N cycles back to back and exit")
    sp.add_argument("--no-desk", action="store_true")
    sp.add_argument("--no-macro", action="store_true")
    sp.add_argument("--llm", action="store_true", help="Claude strategist with live headlines")
    sp.add_argument("--max-lots", type=float, default=2.0)
    sp.add_argument("--allow-live", action="store_true",
                    help="permit a real money account (also needs AURUM_LIVE_ACK)")

    sp = sub.add_parser("paper", help="paper account status or reset")
    sp.set_defaults(fn=cmd_paper)
    sp.add_argument("action", nargs="?", default="status", choices=["status", "reset"])

    sp = sub.add_parser("optimize", help="walk forward optimisation with PBO and reality check")
    common(sp); sp.set_defaults(fn=cmd_optimize)
    sp.add_argument("--strategy", default="london_breakout")
    sp.add_argument("--train-days", type=int, default=365)
    sp.add_argument("--test-days", type=int, default=91)

    sp = sub.add_parser("meta", help="meta-label a strategy's trades")
    common(sp); sp.set_defaults(fn=cmd_meta)
    sp.add_argument("--strategy", default="london_breakout", choices=list(STRATEGIES))
    sp.add_argument("--horizon", type=int, default=24)
    sp.add_argument("--k-atr", type=float, default=1.5)
    sp.add_argument("--threshold", type=float, default=0.55)

    sp = sub.add_parser("review", help="post-trade review of the paper account")
    sp.set_defaults(fn=cmd_review)
    sp.add_argument("--no-llm", action="store_true")

    sp = sub.add_parser("news", help="gold relevant headlines")
    sp.set_defaults(fn=cmd_news)
    sp.add_argument("--limit", type=int, default=30)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
