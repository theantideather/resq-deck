"""Command line: python -m aurum <command>

    compare    run every rule based strategy and print a league table
    backtest   one strategy, with validation scorecard and HTML tear sheet
    ml         walk forward gradient boosting model, validated the same way
    brief      today's desk: snapshot, analyst views, Claude strategist (if
               ANTHROPIC_API_KEY is set) and the risk manager's final size
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
from .strategies import STRATEGIES


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
        sig = strat.signal(md, feats)
        st = run_backtest(md, sig, strat.config).stats
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
    sig = strat.signal(md, feats)
    res = run_backtest(md, sig, strat.config)
    print(f"\n{strat.name}: {strat.description}\n")
    print(res.summary())
    card = scorecard(md, sig, res, n_trials=args.trials)
    print("\n" + card.text())
    _save(args, strat.name, md, sig, res, card)


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

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
