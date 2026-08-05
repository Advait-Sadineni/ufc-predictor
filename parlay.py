"""Parlay generator from the latest saved predictions + FanDuel lines.

Rules (per project owner + standard practice):
  - 2-4 legs, chosen by model EV at FanDuel's price
  - no coin-flip legs (model < 58%) — EXCEPT the main event, which may ride
  - hedge ticket: same parlay with the least-confident leg flipped, so the
    two tickets split the riskiest fight between them
  - live-hedge math printed for the final leg (lock profit if legs 1..N-1 hit)

Honest math is printed with every ticket: the model's probability the parlay
hits, the market's no-vig probability, and the EV under each. Parlays multiply
each leg's vig — the market-EV line shows exactly what that costs.

Run predict.py first (it saves the snapshot this reads).
Usage: python parlay.py [--legs=3,4,6] [--stake=25] [--date=YYYY-MM-DD]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PREDS = Path(__file__).parent / "predictions"
COIN_FLIP = 0.58  # model-probability floor for a non-main-event leg


def american_to_dec(o):
    return 1 + (o / 100 if o > 0 else 100 / -o)


def dec_to_american(d):
    b = d - 1
    return b * 100 if b >= 1 else -100 / b


def implied(o):
    return -o / (-o + 100) if o < 0 else 100 / (o + 100)


def no_vig_side(o_side, o_other):
    pa, pb = implied(o_side), implied(o_other)
    return pa / (pa + pb)


def load_snapshot(date_filter):
    files = sorted(PREDS.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if date_filter:
        files = [f for f in files if f.name.startswith(date_filter)]
    if not files:
        sys.exit("No prediction snapshot found — run predict.py first.")
    f = files[-1]
    print(f"Using snapshot: {f.name}")
    return pd.read_csv(f)


def ticket_report(name, legs, stake):
    dec = float(np.prod([american_to_dec(l["odds"]) for l in legs]))
    p_model = float(np.prod([l["p_model"] for l in legs]))
    p_market = float(np.prod([l["p_market"] for l in legs]))
    payout = stake * dec
    print(f"\n{name}  ({len(legs)} legs, ${stake:.0f} stake)")
    for l in legs:
        print(f"  {l['side']:26} {l['odds']:+.0f}   model {l['p_model']:.0%} | "
              f"market no-vig {l['p_market']:.0%}{'   [main event]' if l['main'] else ''}")
    print(f"  Combined: {dec_to_american(dec):+.0f} (decimal {dec:.2f}) -> "
          f"${payout:.2f} payout (${payout - stake:.2f} profit)")
    print(f"  P(hits): model {p_model:.1%} | market no-vig {p_market:.1%}")
    print(f"  EV on ${stake:.0f}: model {stake * (p_model * dec - 1):+.2f} | "
          f"market {stake * (p_market * dec - 1):+.2f}  <- the market number is the vig, compounded")
    return dec, payout


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    leg_sizes = [int(x) for x in str(args.get("--legs", "3,4,6")).split(",")]
    stake = float(args.get("--stake", 25))
    df = load_snapshot(args.get("--date"))

    pool = []
    main_idx = len(df) - 1  # snapshots preserve card order; main event is last
    for i, r in df.iterrows():
        if pd.isna(r.get("fanduel_a")) or pd.isna(r.get("fanduel_b")):
            continue
        pick_a = r["p_a"] >= 0.5
        side = r["fighter_a"] if pick_a else r["fighter_b"]
        other = r["fighter_b"] if pick_a else r["fighter_a"]
        p = r["p_a"] if pick_a else 1 - r["p_a"]
        o = r["fanduel_a"] if pick_a else r["fanduel_b"]
        o_other = r["fanduel_b"] if pick_a else r["fanduel_a"]
        pre = "a" if pick_a else "b"
        pool.append({
            "side": side, "other": other, "odds": o, "odds_other": o_other,
            "p_model": p, "p_market": no_vig_side(o, o_other),
            "p_market_other": no_vig_side(o_other, o),
            "ev": p * american_to_dec(o) - 1, "main": i == main_idx,
            "methods": {"KO/TKO": r[f"p_{pre}_ko"], "Sub": r[f"p_{pre}_sub"],
                        "Dec": r[f"p_{pre}_dec"]},
        })
    if not pool:
        sys.exit("No fights with FanDuel odds in the snapshot.")

    eligible = [l for l in pool if l["p_model"] >= COIN_FLIP or l["main"]]
    dropped = [l for l in pool if l not in eligible]
    if dropped:
        print("Excluded coin flips (model < 58%, non-main-event): "
              + ", ".join(f"{l['side']} ({l['p_model']:.0%})" for l in dropped))

    for n_legs in leg_sizes:
        legs = sorted(eligible, key=lambda l: l["ev"], reverse=True)[:n_legs]
        legs.sort(key=lambda l: l["p_model"], reverse=True)
        if len(legs) < 2:
            print(f"\n[{n_legs} legs] fewer than 2 eligible legs — no ticket.")
            continue
        if len(legs) < n_legs:
            print(f"\n[{n_legs} legs] only {len(legs)} eligible legs on this card "
                  f"— building a {len(legs)}-legger instead.")
        print("\n" + "#" * 74)
        dec, payout = ticket_report(f"PRIMARY PARLAY — {len(legs)} legs", legs, stake)

        # hedge ticket: flip the least-confident leg
        flip = min(legs, key=lambda l: l["p_model"])
        hedge_legs = [dict(l) for l in legs]
        for l in hedge_legs:
            if l["side"] == flip["side"]:
                l["side"], l["other"] = l["other"], l["side"]
                l["odds"], l["odds_other"] = l["odds_other"], l["odds"]
                l["p_model"] = 1 - l["p_model"]
                l["p_market"], l["p_market_other"] = l["p_market_other"], l["p_market"]
        ticket_report(f"HEDGE PARLAY (flips {flip['side']} -> {flip['other']})",
                      hedge_legs, stake)
        print(f"  The {flip['side']} vs {flip['other']} fight can't kill both tickets.")

        # live hedge on the final leg of the primary
        last = legs[-1]
        d_h = american_to_dec(last["odds_other"])
        h = payout / d_h
        locked = payout - h - stake  # both outcomes pay `locked` net of both stakes
        print(f"  LIVE HEDGE: if every leg except {last['side']} has hit, "
              f"${h:.2f} on {last['other']} at {last['odds_other']:+.0f} locks ${locked:+.2f}.")

    # ---- full method ticket: primary-parlay fighters, each leg = their best method ----
    def fair_american(p):
        return -p / (1 - p) * 100 if p >= 0.5 else (1 - p) / p * 100

    n_full = min(max(leg_sizes), len(eligible))
    full_legs = sorted(eligible, key=lambda l: l["ev"], reverse=True)[:n_full]
    full_legs.sort(key=lambda l: l["p_model"], reverse=True)
    print("\n" + "#" * 74)
    print(f"\nFULL METHOD PARLAY — every leg is winner + method ({n_full} legs):")
    p_full = 1.0
    for l in full_legs:
        m, pm = max(l["methods"].items(), key=lambda kv: kv[1])
        p_full *= pm
        print(f"  {l['side']:26} by {m:7}  model {pm:.0%}   fair odds {fair_american(pm):+.0f}")
    print(f"  P(all hit): model {p_full:.1%}   fair combined {fair_american(p_full):+.0f}")
    print(f"  On FanDuel, only take it above {fair_american(p_full * 0.8):+.0f}"
          f" (fair + 25% cushion) -> ${25 * (1 + (fair_american(p_full * 0.8)) / 100):.0f}+ payout on $25.")

    # ---- double-chance parlay: each leg drops only the pick's LEAST likely method ----
    print("\n" + "#" * 74)
    print(f"\nDOUBLE CHANCE PARLAY — winner by either of their two best methods ({n_full} legs):")
    p_dc = 1.0
    for l in full_legs:
        ranked = sorted(l["methods"].items(), key=lambda kv: kv[1], reverse=True)
        (m1, p1), (m2, p2) = ranked[0], ranked[1]
        p_leg = p1 + p2
        p_dc *= p_leg
        print(f"  {l['side']:26} by {m1} or {m2:7}  model {p_leg:.0%}   "
              f"fair odds {fair_american(p_leg):+.0f}")
    print(f"  P(all hit): model {p_dc:.1%}   fair combined {fair_american(p_dc):+.0f}")
    print(f"  On FanDuel, only take it above {fair_american(p_dc * 0.8):+.0f}"
          f" (fair + 25% cushion) -> ${25 * (1 + (fair_american(p_dc * 0.8)) / 100):.0f}+ payout on $25.")

    methods = []
    for _, r in df.iterrows():
        for side, tag, p in [(r["fighter_a"], "by KO/TKO", r["p_a_ko"]),
                             (r["fighter_a"], "by Sub", r["p_a_sub"]),
                             (r["fighter_a"], "by Dec", r["p_a_dec"]),
                             (r["fighter_b"], "by KO/TKO", r["p_b_ko"]),
                             (r["fighter_b"], "by Sub", r["p_b_sub"]),
                             (r["fighter_b"], "by Dec", r["p_b_dec"])]:
            methods.append((p, f"{side} {tag}"))
    methods.sort(reverse=True)
    top = [m for m in methods if m[0] >= 0.35][:3] or methods[:2]
    p_combo = float(np.prod([p for p, _ in top]))
    print("\n" + "#" * 74)
    print(f"\nMETHOD PARLAY — model-priced (FanDuel prop odds not in the API):")
    for p, leg in top:
        print(f"  {leg:38} model {p:.0%}   fair odds {fair_american(p):+.0f}")
    print(f"  P(all hit): model {p_combo:.1%}   fair combined {fair_american(p_combo):+.0f}")
    print(f"  Only worth a ticket if FanDuel pays MORE than {fair_american(p_combo * 0.8):+.0f}")
    print("  (fair + a 25% cushion for the model knowing less than the market).")
    print("  Method models are weaker than the winner model — see report.md.")

    print("\nReality check: the market-EV lines above are what these tickets cost in")
    print("expectation. Parlays compound the vig on every leg — they are the most")
    print("expensive way to bet the same opinions. Entertainment pricing, not edge.")


if __name__ == "__main__":
    main()
