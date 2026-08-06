"""Personal bet log with an honest report. No recommendations — just the truth.

The one stat that predicts long-run profitability is closing-line value (CLV):
did you consistently get better odds than the closing line? Record the closing
odds when you can and the report will tell you.

Usage:
  python track_bets.py add --pick "Sean O'Malley" --odds -150 --stake 50 \
      --event "UFC 320" [--date 2026-06-28] [--result W|L|P] [--close -170]
  python track_bets.py result <bet_id> W|L|P     # settle a pending bet
  python track_bets.py report

Bets are stored in bets.csv (gitignored — it's personal data).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BETS = Path(__file__).parent / "bets.csv"
COLS = ["id", "date", "event", "pick", "odds", "stake", "result", "close"]


def load():
    if BETS.exists():
        return pd.read_csv(BETS)
    return pd.DataFrame(columns=COLS)


def profit(odds, stake, result):
    if result == "W":
        return stake * (odds / 100 if odds > 0 else 100 / -odds)
    if result == "L":
        return -stake
    return 0.0  # push / pending


def implied(odds):
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def snapshot_line(pick_name):
    """Reference line for a pick from the newest snapshot that priced them.

    Not a true closing line — it's the Thursday/Saturday-morning FanDuel line
    from the last prediction snapshot. Label accordingly wherever shown.
    """
    from build_features import norm_name
    key = norm_name(pick_name)
    preds = Path(__file__).parent / "predictions"
    for f in sorted(preds.glob("*.csv"), key=lambda p: p.name, reverse=True):
        df = pd.read_csv(f)
        if "fanduel_a" not in df.columns:
            continue
        for _, r in df.iterrows():
            if norm_name(r["fighter_a"]) == key and pd.notna(r["fanduel_a"]):
                return float(r["fanduel_a"])
            if norm_name(r["fighter_b"]) == key and pd.notna(r["fanduel_b"]):
                return float(r["fanduel_b"])
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--pick", required=True)
    a.add_argument("--odds", type=float, required=True, help="American odds you took")
    a.add_argument("--stake", type=float, required=True)
    a.add_argument("--event", default="")
    a.add_argument("--date", default=str(pd.Timestamp.today().date()))
    a.add_argument("--result", choices=["W", "L", "P"], default="")
    a.add_argument("--close", type=float, default=np.nan,
                   help="closing odds on your side (for CLV)")
    r = sub.add_parser("result")
    r.add_argument("bet_id", type=int)
    r.add_argument("outcome", choices=["W", "L", "P"])
    sub.add_parser("report")
    args = ap.parse_args()

    df = load()
    if args.cmd == "add":
        row = {"id": (df["id"].max() + 1 if len(df) else 1), "date": args.date,
               "event": args.event, "pick": args.pick, "odds": args.odds,
               "stake": args.stake, "result": args.result, "close": args.close}
        df = pd.DataFrame([row]) if df.empty else pd.concat(
            [df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(BETS, index=False)
        print(f"Logged bet #{row['id']}: {args.pick} at {args.odds:+.0f} for ${args.stake:.2f}")
        return
    if args.cmd == "result":
        if args.bet_id not in df["id"].values:
            sys.exit(f"No bet #{args.bet_id}")
        df.loc[df["id"] == args.bet_id, "result"] = args.outcome
        df.to_csv(BETS, index=False)
        print(f"Bet #{args.bet_id} settled: {args.outcome}")
        return

    # ---- report ----
    if not len(df):
        sys.exit("No bets logged yet.")
    settled = df[df["result"].isin(["W", "L", "P"])].copy()
    pending = df[~df["result"].isin(["W", "L", "P"])]
    settled["pl"] = [profit(o, s, r) for o, s, r in
                     zip(settled["odds"], settled["stake"], settled["result"])]
    staked = settled["stake"].sum()
    pl = settled["pl"].sum()
    wins = (settled["result"] == "W").sum()
    dec = (settled["result"] != "P").sum()
    print(f"Bets: {len(df)} logged, {len(settled)} settled, {len(pending)} pending")
    print(f"Staked: ${staked:.2f}   P/L: ${pl:+.2f}   ROI: {pl/staked:+.1%}" if staked else "")
    if dec:
        avg_imp = np.mean([implied(o) for o in settled.loc[settled['result'] != 'P', 'odds']])
        print(f"Record: {wins}-{dec-wins}  ({wins/dec:.0%} vs {avg_imp:.0%} break-even at your avg odds)")
    settled["ref_close"] = settled["close"]
    fill = settled["ref_close"].isna()
    settled.loc[fill, "ref_close"] = [snapshot_line(p) for p in settled.loc[fill, "pick"]]
    n_filled = int(fill.sum() - settled.loc[fill, "ref_close"].isna().sum())
    clv = settled.dropna(subset=["ref_close"])
    if len(clv):
        edges = [implied(c) - implied(o) for o, c in zip(clv["odds"], clv["ref_close"])]
        note = f", {n_filled} auto-filled from last snapshot line" if n_filled else ""
        print(f"CLV: {np.mean(edges):+.1%} avg ({len(clv)} bets{note}) — "
              f"{'you are beating the close (the good sign)' if np.mean(edges) > 0 else 'the close beats you (the market moved against your picks)'}")
    else:
        print("CLV: no closing odds recorded yet — add --close to future bets; it's the "
              "one number that predicts whether this stays profitable.")
    if dec >= 20:
        se = np.std(settled.loc[settled["result"] != "P", "pl"]) / np.sqrt(dec)
        print(f"Luck check: P/L per bet {pl/dec:+.2f} ± {1.96*se:.2f} (95% CI). "
              f"{'CI includes zero — results so far are indistinguishable from luck.' if abs(pl/dec) < 1.96*se else 'CI excludes zero.'}")
    else:
        print(f"Luck check: {dec} settled bets is far too few to distinguish skill "
              f"from luck — judge nothing before ~50.")


if __name__ == "__main__":
    main()
