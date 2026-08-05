"""UFC predictor dashboard — picks, parlays, and bet log in the browser.

Reads the latest prediction snapshots saved by predict.py; the "regenerate"
button runs the pipeline. Run: streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from parlay import (COIN_FLIP, american_to_dec, dec_to_american, fair_american,
                    implied, no_vig_side)
from track_bets import BETS, implied as bet_implied, load as load_bets, profit

ROOT = Path(__file__).parent
PREDS = ROOT / "predictions"

st.set_page_config(page_title="UFC Predictor", page_icon="🥊", layout="wide")

BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"


def snapshots():
    files = sorted(PREDS.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    cards = {}
    for f in files:  # newest snapshot per card date
        card = f.name.split("_")[0]
        cards.setdefault(card, f)
    return cards


cards = snapshots()
st.sidebar.title("🥊 UFC Predictor")
if not cards:
    st.warning("No prediction snapshots yet — run `python predict.py` first.")
    st.stop()
card = st.sidebar.selectbox("Card", list(cards), format_func=lambda d: f"UFC card — {d}")
df = pd.read_csv(cards[card])
has_odds = df["fanduel_a"].notna() if "fanduel_a" in df else pd.Series(False, index=df.index)

is_local = (ROOT / "data" / "ufc-master.csv").exists()
if is_local and st.sidebar.button("Regenerate picks (2-3 min)"):
    with st.spinner("Replaying history, training, fetching lines..."):
        arg = [] if card == "next" else [f"--date={card.replace('-', '')}"]
        subprocess.run([sys.executable, "predict.py", "--refresh", *arg], cwd=ROOT)
    st.rerun()
if not is_local:
    st.sidebar.caption("Cloud mode: picks auto-update via GitHub Actions "
                       "(Thu + Sat mornings).")

st.sidebar.caption(
    "Model does **not** beat closing odds (see report.md). Probabilities are "
    "calibrated to ~±3%. Nothing here is betting advice.")

tab_picks, tab_parlay, tab_log = st.tabs(["Picks", "Parlay builder", "Bet log"])

# ------------------------------- Picks ---------------------------------------
with tab_picks:
    st.subheader(f"Card: {card}")
    rows = []
    for _, r in df.iterrows():
        pick_a = r["p_a"] >= 0.5
        p = r["p_a"] if pick_a else 1 - r["p_a"]
        pre = "a" if pick_a else "b"
        raw = [r[f"p_{pre}_ko"], r[f"p_{pre}_sub"], r[f"p_{pre}_dec"]]
        ko, sub, dec = [x * p / sum(raw) for x in raw]
        fd = ""
        if pd.notna(r.get("fanduel_a")):
            fd = f"{r['fanduel_a']:+.0f} / {r['fanduel_b']:+.0f}"
        rows.append({
            "Fight": f"{r['fighter_a']} vs {r['fighter_b']}",
            "Pick": r["fighter_a"] if pick_a else r["fighter_b"],
            "Win %": p,
            "KO/Sub/Dec": f"{ko:.0%} / {sub:.0%} / {dec:.0%}",
            "Distance": r["p_a_dec"] + r["p_b_dec"],
            "Exp. TD": r["exp_td"],
            "FanDuel (A/B)": fd,
        })
    st.dataframe(
        pd.DataFrame(rows),
        column_config={
            "Win %": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Distance": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Exp. TD": st.column_config.NumberColumn(format="%.1f"),
        },
        hide_index=True, use_container_width=True)

    edges = []
    for _, r in df[has_odds].iterrows():
        for side, other, o, oo, p in [
            (r["fighter_a"], r["fighter_b"], r["fanduel_a"], r["fanduel_b"], r["p_a"]),
            (r["fighter_b"], r["fighter_a"], r["fanduel_b"], r["fanduel_a"], 1 - r["p_a"]),
        ]:
            gap = p - no_vig_side(o, oo)
            if gap > 0.02:
                edges.append({"Side": side, "FanDuel": f"{o:+.0f}",
                              "Model": p, "Market (no-vig)": no_vig_side(o, oo),
                              "Gap": gap})
    if edges:
        st.subheader("Model vs FanDuel disagreements")
        st.dataframe(
            pd.DataFrame(edges).sort_values("Gap", ascending=False),
            column_config={c: st.column_config.NumberColumn(format="percent")
                           for c in ("Model", "Market (no-vig)", "Gap")},
            hide_index=True, use_container_width=True)
        st.caption("The market side of large gaps has historically been right more "
                   "often than the model side — treat these as questions, not answers.")

# --------------------------- Parlay builder ----------------------------------
with tab_parlay:
    c1, c2, c3 = st.columns(3)
    stake = c1.number_input("Stake ($)", 5.0, 1000.0, 25.0, 5.0)
    n_legs = c2.slider("Legs", 2, 6, 3)
    leg_type = c3.radio("Leg type", ["Moneyline", "Double chance (2 methods)",
                                     "Single method"], horizontal=False)

    pool = []
    main_idx = len(df) - 1
    for i, r in df.iterrows():
        if pd.isna(r.get("fanduel_a")):
            continue
        pick_a = r["p_a"] >= 0.5
        p = r["p_a"] if pick_a else 1 - r["p_a"]
        pre = "a" if pick_a else "b"
        raw = {"KO/TKO": r[f"p_{pre}_ko"], "Sub": r[f"p_{pre}_sub"], "Dec": r[f"p_{pre}_dec"]}
        scale = p / sum(raw.values())
        o = r["fanduel_a"] if pick_a else r["fanduel_b"]
        oo = r["fanduel_b"] if pick_a else r["fanduel_a"]
        pool.append({"side": r["fighter_a"] if pick_a else r["fighter_b"],
                     "p": p, "odds": o, "ev": p * american_to_dec(o) - 1,
                     "main": i == main_idx,
                     "methods": {k: v * scale for k, v in raw.items()}})
    eligible = [l for l in pool if l["p"] >= COIN_FLIP or l["main"]]
    legs = sorted(eligible, key=lambda l: l["ev"], reverse=True)[:n_legs]
    legs.sort(key=lambda l: l["p"], reverse=True)

    if len(legs) < 2:
        st.info("Fewer than 2 eligible legs with odds on this card.")
    else:
        out, p_hit, dec_odds, priced = [], 1.0, 1.0, True
        for l in legs:
            ranked = sorted(l["methods"].items(), key=lambda kv: kv[1], reverse=True)
            if leg_type == "Moneyline":
                label, p_leg = l["side"], l["p"]
                dec_odds *= american_to_dec(l["odds"])
            elif leg_type.startswith("Double"):
                label = f"{l['side']} by {ranked[0][0]} or {ranked[1][0]}"
                p_leg, priced = ranked[0][1] + ranked[1][1], False
            else:
                label = f"{l['side']} by {ranked[0][0]}"
                p_leg, priced = ranked[0][1], False
            p_hit *= p_leg
            out.append({"Leg": label, "Model": p_leg, "Fair odds": f"{fair_american(p_leg):+.0f}"})
        st.dataframe(pd.DataFrame(out),
                     column_config={"Model": st.column_config.NumberColumn(format="percent")},
                     hide_index=True, use_container_width=True)
        a, b, c = st.columns(3)
        a.metric("Model P(all hit)", f"{p_hit:.1%}")
        if priced:
            b.metric("FanDuel pays", f"{dec_to_american(dec_odds):+.0f}")
            c.metric(f"Payout on ${stake:.0f}", f"${stake * dec_odds:,.2f}")
            st.caption(f"Model EV ${stake * (p_hit * dec_odds - 1):+,.2f} — but parlays "
                       "compound the vig; at market probabilities every ticket is negative.")
        else:
            b.metric("Fair combined", f"{fair_american(p_hit):+.0f}")
            c.metric("Only take above", f"{fair_american(p_hit * 0.8):+.0f}")
            st.caption("Method odds aren't in the API — build this on FanDuel and only "
                       "take it if they pay MORE than the threshold (fair + 25% cushion).")

# ------------------------------ Bet log --------------------------------------
with tab_log:
    with st.form("add_bet"):
        st.write("Log a bet")
        f1, f2, f3, f4, f5 = st.columns([2, 1, 1, 1, 1])
        pick = f1.text_input("Pick")
        odds = f2.number_input("Odds", -2000, 2000, -150, 5)
        stake_b = f3.number_input("Stake", 1.0, 10000.0, 25.0, 5.0)
        result = f4.selectbox("Result", ["pending", "W", "L", "P"])
        close = f5.number_input("Closing odds (CLV)", -2000, 2000, 0, 5)
        if st.form_submit_button("Add") and pick:
            dfb = load_bets()
            row = {"id": (dfb["id"].max() + 1 if len(dfb) else 1),
                   "date": str(pd.Timestamp.today().date()), "event": card,
                   "pick": pick, "odds": odds, "stake": stake_b,
                   "result": "" if result == "pending" else result,
                   "close": close if close else np.nan}
            dfb = pd.DataFrame([row]) if dfb.empty else pd.concat(
                [dfb, pd.DataFrame([row])], ignore_index=True)
            dfb.to_csv(BETS, index=False)
            st.rerun()

    dfb = load_bets()
    if len(dfb):
        settled = dfb[dfb["result"].isin(["W", "L", "P"])].copy()
        if len(settled):
            settled["pl"] = [profit(o, s, r) for o, s, r in
                             zip(settled["odds"], settled["stake"], settled["result"])]
            pl, staked = settled["pl"].sum(), settled["stake"].sum()
            wins = (settled["result"] == "W").sum()
            decd = (settled["result"] != "P").sum()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P/L", f"${pl:+,.2f}")
            m2.metric("ROI", f"{pl / staked:+.1%}" if staked else "—")
            m3.metric("Record", f"{wins}-{decd - wins}")
            clv = settled.dropna(subset=["close"])
            if len(clv):
                edges_clv = [bet_implied(c) - bet_implied(o)
                             for o, c in zip(clv["odds"], clv["close"])]
                m4.metric("Avg CLV", f"{np.mean(edges_clv):+.1%}")
            if decd < 20:
                st.caption(f"{decd} settled bets — far too few to distinguish skill "
                           "from luck. Judge nothing before ~50.")
        st.dataframe(dfb, hide_index=True, use_container_width=True)
    else:
        st.caption("No bets logged yet.")
