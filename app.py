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

tab_picks, tab_parlay, tab_results, tab_log, tab_model = st.tabs(
    ["Picks", "Parlay builder", "Results", "Bet log", "Model report"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_results(card_date):
    """Completed-fight winners for a card date from ESPN. {frozenset: winner_name}"""
    import json as _json
    import urllib.request as _rq
    from build_features import norm_name as _norm
    url = ("https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
           f"?dates={card_date.replace('-', '')}")
    try:
        data = _json.loads(_rq.urlopen(url, timeout=30).read())
    except Exception:
        return {}
    winners = {}
    for ev in data.get("events", []):
        for c in ev.get("competitions", []):
            comps = c.get("competitors", [])
            if len(comps) != 2 or not any(x.get("winner") for x in comps):
                continue
            names = [x.get("athlete", {}).get("displayName", "") for x in comps]
            w = next(x for x in comps if x.get("winner"))
            winners[frozenset(_norm(n) for n in names)] = w.get("athlete", {}).get("displayName")
    return winners

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
    c1, c2 = st.columns(2)
    stake = c1.number_input("Stake ($)", 5.0, 1000.0, 25.0, 5.0)
    n_legs = c2.slider("Candidate legs", 2, 6, 4)

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

    if st.toggle("✨ Suggested parlay — model picks the best bet type per leg"):
        # Per fight, choose the most confident option that clears its floor:
        # single method >= 42%, double chance >= 60%, moneyline >= 62%,
        # fight-level distance/finish >= 62% (no fighter side taken at all).
        options = []
        for _, r in df.iterrows():
            pick_a = r["p_a"] >= 0.5
            p_win = r["p_a"] if pick_a else 1 - r["p_a"]
            side = r["fighter_a"] if pick_a else r["fighter_b"]
            fight = f"{r['fighter_a']} vs {r['fighter_b']}"
            pre = "a" if pick_a else "b"
            raw = {"KO/TKO": r[f"p_{pre}_ko"], "Sub": r[f"p_{pre}_sub"],
                   "Dec": r[f"p_{pre}_dec"]}
            sc = p_win / sum(raw.values())
            meth = sorted(((k, v * sc) for k, v in raw.items()),
                          key=lambda kv: kv[1], reverse=True)
            p_dist = r["p_a_dec"] + r["p_b_dec"]
            o = (r["fanduel_a"] if pick_a else r["fanduel_b"]) if pd.notna(r.get("fanduel_a")) else np.nan
            cands = []
            if meth[0][1] >= 0.42:
                cands.append((meth[0][1], f"{side} by {meth[0][0]}", "method", np.nan))
            if meth[0][1] + meth[1][1] >= 0.60:
                cands.append((meth[0][1] + meth[1][1],
                              f"{side} by {meth[0][0]} or {meth[1][0]}", "double chance", np.nan))
            if p_win >= 0.62:
                cands.append((p_win, side, "moneyline", o))
            if p_dist >= 0.62:
                cands.append((p_dist, f"Goes the distance: {fight}", "fight-level", np.nan))
            if 1 - p_dist >= 0.62:
                cands.append((1 - p_dist, f"Doesn't go the distance: {fight}", "fight-level", np.nan))
            if cands:
                options.append(max(cands))
        options.sort(reverse=True)
        chosen = options[:n_legs]
        if len(chosen) < 2:
            st.info("Not enough confident legs on this card for a suggested ticket.")
        else:
            p_hit, dec_known, fair_unpriced, rows_s = 1.0, 1.0, 1.0, []
            for p_leg, label, kind, o in chosen:
                p_hit *= p_leg
                if kind == "moneyline" and pd.notna(o):
                    dec_known *= american_to_dec(o)
                    price = f"FanDuel {o:+.0f}"
                else:
                    fair_unpriced /= p_leg
                    price = f"fair {fair_american(p_leg):+.0f} (price on FanDuel)"
                rows_s.append({"Leg": label, "Type": kind, "Model": p_leg, "Price": price})
            st.dataframe(pd.DataFrame(rows_s),
                         column_config={"Model": st.column_config.NumberColumn(format="percent")},
                         hide_index=True, use_container_width=True)
            a, b, c = st.columns(3)
            a.metric("Model P(all hit)", f"{p_hit:.1%}")
            fair_dec = dec_known * fair_unpriced
            if fair_unpriced == 1.0:
                b.metric("FanDuel pays", f"{dec_to_american(fair_dec):+.0f}")
                c.metric(f"Payout on ${stake:.0f}", f"${stake * fair_dec:,.2f}")
            else:
                b.metric("Fair combined", f"{dec_to_american(fair_dec):+.0f}")
                c.metric("Only take above",
                         f"{dec_to_american(dec_known * fair_unpriced / 0.8):+.0f}")
            st.caption("Legs chosen for confidence, not payout — method when the "
                       "finish is predictable, moneyline when only the winner is, "
                       "and fight-level (distance/finish) when the model trusts the "
                       "fight's shape more than either fighter. Same rule as always: "
                       "only take it if FanDuel pays above the threshold.")
        st.divider()

    with st.expander("🎯 Same-fight combo (SGP) calculator — e.g. Salkilld wins + inside the distance"):
        fight_names = [f"{r['fighter_a']} vs {r['fighter_b']}" for _, r in df.iterrows()]
        sel = st.selectbox("Fight", fight_names, index=len(fight_names) - 1)
        r = df.iloc[fight_names.index(sel)]
        p6 = np.array([r["p_a_ko"], r["p_a_sub"], r["p_a_dec"],
                       r["p_b_ko"], r["p_b_sub"], r["p_b_dec"]], dtype=float)
        p6 = p6 / p6.sum()
        s1, s2 = st.columns(2)
        winner = s1.selectbox("Winner", ["Either", r["fighter_a"], r["fighter_b"]])
        ending = s2.selectbox("How it ends", ["Any", "Inside the distance",
                                              "Goes the distance", "KO/TKO",
                                              "Submission", "Decision"])
        w_mask = {"Either": {0, 1, 2, 3, 4, 5}, r["fighter_a"]: {0, 1, 2},
                  r["fighter_b"]: {3, 4, 5}}[winner]
        e_mask = {"Any": {0, 1, 2, 3, 4, 5}, "Inside the distance": {0, 1, 3, 4},
                  "Goes the distance": {2, 5}, "KO/TKO": {0, 3},
                  "Submission": {1, 4}, "Decision": {2, 5}}[ending]
        mask = list(w_mask & e_mask)
        p_joint = float(p6[mask].sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Exact joint probability", f"{p_joint:.1%}")
        m2.metric("Fair odds", f"{fair_american(p_joint):+.0f}" if 0 < p_joint < 1 else "—")
        if winner != "Either" and ending != "Any":
            p_w = float(p6[list(w_mask)].sum())
            p_e = float(p6[list(e_mask)].sum())
            naive = p_w * p_e
            m3.metric("Naive (wrong) product", f"{naive:.1%}",
                      delta=f"{p_joint - naive:+.1%} correlation", delta_color="off")
        st.caption("Joint probability read directly off the 6-way outcome model — "
                   "no independence assumption. Within-fight legs are correlated, "
                   "which is exactly why books discount SGP payouts; compare "
                   "FanDuel's quoted price to the fair odds above (+25% cushion "
                   "still applies). Probabilities come from the method model, which "
                   "is weaker than the winner model — see the report.")

    if len(legs) < 2:
        st.info("Fewer than 2 eligible legs with odds on this card.")
    else:
        st.caption("Pick a type per leg — mix moneyline and method legs freely. "
                   "'Skip' drops a leg from the ticket.")
        out, p_hit, dec_known, fair_unpriced = [], 1.0, 1.0, 1.0
        n_used = 0
        for i, l in enumerate(legs):
            ranked = sorted(l["methods"].items(), key=lambda kv: kv[1], reverse=True)
            lc1, lc2 = st.columns([2, 1])
            lc1.markdown(f"**{l['side']}**  ({l['odds']:+.0f}, model {l['p']:.0%})"
                         + ("  · main event" if l["main"] else ""))
            choice = lc2.selectbox(
                "Leg type", ["Moneyline", "Double chance", "Single method", "Skip"],
                key=f"leg_{i}", label_visibility="collapsed")
            if choice == "Skip":
                continue
            n_used += 1
            if choice == "Moneyline":
                label, p_leg = l["side"], l["p"]
                dec_known *= american_to_dec(l["odds"])
                price = f"FanDuel {l['odds']:+.0f}"
            elif choice == "Double chance":
                label = f"{l['side']} by {ranked[0][0]} or {ranked[1][0]}"
                p_leg = ranked[0][1] + ranked[1][1]
                fair_unpriced /= p_leg
                price = f"fair {fair_american(p_leg):+.0f} (price on FanDuel)"
            else:
                label = f"{l['side']} by {ranked[0][0]}"
                p_leg = ranked[0][1]
                fair_unpriced /= p_leg
                price = f"fair {fair_american(p_leg):+.0f} (price on FanDuel)"
            p_hit *= p_leg
            out.append({"Leg": label, "Model": p_leg, "Price": price})
        if n_used < 2:
            st.info("Keep at least 2 legs on the ticket.")
        else:
            st.dataframe(pd.DataFrame(out),
                         column_config={"Model": st.column_config.NumberColumn(format="percent")},
                         hide_index=True, use_container_width=True)
            fully_priced = fair_unpriced == 1.0
            a, b, c = st.columns(3)
            a.metric("Model P(all hit)", f"{p_hit:.1%}")
            if fully_priced:
                b.metric("FanDuel pays", f"{dec_to_american(dec_known):+.0f}")
                c.metric(f"Payout on ${stake:.0f}", f"${stake * dec_known:,.2f}")
                st.caption(f"Model EV ${stake * (p_hit * dec_known - 1):+,.2f} — but "
                           "parlays compound the vig; at market probabilities every "
                           "ticket is negative.")
            else:
                fair_dec = dec_known * fair_unpriced
                need_dec = dec_known * fair_unpriced / 0.8
                b.metric("Fair combined", f"{dec_to_american(fair_dec):+.0f}")
                c.metric("Only take above", f"{dec_to_american(need_dec):+.0f}")
                st.caption("Ticket includes method legs FanDuel doesn't price via the "
                           "API — build it in the app and only take it if the quoted "
                           "payout beats the threshold (fair + 25% cushion on the "
                           "model-priced legs).")

# ------------------------------ Results --------------------------------------
with tab_results:
    from build_features import norm_name as _nn
    today = str(pd.Timestamp.today().date())
    past_cards = [c for c in cards if c != "next" and c < today]
    if not past_cards:
        st.info("No completed cards yet — grading appears here automatically "
                "the day after each event.")
    for pc in past_cards:
        winners = fetch_results(pc)
        snap = pd.read_csv(cards[pc])
        graded, briers, mkt_briers, mkt_hits = [], [], [], []
        for _, r in snap.iterrows():
            key = frozenset((_nn(r["fighter_a"]), _nn(r["fighter_b"])))
            if key not in winners:
                continue
            actual = winners[key]
            a_won = _nn(actual) == _nn(r["fighter_a"])
            pick_a = r["p_a"] >= 0.5
            p_pick = r["p_a"] if pick_a else 1 - r["p_a"]
            hit = pick_a == a_won
            brier = (r["p_a"] - (1 if a_won else 0)) ** 2
            briers.append(brier)
            row = {"Fight": f"{r['fighter_a']} vs {r['fighter_b']}",
                   "Model pick": f"{r['fighter_a'] if pick_a else r['fighter_b']} ({p_pick:.0%})",
                   "Winner": actual, "Hit": "✅" if hit else "❌",
                   "Brier": round(brier, 3)}
            if pd.notna(r.get("fanduel_a")):
                p_mkt_a = no_vig_side(r["fanduel_a"], r["fanduel_b"])
                mkt_briers.append((p_mkt_a - (1 if a_won else 0)) ** 2)
                mkt_hits.append((p_mkt_a >= 0.5) == a_won)
            graded.append(row)
        if not graded:
            st.info(f"{pc}: results not posted yet.")
            continue
        st.subheader(f"Card {pc} — graded")
        st.dataframe(pd.DataFrame(graded), hide_index=True, use_container_width=True)
        g1, g2, g3 = st.columns(3)
        hits = sum(1 for g in graded if g["Hit"] == "✅")
        g1.metric("Model record", f"{hits}/{len(graded)}")
        g2.metric("Model Brier (lower = better)", f"{np.mean(briers):.3f}")
        if mkt_briers:
            g3.metric("FanDuel Brier (same fights)", f"{np.mean(mkt_briers):.3f}",
                      delta=f"market {sum(mkt_hits)}/{len(mkt_hits)} picks", delta_color="off")
        st.caption("One card proves nothing either way — the season-long Brier "
                   "comparison is the scoreboard that matters.")

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

# ---------------------------- Model report -----------------------------------
with tab_model:
    report_path = ROOT / "report.md"
    if report_path.exists():
        for plot in ("reliability.png", "importance.png"):
            p = ROOT / "plots" / plot
            if p.exists():
                st.image(str(p))
        st.markdown(report_path.read_text(encoding="utf-8")
                    .replace("![Reliability diagram](plots/reliability.png)", "")
                    .replace("![Permutation importance](plots/importance.png)", ""))
    else:
        st.info("report.md not found — run train_report.py.")

with tab_log:
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
