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
from track_bets import (BETS, implied as bet_implied, load as load_bets, profit,
                        snapshot_line)

ROOT = Path(__file__).parent
PREDS = ROOT / "predictions"

st.set_page_config(page_title="UFC Predictor", page_icon="🥊", layout="wide")

BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"


def snapshots():
    # sort by filename (card_YYYYMMDD_HHMM.csv) — mtime is useless on cloud,
    # where a fresh clone stamps every file with the same time
    files = sorted(PREDS.glob("*.csv"), key=lambda p: p.name, reverse=True)
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
    st.write("")
    for i, r in df.iloc[::-1].iterrows():  # main event first
        pick_a = r["p_a"] >= 0.5
        p = r["p_a"] if pick_a else 1 - r["p_a"]
        pre = "a" if pick_a else "b"
        raw = [r[f"p_{pre}_ko"], r[f"p_{pre}_sub"], r[f"p_{pre}_dec"]]
        ko, sub, dec = [x * p / sum(raw) for x in raw]
        dist = r["p_a_dec"] + r["p_b_dec"]
        pick_name = r["fighter_a"] if pick_a else r["fighter_b"]

        def price_bits(side):
            bits = []
            if str(r.get(f"rec_{side}", "")) not in ("", "nan"):
                bits.append(f"UFC record **{r[f'rec_{side}']}**")
            fd = r.get(f"fanduel_{side}")
            if pd.notna(fd):
                bits.append(f"FanDuel {fd:+.0f}")
            best, bk = r.get(f"best_{side}"), r.get(f"best_{side}_book")
            if pd.notna(best) and str(bk) != "fanduel" and (
                    pd.isna(fd) or american_to_dec(best) > american_to_dec(fd) * 1.01):
                bits.append(f"best {best:+.0f} @ {bk}")
            return bits

        ca, cm, cb = st.columns([3, 4, 3])
        with ca:
            mark = " ✅" if pick_a else ""
            st.markdown(f"### {r['fighter_a']}{mark}")
            bits = price_bits("a")
            st.markdown(" · ".join(bits) if bits else "&nbsp;")
        with cb:
            mark = "" if pick_a else " ✅"
            st.markdown(f"### {r['fighter_b']}{mark}")
            bits = price_bits("b")
            st.markdown(" · ".join(bits) if bits else "&nbsp;")
        with cm:
            st.markdown(f"<div style='text-align:center'>"
                        f"<span style='font-size:1.4em'>Pick: <b>{pick_name}</b> "
                        f"<b>{p:.0%}</b></span></div>", unsafe_allow_html=True)
            st.progress(min(max(p, 0.0), 1.0))
            st.markdown(f"<div style='text-align:center;color:#898781'>"
                        f"KO {ko:.0%} &nbsp;·&nbsp; Sub {sub:.0%} &nbsp;·&nbsp; "
                        f"Dec {dec:.0%} &nbsp;&nbsp;|&nbsp;&nbsp; distance {dist:.0%}"
                        f" &nbsp;&nbsp;|&nbsp;&nbsp; exp. takedowns {r['exp_td']:.1f}"
                        f"</div>", unsafe_allow_html=True)
            why = str(r.get("why", ""))
            if why and why != "nan":
                st.markdown(f"<div style='text-align:center;color:#898781'>"
                            f"<i>edge: {why}</i></div>", unsafe_allow_html=True)
            if pd.notna(r.get("total_point")):
                tb = []
                if pd.notna(r.get("fd_over")):
                    tb.append(f"FanDuel O {r['fd_over']:+.0f} / U {r['fd_under']:+.0f}")
                if pd.notna(r.get("best_under")):
                    tb.append(f"best U {r['best_under']:+.0f} @ {r['best_under_book']}")
                if pd.notna(r.get("best_over")):
                    tb.append(f"best O {r['best_over']:+.0f} @ {r['best_over_book']}")
                st.markdown(f"<div style='text-align:center;color:#898781'>"
                            f"Round total {r['total_point']}: {' · '.join(tb)}"
                            f"</div>", unsafe_allow_html=True)
        st.divider()

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
    with st.expander("📈 Line movement — FanDuel across snapshots"):
        mov_files = sorted(PREDS.glob(f"{card}_*.csv"), key=lambda p: p.name)
        if len(mov_files) < 2:
            st.caption("Movement appears once there are two or more snapshots for "
                       "this card (they accumulate Thu/Fri/Sat).")
        else:
            frames = [(f.name.split("_", 1)[1][:-4], pd.read_csv(f)) for f in mov_files]
            rows_m = []
            for _, r in df.iterrows():
                pick_a = r["p_a"] >= 0.5
                side = "a" if pick_a else "b"
                series = {}
                for tlabel, fr in frames:
                    m = fr[(fr["fighter_a"] == r["fighter_a"])
                           & (fr["fighter_b"] == r["fighter_b"])]
                    if len(m) and f"fanduel_{side}" in m.columns:
                        v = m.iloc[0][f"fanduel_{side}"]
                        if pd.notna(v):
                            series[tlabel] = v
                if len(series) >= 2:
                    vals = list(series.values())
                    d0, d1 = bet_implied(vals[0]), bet_implied(vals[-1])
                    steam = ("🟢 toward model pick" if d1 > d0 + 0.005 else
                             "🔴 against model pick" if d1 < d0 - 0.005 else "flat")
                    rows_m.append({"Pick": r["fighter_a"] if pick_a else r["fighter_b"],
                                   **{k: f"{v:+.0f}" for k, v in series.items()},
                                   "Steam": steam})
            if rows_m:
                st.dataframe(pd.DataFrame(rows_m), hide_index=True, width="stretch")
                st.caption("Steam toward the model's side = the market agreeing "
                           "late; against = the market learning something the "
                           "stats can't see. Neither is proof — both are signal.")
            else:
                st.caption("No overlapping priced fights across snapshots yet.")

    if edges:
        st.subheader("Model vs FanDuel disagreements")
        st.dataframe(
            pd.DataFrame(edges).sort_values("Gap", ascending=False),
            column_config={c: st.column_config.NumberColumn(format="percent")
                           for c in ("Model", "Market (no-vig)", "Gap")},
            hide_index=True, width="stretch")
        st.caption("The market side of large gaps has historically been right more "
                   "often than the model side — treat these as questions, not answers.")

# --------------------------- Parlay builder ----------------------------------
with tab_parlay:
    _book = load_bets()
    _pend = _book[~_book["result"].isin(["W", "L", "P"])] if len(_book) else _book
    if len(_pend):
        st.markdown("#### 📌 Your book — placed and pending")
        for _, b in _pend.iterrows():
            tag = b.get("tag", "model") if "tag" in _pend.columns else "model"
            tag = tag if str(tag) != "nan" else "model"
            payout = b["stake"] * american_to_dec(b["odds"])
            st.markdown(f"- **{b['pick']}** — ${b['stake']:.0f} at {b['odds']:+.0f} "
                        f"→ ${payout:,.2f} if it hits · `{tag}`")
        st.caption("Frozen at placement — the suggestions below keep moving with "
                   "prices and snapshots; your book doesn't.")
        st.divider()
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

    if st.toggle("🎯 The Play — the model's fight-night plan", value=True):
        target = st.number_input(
            "Goal mode — profit target ($). 0 = the standard two-ticket plan.",
            0.0, 100000.0, 0.0, 25.0,
            help="Set a target and it builds the cheapest-probability single "
                 "ticket that reaches it. Leave 0 for the main + side plan.")
        # Books price props flatter than our model (observed 2026-08-08 card:
        # our 76%/71% distance calls priced as ~65%/64%). Estimate their price
        # by shrinking our probability toward 50%, minus a small vig cut.
        est_prop_dec = lambda p: 0.95 / (0.5 + (p - 0.5) * 0.55)

        def build_cands(TH):
            """One candidate leg per fight, type chosen by confidence structure."""
            out = []
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
                ml_dec = american_to_dec(o) if pd.notna(o) else np.nan
                why = str(r.get("why", "")) if "why" in df.columns else ""
                leg = None
                if p_win >= 0.62:
                    m1_share = meth[0][1] / p_win
                    if pd.notna(ml_dec) and p_win * ml_dec >= 1.05:
                        leg = (p_win, side, "moneyline", ml_dec, f"FanDuel {o:+.0f}",
                               f"FanDuel's price is generous vs the model — edge: {why}", fight)
                    elif m1_share >= TH["m_share"] and meth[0][1] >= TH["m_min"]:
                        p_leg = meth[0][1]
                        leg = (p_leg, f"{side} by {meth[0][0]}", "method",
                               est_prop_dec(p_leg),
                               f"fair {fair_american(p_leg):+.0f} (price on FanDuel)",
                               f"{m1_share:.0%} of his win paths end this way — edge: {why}", fight)
                    elif (meth[0][1] + meth[1][1]) / p_win >= TH["dc_share"]:
                        p_leg = meth[0][1] + meth[1][1]
                        leg = (p_leg, f"{side} by {meth[0][0]} or {meth[1][0]}", "double chance",
                               est_prop_dec(p_leg),
                               f"fair {fair_american(p_leg):+.0f} (price on FanDuel)",
                               f"two live finish paths cover {(p_leg / p_win):.0%} of his wins — edge: {why}", fight)
                    elif pd.notna(ml_dec):
                        leg = (p_win, side, "moneyline", ml_dec, f"FanDuel {o:+.0f}",
                               f"clear pick, no dominant method — edge: {why}", fight)
                elif max(p_dist, 1 - p_dist) >= TH["shape"]:
                    # Real-priced rounds leg (Under/Over 2.5) beats an estimated
                    # distance leg when FanDuel has posted the total.
                    pu = r.get("p_u25")
                    if (pd.notna(pu) and pd.notna(r.get("fd_under"))
                            and max(pu, 1 - pu) >= TH["shape"] - 0.04):
                        under = pu >= 0.5
                        p_leg = pu if under else 1 - pu
                        o_r = r["fd_under"] if under else r["fd_over"]
                        leg = (p_leg, f"{'Under' if under else 'Over'} 2.5 rounds: {fight}",
                               "rounds", american_to_dec(o_r),
                               f"FanDuel {o_r:+.0f} ({'U' if under else 'O'}2.5)",
                               "winner too close to call — betting fight length at a real price", fight)
                    else:
                        p_leg = max(p_dist, 1 - p_dist)
                        lbl = ("Goes the distance: " if p_dist >= 0.5 else
                               "Doesn't go the distance: ") + fight
                        leg = (p_leg, lbl, "fight-level", est_prop_dec(p_leg),
                               f"fair {fair_american(p_leg):+.0f} (price on FanDuel)",
                               "winner too close to call — betting the fight's shape, not a side", fight)
                if leg:
                    out.append(leg)
            return out

        def render_ticket(title, chosen, stake_t, note):
            if not chosen:
                st.info(f"{title}: not enough eligible legs on this card.")
                return
            p_hit = float(np.prod([c[0] for c in chosen]))
            est_dec = float(np.prod([c[3] for c in chosen]))
            st.markdown(f"**{title}**")
            st.dataframe(pd.DataFrame(
                [{"Leg": c[1], "Type": c[2], "Model": c[0],
                  "Price": c[4], "Why this leg": c[5]} for c in chosen]),
                column_config={"Model": st.column_config.NumberColumn(format="percent")},
                hide_index=True, width="stretch")
            a, b, c = st.columns(3)
            a.metric("Model P(hit)", f"{p_hit:.1%}")
            b.metric("Est. payout", f"{dec_to_american(est_dec):+.0f}",
                     help="Moneyline legs at FanDuel's real price; other legs use "
                          "a market-anchored estimate. Their quote decides.")
            c.metric(f"Est. ${stake_t:.0f} returns", f"${stake_t * est_dec:,.2f}")
            if note:
                st.caption(note)
            return p_hit, est_dec

        def variant_pool():
            """Every leg variant for every fight (ML, method, DC, distance sides,
            real-priced rounds) — the full menu the optimizers choose from."""
            out = []
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
                why = str(r.get("why", "")) if "why" in df.columns else ""
                if pd.notna(o) and p_win >= 0.55:
                    out.append((p_win, side, "moneyline", american_to_dec(o),
                               f"FanDuel {o:+.0f}", f"edge: {why}", fight))
                if meth[0][1] >= 0.35:
                    p_leg = meth[0][1]
                    out.append((p_leg, f"{side} by {meth[0][0]}", "method",
                               est_prop_dec(p_leg),
                               f"fair {fair_american(p_leg):+.0f} (price on FanDuel)",
                               f"{meth[0][1]/p_win:.0%} of his win paths — edge: {why}", fight))
                p_dc = meth[0][1] + meth[1][1]
                if p_dc >= 0.50:
                    out.append((p_dc, f"{side} by {meth[0][0]} or {meth[1][0]}", "double chance",
                               est_prop_dec(p_dc),
                               f"fair {fair_american(p_dc):+.0f} (price on FanDuel)",
                               f"two live paths cover {p_dc/p_win:.0%} of his wins — edge: {why}", fight))
                for p_leg, lbl in [(p_dist, "Goes the distance: " + fight),
                                   (1 - p_dist, "Doesn't go the distance: " + fight)]:
                    if p_leg >= 0.55:
                        out.append((p_leg, lbl, "fight-level", est_prop_dec(p_leg),
                                   f"fair {fair_american(p_leg):+.0f} (price on FanDuel)",
                                   "betting the fight's shape, not a side", fight))
                pu = r.get("p_u25")
                if pd.notna(pu) and pd.notna(r.get("fd_under")):
                    for p_leg, o_r, tag in [(pu, r["fd_under"], "Under"),
                                            (1 - pu, r["fd_over"], "Over")]:
                        if p_leg >= 0.55:
                            out.append((p_leg, f"{tag} 2.5 rounds: {fight}", "rounds",
                                       american_to_dec(o_r),
                                       f"FanDuel {o_r:+.0f} ({tag[0]}2.5)",
                                       "fight-length bet at a real price", fight))
            return out

        if target > 0:
            gc = variant_pool()
            required = (stake + target) / stake
            gc.sort(key=lambda t: np.log(t[0]) / np.log(t[3]), reverse=True)
            chosen, est_prod, p_hit, used_f = [], 1.0, 1.0, set()
            for c_leg in gc:
                if est_prod >= required or len(chosen) >= 6:
                    break
                if c_leg[6] in used_f:
                    continue
                chosen.append(c_leg)
                p_hit *= c_leg[0]
                est_prod *= c_leg[3]
                used_f.add(c_leg[6])
            note = (f"Turning ${stake:.0f} into ${stake + target:.0f} needs about "
                    f"{dec_to_american(required):+.0f}; the cheapest route hits "
                    f"{p_hit:.0%} of the time. The target sets the risk — the honest "
                    f"levers are a smaller target or bigger stake, never worse legs."
                    if est_prod >= required else
                    f"This card can't honestly reach ${target:.0f} on ${stake:.0f} — "
                    f"the best six legs top out near {dec_to_american(est_prod):+.0f}. "
                    f"Lower the target or raise the stake.")
            render_ticket(f"Goal ticket — ${stake:.0f} chasing ${target:.0f} profit",
                          chosen, stake, note)
        if target > 0:
            st.divider()
        # The standing two-ticket plan always renders; the goal ticket
        # (when a target is set) appears above it as an extra view.
        s1, s2 = st.columns(2)
        main_stake = s1.number_input("Main ticket stake ($)", 25.0, 500.0, 75.0, 25.0)
        side_stake = s2.number_input("Side ticket stake ($)", 5.0, 100.0, 25.0, 5.0)
        # Main ticket: 2-3 chunkiest convictions, methods preferred.
        mc = build_cands(dict(m_share=0.50, m_min=0.40, dc_share=0.80, shape=0.64))
        mc.sort(key=lambda t: t[0], reverse=True)
        main, p = [], 1.0
        for c_leg in mc:
            if len(main) >= 3 or (len(main) >= 2 and p * c_leg[0] < 0.25):
                break
            main.append(c_leg)
            p *= c_leg[0]
        res_main = render_ticket("Main ticket — the convictions, with their methods",
                                 main, main_stake, None)
        st.divider()
        # Side ticket: the stretch play — best-EV leg per fight from the full
        # variant pool (real prices beat estimates by construction), 4 legs max.
        best_by_fight = {}
        for c_leg in variant_pool():
            if c_leg[0] < 0.5:
                continue  # no underdog legs on tickets
            cur = best_by_fight.get(c_leg[6])
            if cur is None or c_leg[0] * c_leg[3] > cur[0] * cur[3]:
                best_by_fight[c_leg[6]] = c_leg
        sc_ = sorted(best_by_fight.values(), key=lambda t: t[0] * t[3], reverse=True)
        side, p = [], 1.0
        for c_leg in sc_:
            if len(side) >= 4:
                break
            if p * c_leg[0] >= 0.15 or len(side) < 3:
                side.append(c_leg)
                p *= c_leg[0]
        res_side = render_ticket("Side ticket — the stretch play", side, side_stake,
                                 "Best-EV leg per fight, real prices first, four legs max — "
                                 "the same logic that built the analysis-winning tickets. "
                                 "Shares the strongest legs with the main on purpose. Price "
                                 "each on FanDuel against its estimate; below it, pass.")
        if res_main and res_side and main and side:
            # exact joint across the union of fights (legs matched by fight)
            mmap = {c[6]: c for c in main}
            smap = {c[6]: c for c in side}
            p_both = 1.0
            for f_id in set(mmap) | set(smap):
                if f_id in mmap and f_id in smap:
                    a_leg, b_leg = mmap[f_id], smap[f_id]
                    p_both *= (a_leg[0] if a_leg[1] == b_leg[1]
                               else min(a_leg[0], b_leg[0]) * 0.9)
                else:
                    p_both *= (mmap.get(f_id) or smap[f_id])[0]
            pm, dm = res_main
            ps, ds = res_side
            st.divider()
            st.markdown("**Portfolio readout — the night's four outcomes**")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Both tickets hit", f"{p_both:.0%}",
                      delta=f"+${main_stake*(dm-1)+side_stake*(ds-1):,.0f}",
                      delta_color="off")
            q2.metric("Main only", f"{max(pm - p_both, 0):.0%}",
                      delta=f"+${main_stake*(dm-1)-side_stake:,.0f}", delta_color="off")
            q3.metric("Side only", f"{max(ps - p_both, 0):.0%}",
                      delta=f"+${side_stake*(ds-1)-main_stake:,.0f}", delta_color="off")
            q4.metric("Both lose", f"{max(1 - pm - ps + p_both, 0):.0%}",
                      delta=f"-${main_stake+side_stake:,.0f}", delta_color="off")
            budget = main_stake + side_stake

            def qkelly(p_t, d_t):
                b = d_t - 1
                return max((p_t * d_t - 1) / b, 0) / 4 if b > 0 else 0
            km, ks = qkelly(pm, dm), qkelly(ps, ds)
            if km + ks > 0:
                am = budget * km / (km + ks)
                st.caption(f"Quarter-Kelly split of your ${budget:.0f} (per model "
                           f"estimates): main ${am:.0f} / side ${budget - am:.0f}. "
                           f"Suggestion only — Kelly assumes the model's "
                           f"probabilities are exactly right, and they aren't.")
            else:
                st.caption(f"At current estimated prices neither ticket is +EV per "
                           f"the model — quarter-Kelly says $0/$0. Whatever you "
                           f"stake is entertainment budget; size it accordingly.")
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
                         hide_index=True, width="stretch")
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
def grade_card(snap, winners):
    """Grade one snapshot against ESPN winners. Returns (rows, briers, mkt_briers, mkt_hits)."""
    from build_features import norm_name as _nn
    graded, briers, mkt_briers, mkt_hits = [], [], [], []
    for _, r in snap.iterrows():
        key = frozenset((_nn(r["fighter_a"]), _nn(r["fighter_b"])))
        if key not in winners:
            continue
        actual = winners[key]
        a_won = _nn(actual) == _nn(r["fighter_a"])
        pick_a = r["p_a"] >= 0.5
        p_pick = r["p_a"] if pick_a else 1 - r["p_a"]
        briers.append((r["p_a"] - (1 if a_won else 0)) ** 2)
        if pd.notna(r.get("fanduel_a")):
            p_mkt_a = no_vig_side(r["fanduel_a"], r["fanduel_b"])
            mkt_briers.append((p_mkt_a - (1 if a_won else 0)) ** 2)
            mkt_hits.append((p_mkt_a >= 0.5) == a_won)
        graded.append({"Fight": f"{r['fighter_a']} vs {r['fighter_b']}",
                       "Model pick": f"{r['fighter_a'] if pick_a else r['fighter_b']} ({p_pick:.0%})",
                       "Winner": actual, "Hit": "✅" if pick_a == a_won else "❌",
                       "Brier": round(briers[-1], 3)})
    return graded, briers, mkt_briers, mkt_hits


with tab_results:
    today = str(pd.Timestamp.today().date())
    past_cards = sorted([c for c in cards if c != "next" and c < today])
    if not past_cards:
        st.info("No completed cards yet — grading appears here automatically "
                "the day after each event.")
    season = {"hits": 0, "n": 0, "briers": [], "mkt_briers": [], "mkt_hits": 0, "mkt_n": 0}
    for pc in past_cards:
        winners = fetch_results(pc)
        graded, briers, mkt_briers, mkt_hits = grade_card(pd.read_csv(cards[pc]), winners)
        if not graded:
            st.info(f"{pc}: results not posted yet.")
            continue
        season["hits"] += sum(1 for g in graded if g["Hit"] == "✅")
        season["n"] += len(graded)
        season["briers"] += briers
        season["mkt_briers"] += mkt_briers
        season["mkt_hits"] += sum(mkt_hits)
        season["mkt_n"] += len(mkt_hits)
        with st.expander(f"Card {pc} — graded", expanded=(pc == past_cards[-1])):
            st.dataframe(pd.DataFrame(graded), hide_index=True, width="stretch")
            g1, g2, g3 = st.columns(3)
            g1.metric("Model record", f"{sum(1 for g in graded if g['Hit'] == '✅')}/{len(graded)}")
            g2.metric("Model Brier (lower = better)", f"{np.mean(briers):.3f}")
            if mkt_briers:
                g3.metric("FanDuel Brier (same fights)", f"{np.mean(mkt_briers):.3f}",
                          delta=f"market {sum(mkt_hits)}/{len(mkt_hits)} picks", delta_color="off")
    if season["n"]:
        st.subheader("📊 Season scoreboard — every graded fight")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Model record", f"{season['hits']}/{season['n']}"
                  f" ({season['hits']/season['n']:.0%})")
        s2.metric("Model Brier", f"{np.mean(season['briers']):.3f}")
        if season["mkt_n"]:
            s3.metric("FanDuel record (same fights)",
                      f"{season['mkt_hits']}/{season['mkt_n']}"
                      f" ({season['mkt_hits']/season['mkt_n']:.0%})")
            s4.metric("FanDuel Brier", f"{np.mean(season['mkt_briers']):.3f}")
            lead = np.mean(season["mkt_briers"]) - np.mean(season["briers"])
            st.caption(f"Brier difference (market − model): {lead:+.4f} — "
                       f"{'model ahead' if lead > 0 else 'market ahead'} so far. "
                       "One card proves nothing; this number matters after ~10 cards.")

# ------------------------------ Bet log --------------------------------------
with tab_log:
    with st.form("add_bet"):
        st.write("Log a bet")
        f1, f2, f3, f4, f5, f6 = st.columns([2, 1, 1, 1, 1, 1])
        pick = f1.text_input("Pick")
        odds = f2.number_input("Odds", -2000, 2000, -150, 5)
        stake_b = f3.number_input("Stake", 1.0, 10000.0, 25.0, 5.0)
        result = f4.selectbox("Result", ["pending", "W", "L", "P"])
        close = f5.number_input("Closing odds (CLV)", -2000, 2000, 0, 5)
        bet_tag = f6.selectbox("Tag", ["model", "hunch"])
        if st.form_submit_button("Add") and pick:
            dfb = load_bets()
            row = {"id": (dfb["id"].max() + 1 if len(dfb) else 1),
                   "date": str(pd.Timestamp.today().date()), "event": card,
                   "pick": pick, "odds": odds, "stake": stake_b,
                   "result": "" if result == "pending" else result,
                   "close": close if close else np.nan, "tag": bet_tag}
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
            settled["ref_close"] = settled["close"]
            fill_m = settled["ref_close"].isna()
            settled.loc[fill_m, "ref_close"] = [snapshot_line(p)
                                                for p in settled.loc[fill_m, "pick"]]
            clv = settled.dropna(subset=["ref_close"])
            if len(clv):
                edges_clv = [bet_implied(c) - bet_implied(o)
                             for o, c in zip(clv["odds"], clv["ref_close"])]
                m4.metric("Avg CLV", f"{np.mean(edges_clv):+.1%}",
                          help="Closing-line value; blanks auto-filled from the "
                               "last snapshot's FanDuel line (not a true close).")
            if decd < 20:
                st.caption(f"{decd} settled bets — far too few to distinguish skill "
                           "from luck. Judge nothing before ~50.")
        st.dataframe(dfb, hide_index=True, width="stretch")
    else:
        st.caption("No bets logged yet.")
