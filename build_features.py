"""Build leak-free pre-fight features from raw ufcstats.com data.

Reads the cached raw CSVs (Greco1899/scrape_ufc_stats mirror) and replays UFC
history in date order, maintaining per-fighter career state: records, per-minute
rates, strike-target and position mixes, recent-form windows, opponent quality
(average opponent Elo), and a finish-weighted Elo. Every feature for a fight is
computed strictly from fights BEFORE it. Closing odds and official rankings are
joined from ufc-master.csv by date + fighter names.

Fighter orientation (who is "A" vs "B") is randomized with a fixed seed because
ufcstats lists winners first 64% of the time; a red_corner feature preserves the
legitimate corner information.

Run: python build_features.py   ->  data/features.csv
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA = Path(__file__).parent / "data"
FORM_N = 3          # "recent form" window (fights)
FINISH_K_MULT = 1.4  # Elo update boost for KO/sub wins

WEIGHT_LBS = {
    "Strawweight": 115, "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145,
    "Lightweight": 155, "Welterweight": 170, "Middleweight": 185,
    "Light Heavyweight": 205, "Heavyweight": 250,
}


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def parse_of(s):
    """'11 of 14' -> (11, 14)."""
    m = re.match(r"\s*(\d+)\s+of\s+(\d+)", str(s))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def parse_ctrl(s):
    m = re.match(r"\s*(\d+):(\d+)", str(s))
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def parse_height_cm(s):
    m = re.match(r"\s*(\d+)'\s*(\d+)", str(s))
    return (int(m.group(1)) * 12 + int(m.group(2))) * 2.54 if m else np.nan


def parse_reach_cm(s):
    m = re.match(r'\s*(\d+)"', str(s))
    return int(m.group(1)) * 2.54 if m else np.nan


def fight_seconds(round_num, time_str):
    """Approximate elapsed fight time; assumes 5-minute rounds (true post-2000)."""
    m = re.match(r"\s*(\d+):(\d+)", str(time_str))
    last = int(m.group(1)) * 60 + int(m.group(2)) if m else 0
    try:
        return (int(round_num) - 1) * 300 + last
    except (TypeError, ValueError):
        return last


def load_raw():
    ev = pd.read_csv(DATA / "ufc_event_details.csv")
    ev["date"] = pd.to_datetime(ev["DATE"], format="%B %d, %Y")
    event_date = dict(zip(ev["EVENT"].str.strip(), ev["date"]))

    res = pd.read_csv(DATA / "ufc_fight_results.csv")
    res["EVENT"] = res["EVENT"].str.strip()
    res["date"] = res["EVENT"].map(event_date)

    st = pd.read_csv(DATA / "ufc_fight_stats.csv")
    st["EVENT"] = st["EVENT"].str.strip()
    both = [("SIG.STR.", "sig"), ("TOTAL STR.", "tot"), ("TD", "td")]
    landed_only = [("HEAD", "head"), ("BODY", "body"), ("LEG", "leg"),
                   ("DISTANCE", "dist"), ("CLINCH", "clinch"), ("GROUND", "ground")]
    for col, out in both:
        parsed = st[col].map(parse_of)
        st[f"{out}_l"] = [p[0] for p in parsed]
        st[f"{out}_a"] = [p[1] for p in parsed]
    for col, out in landed_only:
        st[f"{out}_l"] = [parse_of(v)[0] for v in st[col]]
    st["kd"] = pd.to_numeric(st["KD"], errors="coerce").fillna(0)
    st["sub_att"] = pd.to_numeric(st["SUB.ATT"], errors="coerce").fillna(0)
    st["ctrl"] = st["CTRL"].map(parse_ctrl)
    stat_cols = ["kd", "sig_l", "sig_a", "tot_l", "tot_a", "td_l", "td_a",
                 "sub_att", "ctrl", "head_l", "body_l", "leg_l",
                 "dist_l", "clinch_l", "ground_l"]
    agg = st.groupby(["EVENT", "BOUT", "FIGHTER"], sort=False)[stat_cols].sum().reset_index()
    stats = {(e, b): g.set_index(g["FIGHTER"].map(norm_name))
             for (e, b), g in agg.groupby(["EVENT", "BOUT"], sort=False)}

    tott = pd.read_csv(DATA / "ufc_fighter_tott.csv")
    tott["key"] = tott["FIGHTER"].map(norm_name)
    tott = tott.drop_duplicates("key", keep="first").set_index("key")
    phys = {
        k: {
            "height": parse_height_cm(r["HEIGHT"]),
            "reach": parse_reach_cm(r["REACH"]),
            "stance": str(r["STANCE"]).strip(),
            "dob": pd.to_datetime(r["DOB"], format="%b %d, %Y", errors="coerce"),
        }
        for k, r in tott.iterrows()
    }
    return res, stats, phys


def load_market():
    """(date, name-pair) -> per-fighter closing odds and weight-class rank."""
    m = pd.read_csv(DATA / "ufc-master.csv", low_memory=False)
    m["date"] = pd.to_datetime(m["date"])
    market = {}
    for r in m.itertuples():
        kr, kb = norm_name(r.R_fighter), norm_name(r.B_fighter)
        market[(r.date, frozenset((kr, kb)))] = {
            kr: (r.R_odds, getattr(r, "R_match_weightclass_rank", np.nan)),
            kb: (r.B_odds, getattr(r, "B_match_weightclass_rank", np.nan)),
        }
    return market


def new_state():
    return {
        "elo": 1500.0, "peak_elo": 1500.0, "n": 0, "wins": 0, "losses": 0, "draws": 0,
        "win_streak": 0, "lose_streak": 0, "ko_wins": 0, "sub_wins": 0, "dec_wins": 0,
        "sig_l": 0, "sig_a": 0, "opp_sig_l": 0, "opp_sig_a": 0,
        "td_l": 0, "td_a": 0, "opp_td_l": 0, "opp_td_a": 0,
        "kd": 0, "kd_taken": 0, "sub_att": 0, "ctrl": 0, "secs": 0,
        "head_l": 0, "body_l": 0, "leg_l": 0, "dist_l": 0, "clinch_l": 0, "ground_l": 0,
        "opp_head_l": 0, "opp_ground_l": 0,
        "opp_elo_sum": 0.0, "five_rd": 0, "last_weight": np.nan,
        "last_date": None, "hist": [],
    }


def snapshot(s, date, ph, cur_weight):
    """Pre-fight feature vector for one fighter. Rates are NaN before debut."""
    mins = s["secs"] / 60
    p15 = s["secs"] / 900
    n = s["n"]
    age = (date - ph["dob"]).days / 365.25 if pd.notna(ph.get("dob")) else np.nan
    hist = s["hist"]
    form = hist[-FORM_N:]
    f_secs = sum(h["secs"] for h in form)
    f_mins = f_secs / 60
    out = {
        "elo": s["elo"], "n_fights": n,
        "win_pct": s["wins"] / n if n else np.nan,
        "win_streak": s["win_streak"], "lose_streak": s["lose_streak"],
        "slpm": s["sig_l"] / mins if mins else np.nan,
        "sapm": s["opp_sig_l"] / mins if mins else np.nan,
        "str_acc": s["sig_l"] / s["sig_a"] if s["sig_a"] else np.nan,
        "str_def": 1 - s["opp_sig_l"] / s["opp_sig_a"] if s["opp_sig_a"] else np.nan,
        "td_avg": s["td_l"] / p15 if p15 else np.nan,
        "td_acc": s["td_l"] / s["td_a"] if s["td_a"] else np.nan,
        "td_def": 1 - s["opp_td_l"] / s["opp_td_a"] if s["opp_td_a"] else np.nan,
        "sub_att_p15": s["sub_att"] / p15 if p15 else np.nan,
        "kd_p15": s["kd"] / p15 if p15 else np.nan,
        "kd_taken_p15": s["kd_taken"] / p15 if p15 else np.nan,
        "ctrl_min_share": s["ctrl"] / s["secs"] if s["secs"] else np.nan,
        "finish_rate": (s["ko_wins"] + s["sub_wins"]) / s["wins"] if s["wins"] else np.nan,
        "dec_win_rate": s["dec_wins"] / s["wins"] if s["wins"] else np.nan,
        "layoff_days": (date - s["last_date"]).days if s["last_date"] is not None else np.nan,
        "age": age, "height": ph.get("height", np.nan), "reach": ph.get("reach", np.nan),
        # --- strike-target and position mixes (style fingerprint) ---
        "head_share": s["head_l"] / s["sig_l"] if s["sig_l"] else np.nan,
        "leg_share": s["leg_l"] / s["sig_l"] if s["sig_l"] else np.nan,
        "ground_str_share": s["ground_l"] / s["sig_l"] if s["sig_l"] else np.nan,
        "clinch_str_share": s["clinch_l"] / s["sig_l"] if s["sig_l"] else np.nan,
        "absorbed_head_share": s["opp_head_l"] / s["opp_sig_l"] if s["opp_sig_l"] else np.nan,
        "absorbed_ground_share": s["opp_ground_l"] / s["opp_sig_l"] if s["opp_sig_l"] else np.nan,
        # --- recent form (last FORM_N fights) ---
        "form_win": np.mean([h["win"] for h in form]) if form else np.nan,
        "form_slpm": sum(h["sig_l"] for h in form) / f_mins if f_mins else np.nan,
        "form_sapm": sum(h["opp_sig_l"] for h in form) / f_mins if f_mins else np.nan,
        "form_td15": sum(h["td_l"] for h in form) / (f_secs / 900) if f_secs else np.nan,
        "form_finished": np.mean([h["finished"] for h in form]) if form else np.nan,
        "form_was_finished": np.mean([h["was_finished"] for h in form]) if form else np.nan,
        # --- opponent quality (strength of schedule) ---
        "avg_opp_elo": s["opp_elo_sum"] / n if n else np.nan,
        "form_opp_elo": np.mean([h["opp_elo"] for h in form]) if form else np.nan,
        # --- trajectory / experience ---
        "peak_elo": s["peak_elo"],
        "elo_decline": s["elo"] - s["peak_elo"],
        "five_rd_fights": s["five_rd"],
        "weight_change": (np.sign(cur_weight - s["last_weight"])
                          if pd.notna(cur_weight) and pd.notna(s["last_weight"]) else 0.0),
    }
    return out


def update_state(s, my, opp, date, result, method, secs, opp_elo_pre, sched5, cur_weight):
    """Fold one completed fight into a fighter's career state."""
    if my is not None:
        s["sig_l"] += my["sig_l"]; s["sig_a"] += my["sig_a"]
        s["td_l"] += my["td_l"]; s["td_a"] += my["td_a"]
        s["kd"] += my["kd"]; s["sub_att"] += my["sub_att"]; s["ctrl"] += my["ctrl"]
        s["head_l"] += my["head_l"]; s["body_l"] += my["body_l"]; s["leg_l"] += my["leg_l"]
        s["dist_l"] += my["dist_l"]; s["clinch_l"] += my["clinch_l"]; s["ground_l"] += my["ground_l"]
    if opp is not None:
        s["opp_sig_l"] += opp["sig_l"]; s["opp_sig_a"] += opp["sig_a"]
        s["opp_td_l"] += opp["td_l"]; s["opp_td_a"] += opp["td_a"]
        s["kd_taken"] += opp["kd"]
        s["opp_head_l"] += opp["head_l"]; s["opp_ground_l"] += opp["ground_l"]
    s["secs"] += secs
    s["n"] += 1
    s["last_date"] = date
    s["opp_elo_sum"] += opp_elo_pre
    s["five_rd"] += int(sched5)
    if pd.notna(cur_weight):
        s["last_weight"] = cur_weight
    m = str(method)
    finish = ("KO" in m or "Submission" in m)
    if result == 1:
        s["wins"] += 1
        s["win_streak"] += 1; s["lose_streak"] = 0
        if "KO" in m: s["ko_wins"] += 1
        elif "Submission" in m: s["sub_wins"] += 1
        elif "Decision" in m: s["dec_wins"] += 1
    elif result == 0:
        s["losses"] += 1
        s["lose_streak"] += 1; s["win_streak"] = 0
    else:  # draw
        s["draws"] += 1
        s["win_streak"] = 0; s["lose_streak"] = 0
    s["hist"].append({
        "win": result, "secs": secs,
        "sig_l": my["sig_l"] if my is not None else 0,
        "opp_sig_l": opp["sig_l"] if opp is not None else 0,
        "td_l": my["td_l"] if my is not None else 0,
        "finished": int(result == 1 and finish),
        "was_finished": int(result == 0 and finish),
        "opp_elo": opp_elo_pre,
    })


def main():
    rng = np.random.default_rng(SEED)
    res, stats, phys = load_raw()
    market = load_market()

    res = res[res["date"].notna()].sort_values(["date", "EVENT", "BOUT"]).reset_index(drop=True)
    states, rows = {}, []
    matched_odds = 0

    for r in res.itertuples():
        bout = str(r.BOUT)
        if " vs. " not in bout:
            continue
        f1, f2 = [x.strip() for x in bout.split(" vs. ", 1)]
        k1, k2 = norm_name(f1), norm_name(f2)
        if k1 == k2:
            continue
        outcome = str(r.OUTCOME)
        if outcome not in ("W/L", "L/W", "D/D"):
            continue  # NC etc. — no state update, no row
        s1 = states.setdefault(k1, new_state())
        s2 = states.setdefault(k2, new_state())

        wc = str(r.WEIGHTCLASS)
        weight = next((v for k, v in WEIGHT_LBS.items()
                       if k in wc and not ("Light Heavyweight" in wc and k == "Heavyweight")), np.nan)
        sched5 = "5 Rnd" in str(r._8)

        # ---- pre-fight snapshot (this is all the model may see) ----
        if outcome != "D/D":
            a_first = bool(rng.integers(2))  # seeded random orientation
            (ka, kb) = (k1, k2) if a_first else (k2, k1)
            (fa, fb) = (f1, f2) if a_first else (f2, f1)
            sa = snapshot(states[ka], r.date, phys.get(ka, {}), weight)
            sb = snapshot(states[kb], r.date, phys.get(kb, {}), weight)
            stance_a = phys.get(ka, {}).get("stance", "")
            stance_b = phys.get(kb, {}).get("stance", "")
            row = {
                "date": r.date, "event": r.EVENT, "fighter_a": fa, "fighter_b": fb,
                "a_wins": int((outcome == "W/L") == a_first),
                "red_corner": 1 if a_first else -1,
                "southpaw_vs_orthodox": int(stance_a == "Southpaw" and stance_b == "Orthodox")
                                      - int(stance_a == "Orthodox" and stance_b == "Southpaw"),
                "title_bout": int("Title" in wc),
                "women": int("Women" in wc),
                "sched_rounds": 5 if sched5 else 3,
                "weight_lbs": weight,
            }
            for feat in sa:
                row[f"{feat}_diff"] = sa[feat] - sb[feat]
                row[f"{feat}_a"] = sa[feat]
                row[f"{feat}_b"] = sb[feat]
            okey = (r.date, frozenset((ka, kb)))
            if okey in market:
                odds_a, rank_a = market[okey].get(ka, (np.nan, np.nan))
                odds_b, rank_b = market[okey].get(kb, (np.nan, np.nan))
                row["odds_a"], row["odds_b"] = odds_a, odds_b
                # lower rank number is better; positive = A advantage
                row["rank_adv"] = (rank_b - rank_a) if pd.notna(rank_a) and pd.notna(rank_b) else np.nan
                row["ranked_diff"] = int(pd.notna(rank_a)) - int(pd.notna(rank_b))
                matched_odds += 1
            else:
                row["rank_adv"], row["ranked_diff"] = np.nan, 0
            rows.append(row)

        # ---- fold results into state ----
        g = stats.get((r.EVENT, bout))
        st1 = g.loc[k1].to_dict() if g is not None and k1 in g.index else None
        st2 = g.loc[k2].to_dict() if g is not None and k2 in g.index else None
        if st1 is not None and isinstance(st1.get("sig_l"), pd.Series):
            st1 = None  # duplicate-name collision inside one bout; skip stats
        if st2 is not None and isinstance(st2.get("sig_l"), pd.Series):
            st2 = None
        secs = fight_seconds(r.ROUND, r.TIME)
        res1 = 1 if outcome == "W/L" else (0 if outcome == "L/W" else 0.5)
        # Elo before records mutate; finishes move ratings harder
        elo1_pre, elo2_pre = s1["elo"], s2["elo"]
        exp1 = 1 / (1 + 10 ** ((elo2_pre - elo1_pre) / 400))
        finish_mult = FINISH_K_MULT if ("KO" in str(r.METHOD) or "Submission" in str(r.METHOD)) else 1.0
        k_elo1 = (40 if s1["n"] < 5 else 24) * finish_mult
        k_elo2 = (40 if s2["n"] < 5 else 24) * finish_mult
        s1_new_elo = elo1_pre + k_elo1 * (res1 - exp1)
        s2_new_elo = elo2_pre + k_elo2 * ((1 - res1) - (1 - exp1))
        update_state(s1, st1, st2, r.date, res1, r.METHOD, secs, elo2_pre, sched5, weight)
        update_state(s2, st2, st1, r.date, 1 - res1, r.METHOD, secs, elo1_pre, sched5, weight)
        s1["elo"], s2["elo"] = s1_new_elo, s2_new_elo
        s1["peak_elo"] = max(s1["peak_elo"], s1_new_elo)
        s2["peak_elo"] = max(s2["peak_elo"], s2_new_elo)

    df = pd.DataFrame(rows)
    out = DATA / "features.csv"
    df.to_csv(out, index=False)
    n_feat = len([c for c in df.columns if c.endswith("_diff")]) + 6
    print(f"{len(df)} fights -> {out}  ({n_feat}+ feature columns)")
    print(f"  date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  with odds:  {matched_odds} ({matched_odds/len(df):.1%})")
    print(f"  a_wins base rate: {df['a_wins'].mean():.3f} (should be ~0.5 by construction)")


if __name__ == "__main__":
    main()
