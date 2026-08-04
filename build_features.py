"""Build leak-free pre-fight features from raw ufcstats.com data.

Reads the cached raw CSVs (Greco1899/scrape_ufc_stats mirror) and replays UFC
history in date order, maintaining per-fighter career state (records, per-minute
rates, Elo). Every feature for a fight is computed strictly from fights BEFORE
it. Closing odds are joined from ufc-master.csv by date + fighter names.

Fighter orientation (who is "A" vs "B") is randomized with a fixed seed because
ufcstats lists winners first 64% of the time; a red_corner feature preserves the
legitimate corner information.

Run: python build_features.py   ->  data/features.csv
"""

import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA = Path(__file__).parent / "data"

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


def fight_seconds(round_num, time_str, time_format):
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
    for col, out in [("SIG.STR.", "sig"), ("TOTAL STR.", "tot"), ("TD", "td")]:
        parsed = st[col].map(parse_of)
        st[f"{out}_l"] = [p[0] for p in parsed]
        st[f"{out}_a"] = [p[1] for p in parsed]
    st["kd"] = pd.to_numeric(st["KD"], errors="coerce").fillna(0)
    st["sub_att"] = pd.to_numeric(st["SUB.ATT"], errors="coerce").fillna(0)
    st["ctrl"] = st["CTRL"].map(parse_ctrl)
    agg = st.groupby(["EVENT", "BOUT", "FIGHTER"], sort=False)[
        ["kd", "sig_l", "sig_a", "tot_l", "tot_a", "td_l", "td_a", "sub_att", "ctrl"]
    ].sum().reset_index()
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


def load_odds():
    m = pd.read_csv(DATA / "ufc-master.csv", low_memory=False)
    m["date"] = pd.to_datetime(m["date"])
    odds = {}
    for r in m.itertuples():
        key = (r.date, frozenset((norm_name(r.R_fighter), norm_name(r.B_fighter))))
        odds[key] = {norm_name(r.R_fighter): r.R_odds, norm_name(r.B_fighter): r.B_odds}
    return odds


def new_state():
    return {
        "elo": 1500.0, "n": 0, "wins": 0, "losses": 0, "draws": 0,
        "win_streak": 0, "lose_streak": 0, "ko_wins": 0, "sub_wins": 0, "dec_wins": 0,
        "sig_l": 0, "sig_a": 0, "opp_sig_l": 0, "opp_sig_a": 0,
        "td_l": 0, "td_a": 0, "opp_td_l": 0, "opp_td_a": 0,
        "kd": 0, "kd_taken": 0, "sub_att": 0, "ctrl": 0, "secs": 0,
        "last_date": None,
    }


def snapshot(s, date, ph):
    """Pre-fight feature vector for one fighter. Rates are NaN before debut."""
    mins = s["secs"] / 60
    p15 = s["secs"] / 900
    n = s["n"]
    age = (date - ph["dob"]).days / 365.25 if pd.notna(ph.get("dob")) else np.nan
    return {
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
    }


def update_state(s, my, opp, date, result, method, secs):
    """Fold one completed fight into a fighter's career state."""
    if my is not None:
        s["sig_l"] += my["sig_l"]; s["sig_a"] += my["sig_a"]
        s["td_l"] += my["td_l"]; s["td_a"] += my["td_a"]
        s["kd"] += my["kd"]; s["sub_att"] += my["sub_att"]; s["ctrl"] += my["ctrl"]
    if opp is not None:
        s["opp_sig_l"] += opp["sig_l"]; s["opp_sig_a"] += opp["sig_a"]
        s["opp_td_l"] += opp["td_l"]; s["opp_td_a"] += opp["td_a"]
        s["kd_taken"] += opp["kd"]
    s["secs"] += secs
    s["n"] += 1
    s["last_date"] = date
    m = str(method)
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


def main():
    rng = np.random.default_rng(SEED)
    res, stats, phys = load_raw()
    odds = load_odds()

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
        ph1, ph2 = phys.get(k1, {}), phys.get(k2, {})

        # ---- pre-fight snapshot (this is all the model may see) ----
        if outcome != "D/D":
            a_first = bool(rng.integers(2))  # seeded random orientation
            (ka, kb) = (k1, k2) if a_first else (k2, k1)
            (fa, fb) = (f1, f2) if a_first else (f2, f1)
            sa = snapshot(states[ka], r.date, phys.get(ka, {}))
            sb = snapshot(states[kb], r.date, phys.get(kb, {}))
            wc = str(r.WEIGHTCLASS)
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
                "sched_rounds": 5 if "5 Rnd" in str(r._8) else 3,
                "weight_lbs": next((v for k, v in WEIGHT_LBS.items()
                                    if k in wc and not ("Light Heavyweight" in wc and k == "Heavyweight")), np.nan),
            }
            for feat in sa:
                row[f"{feat}_diff"] = sa[feat] - sb[feat]
                row[f"{feat}_a"] = sa[feat]
                row[f"{feat}_b"] = sb[feat]
            okey = (r.date, frozenset((ka, kb)))
            if okey in odds:
                row["odds_a"] = odds[okey].get(ka, np.nan)
                row["odds_b"] = odds[okey].get(kb, np.nan)
                matched_odds += 1
            rows.append(row)

        # ---- fold results into state ----
        g = stats.get((r.EVENT, bout))
        st1 = g.loc[k1].to_dict() if g is not None and k1 in g.index else None
        st2 = g.loc[k2].to_dict() if g is not None and k2 in g.index else None
        if st1 is not None and isinstance(st1.get("sig_l"), pd.Series):
            st1 = None  # duplicate-name collision inside one bout; skip stats
        if st2 is not None and isinstance(st2.get("sig_l"), pd.Series):
            st2 = None
        secs = fight_seconds(r.ROUND, r.TIME, r._8)
        res1 = 1 if outcome == "W/L" else (0 if outcome == "L/W" else 0.5)
        # Elo before records mutate
        exp1 = 1 / (1 + 10 ** ((s2["elo"] - s1["elo"]) / 400))
        k_elo1 = 40 if s1["n"] < 5 else 24
        k_elo2 = 40 if s2["n"] < 5 else 24
        s1_new_elo = s1["elo"] + k_elo1 * (res1 - exp1)
        s2_new_elo = s2["elo"] + k_elo2 * ((1 - res1) - (1 - exp1))
        update_state(s1, st1, st2, r.date, 1 if res1 == 1 else (0 if res1 == 0 else 0.5), r.METHOD, secs)
        update_state(s2, st2, st1, r.date, 1 if res1 == 0 else (0 if res1 == 1 else 0.5), r.METHOD, secs)
        s1["elo"], s2["elo"] = s1_new_elo, s2_new_elo

    df = pd.DataFrame(rows)
    out = DATA / "features.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} fights -> {out}")
    print(f"  date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  with odds:  {matched_odds} ({matched_odds/len(df):.1%})")
    print(f"  a_wins base rate: {df['a_wins'].mean():.3f} (should be ~0.5 by construction)")


if __name__ == "__main__":
    main()
