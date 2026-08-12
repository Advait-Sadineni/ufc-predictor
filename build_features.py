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
import sys
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA = Path(__file__).parent / "data"
RAW_FILES = ["ufc_fight_stats.csv", "ufc_fight_results.csv",
             "ufc_fighter_tott.csv", "ufc_event_details.csv"]
RAW_BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
MASTER_MIRRORS = [
    "https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/main/ufc-master.csv",
    "https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/master/ufc-master.csv",
]


def ensure_data(refresh=False):
    """Download any missing source CSVs (or all of them with refresh=True)."""
    DATA.mkdir(exist_ok=True)
    for f in RAW_FILES:
        path = DATA / f
        if refresh or not path.exists():
            print(f"Downloading {f} ...")
            path.write_bytes(urllib.request.urlopen(RAW_BASE + f, timeout=120).read())
    path = DATA / "ufc-master.csv"
    if refresh or not path.exists():
        for url in MASTER_MIRRORS:
            try:
                print("Downloading ufc-master.csv ...")
                path.write_bytes(urllib.request.urlopen(url, timeout=120).read())
                break
            except Exception as e:
                print(f"  mirror failed: {e}")
# Two different UFC fighters share each of these normalized names; their merged
# career stats are noise, so their fights are excluded from training rows
# (their OPPONENTS' states still update normally).
KNOWN_DUPES = {"brunosilva", "jeansilva", "joeygomez", "michaelmcdonald",
               "mikedavis", "victorvalenzuela"}
FORM_N = 3          # "recent form" window (fights)
SCORE_RE = re.compile(r"(\d{1,3})\s*-\s*(\d{1,3})")  # judge scorecards in DETAILS
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
    st["rev"] = pd.to_numeric(st["REV."], errors="coerce").fillna(0)
    stat_cols = ["kd", "sig_l", "sig_a", "tot_l", "tot_a", "td_l", "td_a",
                 "sub_att", "ctrl", "rev", "head_l", "body_l", "leg_l",
                 "dist_l", "clinch_l", "ground_l"]
    st["rnd"] = pd.to_numeric(st["ROUND"].str.extract(r"(\d+)")[0], errors="coerce")
    # round-split output: early = R1, late = R3+ (cardio / fade signal that the
    # fight-total aggregation destroys)
    st["early_sig"] = st["sig_l"].where(st["rnd"] == 1, 0)
    st["late_sig"] = st["sig_l"].where(st["rnd"] >= 3, 0)
    st["early_rounds"] = (st["rnd"] == 1).astype(float)
    st["late_rounds"] = (st["rnd"] >= 3).astype(float)
    stat_cols = stat_cols + ["early_sig", "late_sig", "early_rounds", "late_rounds"]
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
    # Phase 13/1D note: backfilling missing reach/DOB from ufc-master's
    # always-populated physical columns was tested and REJECTED — CV degraded
    # 0.6466 -> 0.6488. Those columns are themselves imputed, and LightGBM was
    # already using the missingness as signal. _backfill_phys kept for the record.
    return res, stats, phys


def _backfill_phys(phys):
    """REJECTED (Phase 13/1D, see note in load_raw) — kept unused for the record.
    Fill missing reach/height/DOB from ufc-master's physical columns."""
    try:
        m = pd.read_csv(DATA / "ufc-master.csv", low_memory=False,
                        usecols=["R_fighter", "B_fighter", "date",
                                 "R_Reach_cms", "B_Reach_cms",
                                 "R_Height_cms", "B_Height_cms", "R_age", "B_age"])
    except (FileNotFoundError, ValueError):
        return
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("date")
    for side in ("R", "B"):
        for r in m.itertuples():
            k = norm_name(getattr(r, f"{side}_fighter"))
            e = phys.setdefault(k, {"height": np.nan, "reach": np.nan,
                                    "stance": "nan", "dob": pd.NaT})
            reach, height = getattr(r, f"{side}_Reach_cms"), getattr(r, f"{side}_Height_cms")
            age = getattr(r, f"{side}_age")
            if pd.isna(e.get("reach")) and pd.notna(reach):
                e["reach"] = float(reach)
            if pd.isna(e.get("height")) and pd.notna(height):
                e["height"] = float(height)
            if pd.isna(e.get("dob")) and pd.notna(age):
                # integer age at fight date -> DOB estimate good to ~±6 months
                e["dob"] = r.date - pd.Timedelta(days=(age + 0.5) * 365.25)


def load_pre_ufc(res):
    """Static pre-UFC record per fighter: ESPN pro record minus their full UFC
    record from our own data. A regional career never changes, so this is
    leak-free for every fight (unlike the current total, which does leak)."""
    import json
    path = DATA / "pro_records.json"
    if not path.exists():
        return {}
    pro = json.loads(path.read_text())
    pro.pop("_dates", None)
    ufc = {}
    for r in res.itertuples():
        bout, outcome = str(r.BOUT), str(r.OUTCOME)
        if " vs. " not in bout or outcome not in ("W/L", "L/W", "D/D"):
            continue
        k1, k2 = [norm_name(x) for x in bout.split(" vs. ", 1)]
        for k, won in ((k1, outcome == "W/L"), (k2, outcome == "L/W")):
            w, l = ufc.get(k, (0, 0))
            ufc[k] = (w + int(won), l + int(outcome != "D/D" and not won))
    out = {}
    for k, summary in pro.items():
        try:
            pw, pl = (int(x) for x in str(summary).split("-")[:2])
        except ValueError:
            continue
        uw, ul = ufc.get(k, (0, 0))
        out[k] = (max(pw - uw, 0), max(pl - ul, 0))
    return out


def load_market():
    """(date, name-pair) -> per-fighter closing odds and weight-class rank."""
    m = pd.read_csv(DATA / "ufc-master.csv", low_memory=False)
    m["date"] = pd.to_datetime(m["date"])
    market = {}
    for r in m.itertuples():
        kr, kb = norm_name(r.R_fighter), norm_name(r.B_fighter)
        br = getattr(r, "better_rank", None)
        market[(r.date, frozenset((kr, kb)))] = {
            kr: (r.R_odds, getattr(r, "R_match_weightclass_rank", np.nan),
                 getattr(r, "r_ko_odds", np.nan), getattr(r, "r_sub_odds", np.nan),
                 getattr(r, "r_dec_odds", np.nan)),
            kb: (r.B_odds, getattr(r, "B_match_weightclass_rank", np.nan),
                 getattr(r, "b_ko_odds", np.nan), getattr(r, "b_sub_odds", np.nan),
                 getattr(r, "b_dec_odds", np.nan)),
            # 93.9%-covered "who is better ranked" flag vs 19-27% for the
            # numeric match_weightclass_rank pair (Phase 13 / 1D)
            "_better": kr if br == "Red" else (kb if br == "Blue" else None),
        }
    return market


GLICKO_Q = 0.0057565  # ln(10)/400
GLICKO_C2 = 2000.0    # RD^2 inflation per month idle (50->350 in ~5y)


def _glicko_g(rd):
    return 1.0 / np.sqrt(1 + 3 * GLICKO_Q**2 * rd**2 / np.pi**2)


def glicko_rd_now(s, date):
    """Pre-fight rating deviation, inflated by inactivity."""
    if s["last_date"] is None:
        return 350.0
    months = (date - s["last_date"]).days / 30.0
    return float(min(np.sqrt(s["g_rd"] ** 2 + GLICKO_C2 * months), 350.0))


def glicko_update(r, rd, r_opp, rd_opp, score):
    g = _glicko_g(rd_opp)
    e = 1.0 / (1 + 10 ** (-g * (r - r_opp) / 400))
    d2 = 1.0 / (GLICKO_Q**2 * g**2 * e * (1 - e))
    denom = 1.0 / rd**2 + 1.0 / d2
    r_new = r + GLICKO_Q / denom * g * (score - e)
    rd_new = max(np.sqrt(1.0 / denom), 30.0)
    return float(r_new), float(rd_new)


def new_state():
    return {
        "g_r": 1500.0, "g_rd": 350.0,
        "elo": 1500.0, "peak_elo": 1500.0, "n": 0, "wins": 0, "losses": 0, "draws": 0,
        "win_streak": 0, "lose_streak": 0, "ko_wins": 0, "sub_wins": 0, "dec_wins": 0,
        "ko_losses": 0, "sub_losses": 0, "dec_losses": 0,
        "sig_l": 0, "sig_a": 0, "opp_sig_l": 0, "opp_sig_a": 0,
        "td_l": 0, "td_a": 0, "opp_td_l": 0, "opp_td_a": 0,
        "kd": 0, "kd_taken": 0, "sub_att": 0, "ctrl": 0, "secs": 0,
        "tot_l": 0, "tot_a": 0, "rev": 0,
        "head_l": 0, "body_l": 0, "leg_l": 0, "dist_l": 0, "clinch_l": 0, "ground_l": 0,
        "early_sig": 0, "late_sig": 0, "early_rounds": 0, "late_rounds": 0,
        "opp_early_sig": 0, "opp_late_sig": 0,
        "opp_head_l": 0, "opp_ground_l": 0, "opp_body_l": 0, "opp_leg_l": 0,
        "opp_clinch_l": 0, "opp_dist_l": 0, "opp_ctrl": 0, "opp_sub_att": 0,
        "opp_elo_sum": 0.0, "five_rd": 0, "last_weight": np.nan,
        "sc_n": 0, "sc_close": 0, "sc_split": 0, "sc_share": 0.0,
        "fin_choke": 0, "fin_ground": 0, "fin_mech_n": 0,
        "last_date": None, "hist": [],
    }


def _rate(num, secs):
    return num / (secs / 900) if secs else np.nan


def opp_traits(s, own_height, opp_height, opp_stance):
    """Archetype of an opponent, judged from their PRE-fight career state."""
    return {
        "taller": pd.notna(opp_height) and pd.notna(own_height) and opp_height >= own_height + 2.5,
        "wrestler": s["secs"] > 0 and _rate(s["td_l"], s["secs"]) >= 1.5,
        "power": s["secs"] > 0 and _rate(s["kd"], s["secs"]) >= 0.4,
        "southpaw": opp_stance == "Southpaw",
    }


def _vs_split(hist, key, min_n=2):
    sub = [h["win"] for h in hist if h["opp"][key]]
    return np.mean(sub) if len(sub) >= min_n else np.nan


def method_class(m):
    m = str(m)
    if "KO" in m or "TKO" in m:
        return "ko"
    if "Submission" in m:
        return "sub"
    if "Decision" in m:
        return "dec"
    return None  # DQ, overturned, etc.


def snapshot(s, date, ph, cur_weight, today_opp, div, pre=(0, 0)):
    """Pre-fight feature vector for one fighter. Rates are NaN before debut.

    today_opp: archetype dict of TODAY's opponent (activates matchup splits).
    div: running (sum, n) of height/reach for the division, pre-fight.
    """
    mins = s["secs"] / 60
    p15 = s["secs"] / 900
    n = s["n"]
    age = (date - ph["dob"]).days / 365.25 if pd.notna(ph.get("dob")) else np.nan
    hist = s["hist"]
    form = hist[-FORM_N:]
    f_secs = sum(h["secs"] for h in form)
    f_mins = f_secs / 60
    pre_w, pre_l = pre
    out = {
        # regional career before the UFC — static, so leak-free; matters most
        # for debutants where every other career feature is empty
        "pre_ufc_wins": pre_w, "pre_ufc_losses": pre_l,
        "pre_ufc_fights": pre_w + pre_l,
        "pre_ufc_winpct": pre_w / (pre_w + pre_l) if (pre_w + pre_l) else np.nan,
        "glicko": s["g_r"], "glicko_rd": glicko_rd_now(s, date),
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
        # how he LOSES — chin and submission vulnerability over the whole career
        "ko_loss_rate": s["ko_losses"] / n if n else np.nan,
        "sub_loss_rate": s["sub_losses"] / n if n else np.nan,
        "finished_rate": (s["ko_losses"] + s["sub_losses"]) / n if n else np.nan,
        "never_finished": float(s["ko_losses"] + s["sub_losses"] == 0) if n else np.nan,
        # cardio: output per round early (R1) vs late (R3+). A fighter who
        # halves his output by round three is a different bet from a steady one.
        "r1_output": s["early_sig"] / s["early_rounds"] if s["early_rounds"] else np.nan,
        "late_output": s["late_sig"] / s["late_rounds"] if s["late_rounds"] else np.nan,
        "cardio_ratio": ((s["late_sig"] / s["late_rounds"]) /
                         (s["early_sig"] / s["early_rounds"])
                         if s["late_rounds"] and s["early_rounds"] and s["early_sig"] else np.nan),
        "late_absorbed": (s["opp_late_sig"] / s["late_rounds"]
                          if s["late_rounds"] else np.nan),
        "deep_water_rounds": s["late_rounds"],
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
        # --- method fingerprint (Phase 13 / 2C+1E, for the 6-way model) ---
        "ko_win_rate": s["ko_wins"] / s["wins"] if s["wins"] else np.nan,
        "sub_win_rate": s["sub_wins"] / s["wins"] if s["wins"] else np.nan,
        "choke_fin_rate": s["fin_choke"] / s["fin_mech_n"] if s["fin_mech_n"] else np.nan,
        "ground_fin_rate": s["fin_ground"] / s["fin_mech_n"] if s["fin_mech_n"] else np.nan,
        # --- judges' scorecards (Phase 13 / 1C, adopted on the >=3-decisions
        # slice like pre-UFC records): decision narrowness as decline signal ---
        "sc_close": s["sc_close"] / s["sc_n"] if s["sc_n"] else np.nan,
        "sc_split": s["sc_split"] / s["sc_n"] if s["sc_n"] else np.nan,
        "sc_share": s["sc_share"] / s["sc_n"] if s["sc_n"] else np.nan,
        # --- absorbed-side / grappling-defense (Phase 13 / 1B) ---
        "ctrl_conceded_share": s["opp_ctrl"] / s["secs"] if s["secs"] else np.nan,
        "absorbed_body_share": s["opp_body_l"] / s["opp_sig_l"] if s["opp_sig_l"] else np.nan,
        "absorbed_leg_share": s["opp_leg_l"] / s["opp_sig_l"] if s["opp_sig_l"] else np.nan,
        "subs_survived_p15": _rate(s["opp_sub_att"], s["secs"]),
        "nonsig_volume": (s["tot_l"] - s["sig_l"]) / mins if mins else np.nan,
        "rev_p15": _rate(s["rev"], s["secs"]),
        # --- quality-adjusted recent form (Phase 12) ---
        "form_perf_vs_exp": np.mean([h["win"] - h["exp"] for h in form]) if form else np.nan,
        "form_opp_winpct": (lambda v: np.mean(v) if v else np.nan)(
            [h["opp_winpct"] for h in form if pd.notna(h["opp_winpct"])]),
        "form_opp_adj_out": (lambda v: np.mean(v) if v else np.nan)(
            [h["opp_adj_out"] for h in form if pd.notna(h["opp_adj_out"])]),
        # --- trajectory / experience ---
        "peak_elo": s["peak_elo"],
        "elo_decline": s["elo"] - s["peak_elo"],
        "five_rd_fights": s["five_rd"],
        "weight_change": (np.sign(cur_weight - s["last_weight"])
                          if pd.notna(cur_weight) and pd.notna(s["last_weight"]) else 0.0),
    }
    # --- style-matchup splits: record vs opponent archetypes ---
    splits = {k: _vs_split(hist, k) for k in ("taller", "wrestler", "power", "southpaw")}
    out["vs_taller_win"] = splits["taller"]
    out["vs_wrestler_win"] = splits["wrestler"]
    out["vs_power_win"] = splits["power"]
    out["vs_southpaw_win"] = splits["southpaw"]
    # win rate vs the archetypes TODAY's opponent actually matches
    active = [splits[k] for k, v in today_opp.items() if v and pd.notna(splits[k])]
    out["matchup_win"] = np.mean(active) if active else np.nan
    # --- cut-severity proxy: physical size relative to the division so far ---
    h_sum, h_n, r_sum, r_n = div
    out["size_vs_div_h"] = (ph.get("height", np.nan) - h_sum / h_n) if h_n >= 20 else np.nan
    out["size_vs_div_r"] = (ph.get("reach", np.nan) - r_sum / r_n) if r_n >= 20 else np.nan
    return out


def update_state(s, my, opp, date, result, method, secs, opp_elo_pre, sched5, cur_weight,
                 opp_archetype, own_exp=np.nan, opp_winpct_pre=np.nan, opp_sapm_pre=np.nan,
                 card=None, mech=None):
    """Fold one completed fight into a fighter's career state."""
    if my is not None:
        s["sig_l"] += my["sig_l"]; s["sig_a"] += my["sig_a"]
        s["td_l"] += my["td_l"]; s["td_a"] += my["td_a"]
        s["kd"] += my["kd"]; s["sub_att"] += my["sub_att"]; s["ctrl"] += my["ctrl"]
        s["tot_l"] += my["tot_l"]; s["tot_a"] += my["tot_a"]
        s["rev"] += my.get("rev", 0)
        s["head_l"] += my["head_l"]; s["body_l"] += my["body_l"]; s["leg_l"] += my["leg_l"]
        for k in ("early_sig", "late_sig", "early_rounds", "late_rounds"):
            s[k] += my.get(k, 0)
        s["dist_l"] += my["dist_l"]; s["clinch_l"] += my["clinch_l"]; s["ground_l"] += my["ground_l"]
    if opp is not None:
        s["opp_sig_l"] += opp["sig_l"]; s["opp_sig_a"] += opp["sig_a"]
        s["opp_td_l"] += opp["td_l"]; s["opp_td_a"] += opp["td_a"]
        s["kd_taken"] += opp["kd"]
        s["opp_head_l"] += opp["head_l"]; s["opp_ground_l"] += opp["ground_l"]
        s["opp_body_l"] += opp["body_l"]; s["opp_leg_l"] += opp["leg_l"]
        s["opp_clinch_l"] += opp["clinch_l"]; s["opp_dist_l"] += opp["dist_l"]
        s["opp_ctrl"] += opp["ctrl"]; s["opp_sub_att"] += opp["sub_att"]
        s["opp_early_sig"] += opp.get("early_sig", 0)
        s["opp_late_sig"] += opp.get("late_sig", 0)
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
        if "KO" in m: s["ko_losses"] += 1
        elif "Submission" in m: s["sub_losses"] += 1
        elif "Decision" in m: s["dec_losses"] += 1
    else:  # draw
        s["draws"] += 1
        s["win_streak"] = 0; s["lose_streak"] = 0
    if mech is not None and result == 1:
        # finish-mechanism fingerprint from DETAILS (winner side only)
        s["fin_mech_n"] += 1
        s["fin_choke"] += mech[0]
        s["fin_ground"] += mech[1]
    if card is not None and result in (0, 1):
        w_share, split, close = card
        s["sc_n"] += 1
        s["sc_close"] += close
        s["sc_split"] += split
        s["sc_share"] += w_share if result == 1 else 1 - w_share
    s["hist"].append({
        "win": result, "secs": secs,
        "sig_l": my["sig_l"] if my is not None else 0,
        "opp_sig_l": opp["sig_l"] if opp is not None else 0,
        "td_l": my["td_l"] if my is not None else 0,
        "finished": int(result == 1 and finish),
        "was_finished": int(result == 0 and finish),
        "opp_elo": opp_elo_pre,
        "opp": opp_archetype,
        "exp": own_exp,
        "opp_winpct": opp_winpct_pre,
        "opp_adj_out": (my["sig_l"] / (secs / 60) - opp_sapm_pre
                        if my is not None and secs and pd.notna(opp_sapm_pre) else np.nan),
    })


def replay(res, stats, phys, market, rng, pre_ufc=None):
    """Replay history in date order.
    pre_ufc: {fighter_key: (wins, losses)} regional record before the UFC. Returns (feature rows, fighter states,
    division size stats) — states/div_stats reflect everything up to today,
    which is what predict.py needs for upcoming fights."""
    pre_ufc = pre_ufc or {}
    res = res[res["date"].notna()].sort_values(["date", "EVENT", "BOUT"]).reset_index(drop=True)
    states, rows = {}, []
    div_stats = {}  # weight_lbs -> [height_sum, height_n, reach_sum, reach_n], pre-fight
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
        h1 = phys.get(k1, {}).get("height", np.nan)
        h2 = phys.get(k2, {}).get("height", np.nan)
        stance1 = phys.get(k1, {}).get("stance", "")
        stance2 = phys.get(k2, {}).get("stance", "")
        # archetype of each fighter's opponent, from PRE-fight state
        traits_of_2 = opp_traits(s2, h1, h2, stance2)   # fighter1's opponent
        traits_of_1 = opp_traits(s1, h2, h1, stance1)   # fighter2's opponent
        div = div_stats.setdefault(weight if pd.notna(weight) else 0,
                                   [0.0, 0, 0.0, 0])

        # post-fight facts, used ONLY as labels and state updates — never features
        g = stats.get((r.EVENT, bout))
        st1 = g.loc[k1].to_dict() if g is not None and k1 in g.index else None
        st2 = g.loc[k2].to_dict() if g is not None and k2 in g.index else None
        if st1 is not None and isinstance(st1.get("sig_l"), pd.Series):
            st1 = None  # duplicate-name collision inside one bout; skip stats
        if st2 is not None and isinstance(st2.get("sig_l"), pd.Series):
            st2 = None
        secs = fight_seconds(r.ROUND, r.TIME)

        # ---- pre-fight snapshot (this is all the model may see) ----
        if outcome != "D/D":
            # drawn for every decided fight (dupes included) so each fight's
            # orientation is stable regardless of exclusions
            a_first = bool(rng.integers(2))
        if outcome != "D/D" and k1 not in KNOWN_DUPES and k2 not in KNOWN_DUPES:
            (ka, kb) = (k1, k2) if a_first else (k2, k1)
            (fa, fb) = (f1, f2) if a_first else (f2, f1)
            (ta, tb) = (traits_of_2, traits_of_1) if a_first else (traits_of_1, traits_of_2)
            sa = snapshot(states[ka], r.date, phys.get(ka, {}), weight, ta, div, pre_ufc.get(ka, (0, 0)))
            sb = snapshot(states[kb], r.date, phys.get(kb, {}), weight, tb, div, pre_ufc.get(kb, (0, 0)))
            stance_a = phys.get(ka, {}).get("stance", "")
            stance_b = phys.get(kb, {}).get("stance", "")
            row = {
                "date": r.date, "event": r.EVENT, "fighter_a": fa, "fighter_b": fb,
                "a_wins": int((outcome == "W/L") == a_first),
                "method_cls": method_class(r.METHOD),
                "finish_round": pd.to_numeric(r.ROUND, errors="coerce"),
                "fight_secs": secs,
                "total_td": (st1["td_l"] + st2["td_l"]
                             if st1 is not None and st2 is not None else np.nan),
                # per-fighter takedowns landed, oriented to A/B (prop markets
                # are per fighter, not per fight)
                "td_landed_a": ((st1 if a_first else st2) or {}).get("td_l", np.nan),
                "td_landed_b": ((st2 if a_first else st1) or {}).get("td_l", np.nan),
                "sig_landed_a": ((st1 if a_first else st2) or {}).get("sig_l", np.nan),
                "sig_landed_b": ((st2 if a_first else st1) or {}).get("sig_l", np.nan),
                "kd_landed_a": ((st1 if a_first else st2) or {}).get("kd", np.nan),
                "kd_landed_b": ((st2 if a_first else st1) or {}).get("kd", np.nan),
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
            # explicit trait interactions (antisymmetric by construction)
            row["chin_vs_power_diff"] = (sa["kd_taken_p15"] * sb["kd_p15"]
                                         - sb["kd_taken_p15"] * sa["kd_p15"])
            row["grapple_threat_diff"] = (sa["td_avg"] * (1 - sb["td_def"])
                                          - sb["td_avg"] * (1 - sa["td_def"]))
            okey = (r.date, frozenset((ka, kb)))
            if okey in market:
                odds_a, rank_a, *mo_a = market[okey].get(ka, (np.nan,) * 5)
                odds_b, rank_b, *mo_b = market[okey].get(kb, (np.nan,) * 5)
                row["odds_a"], row["odds_b"] = odds_a, odds_b
                # closing method-of-victory odds (market prior for the 6-way)
                for s, mo in (("a", mo_a), ("b", mo_b)):
                    row[f"mo_ko_{s}"], row[f"mo_sub_{s}"], row[f"mo_dec_{s}"] = mo
                # lower rank number is better; positive = A advantage
                row["rank_adv"] = (rank_b - rank_a) if pd.notna(rank_a) and pd.notna(rank_b) else np.nan
                row["ranked_diff"] = int(pd.notna(rank_a)) - int(pd.notna(rank_b))
                better = market[okey].get("_better")
                row["better_rank_adv"] = (1 if better == ka
                                          else (-1 if better == kb else 0))
                matched_odds += 1
            else:
                row["rank_adv"], row["ranked_diff"] = np.nan, 0
                row["better_rank_adv"] = 0
            rows.append(row)

        # ---- fold results into state ----
        res1 = 1 if outcome == "W/L" else (0 if outcome == "L/W" else 0.5)
        # Elo before records mutate; finishes move ratings harder
        elo1_pre, elo2_pre = s1["elo"], s2["elo"]
        exp1 = 1 / (1 + 10 ** ((elo2_pre - elo1_pre) / 400))
        finish_mult = FINISH_K_MULT if ("KO" in str(r.METHOD) or "Submission" in str(r.METHOD)) else 1.0
        k_elo1 = (40 if s1["n"] < 5 else 24) * finish_mult
        k_elo2 = (40 if s2["n"] < 5 else 24) * finish_mult
        s1_new_elo = elo1_pre + k_elo1 * (res1 - exp1)
        s2_new_elo = elo2_pre + k_elo2 * ((1 - res1) - (1 - exp1))
        rd1, rd2 = glicko_rd_now(s1, r.date), glicko_rd_now(s2, r.date)  # pre-fight
        # pre-fight opponent-quality scalars, captured before either state mutates
        wp1 = s1["wins"] / s1["n"] if s1["n"] else np.nan
        wp2 = s2["wins"] / s2["n"] if s2["n"] else np.nan
        sapm1 = s1["opp_sig_l"] / (s1["secs"] / 60) if s1["secs"] else np.nan
        sapm2 = s2["opp_sig_l"] / (s2["secs"] / 60) if s2["secs"] else np.nan
        # judges' scorecards (Phase 13/1C): DETAILS stores loser-first /
        # winner-second (verified vs UFC 284/293 official cards)
        card = None
        if "Decision" in str(r.METHOD) and res1 in (0, 1):
            sc = SCORE_RE.findall(str(r.DETAILS))[:3]
            if len(sc) == 3:
                a = np.array([[int(x), int(y)] for x, y in sc], dtype=float)
                margin = float((a[:, 1] - a[:, 0]).mean())
                card = (float((a[:, 1] / a.sum(axis=1)).mean()),   # winner share
                        int((a[:, 1] < a[:, 0]).any()),            # split
                        int(margin <= 1.01))                       # close
        mech = None
        if "KO" in str(r.METHOD) or "Submission" in str(r.METHOD):
            dtl = str(r.DETAILS)
            mech = (int("Choke" in dtl), int("Ground" in dtl))
        update_state(s1, st1, st2, r.date, res1, r.METHOD, secs, elo2_pre, sched5, weight,
                     traits_of_2, exp1, wp2, sapm2, card, mech)
        update_state(s2, st2, st1, r.date, 1 - res1, r.METHOD, secs, elo1_pre, sched5, weight,
                     traits_of_1, 1 - exp1, wp1, sapm1, card, mech)
        s1["elo"], s2["elo"] = s1_new_elo, s2_new_elo
        s1["peak_elo"] = max(s1["peak_elo"], s1_new_elo)
        s2["peak_elo"] = max(s2["peak_elo"], s2_new_elo)
        g1_r, g1_rd = glicko_update(s1["g_r"], rd1, s2["g_r"], rd2, res1)
        g2_r, g2_rd = glicko_update(s2["g_r"], rd2, s1["g_r"], rd1, 1 - res1)
        s1["g_r"], s1["g_rd"] = g1_r, g1_rd
        s2["g_r"], s2["g_rd"] = g2_r, g2_rd
        for h, rr in ((h1, phys.get(k1, {}).get("reach", np.nan)),
                      (h2, phys.get(k2, {}).get("reach", np.nan))):
            if pd.notna(h):
                div[0] += h; div[1] += 1
            if pd.notna(rr):
                div[2] += rr; div[3] += 1
    return rows, states, div_stats, matched_odds


def main():
    ensure_data(refresh="--refresh" in sys.argv)
    rng = np.random.default_rng(SEED)
    res, stats, phys = load_raw()
    market = load_market()
    pre_ufc = load_pre_ufc(res)
    print(f"pre-UFC records: {len(pre_ufc)} fighters")
    rows, states, div_stats, matched_odds = replay(res, stats, phys, market, rng, pre_ufc)

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
