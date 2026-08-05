"""Picks mode: calibrated win probabilities for the next UFC card.

Pulls the upcoming card from ESPN's public scoreboard API (cached 6h), replays
all completed fights to get every fighter's current career state, trains the
stacked ensemble on all completed modern fights (last 365 days held out for
early stopping and stacker fitting, then bases are refit on everything), and
prints a win probability for each announced bout.

Not betting advice: per report.md the model does NOT beat closing odds, and its
edge blended with the market is far too small to clear a sportsbook's vig.

Run: python predict.py [--refresh]
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from build_features import (DATA, ensure_data, load_raw, load_market, new_state,
                            norm_name, opp_traits, replay, snapshot, SEED)
from train_report import (ANTISYM, CONTEXT, MODERN, SNAP_FEATS, fit_lgb, fit_xgb,
                          lgb_params, logit, make_logistic, mirror, xgb_params)
import lightgbm as lgb  # noqa: F401  (via fit_lgb)
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
CACHE_TTL_S = 6 * 3600
BEST_LGB, BEST_XGB = (15, 0.06, 60), (5, 0.03, 5)  # tuned in train_report.py CV

WC_LBS = {"Strawweight": 115, "Flyweight": 125, "Bantamweight": 135,
          "Featherweight": 145, "Lightweight": 155, "Welterweight": 170,
          "Middleweight": 185, "Light Heavyweight": 205, "Heavyweight": 250}


def fetch_card(refresh, date=None):
    """Next upcoming card, or the card on a specific YYYYMMDD date."""
    cache = DATA / f"upcoming_espn_{date or 'next'}.json"
    url = ESPN_URL + (f"?dates={date}" if date else "")
    fresh = cache.exists() and time.time() - cache.stat().st_mtime < CACHE_TTL_S
    if refresh or not fresh:
        print("Fetching card from ESPN...")
        cache.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    events = json.loads(cache.read_text(encoding="utf-8")).get("events", [])
    if not events:
        sys.exit(f"No UFC event found{' on ' + date if date else ''}.")
    return events[0]


def parse_weight(abbrev):
    women = abbrev.startswith("W ")
    name = abbrev[2:] if women else abbrev
    return WC_LBS.get(name, np.nan), int(women)


def bout_features(name1, name2, order1_first, weight, women, periods, title,
                  states, div_stats, phys):
    """Build one feature row for an upcoming bout; fighter A = red corner."""
    k1, k2 = norm_name(name1), norm_name(name2)
    s1 = states.get(k1) or new_state()
    s2 = states.get(k2) or new_state()
    ph1, ph2 = phys.get(k1, {}), phys.get(k2, {})
    today = pd.Timestamp.today().normalize()
    div = div_stats.get(weight if pd.notna(weight) else 0, [0.0, 0, 0.0, 0])
    h1, h2 = ph1.get("height", np.nan), ph2.get("height", np.nan)
    t2 = opp_traits(s2, h1, h2, ph2.get("stance", ""))  # fighter1's opponent
    t1 = opp_traits(s1, h2, h1, ph1.get("stance", ""))
    sa = snapshot(s1, today, ph1, weight, t2, div)
    sb = snapshot(s2, today, ph2, weight, t1, div)
    stance_a, stance_b = ph1.get("stance", ""), ph2.get("stance", "")
    row = {
        "red_corner": 1 if order1_first else -1,
        "southpaw_vs_orthodox": int(stance_a == "Southpaw" and stance_b == "Orthodox")
                              - int(stance_a == "Orthodox" and stance_b == "Southpaw"),
        "title_bout": title, "women": women,
        "sched_rounds": periods, "weight_lbs": weight,
        "rank_adv": np.nan, "ranked_diff": 0,
    }
    for feat in sa:
        row[f"{feat}_diff"] = sa[feat] - sb[feat]
        row[f"{feat}_a"] = sa[feat]
        row[f"{feat}_b"] = sb[feat]
    row["chin_vs_power_diff"] = (sa["kd_taken_p15"] * sb["kd_p15"]
                                 - sb["kd_taken_p15"] * sa["kd_p15"])
    row["grapple_threat_diff"] = (sa["td_avg"] * (1 - sb["td_def"])
                                  - sb["td_avg"] * (1 - sa["td_def"]))
    return row, s1["n"], s2["n"]


def train_stack(df, feats):
    """Train bases with a 365-day holdout for early stopping + stacker, then
    refit bases on everything at the found iteration counts."""
    holdout_start = df["date"].max() - pd.Timedelta(days=365)
    tr, va = df[df["date"] < holdout_start], df[df["date"] >= holdout_start]
    X_tr, y_tr = mirror(tr[feats], tr["a_wins"])
    m_lgb = fit_lgb(lgb_params(*BEST_LGB), X_tr, y_tr, 3000, va[feats], va["a_wins"])
    m_xgb = fit_xgb(xgb_params(*BEST_XGB), X_tr, y_tr, 3000, va[feats], va["a_wins"])
    m_log = make_logistic().fit(X_tr, y_tr)
    stacker = LogisticRegression(random_state=SEED)
    stacker.fit(np.column_stack([logit(m_lgb.predict(va[feats])),
                                 logit(m_xgb.predict(xgb.DMatrix(va[feats]))),
                                 logit(m_log.predict_proba(va[feats])[:, 1])]),
                va["a_wins"])
    X_all, y_all = mirror(df[feats], df["a_wins"])
    f_lgb = fit_lgb(lgb_params(*BEST_LGB), X_all, y_all, m_lgb.best_iteration)
    f_xgb = fit_xgb(xgb_params(*BEST_XGB), X_all, y_all, m_xgb.best_iteration)
    f_log = make_logistic().fit(X_all, y_all)

    def predict(X):
        return stacker.predict_proba(np.column_stack([
            logit(f_lgb.predict(X)), logit(f_xgb.predict(xgb.DMatrix(X))),
            logit(f_log.predict_proba(X)[:, 1])]))[:, 1]
    return predict


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    refresh = "--refresh" in sys.argv
    date = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--date=")), None)
    ensure_data(refresh)
    event = fetch_card(refresh, date)

    print("Replaying fight history...")
    rng = np.random.default_rng(SEED)
    res, stats, phys = load_raw()
    market = load_market()
    rows, states, div_stats, _ = replay(res, stats, phys, market, rng)
    df = pd.DataFrame(rows)
    df = df[df["date"] >= MODERN].sort_values("date").reset_index(drop=True)
    for f in SNAP_FEATS:
        df[f"{f}_mean"] = (df[f"{f}_a"] + df[f"{f}_b"]) / 2
    feats = ANTISYM + [f"{f}_mean" for f in SNAP_FEATS] + CONTEXT

    print(f"Training on {len(df)} completed fights through {df['date'].max().date()}...")
    predict = train_stack(df, feats)
    from props import fit_props
    pred6, predtd = fit_props(df, feats)

    date = event.get("date", "")[:10]
    print(f"\n{event.get('name')}  —  {date}")
    print("=" * 74)
    brows = []
    for c in event.get("competitions", []):
        comps = sorted(c.get("competitors", []), key=lambda x: x.get("order", 9))
        if len(comps) != 2:
            continue
        n1 = comps[0].get("athlete", {}).get("displayName", "?")
        n2 = comps[1].get("athlete", {}).get("displayName", "?")
        abbrev = c.get("type", {}).get("abbreviation", "")
        weight, women = parse_weight(abbrev)
        periods = c.get("format", {}).get("regulation", {}).get("periods", 3)
        title = int("Title" in abbrev or "title" in str(c.get("notes", "")))
        row, nf1, nf2 = bout_features(n1, n2, True, weight, women, periods, title,
                                      states, div_stats, phys)
        brows.append((n1, n2, abbrev, row, nf1, nf2))

    X = pd.DataFrame([b[3] for b in brows])
    for f in SNAP_FEATS:
        X[f"{f}_mean"] = (X[f"{f}_a"] + X[f"{f}_b"]) / 2
    probs = predict(X[feats])
    P6 = pred6(X[feats])
    exp_td = predtd(X[feats])

    for (n1, n2, abbrev, _, nf1, nf2), p, p6, td in zip(brows, probs, P6, exp_td):
        pick, pp = (n1, p) if p >= 0.5 else (n2, 1 - p)
        # method split for the picked side, renormalized to its win probability
        side = p6[:3] if p >= 0.5 else p6[3:]
        ko, sub, dec = pp * side / side.sum()
        dist = p6[2] + p6[5]
        flags = "".join(f"  [{n}: {k} prior UFC fights]"
                        for n, k in ((n1, nf1), (n2, nf2)) if k < 3)
        print(f"{n1:24} vs {n2:24} {abbrev}")
        print(f"    -> {pick} {pp:.0%}  (KO {ko:.0%} | Sub {sub:.0%} | Dec {dec:.0%})"
              f"   distance {dist:.0%}   exp. takedowns {td:.1f}{flags}")
    print("=" * 74)
    print("Model probabilities are calibrated to ~±3% (see report.md) but do NOT")
    print("beat closing odds. This is not betting advice.")


if __name__ == "__main__":
    main()
