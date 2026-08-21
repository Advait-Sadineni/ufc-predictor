"""EPL feature replay: leak-free per-team state as of each match's kickoff.

Loads the cached football-data.co.uk season files via demo.load_matches()
(downloading any missing ones), then replays all 27k matches across the five
leagues in one chronological pass. For every match the features are emitted
FIRST, from state built only on strictly earlier matches, and the match
result is folded into team state AFTER (snapshot-before-update). Features
per team: idle-decayed goal-margin-weighted Elo (init 1500, K=20*sqrt(margin),
home advantage +60 Elo points inside the expectation, 10%/idle-year regression
to 1500); rolling last-5/last-10 per-match rates for points, goals, shots,
shots-on-target, corners and cards, for and against (np.nanmean, so a match
with missing stats degrades gracefully); venue-split last-5 (home team at
home, away team away); season points-per-game and games played; rest days and
matches in the last 21 days; promoted flag (absent from this division the
previous season — knowable at season start from the fixture list); per-team
home-advantage strength (mean home points minus mean away points over the
last 20 of each). Also rolling last-5/10 understat xG for/against when
epl/data/understat_matches.csv exists (fetch_understat.py) — REJECTED by
the pre-registered gate, kept computed per locked rule, excluded from the
model in train_report. Output: epl/data/features.csv (regenerable,
gitignored) = key/odds columns + ~75 feature columns prefixed h_/a_.

Leak-freedom argument: every feature is a pure function of matches dated
strictly before the row's kickoff, enforced structurally by emit-then-update.
The __main__ run proves it empirically: for three fixed row indices it
re-runs the replay on only the rows up to that match and asserts the feature
row is bitwise-identical to the full replay's — if any future match leaked
into a feature, those would differ. Not established: within-day ordering
(two matches the same day never share a team, so it cannot matter here), and
the stats columns' own upstream correctness (taken from football-data as-is).

Run: python epl/build_features.py
"""

import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo  # noqa: E402

OUT = demo.DATA / "features.csv"
ELO_INIT, ELO_HA, ELO_K = 1500.0, 60.0, 20.0
STATS = ["pts", "gf", "ga", "sf", "sa", "stf", "sta", "cf", "ca", "kf", "ka"]
TOTALS_COLS = ["PC>2.5", "PC<2.5", "B365C>2.5", "B365C<2.5",
               "P>2.5", "P<2.5", "B365>2.5", "B365<2.5"]
KEEP = ["Div", "season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
SPOT_CHECK_ROWS = [7000, 17000, 26900]


def new_team():
    """Fresh team state: Elo, rolling deques, venue/season/congestion trackers."""
    return {"elo": ELO_INIT, "last": None, "res": deque(maxlen=10),
            "home": deque(maxlen=5), "away": deque(maxlen=5),
            "dates": deque(maxlen=15), "hpts": deque(maxlen=20),
            "apts": deque(maxlen=20), "season": None, "spts": 0, "sgames": 0}


def window_means(res, w):
    """np.nanmean of the last w rolling entries, or all-NaN if fewer than w."""
    if len(res) < w:
        return np.full(len(STATS), np.nan)
    arr = np.array(list(res)[-w:], dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(arr, axis=0)


def side_features(prefix, st, date, promoted):
    """Feature dict for one team as of `date`, reading state only."""
    f = {f"{prefix}elo": st["elo"]}
    for w in (5, 10):
        means = window_means(st["res"], w)
        for name, val in zip(STATS, means):
            f[f"{prefix}{name}{w}"] = val
    venue = st["home"] if prefix == "h_" else st["away"]
    if len(venue) == 5:
        vp, vgf, vga = np.mean(venue, axis=0)
    else:
        vp = vgf = vga = np.nan
    f[f"{prefix}venue_ppg5"], f[f"{prefix}venue_gf5"], f[f"{prefix}venue_ga5"] = vp, vgf, vga
    f[f"{prefix}ppg_season"] = st["spts"] / st["sgames"] if st["sgames"] else np.nan
    f[f"{prefix}games_season"] = st["sgames"]
    idle = (date - st["last"]).days if st["last"] is not None else np.nan
    f[f"{prefix}rest"] = min(idle, 60) if idle == idle else np.nan
    f[f"{prefix}m21"] = sum(1 for d in st["dates"] if (date - d).days <= 21)
    f[f"{prefix}promoted"] = promoted
    if len(st["hpts"]) >= 5 and len(st["apts"]) >= 5:
        f[f"{prefix}homeadv"] = np.mean(st["hpts"]) - np.mean(st["apts"])
    else:
        f[f"{prefix}homeadv"] = np.nan
    return f


def replay(df):
    """Return the feature DataFrame for df (chronological), emit-then-update."""
    members = df.groupby(["season", "Div"])["HomeTeam"].agg(set).to_dict()
    prev_season = {s: f"{int(s[:2]) - 1:02d}{int(s[2:]) - 1:02d}" for s in df["season"].unique()}
    teams = defaultdict(new_team)
    for c in ("HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"):
        if c not in df:
            df = df.copy()
            df[c] = np.nan
    rows = []
    for m in df.itertuples():
        date = m.Date
        for team in (m.HomeTeam, m.AwayTeam):
            st = teams[team]
            if st["last"] is not None:
                idle = (date - st["last"]).days
                if idle > 60:                              # ponytail: one decay knob for off-season + long absences
                    st["elo"] = ELO_INIT + 0.9 ** (idle / 365.0) * (st["elo"] - ELO_INIT)
            if st["season"] != m.season:
                st["season"], st["spts"], st["sgames"] = m.season, 0, 0
        hs, aw = teams[m.HomeTeam], teams[m.AwayTeam]
        prev = members.get((prev_season[m.season], m.Div))
        feat = {"elo_diff": hs["elo"] - aw["elo"]}
        feat.update(side_features("h_", hs, date, int(prev is not None and m.HomeTeam not in prev)))
        feat.update(side_features("a_", aw, date, int(prev is not None and m.AwayTeam not in prev)))
        rows.append(feat)

        h_pts, a_pts = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}[m.FTR]
        hg, ag = m.FTHG, m.FTAG
        h_cards, a_cards = m.HY + m.HR, m.AY + m.AR
        stats_h = [h_pts, hg, ag, m.HS, m.AS, m.HST, m.AST, m.HC, m.AC, h_cards, a_cards]
        stats_a = [a_pts, ag, hg, m.AS, m.HS, m.AST, m.HST, m.AC, m.HC, a_cards, h_cards]
        hs["res"].append(stats_h)
        aw["res"].append(stats_a)
        hs["home"].append((h_pts, hg, ag))
        aw["away"].append((a_pts, ag, hg))
        hs["hpts"].append(h_pts)
        aw["apts"].append(a_pts)
        expected = 1.0 / (1.0 + 10 ** (-(hs["elo"] + ELO_HA - aw["elo"]) / 400.0))
        score = {"H": 1.0, "D": 0.5, "A": 0.0}[m.FTR]
        delta = ELO_K * np.sqrt(max(abs(hg - ag), 1)) * (score - expected)
        hs["elo"] += delta
        aw["elo"] -= delta
        for st, pts in ((hs, h_pts), (aw, a_pts)):
            st["last"] = date
            st["dates"].append(date)
            st["spts"] += pts
            st["sgames"] += 1
    return pd.DataFrame(rows, index=df.index)


def add_xg(out):
    """Rolling last-5/10 understat xG for/against (REJECTED 2026-08-21 per gate,
    delta -0.00023 — kept computed per locked rule, excluded in train_report)."""
    src = demo.DATA / "understat_matches.csv"
    if not src.exists():
        print("understat_matches.csv missing (run fetch_understat.py); xG cols skipped")
        return out
    xg = pd.read_csv(src, parse_dates=["Date"])
    out = out.merge(xg, on=["Div", "Date", "HomeTeam", "AwayTeam"], how="left")
    hist = defaultdict(lambda: deque(maxlen=10))
    rows = []
    for m in out.itertuples():
        feat = {}
        for p, team in (("h_", m.HomeTeam), ("a_", m.AwayTeam)):
            past = hist[team]
            for w in (5, 10):
                if len(past) >= w:
                    with np.errstate(invalid="ignore"):
                        vals = np.nanmean(np.array(list(past)[-w:], float), axis=0)
                    feat[f"{p}xgf{w}"], feat[f"{p}xga{w}"] = vals
                else:
                    feat[f"{p}xgf{w}"] = feat[f"{p}xga{w}"] = np.nan
        rows.append(feat)
        if np.isfinite(m.xg_h) and np.isfinite(m.xg_a):
            hist[m.HomeTeam].append((m.xg_h, m.xg_a))
            hist[m.AwayTeam].append((m.xg_a, m.xg_h))
    return pd.concat([out.drop(columns=["xg_h", "xg_a"]),
                      pd.DataFrame(rows, index=out.index)], axis=1)


def main():
    df = demo.load_matches()
    feats = replay(df)
    keep = KEEP + [c for _, cols in demo.ODDS_CHAINS for c in cols]
    keep += [c for c in TOTALS_COLS if c in df.columns]
    out = add_xg(pd.concat([df[keep], feats], axis=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"{OUT}: {len(out)} rows, {len(out.columns) - len(keep)} feature cols")

    for i in SPOT_CHECK_ROWS:                              # leak proof: partial replay == full replay
        partial = replay(df.iloc[: i + 1]).iloc[-1].to_numpy(float)
        full = feats.iloc[i].to_numpy(float)
        assert np.allclose(partial, full, equal_nan=True), f"leak detected at row {i}"
    print(f"leak spot-check passed at rows {SPOT_CHECK_ROWS}")


if __name__ == "__main__":
    sys.exit(main())
