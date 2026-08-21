"""EPL demo: 3-way match-outcome model vs no-vig closing odds, top-5 leagues.

Downloads 26 seasons (2000-01 .. 2025-26) of results for E0/SP1/I1/D1/F1 from
football-data.co.uk into epl/data/ (cached forever: a season file already on
disk is never re-fetched), replays all matches chronologically to build basic
pre-match form features (points-per-game and goals for/against over each
team's last 5 matches, computed strictly from matches before kickoff), trains
a LightGBM 3-way classifier (home/draw/away, seed 42) on seasons up to
2023-24, and prints RPS, log loss, and 10-bin per-class calibration on the
2024-25 + 2025-26 test seasons — side by side with the no-vig closing odds
(Pinnacle closing preferred, then Bet365 closing, then pre-close prices,
margin stripped by proportional normalization).

Demo tier, not the real pipeline: form-only features (no Elo, no venue
splits, no rest days), a single temporal split (no CV), and no stacking or
calibration layer. Expect the model NOT to beat the market — closing 1X2
prices are the honest benchmark and matching their calibration is the bar.
All 26 seasons are complete; the in-progress 2026-27 season is deliberately
excluded so both test seasons are full ones. Odds and match-stat columns
are sparse before ~2005 — extra seasons serve as training rows only.

Run: python epl/demo.py
"""

import sys
import urllib.request
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BASE_URL = "https://www.football-data.co.uk/mmz4281"
DIVISIONS = ["E0", "SP1", "I1", "D1", "F1"]        # EPL, La Liga, Serie A, Bundesliga, Ligue 1
SEASONS = [f"{y:02d}{y + 1:02d}" for y in range(0, 26)]    # 0001 .. 2526 (deep history ADOPTED +0.00241)
TEST_START = pd.Timestamp("2024-07-01")            # holdout = 2024-25 + 2025-26 seasons
SEED = 42
ODDS_CHAINS = [                                    # preference order, first complete triple wins
    ("PSC", ("PSCH", "PSCD", "PSCA")),             # Pinnacle closing
    ("B365C", ("B365CH", "B365CD", "B365CA")),     # Bet365 closing
    ("PS", ("PSH", "PSD", "PSA")),                 # Pinnacle pre-close
    ("B365", ("B365H", "B365D", "B365A")),         # Bet365 pre-close
]


def download_season(season, div):
    """Fetch one season CSV to epl/data/<season>_<div>.csv; skip if cached."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"{season}_{div}.csv"
    if out.exists() and out.stat().st_size > 1024:  # re-fetch truncated/empty caches
        return out
    url = f"{BASE_URL}/{season}/{div}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        out.write_bytes(resp.read())
    return out


def load_matches():
    """Return one chronological DataFrame of all matches across leagues/seasons."""
    frames = []
    for season in SEASONS:
        for div in DIVISIONS:
            path = download_season(season, div)
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
            df.columns = [c.replace("﻿", "").replace("ï»¿", "") for c in df.columns]
            df["season"] = season
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Div", "HomeTeam", "AwayTeam", "FTR"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    for _, cols in ODDS_CHAINS:
        for c in cols:
            if c not in df:
                df[c] = np.nan
    df = df.sort_values(["Date", "Div", "HomeTeam"], kind="stable").reset_index(drop=True)
    return df


def market_probs(df):
    """Return (n,3) no-vig probs H/D/A + source label per row (NaN where no odds)."""
    probs = np.full((len(df), 3), np.nan)
    source = np.array([""] * len(df), dtype=object)
    for name, cols in ODDS_CHAINS:
        odds = df[list(cols)].to_numpy(float)
        ok = (source == "") & np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1)
        inv = 1.0 / odds[ok]
        probs[ok] = inv / inv.sum(axis=1, keepdims=True)   # proportional vig strip
        source[ok] = name
    return probs, source


def build_form_features(df):
    """Return DataFrame of last-5 form features per match (NaN until 5 played).

    Per-team state is read BEFORE the row's result is appended, so every
    feature is a pure function of matches strictly before kickoff.
    """
    hist = defaultdict(lambda: deque(maxlen=5))    # team -> last 5 (points, gf, ga)
    rows = []
    for m in df.itertuples():
        feat = {}
        for side, team in (("h", m.HomeTeam), ("a", m.AwayTeam)):
            past = hist[team]
            if len(past) == 5:
                pts, gf, ga = np.mean(past, axis=0)
                feat[f"{side}_ppg5"], feat[f"{side}_gf5"], feat[f"{side}_ga5"] = pts, gf, ga
            else:
                feat[f"{side}_ppg5"] = feat[f"{side}_gf5"] = feat[f"{side}_ga5"] = np.nan
        rows.append(feat)
        h_pts, a_pts = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}[m.FTR]
        hist[m.HomeTeam].append((h_pts, m.FTHG, m.FTAG))
        hist[m.AwayTeam].append((a_pts, m.FTAG, m.FTHG))
    return pd.DataFrame(rows, index=df.index)


def rps(y, p):
    """Mean ranked probability score, outcome order H<D<A (0 = perfect)."""
    outcome = np.zeros_like(p)
    outcome[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((np.cumsum(p, 1) - np.cumsum(outcome, 1)) ** 2, 1) / 2))


def log_loss3(y, p):
    """Mean 3-class log loss (natural log), clipped at 1e-12."""
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def calibration_table(y, p, cls, bins=10):
    """Rows of (bin range, n, mean predicted, observed freq) for one class."""
    idx = np.clip((p[:, cls] * bins).astype(int), 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        out.append((f"{b / bins:.1f}-{(b + 1) / bins:.1f}", int(mask.sum()),
                    float(p[mask, cls].mean()), float((y[mask] == cls).mean())))
    return out


def main():
    df = load_matches()
    print(f"{len(df)} matches, {df['Date'].min():%Y-%m-%d} .. {df['Date'].max():%Y-%m-%d}")

    y = df["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    feats = build_form_features(df)
    mkt, src = market_probs(df)

    test = (df["Date"] >= TEST_START).to_numpy()
    usable = feats.notna().all(axis=1).to_numpy()          # both teams have 5 prior matches
    has_mkt = np.isfinite(mkt).all(axis=1)
    train_m = ~test & usable
    eval_m = test & usable & has_mkt                       # identical rows for model vs market

    from lightgbm import LGBMClassifier
    model = LGBMClassifier(objective="multiclass", num_class=3, n_estimators=300,
                           learning_rate=0.05, num_leaves=31, random_state=SEED, verbose=-1)
    model.fit(feats[train_m], y[train_m])
    p = model.predict_proba(feats[eval_m])
    m = mkt[eval_m]
    yt = y[eval_m]

    print(f"train {train_m.sum()} matches (< {TEST_START:%Y-%m-%d}), "
          f"eval {eval_m.sum()} matches with odds "
          f"({(test & usable & ~has_mkt).sum()} test rows dropped for missing odds)")
    counts = pd.Series(src[eval_m]).value_counts()
    print("odds source on eval rows: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    print(f"\n{'':14s}{'RPS':>8s}{'log loss':>10s}")
    print(f"{'model':14s}{rps(yt, p):8.4f}{log_loss3(yt, p):10.4f}")
    print(f"{'market (nv)':14s}{rps(yt, m):8.4f}{log_loss3(yt, m):10.4f}")

    for cls, name in enumerate(["HOME", "DRAW", "AWAY"]):
        print(f"\ncalibration - {name}  (bin | n | mean pred | observed)")
        for label, probs in (("model", p), ("market", m)):
            for rng, n, pred, obs in calibration_table(yt, probs, cls):
                print(f"  {label:7s}{rng:>10s}{n:6d}{pred:11.3f}{obs:11.3f}")

    assert np.allclose(p.sum(axis=1), 1.0), "model probs must sum to 1"
    assert np.allclose(m.sum(axis=1), 1.0), "market probs must sum to 1"
    assert 0.15 < rps(yt, p) < 0.30, f"model RPS {rps(yt, p):.4f} outside sane range"
    assert rps(yt, m) <= rps(yt, p) + 0.02, "market should not be much worse than model"
    print("\nself-checks passed")


if __name__ == "__main__":
    sys.exit(main())
