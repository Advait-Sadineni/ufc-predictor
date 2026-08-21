"""Fetch understat.com per-match xG for the top-5 leagues, 2014-15..2025-26.

Hits understat's own XHR endpoint getLeagueData/{league}/{year} (gzipped
JSON; 60 requests total, 0.5s apart, cached forever in epl/data/ as
understat_{league}_{year}.json), keeps finished matches only, then learns
the understat->football-data team-name mapping automatically: matches are
paired by (division, date, home goals, away goals) against features.csv,
each unambiguous pair votes for its name correspondence, and the majority
vote per understat name wins. Output epl/data/understat_matches.csv:
Div, Date, HomeTeam, AwayTeam (football-data names), xg_h, xg_a.

Epistemic status of xG as a feature (locked rule: name the timestamp):
shot events behind each xG value are static match facts, but understat
scores ALL history with its CURRENT xG model, so historical values embed a
model trained partly on later seasons. That leaks shot-conversion priors,
not future match outcomes — accepted and named here, standard for public
xG. The learned name map is only as good as the scoreline pairing; the run
asserts >= 95% of 2014+ matches join, and unmatched understat names are
printed for eyeballing.

Run: python epl/fetch_understat.py
"""

import gzip
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo  # noqa: E402

OUT = demo.DATA / "understat_matches.csv"
LEAGUES = {"EPL": "E0", "La_liga": "SP1", "Serie_A": "I1",
           "Bundesliga": "D1", "Ligue_1": "F1"}
YEARS = range(2014, 2026)                  # season start years, 2014-15..2025-26


def fetch_league_year(league, year):
    """Cache understat getLeagueData JSON to epl/data/, return parsed dict."""
    demo.DATA.mkdir(parents=True, exist_ok=True)
    path = demo.DATA / f"understat_{league}_{year}.json"
    if not path.exists():
        req = urllib.request.Request(
            f"https://understat.com/getLeagueData/{league}/{year}",
            headers={"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"})
        raw = urllib.request.urlopen(req, timeout=60).read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        json.loads(raw)                    # validate before caching
        path.write_bytes(raw)
        time.sleep(0.5)
    return json.loads(path.read_bytes())


def load_understat():
    """All finished understat matches: Div, Date, us_home, us_away, xg_h, xg_a."""
    rows = []
    for league, div in LEAGUES.items():
        for year in YEARS:
            for m in fetch_league_year(league, year)["dates"]:
                if not m["isResult"]:
                    continue
                rows.append({"Div": div, "Date": pd.Timestamp(m["datetime"][:10]),
                             "us_home": m["h"]["title"], "us_away": m["a"]["title"],
                             "gh": int(m["goals"]["h"]), "ga": int(m["goals"]["a"]),
                             "xg_h": float(m["xG"]["h"]), "xg_a": float(m["xG"]["a"])})
    return pd.DataFrame(rows)


def learn_name_map(us, fd):
    """Majority-vote understat->football-data name map via (Div,Date,score) pairs."""
    fd_keyed = defaultdict(list)
    for r in fd.itertuples():
        fd_keyed[(r.Div, r.Date, int(r.FTHG), int(r.FTAG))].append((r.HomeTeam, r.AwayTeam))
    votes = defaultdict(Counter)
    for r in us.itertuples():
        cands = fd_keyed.get((r.Div, r.Date, r.gh, r.ga), [])
        if len(cands) == 1:                # unambiguous pairing only
            votes[r.us_home][cands[0][0]] += 1
            votes[r.us_away][cands[0][1]] += 1
    return {name: c.most_common(1)[0][0] for name, c in votes.items()}


def main():
    us = load_understat()
    fd = pd.read_csv(demo.DATA / "features.csv", parse_dates=["Date"],
                     usecols=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    name_map = learn_name_map(us, fd)
    unmapped = sorted((set(us["us_home"]) | set(us["us_away"])) - set(name_map))
    if unmapped:
        print(f"unmapped understat names ({len(unmapped)}): {unmapped}")
    us["HomeTeam"] = us["us_home"].map(name_map)
    us["AwayTeam"] = us["us_away"].map(name_map)
    out = us.dropna(subset=["HomeTeam", "AwayTeam"])[
        ["Div", "Date", "HomeTeam", "AwayTeam", "xg_h", "xg_a"]]
    out.to_csv(OUT, index=False)

    fd14 = fd[fd["Date"] >= "2014-07-01"]
    joined = fd14.merge(out, on=["Div", "Date", "HomeTeam", "AwayTeam"], how="inner")
    rate = len(joined) / len(fd14)
    print(f"{OUT}: {len(out)} understat matches; join rate vs football-data "
          f"2014+: {len(joined)}/{len(fd14)} = {rate:.1%}")
    assert rate >= 0.95, f"join rate {rate:.1%} below 95% — name map is broken"


if __name__ == "__main__":
    sys.exit(main())
