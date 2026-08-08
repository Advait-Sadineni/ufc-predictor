"""Backfill full professional records from ESPN for every fighter in the data.

ESPN's scoreboard carries each fighter's total pro record (e.g. 28-1-0).
Those numbers are CURRENT, not as-of-fight-date, so they cannot be used
directly as features. But the PRE-UFC portion is static — a fighter's
regional record never changes — so

    pre_ufc = pro_record_now - ufc_record_now (from our own replay)

is a legitimate leak-free feature, and it is exactly what the model is
missing on debutants and low-experience fighters.

Run: python fetch_pro_records.py   ->  data/pro_records.json
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

from build_features import DATA, norm_name

URL = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard?dates={}"
OUT = DATA / "pro_records.json"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ev = pd.read_csv(DATA / "ufc_event_details.csv")
    dates = sorted({pd.to_datetime(d, format="%B %d, %Y").strftime("%Y%m%d")
                    for d in ev["DATE"]}, reverse=True)
    records = json.loads(OUT.read_text()) if OUT.exists() else {}
    seen_dates = set(records.pop("_dates", []))
    todo = [d for d in dates if d not in seen_dates]
    print(f"{len(records)} fighters cached; fetching {len(todo)} event dates")

    for i, d in enumerate(todo, 1):
        try:
            data = json.loads(urllib.request.urlopen(URL.format(d), timeout=30).read())
        except Exception as e:
            print(f"  {d}: {e}")
            continue
        for event in data.get("events", []):
            for c in event.get("competitions", []):
                for comp in c.get("competitors", []):
                    nm = comp.get("athlete", {}).get("displayName")
                    rec = (comp.get("records") or [{}])[0].get("summary")
                    if nm and rec and norm_name(nm) not in records:
                        records[norm_name(nm)] = rec
        seen_dates.add(d)
        if i % 25 == 0:
            print(f"  {i}/{len(todo)} dates, {len(records)} fighters")
            OUT.write_text(json.dumps({**records, "_dates": sorted(seen_dates)}))
        time.sleep(0.15)  # be polite

    OUT.write_text(json.dumps({**records, "_dates": sorted(seen_dates)}))
    print(f"done: {len(records)} fighters -> {OUT}")


if __name__ == "__main__":
    main()
