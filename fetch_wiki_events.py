"""Weight-miss and short-notice history from Wikipedia UFC event pages.

Wikipedia's per-event articles consistently record weigh-in results in prose
("X missed weight, coming in at Y pounds", "Z stepped in on short notice").
No structured dataset of this exists anywhere, so this text-mines it:

  1. `List of UFC events` -> (article title, event date) for past events
  2. batch-fetch event wikitext via the MediaWiki API (50 titles/request)
  3. per event, flag each fighter on that card (names from our own results
     data) whose surname appears near a miss/short-notice phrase

Output cached to data/wiki_events.json:
  {"events": {"YYYY-MM-DD": {"title": ..., "missed": [fighter keys],
                             "short_notice": [fighter keys]}}}

Run: python fetch_wiki_events.py [--refresh]
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

from build_features import DATA, norm_name

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ufc-predictor/1.0 (personal research project)"}
CACHE = DATA / "wiki_events.json"

MISS_RE = re.compile(
    r"(missed weight|failed to make weight|over the .{0,30}limit|"
    r"pounds? over|weight miss|forfeit(?:ed)? .{0,30}purse|fined .{0,30}per ?cent)",
    re.I)
NOTICE_RE = re.compile(
    r"(short.notice|late replacement|stepped in (?:for|to)|"
    r"replac(?:ed|ing) .{0,40}(?:injur|withdr|pull))", re.I)
WINDOW = 300  # chars of context around a phrase in which a surname must appear


def api_get(params, tries=5):
    q = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{q}", headers=HEADERS)
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            wait = 30 * (i + 1)
            print(f"  429 rate-limited, backing off {wait}s...")
            time.sleep(wait)


def list_events():
    """(title, date) for past UFC events from the two list articles."""
    out = {}
    for page in ("List of UFC events",):
        data = api_get({"action": "parse", "page": page, "prop": "wikitext"})
        text = data["parse"]["wikitext"]["*"]
        # table rows: link then a date somewhere in the same row
        for row in text.split("\n|-"):
            m = re.search(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]", row)
            d = re.search(r"\{\{dts\|(\d{4})\|(\w{3,9})\|(\d{1,2})\}\}", row)
            if not d:
                d = re.search(r"(\w{3,9} \d{1,2}, \d{4})", row)
                ds = d.group(1) if d else None
            else:
                ds = f"{d.group(2)} {d.group(3)}, {d.group(1)}"
            if not (m and ds):
                continue
            title = m.group(1).strip()
            if not title.upper().startswith(("UFC", "THE ULTIMATE")):
                continue
            date = pd.to_datetime(ds, errors="coerce")
            if pd.notna(date):
                out.setdefault(str(date.date()), title)
    return out


def fetch_pages(titles):
    """title -> wikitext, batched 50 per request."""
    out = {}
    titles = list(titles)
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        data = api_get({"action": "query", "prop": "revisions", "rvprop": "content",
                        "rvslots": "main", "titles": "|".join(batch),
                        "redirects": 1})
        for p in data["query"]["pages"].values():
            revs = p.get("revisions")
            if revs:
                out[p["title"]] = revs[0]["slots"]["main"]["*"]
        print(f"  fetched {min(i + 50, len(titles))}/{len(titles)} event pages")
        time.sleep(3.0)
    return out


def surname_hits(text, fighters, regex):
    """Fighter keys whose surname appears within WINDOW chars of a phrase hit."""
    hits = set()
    for m in regex.finditer(text):
        ctx = text[max(m.start() - WINDOW, 0): m.end() + WINDOW].lower()
        for full, key in fighters:
            last = full.split()[-1].lower()
            if len(last) >= 4 and last in ctx:
                hits.add(key)
    return sorted(hits)


def main():
    refresh = "--refresh" in sys.argv
    if CACHE.exists() and not refresh:
        print(f"cache exists ({CACHE}); use --refresh to refetch")
        return
    res = pd.read_csv(DATA / "ufc_fight_results.csv")
    ev = pd.read_csv(DATA / "ufc_event_details.csv")
    dates = dict(zip(ev.EVENT.str.strip(), pd.to_datetime(ev.DATE)))
    res["date"] = res.EVENT.str.strip().map(dates)
    fighters_by_date = {}
    for r in res.dropna(subset=["date"]).itertuples():
        for name in str(r.BOUT).split(" vs. "):
            name = name.strip()
            fighters_by_date.setdefault(str(r.date.date()), set()).add(
                (name, norm_name(name)))

    print("Listing events from Wikipedia...")
    events = list_events()
    both = {d: t for d, t in events.items() if d in fighters_by_date}
    print(f"{len(events)} wiki events, {len(both)} matched to our data by date")

    pages = fetch_pages(set(both.values()))
    out = {}
    for date, title in both.items():
        text = None
        for t, txt in pages.items():  # redirects can rename the title
            if t == title or t.startswith(title.split(":")[0]):
                if t == title:
                    text = txt
                    break
                text = text or txt
        if text is None:
            continue
        fl = fighters_by_date[date]
        out[date] = {"title": title,
                     "missed": surname_hits(text, fl, MISS_RE),
                     "short_notice": surname_hits(text, fl, NOTICE_RE)}
    CACHE.write_text(json.dumps({"events": out}, indent=0), encoding="utf-8")
    n_miss = sum(len(e["missed"]) for e in out.values())
    n_sn = sum(len(e["short_notice"]) for e in out.values())
    print(f"saved {len(out)} events: {n_miss} weight-miss flags, {n_sn} short-notice flags")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
