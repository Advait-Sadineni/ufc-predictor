<div align="center">
  <h1>UFC Fight Outcome Prediction</h1>
  <p>Predicting UFC winners from leak-free historical stats — and honestly measuring whether the model knows anything the betting market doesn't.</p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#tasks">Tasks</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

---

## About The Project

An interview-defensible ML project: train models on ~8,300 UFC fights (2003–2026)
and evaluate them the only way that matters for sports prediction — against the
closing betting line. The deliverable is a **calibration report**
([report.md](report.md)), not picks.

What makes it defensible:

- **Leak-free features.** `build_features.py` replays UFC history in date order,
  maintaining per-fighter career state: records, per-minute striking/grappling
  rates, strike-target and position mixes (head/body/leg, distance/clinch/ground),
  recent-form windows, opponent quality (average opponent Elo), finish-weighted
  Elo with peak-decline tracking, layoffs, weight-class changes, and official
  rankings — 87 features, every one computed strictly from fights *before* the
  one being predicted. Nothing is trusted from pre-aggregated datasets.
- **Strict temporal validation.** Train on fights through 2023-06-03, test on
  the 1,625 fights after. Hyperparameters tuned with 4-fold expanding-window CV
  inside the train period; isotonic calibration and the market blend are fitted
  on out-of-fold predictions only. Fight data is never randomly split.
- **The market is the benchmark.** Closing odds are converted to vig-free
  implied probabilities and scored with the same metrics as the model.

## Results

Test set = 1,200 post-June-2023 fights with closing odds:

| Predictor | Accuracy | Log loss | Brier | ECE |
|---|---|---|---|---|
| Market (no-vig closing odds) | 0.702 | 0.582 | 0.199 | 0.033 |
| Stacked ensemble (5-seed LGB bag + XGB + logistic) | 0.644 | 0.632 | 0.221 | 0.044 |
| **Blend: market + model** | **0.704** | **0.577** | **0.197** | **0.027** |

**The model does not beat the closing line** — no public-data model should be
expected to: closing odds aggregate information (injuries, camp reports, weight
cuts, sharp money) that historical stats cannot see. That is why beating the
closing line, not raw accuracy, is the real benchmark, and why a model claiming
to beat it from public data is usually leaking.

The interesting finding: **the model carries information the market misses.**
Blending model with market beats the market alone (log loss 0.577 vs 0.582) —
better in 98.8% of 10,000 paired bootstrap resamples, 95% CI [+0.0007,
+0.0093]. The significance is seed-sensitive (roughly 95-99% across random
fighter-orientation draws), so read it as a consistent but modest edge — and
one still far too small to clear a sportsbook's vig. Full analysis, reliability diagram,
and feature-importance writeup: **[report.md](report.md)**.

## Built With

[![Python][python-shield]][python-url]

pandas · scikit-learn · LightGBM · XGBoost · matplotlib. Data: raw ufcstats.com scrapes
([Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats))
plus closing odds from the
[Ultimate UFC Dataset](https://github.com/shortlikeafox/ultimate_ufc_dataset).

## Getting Started

### Prerequisites

Python 3.10+.

### Installation

```
pip install -r requirements.txt
```

## Usage

```
python demo.py            # Phase 1 demo: logistic baseline on pre-built data
python build_features.py  # replay fight history -> data/features.csv (leak-free)
python train_report.py    # tune, train, evaluate -> report.md + plots/
python predict.py         # picks for the next card: winner, method, distance, takedowns
python predict.py --date=20260815   # ...or any scheduled card by date
# optional: set ODDS_API_KEY (free, the-odds-api.com) to compare picks
# against FanDuel moneylines, with model-vs-book EV shown per disagreement
python track_bets.py ...  # personal bet log: add / result / report (P/L, CLV)
```

All downloads are cached under `data/` on first run (`--refresh` re-fetches,
including new events). Seeds are fixed (42) — reruns reproduce the reported
numbers. `predict.py` prints calibrated probabilities only — per the report the
model does not beat closing odds, and it will tell you so itself.

## Tasks

- [x] Phase 1: demo slice (logistic regression, temporal split, market baseline)
- [x] Phase 2: leak-free features, Elo, LightGBM, expanding-window CV, calibration report
- [x] Phase 3: strike-location/position mixes, recent form, opponent quality, finish-weighted Elo, rankings, stacked LGB+XGB+logistic ensemble
- [x] Phase 4 (negative result): style-matchup splits, trait interactions, and a cut-severity proxy were built, evaluated, and rejected — too sparse, no test-set improvement (see report.md § Negative results)
- [x] Picks mode: `predict.py` for upcoming cards (probabilities only — no bet sizing, by design)
- [x] Prop models: 6-way outcome (winner × method), goes-the-distance, expected takedowns — all evaluated vs baselines in report.md
- [x] Price intelligence: all-US-books line shopping + round-totals markets; Under-2.5 model (adopted)
- [x] Track-record machine: full-outcome grading (winners/distance/U2.5 + prop-market Brier), line-movement panel, model-vs-hunch bet tags
- [x] 5-seed LightGBM bag (adopted: CV 0.6571→0.6535); Glicko ratings, recency weighting, method calibration all evaluated and rejected (see report.md)
- [x] Data refresh (`--refresh`) and personal bet log with closing-line-value report (`track_bets.py`)
- [ ] Resolve duplicate-name fighters via ufcstats fighter URLs
- [ ] Pre-UFC records for debut fighters (model is weakest on debuts)

## Contact

Advait Sadineni — sadineni.advait@gmail.com

---

<!-- Reference-style links -->
[python-shield]: https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white
[python-url]: https://python.org
