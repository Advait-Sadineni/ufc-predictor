# UFC Fight Outcome Prediction

Predicts UFC fight winners from pre-fight fighter stats, evaluated against the
betting market. Phase 1 demo: logistic regression on feature differentials with
a strict temporal split (train ≤ 2023-06, test after — never random-split).

## Run

```
pip install -r requirements.txt
python demo.py
```

First run downloads the [Ultimate UFC Dataset](https://github.com/shortlikeafox/ultimate_ufc_dataset)
(~7,200 fights with closing odds) to `data/` and caches it; later runs are offline.

## Current results (test = last 20% of fights, 2023–2026)

| Predictor | Accuracy | Log loss |
|---|---|---|
| Logistic regression (7 features) | 0.626 | 0.636 |
| Better career record baseline | 0.614 | — |
| Market favorite (vig-removed) | **0.699** | **0.581** |

The market is the benchmark to beat, and so far it wins — as expected with
public data. Phase 2 (leak-free rolling career features, gradient boosting,
calibration report) tests whether any gap can be closed, reported honestly
either way.

**Known caveat:** the demo trusts the dataset's own pre-fight aggregates.
Phase 2 recomputes all features from raw fight history as of each fight date
to rule out construction leakage.
