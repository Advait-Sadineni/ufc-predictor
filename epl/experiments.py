"""Pre-registered gated experiments on the EPL first-stage LightGBM model.

PRE-REGISTRATION (stated before any result was seen). Metric: mean RPS of
out-of-fold expanding-window LightGBM predictions over the six CV seasons
2018-19..2023-24 (identical fold scheme to train_report.lgb_oof; holdout
never touched). Gate: an experiment is ADOPTED only if it improves that CV
RPS by >= 0.0005 vs the baseline run in the same process; otherwise REJECT,
and the verdict + numbers go into train_report.py's EXPERIMENTS log either
way. Experiments:
  A. league one-hot (Div dummies) as LGB features
  B. walk-forward Dixon-Coles 1X2 probs (half-life 730d, the baseline
     choice) as LGB features — NaN before 2018-19, LGB handles natively
  C. hyperparameter sweep, 4 fixed candidate configs (listed in CONFIGS)
  D. combined adopted changes re-measured together (sanity, same gate vs
     baseline)
  E. (pre-registered 2026-08-21 AFTER A-D ran, BEFORE E ran: A-D all
     REJECTED; B +0.00035 and C3 +0.00048 were near-misses) B+C3 combined —
     DC probs as features AND the leaves15/mc100 config together, same
     >= 0.0005 gate vs the same baseline. One shot; no further
     recombination of A-D pieces after E regardless of outcome.
     OUTCOME: REJECT (0.20340, +0.00036 < gate).
  F. (pre-registered 2026-08-21 before running) per-class isotonic
     calibration of the stack: calibrators fit on the stack's outputs over
     STACK_SEASONS (in-sample for the combiner, but the evaluation set is
     BLEND_SEASONS where both stack and calibrator are out-of-sample).
     Metric: RPS on BLEND_SEASONS, calibrated stack vs raw stack; same
     >= 0.0005 gate. Draws are the known calibration crux, hence the try.

This is a one-off experiment harness, not part of the pipeline: adopted
changes get hardcoded into train_report.py and this script is only evidence.
It reuses train_report's loaders so the fold scheme cannot drift. Results
here are CV-only; nothing in this file reads the holdout.

Run: python epl/experiments.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo  # noqa: E402
import train_report as tr  # noqa: E402

GATE = 0.0005
CONFIGS = [
    ("C1 lr.03 leaves63 n800 mc30", dict(learning_rate=0.03, num_leaves=63,
                                         n_estimators=800, min_child_samples=30)),
    ("C2 lr.02 leaves31 n1000", dict(learning_rate=0.02, n_estimators=1000)),
    ("C3 lr.05 leaves15 n400 mc100", dict(learning_rate=0.05, num_leaves=15,
                                          n_estimators=400, min_child_samples=100)),
    ("C4 colsample.6", dict(colsample_bytree=0.6)),
]


def lgb_cv_rps(df, feat_cols, params):
    """OOF CV RPS over the six CV seasons for one LGB variant."""
    from lightgbm import LGBMClassifier
    y = df["y"].to_numpy()
    probs = np.full((len(df), 3), np.nan)
    for s in tr.CV_SEASONS:
        train = df[df["season"] < s]
        val = df[df["season"] == s]
        model = LGBMClassifier(**params)
        model.fit(train[feat_cols], train["y"])
        probs[val.index] = model.predict_proba(val[feat_cols])
    mask = df["season"].isin(tr.CV_SEASONS).to_numpy()
    return demo.rps(y[mask], probs[mask])


def main():
    only_e = "E" in sys.argv[1:]                   # rerun path: skip A/C/D, they are settled
    print(f"gate (pre-registered): CV RPS improvement >= {GATE}")
    df = tr.load()
    base_cols = ["elo_diff"] + [c for c in df.columns if c.startswith(("h_", "a_"))]

    base = lgb_cv_rps(df, base_cols, tr.LGB_PARAMS)
    print(f"baseline LGB CV RPS: {base:.5f}")

    if "F" in sys.argv[1:]:
        from sklearn.isotonic import IsotonicRegression
        y = df["y"].to_numpy()
        dc, _, _ = tr.dc_predict_seasons(df, tr.CV_SEASONS, 730)
        lgb = np.full((len(df), 3), np.nan)
        from lightgbm import LGBMClassifier
        for s in tr.CV_SEASONS:
            train, val = df[df["season"] < s], df[df["season"] == s]
            m = LGBMClassifier(**tr.LGB_PARAMS)
            m.fit(train[base_cols], train["y"])
            lgb[val.index] = m.predict_proba(val[base_cols])
        sm = df["season"].isin(tr.STACK_SEASONS).to_numpy()
        _, stack = tr.logit_combine([dc[sm], lgb[sm]], y[sm], [dc, lgb])
        bm = df["season"].isin(tr.BLEND_SEASONS).to_numpy()
        cal = np.array(stack)
        for cls in range(3):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
            iso.fit(stack[sm][:, cls], (y[sm] == cls).astype(float))
            cal[:, cls] = iso.predict(np.nan_to_num(stack[:, cls], nan=1 / 3))
        cal /= cal.sum(axis=1, keepdims=True)
        raw_r, cal_r = demo.rps(y[bm], stack[bm]), demo.rps(y[bm], cal[bm])
        print(f"F isotonic stack: raw {raw_r:.5f} cal {cal_r:.5f} "
              f"(delta {raw_r - cal_r:+.5f}) -> "
              f"{'ADOPT' if raw_r - cal_r >= GATE else 'REJECT'}")
        return

    if only_e:
        dc, _, _ = tr.dc_predict_seasons(df, tr.CV_SEASONS, 730)
        df_b = df.copy()
        df_b[["dc_h", "dc_d", "dc_a"]] = dc
        params_e = {**tr.LGB_PARAMS, **dict(CONFIGS)["C3 lr.05 leaves15 n400 mc100"]}
        rps_e = lgb_cv_rps(df_b, base_cols + ["dc_h", "dc_d", "dc_a"], params_e)
        print(f"E B+C3 combined: {rps_e:.5f} (delta {base - rps_e:+.5f}) -> "
              f"{'ADOPT' if base - rps_e >= GATE else 'REJECT'}")
        return

    # A: league one-hot
    div = pd.get_dummies(df["Div"], prefix="div").astype(float)
    df_a = pd.concat([df, div], axis=1)
    rps_a = lgb_cv_rps(df_a, base_cols + list(div.columns), tr.LGB_PARAMS)
    print(f"A league one-hot: {rps_a:.5f} (delta {base - rps_a:+.5f}) -> "
          f"{'ADOPT' if base - rps_a >= GATE else 'REJECT'}")

    # B: walk-forward DC probs as features
    dc, _, _ = tr.dc_predict_seasons(df, tr.CV_SEASONS, 730)
    df_b = df.copy()
    df_b[["dc_h", "dc_d", "dc_a"]] = dc
    rps_b = lgb_cv_rps(df_b, base_cols + ["dc_h", "dc_d", "dc_a"], tr.LGB_PARAMS)
    print(f"B DC probs as features: {rps_b:.5f} (delta {base - rps_b:+.5f}) -> "
          f"{'ADOPT' if base - rps_b >= GATE else 'REJECT'}")

    # C: hyperparameter sweep
    results_c = {}
    for name, over in CONFIGS:
        params = {**tr.LGB_PARAMS, **over}
        results_c[name] = r = lgb_cv_rps(df, base_cols, params)
        print(f"C {name}: {r:.5f} (delta {base - r:+.5f}) -> "
              f"{'ADOPT' if base - r >= GATE else 'REJECT'}")

    # D: combined adopted changes, re-measured under the same gate
    adopted_cols = list(base_cols)
    df_d = df.copy()
    if base - rps_a >= GATE:
        df_d = pd.concat([df_d, div], axis=1)
        adopted_cols += list(div.columns)
    if base - rps_b >= GATE:
        df_d[["dc_h", "dc_d", "dc_a"]] = dc
        adopted_cols += ["dc_h", "dc_d", "dc_a"]
    best_c = min(results_c, key=results_c.get)
    params = {**tr.LGB_PARAMS, **dict(CONFIGS)[best_c]} \
        if base - results_c[best_c] >= GATE else dict(tr.LGB_PARAMS)
    if adopted_cols != base_cols or params != tr.LGB_PARAMS:
        rps_d = lgb_cv_rps(df_d, adopted_cols, params)
        print(f"D combined: {rps_d:.5f} (delta {base - rps_d:+.5f}) -> "
              f"{'ADOPT' if base - rps_d >= GATE else 'REJECT'}")
    else:
        print("D combined: nothing adopted individually, skipped")

    # E: B+C3 combined (pre-registered follow-up, see docstring)
    params_e = {**tr.LGB_PARAMS, **dict(CONFIGS)["C3 lr.05 leaves15 n400 mc100"]}
    rps_e = lgb_cv_rps(df_b, base_cols + ["dc_h", "dc_d", "dc_a"], params_e)
    print(f"E B+C3 combined: {rps_e:.5f} (delta {base - rps_e:+.5f}) -> "
          f"{'ADOPT' if base - rps_e >= GATE else 'REJECT'}")


if __name__ == "__main__":
    sys.exit(main())
