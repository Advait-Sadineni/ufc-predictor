"""Prop models: 6-way outcome (fighter × method) and expected total takedowns.

Classes: 0 A-KO, 1 A-Sub, 2 A-Dec, 3 B-KO, 4 B-Sub, 5 B-Dec.
Everything derives from the 6-way distribution: method of victory, goes the
distance (classes 2+5), any "double chance" combo (e.g. A wins or fight goes
distance = 0+1+2+5). Trained the same way as the winner model: mirrored
antisymmetric features, last-365-day holdout for early stopping, refit on all.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_report import ANTISYM, SEED, mirror

METHOD_IDX = {"ko": 0, "sub": 1, "dec": 2}
CLS_NAMES = ["A by KO/TKO", "A by Sub", "A by Dec", "B by KO/TKO", "B by Sub", "B by Dec"]


def outcome6(df):
    idx = df["method_cls"].map(METHOD_IDX)
    return idx + 3 * (1 - df["a_wins"])


def _mirror_cls(X, y):
    Xm = X.copy()
    Xm[ANTISYM] = -Xm[ANTISYM]
    return (pd.concat([X, Xm], ignore_index=True),
            pd.concat([y, (y + 3) % 6], ignore_index=True))


def _params(objective, extra=None):
    p = dict(objective=objective, num_leaves=15, learning_rate=0.06,
             min_child_samples=60, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, seed=SEED, deterministic=True, force_col_wise=True,
             verbosity=-1)
    p.update(extra or {})
    return p


def _fit(params, X_tr, y_tr, X_va, y_va, X_all, y_all):
    tr = lgb.Dataset(X_tr, y_tr)
    m = lgb.train(params, tr, num_boost_round=3000,
                  valid_sets=[lgb.Dataset(X_va, y_va, reference=tr)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(params, lgb.Dataset(X_all, y_all),
                     num_boost_round=m.best_iteration)


def _fit_binary(df, feats, label, cut):
    """Mirror-augmented binary LightGBM with the standard holdout discipline.
    `label` is a Series aligned to df (symmetric target — same under mirror)."""
    tr_m, va_m = df["date"] < cut, df["date"] >= cut
    X_tr, _ = mirror(df.loc[tr_m, feats], df.loc[tr_m, "a_wins"])
    y_tr = pd.concat([label[tr_m]] * 2, ignore_index=True)
    X_all, _ = mirror(df[feats], df["a_wins"])
    y_all = pd.concat([label] * 2, ignore_index=True)
    p = _params("binary", {"metric": "binary_logloss"})
    return _fit(p, X_tr, y_tr, df.loc[va_m, feats], label[va_m], X_all, y_all)


def fit_props(df, feats, holdout_days=365):
    """Returns a dict of predictors trained on `df`:
      outcome6(X)   calibrated 6-way winner-x-method probabilities
      total_td(X)   expected total takedowns
      distance(X)   P(fight goes the distance) — dedicated binary model
      under25(X)    P(fight ends before 2.5 rounds) — 3-round fights' market bet
    """
    from sklearn.isotonic import IsotonicRegression
    cut = df["date"].max() - pd.Timedelta(days=holdout_days)

    d6 = df.dropna(subset=["method_cls"]).copy()
    y6 = outcome6(d6).astype(int)
    tr_m, va_m = d6["date"] < cut, d6["date"] >= cut
    X_tr, y_tr = _mirror_cls(d6.loc[tr_m, feats], y6[tr_m])
    X_all, y_all = _mirror_cls(d6[feats], y6)
    p6 = _params("multiclass", {"num_class": 6, "metric": "multi_logloss"})
    m6 = _fit(p6, X_tr, y_tr, d6.loc[va_m, feats], y6[va_m], X_all, y_all)
    # per-class isotonic calibration on the holdout, renormalized to sum to 1
    P_va = m6.predict(d6.loc[va_m, feats])
    isos = []
    for k in range(6):
        iso = IsotonicRegression(y_min=1e-3, y_max=0.97, out_of_bounds="clip")
        iso.fit(P_va[:, k], (y6[va_m] == k).astype(int))
        isos.append(iso)

    def pred6_cal(X):
        P = m6.predict(X)
        C = np.column_stack([isos[k].predict(P[:, k]) for k in range(6)])
        return C / C.sum(axis=1, keepdims=True)

    dt = df.dropna(subset=["total_td"]).copy()
    tr_t, va_t = dt["date"] < cut, dt["date"] >= cut
    Xt_tr, _ = mirror(dt.loc[tr_t, feats], dt.loc[tr_t, "a_wins"])
    yt_tr = pd.concat([dt.loc[tr_t, "total_td"]] * 2, ignore_index=True)
    Xt_all, _ = mirror(dt[feats], dt["a_wins"])
    yt_all = pd.concat([dt["total_td"]] * 2, ignore_index=True)
    pt = _params("poisson", {"metric": "rmse"})
    mt = _fit(pt, Xt_tr, yt_tr, dt.loc[va_t, feats], dt.loc[va_t, "total_td"],
              Xt_all, yt_all)

    # dedicated goes-the-distance binary
    dd = df.dropna(subset=["method_cls"]).copy()
    m_dist = _fit_binary(dd, feats, (dd["method_cls"] == "dec").astype(int), cut)

    # under-2.5-rounds (< 750s) on 3-round fights — the market's actual bet
    du = df[(df["sched_rounds"] == 3) & df["fight_secs"].notna()].copy() \
        if "fight_secs" in df.columns else df.iloc[0:0].copy()
    m_u25 = (_fit_binary(du, feats, (du["fight_secs"] < 750).astype(int), cut)
             if len(du) > 500 else None)

    return {
        # raw 6-class stays primary: per-class isotonic calibration was
        # evaluated and REJECTED (test logloss 1.60 -> 2.05; holdout too small
        # per class). Kept under outcome6_cal for the report's record.
        "outcome6": m6.predict,
        "outcome6_cal": pred6_cal,
        "total_td": mt.predict,
        "distance": m_dist.predict,
        "under25": (m_u25.predict if m_u25 is not None else None),
    }
