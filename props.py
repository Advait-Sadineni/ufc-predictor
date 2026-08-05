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


def fit_props(df, feats, holdout_days=365):
    """Returns (predict_outcome6, predict_total_td) trained on `df`."""
    cut = df["date"].max() - pd.Timedelta(days=holdout_days)

    d6 = df.dropna(subset=["method_cls"]).copy()
    y6 = outcome6(d6).astype(int)
    tr_m, va_m = d6["date"] < cut, d6["date"] >= cut
    X_tr, y_tr = _mirror_cls(d6.loc[tr_m, feats], y6[tr_m])
    X_all, y_all = _mirror_cls(d6[feats], y6)
    p6 = _params("multiclass", {"num_class": 6, "metric": "multi_logloss"})
    m6 = _fit(p6, X_tr, y_tr, d6.loc[va_m, feats], y6[va_m], X_all, y_all)

    dt = df.dropna(subset=["total_td"]).copy()
    tr_t, va_t = dt["date"] < cut, dt["date"] >= cut
    Xt_tr, _ = mirror(dt.loc[tr_t, feats], dt.loc[tr_t, "a_wins"])
    yt_tr = pd.concat([dt.loc[tr_t, "total_td"]] * 2, ignore_index=True)
    Xt_all, _ = mirror(dt[feats], dt["a_wins"])
    yt_all = pd.concat([dt["total_td"]] * 2, ignore_index=True)
    pt = _params("poisson", {"metric": "rmse"})
    mt = _fit(pt, Xt_tr, yt_tr, dt.loc[va_t, feats], dt.loc[va_t, "total_td"],
              Xt_all, yt_all)
    return m6.predict, mt.predict
