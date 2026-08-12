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

    # ---- per-fighter counts, duration-aware (Phase 13 / 2A) ----
    # v1 trained on raw full-fight counts, so finish-heavy fighters (short
    # fights) had their totals systematically overestimated (the Luque bug:
    # P(over 24.5 sig) said 70%, reality ~50%). v2 trains PACE (per-minute
    # rate via a log-minutes init_score offset) and puts fight duration back
    # in as a mixture over the distance/under-2.5 heads + empirical duration
    # histograms. Adopted: test Brier beats v1 at sig 24.5/49.5 and TD 0.5/1.5
    # (sides averaged) and on the finisher subset (finish_rate >= 0.6).
    def _fit_rate(col_a, col_b):
        """Mirrored Poisson model of fighter A's per-minute rate of `col`."""
        if "fight_secs" not in df.columns:
            return None
        d = df.dropna(subset=[col_a, col_b, "fight_secs"]).copy()
        d = d[d["fight_secs"] > 0]
        if len(d) <= 500:
            return None
        tr_c, va_c = d["date"] < cut, d["date"] >= cut

        def stack(dd):
            X, _ = mirror(dd[feats], dd["a_wins"])
            y = pd.concat([dd[col_a], dd[col_b]], ignore_index=True)
            off = np.log(pd.concat([dd["fight_secs"]] * 2, ignore_index=True) / 60.0)
            return X, y, off

        X_tr, y_tr, off_tr = stack(d[tr_c])
        X_va, y_va, off_va = stack(d[va_c])
        X_all, y_all, off_all = stack(d)
        p = _params("poisson", {"metric": "poisson"})
        tr_ds = lgb.Dataset(X_tr, y_tr, init_score=off_tr)
        va_ds = lgb.Dataset(X_va, y_va, init_score=off_va, reference=tr_ds)
        m = lgb.train(p, tr_ds, num_boost_round=3000, valid_sets=[va_ds],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        return lgb.train(p, lgb.Dataset(X_all, y_all, init_score=off_all),
                         num_boost_round=m.best_iteration)

    m_tdf = _fit_rate("td_landed_a", "td_landed_b")
    m_sig = _fit_rate("sig_landed_a", "sig_landed_b")
    m_kd = _fit_rate("kd_landed_a", "kd_landed_b")

    # empirical duration histograms from training fights (30s bins)
    def _hist(mask, lo, hi):
        v = df.loc[mask & df["fight_secs"].between(lo, hi - 1), "fight_secs"]
        if len(v) == 0:
            return np.array([(lo + hi) / 2]), np.array([1.0])
        w, edges = np.histogram(v, bins=np.arange(lo, hi + 30, 30))
        centers = (edges[:-1] + edges[1:]) / 2
        keep = w > 0
        return centers[keep], w[keep] / w[keep].sum()

    has_secs = (df["fight_secs"].notna() if "fight_secs" in df.columns
                else pd.Series(False, index=df.index))
    _m3 = (df["sched_rounds"] == 3) & has_secs
    _m5 = (df["sched_rounds"] == 5) & has_secs
    _fin = df["method_cls"] != "dec"
    T3_EARLY, T3_LATE = _hist(_m3, 0, 750), _hist(_m3 & _fin, 750, 900)
    T5_FIN = _hist(_m5 & _fin, 0, 1500)

    def _duration_atoms(X):
        """Per-row (duration atoms in secs, weights) from the distance and
        under-2.5 heads; duration is fight-level so X is never mirrored."""
        p_dist = np.clip(m_dist.predict(X), 0.01, 0.99)
        p_u25 = (np.clip(m_u25.predict(X), 0.01, 0.99) if m_u25 is not None
                 else np.full(len(X), np.nan))
        r5 = X["sched_rounds"].values == 5
        out = []
        for i in range(len(X)):
            if r5[i]:
                atoms = np.append(T5_FIN[0], 1500.0)
                w = np.append((1 - p_dist[i]) * T5_FIN[1], p_dist[i])
            else:
                pe = (p_u25[i] if not np.isnan(p_u25[i])
                      else max(1 - p_dist[i] - 0.15, 0.05))
                pl = max(1 - p_dist[i] - pe, 0.0)
                atoms = np.concatenate([T3_EARLY[0], T3_LATE[0], [900.0]])
                w = np.concatenate([pe * T3_EARLY[1], pl * T3_LATE[1], [p_dist[i]]])
            out.append((atoms, w / w.sum()))
        return out

    def _dispersion(model, col_a):
        """Negative-binomial r from duration-adjusted training residuals —
        duration variance now lives in the mixture, so r captures only the
        skill/style overdispersion around mu = rate x actual minutes."""
        if model is None:
            return None
        d = df.dropna(subset=[col_a, "fight_secs"])
        d = d[d["fight_secs"] > 0]
        mu = model.predict(d[feats]) * (d["fight_secs"] / 60.0)
        mean, var = float(mu.mean()), float(((d[col_a] - mu) ** 2).mean())
        return mean ** 2 / max(var - mean, 1e-6) if var > mean else None

    disp = {"td": _dispersion(m_tdf, "td_landed_a"),
            "sig": _dispersion(m_sig, "sig_landed_a"),
            "kd": _dispersion(m_kd, "kd_landed_a")}
    rate_models = {"td": m_tdf, "sig": m_sig, "kd": m_kd}

    def count_over(X, kind, line, mirrored=False):
        """P(fighter's count > line) via the duration mixture NB tail."""
        from scipy.stats import nbinom, poisson
        model, r = rate_models[kind], disp[kind]
        if model is None:
            return None
        Xq = X.copy()
        if mirrored:
            Xq[ANTISYM] = -Xq[ANTISYM]
        rate = model.predict(Xq)
        out = np.empty(len(X))
        for i, (atoms, w) in enumerate(_duration_atoms(X)):
            mu = np.maximum(rate[i] * atoms / 60.0, 1e-9)
            tail = (1 - nbinom.cdf(np.floor(line), r, r / (r + mu)) if r
                    else 1 - poisson.cdf(np.floor(line), mu))
            out[i] = float((w * tail).sum())
        return out

    def _sided(kind):
        """Expected full-fight count: per-minute rate x expected minutes."""
        model = rate_models[kind]

        def f(X, mirrored=False):
            if model is None:
                return None
            Xq = X.copy()
            if mirrored:
                Xq[ANTISYM] = -Xq[ANTISYM]
            rate = model.predict(Xq)
            exp_min = np.array([(atoms * w).sum() / 60.0
                                for atoms, w in _duration_atoms(X)])
            return rate * exp_min
        return f

    td_fighter = _sided("td")

    return {
        "td_fighter": td_fighter,
        "sig_fighter": _sided("sig"),
        "kd_fighter": _sided("kd"),
        "count_over": count_over,
        "dispersion": disp,
        # raw 6-class stays primary: per-class isotonic calibration was
        # evaluated and REJECTED (test logloss 1.60 -> 2.05; holdout too small
        # per class). Kept under outcome6_cal for the report's record.
        "outcome6": m6.predict,
        "outcome6_cal": pred6_cal,
        "total_td": mt.predict,
        "distance": m_dist.predict,
        "under25": (m_u25.predict if m_u25 is not None else None),
    }
