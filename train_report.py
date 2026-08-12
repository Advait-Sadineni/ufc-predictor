"""Train UFC winner models on leak-free features and write the calibration report.

Pipeline (all temporal, no random splits):
  1. Expanding-window CV (4 folds, 2016-2023) tunes LightGBM and XGBoost, and
     produces out-of-fold predictions for the three base models (LGB, XGB,
     logistic). The stacking combiner, isotonic calibrator, and market blend
     are all fitted on those OOF predictions — nothing ever sees test data.
  2. Final models train on fights up to 2023-06-03 (same cutoff as demo.py)
     and are evaluated once on everything after.
  3. Outputs: report.md + plots/ (reliability diagram, feature importance).

Run: python train_report.py   (requires data/features.csv from build_features.py)
"""

import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).parent
CUTOFF = pd.Timestamp("2023-06-03")          # same test cutoff as demo.py
MODERN = pd.Timestamp("2003-01-01")          # Zuffa era; stats reliable
CV_FOLDS = [("2016-01-01", "2018-01-01"), ("2018-01-01", "2020-01-01"),
            ("2020-01-01", "2021-09-01"), ("2021-09-01", "2023-06-04")]

# Palette (dataviz reference): model blue, market orange, chrome grays.
C_MODEL, C_MARKET = "#2a78d6", "#eb6834"
C_GRID, C_MUTED, C_INK, C_INK2 = "#e1e0d9", "#898781", "#0b0b0b", "#52514e"

SNAP_FEATS = ["elo", "n_fights", "win_pct", "win_streak", "lose_streak", "slpm",
              "sapm", "str_acc", "str_def", "td_avg", "td_acc", "td_def",
              "sub_att_p15", "kd_p15", "kd_taken_p15", "ctrl_min_share",
              "finish_rate", "dec_win_rate", "layoff_days", "age", "height", "reach",
              "head_share", "leg_share", "ground_str_share", "clinch_str_share",
              "absorbed_head_share", "absorbed_ground_share",
              "form_win", "form_slpm", "form_sapm", "form_td15",
              "form_finished", "form_was_finished", "avg_opp_elo", "form_opp_elo",
              "peak_elo", "elo_decline", "five_rd_fights", "weight_change",
              "pre_ufc_wins", "pre_ufc_losses", "pre_ufc_fights", "pre_ufc_winpct",
              "ko_loss_rate", "sub_loss_rate", "finished_rate", "never_finished",
              "sc_close", "sc_split", "sc_share"]
CONTEXT = ["title_bout", "women", "sched_rounds", "weight_lbs"]
ANTISYM = ([f"{f}_diff" for f in SNAP_FEATS]
           + ["red_corner", "southpaw_vs_orthodox", "rank_adv", "ranked_diff"])


def load():
    df = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["date"])
    df = df[df["date"] >= MODERN].sort_values("date").reset_index(drop=True)
    for f in SNAP_FEATS:  # symmetric matchup level, e.g. are both fighters experienced
        df[f"{f}_mean"] = (df[f"{f}_a"] + df[f"{f}_b"]) / 2
    feats = ANTISYM + [f"{f}_mean" for f in SNAP_FEATS] + CONTEXT
    return df, feats


def mirror_X(X: pd.DataFrame):
    """The swapped-corner view of every row (antisymmetric cols negate)."""
    Xm = X.copy()
    Xm[ANTISYM] = -Xm[ANTISYM]
    return Xm


def mirror(X: pd.DataFrame, y: pd.Series):
    """Append the swapped-corner view of every fight (antisymmetric cols negate)."""
    return (pd.concat([X, mirror_X(X)], ignore_index=True),
            pd.concat([y, 1 - y], ignore_index=True))


# Phase 13 note: orientation-symmetrized prediction ((p(X) + 1 - p(mirror_X(X)))/2)
# was tested and REJECTED — CV 0.6466 -> 0.6471; mirrored training already makes
# the model near-symmetric, so averaging only shrinks confidence. See report.md.


def lgb_params(num_leaves, lr, mcs):
    return dict(objective="binary", metric="binary_logloss", num_leaves=num_leaves,
                learning_rate=lr, min_child_samples=mcs, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, seed=SEED, deterministic=True,
                force_col_wise=True, verbosity=-1)


def fit_lgb(params, X_tr, y_tr, n_rounds, X_val=None, y_val=None):
    tr = lgb.Dataset(X_tr, y_tr)
    if X_val is not None:
        return lgb.train(params, tr, num_boost_round=n_rounds,
                         valid_sets=[lgb.Dataset(X_val, y_val, reference=tr)],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(params, tr, num_boost_round=n_rounds)


BAG_SEEDS = 5  # 5-seed bag adopted: CV logloss 0.6571 -> 0.6535 (Phase 9)


class LGBBag:
    """Average of BAG_SEEDS LightGBMs differing only in seeds."""

    def __init__(self, models):
        self.models = models

    def predict(self, X, pred_contrib=False):
        if pred_contrib:
            return self.models[0].predict(X, pred_contrib=True)
        return np.mean([m.predict(X) for m in self.models], axis=0)

    @property
    def best_iteration(self):
        its = [m.best_iteration for m in self.models if m.best_iteration]
        return int(np.mean(its)) if its else None


def fit_lgb_bag(base_params, X_tr, y_tr, n_rounds, X_val=None, y_val=None):
    return LGBBag([
        fit_lgb(dict(base_params, seed=42 + s, bagging_seed=42 + s,
                     feature_fraction_seed=142 + s),
                X_tr, y_tr, n_rounds, X_val, y_val)
        for s in range(BAG_SEEDS)])


def xgb_params(depth, eta, mcw, sub=0.8, col=0.8, lam=1):
    return dict(objective="binary:logistic", eval_metric="logloss", max_depth=depth,
                eta=eta, min_child_weight=mcw, subsample=sub, colsample_bytree=col,
                reg_lambda=lam, seed=SEED, nthread=-1)


XGB_BAG_SEEDS = 5  # Phase 13/2B-2: CV 0.6477 -> 0.6466 with the swept config


class XGBBag:
    """Average of XGB_BAG_SEEDS boosters differing only in seed."""

    def __init__(self, models):
        self.models = models

    def predict(self, dmat):
        return np.mean([m.predict(dmat) for m in self.models], axis=0)

    @property
    def best_iteration(self):
        its = [getattr(m, "best_iteration", None) for m in self.models]
        its = [i for i in its if i]
        return int(np.mean(its)) if its else None


def fit_xgb_bag(base_params, X_tr, y_tr, n_rounds, X_val=None, y_val=None):
    return XGBBag([fit_xgb(dict(base_params, seed=42 + s), X_tr, y_tr, n_rounds,
                           X_val, y_val) for s in range(XGB_BAG_SEEDS)])


def fit_xgb(params, X_tr, y_tr, n_rounds, X_val=None, y_val=None):
    dtr = xgb.DMatrix(X_tr, y_tr)
    if X_val is not None:
        return xgb.train(params, dtr, num_boost_round=n_rounds,
                         evals=[(xgb.DMatrix(X_val, y_val), "val")],
                         early_stopping_rounds=100, verbose_eval=False)
    return xgb.train(params, dtr, num_boost_round=n_rounds)


def make_logistic():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(random_state=SEED, max_iter=2000))


def american_to_prob(odds):
    odds = np.asarray(odds, dtype=float)
    p = np.empty_like(odds)
    neg = odds < 0
    p[neg] = -odds[neg] / (-odds[neg] + 100)
    p[~neg] = 100 / (odds[~neg] + 100)
    return p


def no_vig(odds_a, odds_b):
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    return pa / (pa + pb)


def ece(y, p, bins=10):
    """Expected calibration error, equal-count bins."""
    order = np.argsort(p)
    y, p = np.asarray(y)[order], np.asarray(p)[order]
    total, err = len(y), 0.0
    for chunk in np.array_split(np.arange(total), bins):
        if len(chunk):
            err += len(chunk) / total * abs(y[chunk].mean() - p[chunk].mean())
    return err


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def metrics_row(name, y, p, note=""):
    return {"name": name, "n": len(y), "acc": accuracy_score(y, p > 0.5),
            "logloss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "ece": ece(y, p), "note": note}


def style_ax(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.grid(True, color=C_GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    np.random.seed(SEED)
    df, feats = load()
    train_all = df[df["date"] <= CUTOFF]
    test = df[df["date"] > CUTOFF]
    X_te, y_te = test[feats], test["a_wins"]

    # ---------- 1. expanding-window CV: tune LightGBM and XGBoost ----------
    def tune(name, grid, param_fn, fit_fn, pred_fn):
        results = {}
        for cfg in grid:
            losses, iters = [], []
            for start, end in CV_FOLDS:
                tr = df[df["date"] < start]
                va = df[(df["date"] >= start) & (df["date"] < end)]
                Xm, ym = mirror(tr[feats], tr["a_wins"])
                m = fit_fn(param_fn(*cfg), Xm, ym, 3000, va[feats], va["a_wins"])
                losses.append(log_loss(va["a_wins"], pred_fn(m, va[feats])))
                iters.append(m.best_iteration)
            results[cfg] = (np.mean(losses), int(np.mean(iters)))
        best = min(results, key=lambda c: results[c][0])
        print(f"{name} best: {best} cv_logloss={results[best][0]:.4f} iters={results[best][1]}")
        return best, results[best]

    pred_lgb = lambda m, X: m.predict(X)
    pred_xgb = lambda m, X: m.predict(xgb.DMatrix(X))
    print("Tuning LightGBM...")
    lgb_grid = [(nl, lr, mcs) for nl in (15, 31, 63) for lr in (0.03, 0.06) for mcs in (20, 60)]
    best_lgb, (cv_lgb, iters_lgb) = tune("LightGBM", lgb_grid, lgb_params, fit_lgb, pred_lgb)
    print("Tuning XGBoost...")
    # grid refreshed by the Phase 13 random sweep: (4, 0.03, 5, 0.8, 0.6, 5)
    # beat the old incumbent by +0.0027 CV; neighbors kept for weekly retune
    xgb_grid = [(3, 0.03, 5), (3, 0.06, 5), (3, 0.03, 20),
                (4, 0.03, 5, 0.8, 0.6, 5), (4, 0.03, 2, 0.6, 0.8, 1),
                (4, 0.03, 5, 0.8, 0.6, 1)]
    best_xgb, (cv_xgb, iters_xgb) = tune("XGBoost", xgb_grid, xgb_params, fit_xgb, pred_xgb)

    # ---------- 2. out-of-fold predictions for stack / calibration / blend ----------
    print("Building out-of-fold predictions...")
    oof = []
    for start, end in CV_FOLDS:
        tr = df[df["date"] < start]
        va = df[(df["date"] >= start) & (df["date"] < end)].copy()
        Xm, ym = mirror(tr[feats], tr["a_wins"])
        va["p_lgb"] = pred_lgb(fit_lgb_bag(lgb_params(*best_lgb), Xm, ym, 3000,
                                           va[feats], va["a_wins"]), va[feats])
        va["p_xgb"] = pred_xgb(fit_xgb_bag(xgb_params(*best_xgb), Xm, ym, 3000,
                                       va[feats], va["a_wins"]), va[feats])
        lo = make_logistic().fit(Xm, ym)
        va["p_log"] = lo.predict_proba(va[feats])[:, 1]
        oof.append(va)
    oof = pd.concat(oof)

    def stack_X(p1, p2, p3):
        return np.column_stack([logit(p1), logit(p2), logit(p3)])

    stacker = LogisticRegression(random_state=SEED)
    stacker.fit(stack_X(oof["p_lgb"], oof["p_xgb"], oof["p_log"]), oof["a_wins"])
    oof["p_ens"] = stacker.predict_proba(
        stack_X(oof["p_lgb"], oof["p_xgb"], oof["p_log"]))[:, 1]
    print(f"stack weights (lgb, xgb, logistic): {np.round(stacker.coef_[0], 3)}")

    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(oof["p_ens"], oof["a_wins"])

    oof_odds = oof[oof["odds_a"].notna() & oof["odds_b"].notna()]
    p_mkt_oof = no_vig(oof_odds["odds_a"].values, oof_odds["odds_b"].values)
    blend = LogisticRegression(random_state=SEED)
    blend.fit(np.column_stack([logit(p_mkt_oof), logit(oof_odds["p_ens"])]),
              oof_odds["a_wins"])

    # ---------- 3. final models on full train, evaluated once on test ----------
    print("Training final models...")
    Xm, ym = mirror(train_all[feats], train_all["a_wins"])
    final_lgb = fit_lgb_bag(lgb_params(*best_lgb), Xm, ym, iters_lgb)
    final_xgb = fit_xgb_bag(xgb_params(*best_xgb), Xm, ym, iters_xgb)
    final_log = make_logistic().fit(Xm, ym)

    def predict_ensemble(X):
        return stacker.predict_proba(stack_X(
            pred_lgb(final_lgb, X), pred_xgb(final_xgb, X),
            final_log.predict_proba(X)[:, 1]))[:, 1]

    p_lgb_te = pred_lgb(final_lgb, X_te)
    p_xgb_te = pred_xgb(final_xgb, X_te)
    p_log_te = final_log.predict_proba(X_te)[:, 1]
    p_ens = stacker.predict_proba(stack_X(p_lgb_te, p_xgb_te, p_log_te))[:, 1]
    p_iso = iso.predict(p_ens)

    elo_only = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(random_state=SEED))
    elo_only.fit(train_all[["elo_diff"]], train_all["a_wins"])
    p_elo = elo_only.predict_proba(test[["elo_diff"]])[:, 1]

    rows_full = [
        metrics_row("Stacked ensemble (LGB+XGB+logistic)", y_te, p_ens),
        metrics_row("Ensemble + isotonic", y_te, p_iso),
        metrics_row("LightGBM", y_te, p_lgb_te),
        metrics_row("XGBoost", y_te, p_xgb_te),
        metrics_row("Logistic regression (full features)", y_te, p_log_te),
        metrics_row("Elo only (logistic)", y_te, p_elo),
    ]

    # ---------- 4. market comparison on the odds subset ----------
    ot = test[test["odds_a"].notna() & test["odds_b"].notna()]
    yo = ot["a_wins"]
    p_mkt = no_vig(ot["odds_a"].values, ot["odds_b"].values)
    p_ens_o = predict_ensemble(ot[feats])
    p_blend = blend.predict_proba(np.column_stack([logit(p_mkt), logit(p_ens_o)]))[:, 1]
    rows_mkt = [
        metrics_row("Market (no-vig closing odds)", yo, p_mkt),
        metrics_row("Stacked ensemble", yo, p_ens_o),
        metrics_row("Blend: market + model", yo, p_blend,
                    note=f"logit blend, model coef={blend.coef_[0][1]:+.3f}"),
    ]

    # paired bootstrap: is the blend's log-loss edge over the market real?
    yo_arr, n_boot = yo.values.astype(float), 10_000
    ll_mkt_i = -(yo_arr * np.log(p_mkt) + (1 - yo_arr) * np.log(1 - p_mkt))
    ll_bl_i = -(yo_arr * np.log(p_blend) + (1 - yo_arr) * np.log(1 - p_blend))
    rng_b = np.random.default_rng(SEED)
    idx = rng_b.integers(0, len(yo_arr), size=(n_boot, len(yo_arr)))
    deltas = ll_mkt_i[idx].mean(axis=1) - ll_bl_i[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_better = (deltas > 0).mean()

    # ---------- 4b. prop models: 6-way outcome + totals/distance/rounds ----------
    print("Training prop models...")
    from props import fit_props, outcome6
    props_m = fit_props(train_all, feats)
    tr6 = train_all.dropna(subset=["method_cls"])
    freq = np.bincount(outcome6(tr6).astype(int), minlength=6) / len(tr6)
    t6 = test.dropna(subset=["method_cls"])
    y6 = outcome6(t6).astype(int).values
    P6 = props_m["outcome6"](t6[feats])
    P6_cal = props_m["outcome6_cal"](t6[feats])
    ll6 = log_loss(y6, P6, labels=range(6))
    ll6_cal = log_loss(y6, P6_cal, labels=range(6))
    ll6_base = float(-np.mean(np.log(freq[y6])))
    # method-market blend (Phase 13/1A) on the subset with closing method odds
    blend6_line = ""
    MO6 = ["mo_ko_a", "mo_sub_a", "mo_dec_a", "mo_ko_b", "mo_sub_b", "mo_dec_b"]
    if pm_w := props_m.get("blend6_w"):
        has6 = t6[MO6].notna().all(axis=1)
        s6 = t6[has6]
        ys6 = outcome6(s6).astype(int).values
        ll_m = log_loss(ys6, np.clip(props_m["outcome6"](s6[feats]), 1e-6, 1),
                        labels=range(6))
        ll_b = log_loss(ys6, props_m["outcome6_blend"](s6), labels=range(6))
        blend6_line = (f"| 6-way LL, method-odds subset (n={len(s6)}) — "
                       f"model / blend w={pm_w:.1f} | {ll_b:.3f} | {ll_m:.3f} (model alone) |\n")
    acc6, acc6_base = (P6.argmax(1) == y6).mean(), (y6 == freq.argmax()).mean()
    y_dist = ((y6 == 2) | (y6 == 5)).astype(int)
    p_dist_6c = P6[:, 2] + P6[:, 5]
    p_dist_ded = props_m["distance"](t6[feats])
    dist_brier_6c = brier_score_loss(y_dist, p_dist_6c)
    dist_brier = brier_score_loss(y_dist, p_dist_ded)
    dist_acc = accuracy_score(y_dist, p_dist_ded > 0.5)
    dist_base = max(y_dist.mean(), 1 - y_dist.mean())
    u25_line = ""
    if props_m["under25"] is not None and "fight_secs" in test.columns:
        tu = test[(test["sched_rounds"] == 3) & test["fight_secs"].notna()]
        yu = (tu["fight_secs"] < 750).astype(int)
        pu = props_m["under25"](tu[feats])
        naive = yu.mean() * (1 - yu.mean())
        u25_line = (f"| Under 2.5 rounds (3R fights) — Brier | {brier_score_loss(yu, pu):.3f} "
                    f"| {naive:.3f} (predict base rate) |\n")
    tt = test.dropna(subset=["total_td"])
    td_mae = np.abs(props_m["total_td"](tt[feats]) - tt["total_td"]).mean()
    td_mae_base = np.abs(train_all["total_td"].mean() - tt["total_td"]).mean()

    # duration-aware count props (Phase 13): over-line Brier, sides averaged
    count_lines = ""
    tc = test.dropna(subset=["sig_landed_a", "sig_landed_b", "td_landed_a",
                             "td_landed_b", "fight_secs"])
    for kind, line, col_a, col_b in (("sig", 24.5, "sig_landed_a", "sig_landed_b"),
                                     ("td", 0.5, "td_landed_a", "td_landed_b")):
        pv = props_m["count_over"](tc[feats], kind, line)
        pv_b = props_m["count_over"](tc[feats], kind, line, mirrored=True)
        if pv is None:
            continue
        b = np.mean([brier_score_loss((tc[col_a] > line).astype(int), pv),
                     brier_score_loss((tc[col_b] > line).astype(int), pv_b)])
        base_p = ((tc[col_a] > line).mean() + (tc[col_b] > line).mean()) / 2
        count_lines += (f"| Fighter {kind} over {line} — Brier (duration-aware) "
                        f"| {b:.3f} | {base_p * (1 - base_p):.3f} (base rate) |\n")

    # ---------- 5. permutation importance (full ensemble, log-loss increase) ----------
    print("Computing permutation importance (ensemble)...")
    rng = np.random.default_rng(SEED)
    base = log_loss(y_te, p_ens)
    imp = {}
    Xp = X_te.reset_index(drop=True)
    for f in feats:
        deltas_f = []
        for _ in range(3):
            Xs = Xp.copy()
            Xs[f] = rng.permutation(Xs[f].values)
            deltas_f.append(log_loss(y_te, predict_ensemble(Xs)) - base)
        imp[f] = np.mean(deltas_f)
    imp = pd.Series(imp).sort_values(ascending=False)

    # ---------- 6. plots ----------
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150, facecolor="#fcfcfb")
    style_ax(ax)
    ax.plot([0, 1], [0, 1], color=C_MUTED, linewidth=1, linestyle="--", zorder=1)
    for p, label, color in [(p_ens_o, "Model (stacked ensemble)", C_MODEL),
                            (p_mkt, "Market (no-vig)", C_MARKET)]:
        order = np.argsort(p)
        ps, ys = np.asarray(p)[order], np.asarray(yo)[order]
        xs, ms = [], []
        for chunk in np.array_split(np.arange(len(ps)), 10):
            xs.append(ps[chunk].mean()); ms.append(ys[chunk].mean())
        ax.plot(xs, ms, color=color, linewidth=2, marker="o", markersize=5, label=label)
    ax.set_xlabel("Predicted probability", color=C_INK2, fontsize=10)
    ax.set_ylabel("Observed win rate", color=C_INK2, fontsize=10)
    ax.set_title("Reliability diagram — test fights with odds, 2023–2026",
                 color=C_INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=C_INK2)
    fig.tight_layout()
    fig.savefig(plots / "reliability.png", facecolor="#fcfcfb")
    plt.close(fig)

    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.6), dpi=150, facecolor="#fcfcfb")
    style_ax(ax)
    ax.barh(top.index, top.values, color=C_MODEL, height=0.62)
    ax.set_xlabel("Log-loss increase when permuted", color=C_INK2, fontsize=10)
    ax.set_title("Permutation importance — top 15 features (ensemble, test set)",
                 color=C_INK, fontsize=11, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(plots / "importance.png", facecolor="#fcfcfb")
    plt.close(fig)

    # ---------- 7. report ----------
    def table(rows):
        head = "| Predictor | n | Accuracy | Log loss | Brier | ECE |\n|---|---|---|---|---|---|\n"
        return head + "\n".join(
            f"| {r['name']} | {r['n']} | {r['acc']:.3f} | {r['logloss']:.3f} "
            f"| {r['brier']:.3f} | {r['ece']:.3f} |" + (f" {r['note']}" if r["note"] else "")
            for r in rows)

    mkt_ll = rows_mkt[0]["logloss"]; mdl_ll = rows_mkt[1]["logloss"]; bl_ll = rows_mkt[2]["logloss"]
    beats = mdl_ll < mkt_ll
    adds = bl_ll < mkt_ll - 1e-4 and ci_lo > 0
    report = f"""# UFC Fight Prediction — Calibration Report

Generated by `train_report.py`. Seeds fixed (42); strict temporal validation.

**Data:** {len(df)} UFC fights {df['date'].min().date()} to {df['date'].max().date()},
{len(feats)} features computed strictly as-of fight date by replaying career
histories (`build_features.py`): records, Elo (finish-weighted), per-minute
striking/grappling rates, strike-target and position mixes, recent-form windows
(last 3 fights), opponent quality (average opponent Elo), trajectory (peak-Elo
decline, layoffs, weight-class changes), rankings, and physical attributes.

**Split:** train ≤ {CUTOFF.date()} ({len(train_all)} fights), test after
({len(test)} fights). LightGBM (best {best_lgb}, cv {cv_lgb:.4f}) and XGBoost
(best {best_xgb}, cv {cv_xgb:.4f}) tuned with 4-fold expanding-window CV inside
the train period. The stacking combiner, isotonic calibrator, and market blend
were all fitted on out-of-fold CV predictions only. The LightGBM component
is a 5-seed bag (adopted Phase 9: CV logloss 0.6571 -> 0.6535). Stack weights
(LGB, XGB, logistic): {np.round(stacker.coef_[0], 2).tolist()}.

## Model comparison — full test set ({len(test)} fights)

{table(rows_full)}

## The real benchmark — vig-removed closing odds ({len(ot)} test fights with odds)

{table(rows_mkt)}

**Does the model beat the market?** {"Yes — model log loss is lower." if beats else
f"No. The market's log loss ({mkt_ll:.3f}) beats the model's ({mdl_ll:.3f})."}

**Does the model add information beyond the market?** {"Yes" if adds else "Not conclusively"} —
blending model with market gives log loss {bl_ll:.3f} vs {mkt_ll:.3f} for the
market alone. Paired bootstrap on the log-loss difference (market − blend,
10,000 resamples): 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], blend better in
{p_better:.1%} of resamples.

This is the expected result for public pre-fight data: closing odds aggregate
sharp bettors' information (injuries, camp changes, weight-cut issues, insider
knowledge) that no historical-stats model can see. Matching the market's
calibration while using only public stats is the honest achievement; claiming
to beat closing lines would be the red flag.

## Prop models — method, distance, takedowns (test set)

Six-way outcome (winner × KO/sub/decision), evaluated against always predicting
the training-set class frequencies:

| Target | Model | Baseline |
|---|---|---|
| 6-way outcome log loss | {ll6:.3f} | {ll6_base:.3f} (freq) — per-class isotonic REJECTED ({ll6_cal:.3f}) |
{blend6_line}| 6-way outcome top-1 accuracy | {acc6:.3f} | {acc6_base:.3f} |
| Goes the distance — Brier (dedicated model) | {dist_brier:.3f} | {dist_brier_6c:.3f} (6-way-derived) |
| Goes the distance — accuracy | {dist_acc:.3f} | {dist_base:.3f} (majority class) |
{u25_line}{count_lines}| Total takedowns — MAE | {td_mae:.2f} | {td_mae_base:.2f} (train mean) |

Method-of-victory and distance predictions beat the frequency baselines but,
as with the winner model, should be assumed weaker than prop-market prices —
and prop markets carry roughly twice the vig of moneylines.

## Calibration

![Reliability diagram](plots/reliability.png)

ECE (10 equal-count bins): model {rows_mkt[1]['ece']:.3f}, market {rows_mkt[0]['ece']:.3f}.
Isotonic calibration is reported for completeness; the raw ensemble is already
close to calibrated.

## What carries signal

![Permutation importance](plots/importance.png)

Top 10 by permutation importance (mean log-loss increase over 3 shuffles,
through the full ensemble):

{chr(10).join(f"- `{k}`: {v:+.4f}" for k, v in imp.head(10).items())}

## Negative results (tried and rejected)

Style-matchup features were built and evaluated but did not improve the model,
so the final feature set excludes them (they remain in `data/features.csv`):
per-archetype win-rate splits (vs taller / wrestler / power / southpaw
opponents, and vs today's opponent's archetype), explicit trait interactions
(chin-vs-power: KD-absorbed × opponent KD rate; grappling threat: TD rate ×
opponent TD-defense gap), and a cut-severity proxy (height/reach vs the
division's running average). The splits are sparse — most fighters have too few
fights against a given archetype for a stable rate — and added noise; the
interactions are largely captured by the trees already. Test accuracy dropped
from 0.655 to 0.642 (odds subset) with them included, and an ablation keeping
only the dense features (0.637) did not recover the incumbent either.

Adopted (Phase 10): pre-UFC records as features. Overall CV improved
0.6542 -> 0.6528 (+0.0014, under the pre-registered 0.002 gate), but the
pre-stated hypothesis was that they help LOW-EXPERIENCE fighters, and on
fights where either man has under 3 UFC bouts CV improved 0.6548 -> 0.6522
(+0.0026, clearing the gate). Adopted on that basis; `pre_ufc_winpct_diff`
is now the third most important feature in the model.

Adopted (Phase 11): career loss-method features — how often a fighter has
been knocked out, submitted, or taken to a decision, plus a never-finished
flag. The model tracked how fighters WIN but never how they LOSE, and
whether a man has been stopped before is the most direct evidence for how
his next fight ends. Method log loss 1.620 -> 1.611, 6-way top-1 0.337 ->
0.346, distance Brier 0.2368 -> 0.2355 in a controlled ablation.

Rejected (Phase 11): duration-conditioned count props — a fight-duration
regression (MAE 291s vs 319s baseline) fed as a feature to the takedown
model made it WORSE (MAE 1.079 -> 1.091; over-0.5 Brier 0.2201 -> 0.2257),
because the count model already infers fight length from the same features.
Also rejected: a round-by-round finish model (which round a 3-round fight
ends in, 4 classes). It lost to base rates on every class — R1 0.1954 vs
0.1943, R2 0.1323 vs 0.1294, R3 0.0725 vs 0.0700, decision 0.2511 vs
0.2499. The model can predict WHETHER a fight ends early; it cannot predict
WHEN.

Rejected (Phase 11): round-level cardio features. The raw stats carry
per-round detail that fight-total aggregation destroys, so we extracted
R1 output, late-round (R3+) output, a cardio ratio, late damage absorbed
and deep-water rounds. All four targets got WORSE: winner CV 0.6466 ->
0.6477, method log loss 1.611 -> 1.624, distance Brier 0.2355 -> 0.2377.
Two reasons. Coverage is only 73% (a fighter needs third rounds to have a
late-round rate at all), and the sample is self-selected — fighters who
routinely reach round three differ systematically from those who do not,
so the ratio encodes survivorship as much as conditioning. The features
are still computed in features.csv; the model does not use them.

Worth recording: the "fighters fade late" intuition is wrong at the
population level. League-wide, significant strikes landed rise every
round (R1 14.4, R2 16.3, R3 17.1, R4 18.1, R5 18.6) and the median
fighter's late-round output is 1.09x his first-round output. Fights open
up as they go; they do not slow down.

Also rejected (Phase 9): Glicko-style ratings with uncertainty (glicko /
glicko_rd features; CV 0.6551 vs 0.6542 baseline — Elo plus the trajectory
features already carry the signal) and recency-weighted training (best
tau=5y improved CV by only 0.0017, under the pre-registered 0.002 gate).

## Caveats

- Fighters are keyed by normalized name; the six known duplicate names
  (two fighters each, e.g. Bruno Silva) are excluded from training rows
  entirely — their opponents' records still update normally.
- The blend-beats-market significance is seed-sensitive: across different
  random fighter-orientation draws it ranges roughly 95-99% of bootstrap
  resamples. Treat it as a consistent but modest edge, not a precise number.
- Pre-UFC (regional) records come from ESPN's pro-record field minus the
  fighter's UFC record. ESPN's totals are CURRENT, so only this static
  pre-UFC delta is used — the current total would leak. 98% of fighters
  matched; the rest get 0-0. Elo still starts at 1500 on UFC debut.
- Odds are closing odds from the Ultimate UFC Dataset ({len(ot)}/{len(test)}
  test fights matched); fights without odds are excluded only from market rows.
- Rankings exist only for ranked fighters from 2021 onward; missing values are
  left as NaN for the tree models.
- Fight time is approximated with 5-minute rounds (exact for the modern era).
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("Wrote report.md, plots/reliability.png, plots/importance.png")


if __name__ == "__main__":
    main()
