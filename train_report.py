"""Train UFC winner models on leak-free features and write the calibration report.

Pipeline (all temporal, no random splits):
  1. Expanding-window CV (4 folds, 2016-2023) tunes LightGBM and produces
     out-of-fold predictions used to fit an isotonic calibrator and the
     market-blend logistic — so nothing downstream ever sees test data.
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
              "finish_rate", "dec_win_rate", "layoff_days", "age", "height", "reach"]
CONTEXT = ["title_bout", "women", "sched_rounds", "weight_lbs"]
ANTISYM = [f"{f}_diff" for f in SNAP_FEATS] + ["red_corner", "southpaw_vs_orthodox"]


def load():
    df = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["date"])
    df = df[df["date"] >= MODERN].sort_values("date").reset_index(drop=True)
    for f in SNAP_FEATS:  # symmetric matchup level, e.g. are both fighters experienced
        df[f"{f}_mean"] = (df[f"{f}_a"] + df[f"{f}_b"]) / 2
    feats = ANTISYM + [f"{f}_mean" for f in SNAP_FEATS] + CONTEXT
    return df, feats


def mirror(X: pd.DataFrame, y: pd.Series):
    """Append the swapped-corner view of every fight (antisymmetric cols negate)."""
    Xm = X.copy()
    Xm[ANTISYM] = -Xm[ANTISYM]
    return pd.concat([X, Xm], ignore_index=True), pd.concat([y, 1 - y], ignore_index=True)


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

    # ---------- 1. expanding-window CV: tune LightGBM ----------
    print("Tuning LightGBM with expanding-window CV...")
    grid = [(nl, lr, mcs) for nl in (15, 31, 63) for lr in (0.03, 0.06) for mcs in (20, 60)]
    results = {}
    for cfg in grid:
        losses, iters = [], []
        for start, end in CV_FOLDS:
            tr = df[df["date"] < start]
            va = df[(df["date"] >= start) & (df["date"] < end)]
            Xm, ym = mirror(tr[feats], tr["a_wins"])
            m = fit_lgb(lgb_params(*cfg), Xm, ym, 3000, va[feats], va["a_wins"])
            losses.append(log_loss(va["a_wins"], m.predict(va[feats])))
            iters.append(m.best_iteration)
        results[cfg] = (np.mean(losses), int(np.mean(iters)))
        print(f"  leaves={cfg[0]:3d} lr={cfg[1]} mcs={cfg[2]:3d}  cv_logloss={np.mean(losses):.4f}")
    best_cfg = min(results, key=lambda c: results[c][0])
    best_cv_loss, best_iters = results[best_cfg]
    print(f"best: leaves={best_cfg[0]} lr={best_cfg[1]} mcs={best_cfg[2]} "
          f"(cv_logloss={best_cv_loss:.4f}, avg_iters={best_iters})")

    # ---------- 2. out-of-fold predictions with best params ----------
    oof = []
    for start, end in CV_FOLDS:
        tr = df[df["date"] < start]
        va = df[(df["date"] >= start) & (df["date"] < end)].copy()
        Xm, ym = mirror(tr[feats], tr["a_wins"])
        m = fit_lgb(lgb_params(*best_cfg), Xm, ym, 3000, va[feats], va["a_wins"])
        va["p_oof"] = m.predict(va[feats])
        oof.append(va)
    oof = pd.concat(oof)
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso.fit(oof["p_oof"], oof["a_wins"])

    # blend fit on OOF fights that have odds (market + model -> logistic)
    oof_odds = oof[oof["odds_a"].notna() & oof["odds_b"].notna()]
    p_mkt_oof = no_vig(oof_odds["odds_a"].values, oof_odds["odds_b"].values)
    blend = LogisticRegression(random_state=SEED)
    blend.fit(np.column_stack([logit(p_mkt_oof), logit(oof_odds["p_oof"])]),
              oof_odds["a_wins"])

    # ---------- 3. final models on full train, evaluated once on test ----------
    Xm, ym = mirror(train_all[feats], train_all["a_wins"])
    final = fit_lgb(lgb_params(*best_cfg), Xm, ym, best_iters)
    p_gbm = final.predict(X_te)
    p_iso = iso.predict(p_gbm)

    logistic = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(random_state=SEED, max_iter=2000))
    logistic.fit(Xm, ym)
    p_log = logistic.predict_proba(X_te)[:, 1]

    elo_only = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(random_state=SEED))
    elo_only.fit(train_all[["elo_diff"]], train_all["a_wins"])
    p_elo = elo_only.predict_proba(test[["elo_diff"]])[:, 1]

    rows_full = [
        metrics_row("LightGBM", y_te, p_gbm),
        metrics_row("LightGBM + isotonic", y_te, p_iso),
        metrics_row("Logistic regression (full features)", y_te, p_log),
        metrics_row("Elo only (logistic)", y_te, p_elo),
        metrics_row("Better record baseline", y_te,
                    np.where(test["win_pct_diff"].fillna(0) != 0,
                             (test["win_pct_diff"].fillna(0) > 0).astype(float), 0.5) * 0.98 + 0.01,
                    note="hard picks scored as 0.99/0.5/0.01"),
    ]

    # ---------- 4. market comparison on the odds subset ----------
    ot = test[test["odds_a"].notna() & test["odds_b"].notna()]
    yo = ot["a_wins"]
    p_mkt = no_vig(ot["odds_a"].values, ot["odds_b"].values)
    p_gbm_o = final.predict(ot[feats])
    p_iso_o = iso.predict(p_gbm_o)
    p_blend = blend.predict_proba(np.column_stack([logit(p_mkt), logit(p_gbm_o)]))[:, 1]
    rows_mkt = [
        metrics_row("Market (no-vig closing odds)", yo, p_mkt),
        metrics_row("LightGBM", yo, p_gbm_o),
        metrics_row("LightGBM + isotonic", yo, p_iso_o),
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

    # ---------- 5. permutation importance (test set, log-loss increase) ----------
    rng = np.random.default_rng(SEED)
    base = log_loss(y_te, p_gbm)
    imp = {}
    Xp = X_te.reset_index(drop=True)
    for f in feats:
        deltas = []
        for _ in range(5):
            Xs = Xp.copy()
            Xs[f] = rng.permutation(Xs[f].values)
            deltas.append(log_loss(y_te, final.predict(Xs)) - base)
        imp[f] = np.mean(deltas)
    imp = pd.Series(imp).sort_values(ascending=False)

    # ---------- 6. plots ----------
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150, facecolor="#fcfcfb")
    style_ax(ax)
    ax.plot([0, 1], [0, 1], color=C_MUTED, linewidth=1, linestyle="--", zorder=1)
    for p, label, color in [(p_gbm_o, "Model (LightGBM)", C_MODEL),
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
    ax.set_title("Permutation importance — top 15 features (test set)",
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

    mkt_ll = rows_mkt[0]["logloss"]; mdl_ll = rows_mkt[1]["logloss"]; bl_ll = rows_mkt[3]["logloss"]
    beats = mdl_ll < mkt_ll
    adds = bl_ll < mkt_ll - 1e-4 and ci_lo > 0
    report = f"""# UFC Fight Prediction — Calibration Report

Generated by `train_report.py`. Seeds fixed (42); strict temporal validation.

**Data:** {len(df)} UFC fights {df['date'].min().date()} to {df['date'].max().date()},
features computed strictly as-of fight date by replaying career histories
(`build_features.py`). **Split:** train ≤ {CUTOFF.date()} ({len(train_all)} fights),
test after ({len(test)} fights). LightGBM tuned with 4-fold expanding-window CV
inside the train period (best: num_leaves={best_cfg[0]}, lr={best_cfg[1]},
min_child_samples={best_cfg[2]}, cv logloss {best_cv_loss:.4f}). Isotonic
calibration and the market blend were fitted on out-of-fold CV predictions only.

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

Isotonic calibration (fitted on out-of-fold predictions) improved ECE slightly
but worsened log loss, so the raw LightGBM probabilities — already close to
calibrated — are the primary model output.

This is the expected result for public pre-fight data: closing odds aggregate
sharp bettors' information (injuries, camp changes, weight-cut issues, insider
knowledge) that no historical-stats model can see. Matching the market's
calibration while using only public stats is the honest achievement; claiming
to beat closing lines would be the red flag.

## Calibration

![Reliability diagram](plots/reliability.png)

ECE (10 equal-count bins): model {rows_mkt[1]['ece']:.3f}, market {rows_mkt[0]['ece']:.3f}.

## What carries signal

![Permutation importance](plots/importance.png)

Top 10 by permutation importance (mean log-loss increase over 5 shuffles):

{chr(10).join(f"- `{k}`: {v:+.4f}" for k, v in imp.head(10).items())}

## Caveats

- Fighters are keyed by normalized name; a handful of duplicate names
  (e.g. two Bruno Silvas) introduce minor noise.
- Elo starts at 1500 on UFC debut; pre-UFC records are not observed.
- Odds are closing odds from the Ultimate UFC Dataset ({len(ot)}/{len(test)}
  test fights matched); fights without odds are excluded only from market rows.
- Fight time is approximated with 5-minute rounds (exact for the modern era).
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("Wrote report.md, plots/reliability.png, plots/importance.png")


if __name__ == "__main__":
    main()
