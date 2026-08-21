"""EPL training + honest report: Dixon-Coles x LightGBM stack vs closing odds.

Pipeline (all fitting strictly walk-forward, seeds fixed at 42):
1. load() reads epl/data/features.csv (built by build_features.py; rebuilt
   automatically if missing) and derives no-vig market probs via
   demo.market_probs (Pinnacle closing preferred).
2. dc_predict_seasons() fits a time-decayed Dixon-Coles model per league,
   refit every 30 days walk-forward (protocol v2 — deployment-realistic
   freshness for every model row) — two-stage: weighted Poisson GLM (sklearn
   PoissonRegressor on attack/defence one-hots + home flag, exp decay with
   half-life chosen from {365, 730} days by CV RPS inside the train
   period only) then the low-score correlation rho by 1-D weighted MLE.
   Score grid 0..10 gives 1X2, over/under 2.5 and BTTS probabilities.
   Unseen (promoted) teams get the mean parameters of the 5 weakest fitted
   teams. This two-stage fit is a standard practical approximation to the
   joint DC MLE, chosen for speed; rho is conditional on the GLM rates.
3. lgb_oof() trains LightGBM 3-way on the replay features, expanding-window
   by season for out-of-fold predictions on 2018-19..2023-24, and retrained
   at each holdout season boundary (protocol v2) — the 2025-26 model sees
   2024-25, exactly as a weekly deployment would.
4. Stacking discipline (locked): the logistic combiner is fit on OOF
   first-stage predictions from 2018-19..2021-22 ONLY; the market+model
   logit blend is fit on 2022-23..2023-24, where the stack itself is
   out-of-sample. Nothing downstream ever sees its own training data.
5. Holdout = 2024-25 + 2025-26, untouched by any fitting. report() writes
   epl/report.md: RPS/log-loss for DC, LGB, stack, market, blend; per-class
   reliability; a 10,000-resample paired bootstrap CI on blend-minus-market
   RPS; O/U 2.5 Brier vs no-vig closing totals; BTTS vs base rate.

Not established: this model is NOT expected to beat the closing line — the
closes are the benchmark and matching their calibration is the bar. BTTS has
no market odds in football-data, so its Brier is only compared to the train
base rate. The DC fit window is capped at 6 years; the 30-day refit cadence
still lags the market's match-by-match information, deliberately — anything
fresher than the last completed round adds nothing for pre-match features.

Run: python epl/train_report.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo  # noqa: E402

FEATS = demo.DATA / "features.csv"
REPORT = Path(__file__).resolve().parent / "report.md"
SEED = 42
HOLDOUT_START = pd.Timestamp("2024-07-01")
CV_SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324"]
STACK_SEASONS = CV_SEASONS[:4]           # combiner fit here (first-stage preds are OOF)
BLEND_SEASONS = CV_SEASONS[4:]           # blend fit here (stack itself is OOF)
HOLDOUT_SEASONS = ["2425", "2526"]
HALF_LIVES = [365, 730]                  # days; chosen by CV RPS inside train only
DC_WINDOW_YEARS = 6
DC_REFIT_DAYS = 30                       # protocol v2: refit goal model per 30-day block
GOAL_GRID = 11                           # score matrix 0..10 goals
OU_CHAINS = [("PC>2.5", "PC<2.5"), ("B365C>2.5", "B365C<2.5"),
             ("P>2.5", "P<2.5"), ("B365>2.5", "B365<2.5")]
LGB_PARAMS = dict(objective="multiclass", num_class=3, n_estimators=500,
                  learning_rate=0.03, num_leaves=31, min_child_samples=50,
                  colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                  random_state=SEED, verbose=-1)
EXPERIMENTS = [                            # adopt-or-REJECT ledger (epl/experiments.py)
    "2026-08-21 A league one-hot as LGB features — REJECT (CV RPS 0.20386 vs "
    "baseline 0.20376, delta -0.00011 < gate 0.0005)",
    "2026-08-21 B walk-forward DC 1X2 probs as LGB features — REJECT "
    "(0.20340, +0.00035 < gate)",
    "2026-08-21 C hyperparameter sweep, 4 configs — all REJECT (best C3 "
    "leaves15/mc100 0.20327, +0.00048 < gate; C1 leaves63 -0.00672, "
    "C2 lr.02 -0.00128, C4 colsample.6 -0.00017)",
    "2026-08-21 E near-miss combination B+C3 (pre-registered one-shot) — "
    "REJECT (0.20340, +0.00036 < gate; no better than B alone)",
    "2026-08-21 F per-class isotonic calibration of the stack, evaluated on "
    "blend seasons where stack is out-of-sample — REJECT (raw 0.19956 vs "
    "calibrated 0.19973, -0.00016; the logistic combiner is already "
    "calibrated)",
    "2026-08-21 G understat rolling xG for/against last-5/10 (join rate "
    "99.3% via fetch_understat.py) — REJECT (0.20399, -0.00023 < gate; the "
    "shots/SoT rolling rates already carry the signal). Columns stay "
    "computed in features.csv, excluded from the model.",
    "2026-08-21 PROTOCOL v2 (adopted a priori as deployment realism, not "
    "gated on outcomes): DC refit per 30-day block (was once/season), LGB "
    "retrained at holdout season boundaries, half-life re-chosen on CV "
    "(365d under v2). v1 -> v2 holdout: DC 0.2058 -> 0.2019, LGB 0.2032 -> "
    "0.2028, stack 0.2019 -> 0.2010 (market 0.1956); O/U 2.5 Brier 0.2445 "
    "-> 0.2440, BTTS 0.2488 -> 0.2478. Freshness fixed DC's 1X2; the "
    "remaining goals-market gap is model quality, not staleness.",
]


def load():
    """Return features df (season zero-padded str) with y and market probs."""
    if not FEATS.exists():
        import build_features
        build_features.main()
    df = pd.read_csv(FEATS, parse_dates=["Date"])
    df["season"] = df["season"].astype(str).str.zfill(4)
    df["y"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    return df


def season_cutoff(season):
    """Season '1819' -> its start boundary Timestamp 2018-07-01."""
    return pd.Timestamp(2000 + int(season[:2]), 7, 1)


def fit_dc_league(rows, cutoff, half_life):
    """Fit one league's DC params on rows before cutoff; return predict closure."""
    from scipy.optimize import minimize_scalar
    from sklearn.linear_model import PoissonRegressor
    w = 0.5 ** ((cutoff - rows["Date"]).dt.days / half_life)
    teams = sorted(set(rows["HomeTeam"]) | set(rows["AwayTeam"]))
    ti = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    n = len(rows)
    X = np.zeros((2 * n, 2 * T + 1))
    hi = rows["HomeTeam"].map(ti).to_numpy()
    ai = rows["AwayTeam"].map(ti).to_numpy()
    r = np.arange(n)
    X[r, hi] = 1.0                                   # home rows: attack of home team
    X[r, T + ai] = -1.0                              # ... minus defence of away team
    X[r, 2 * T] = 1.0                                # home-advantage flag
    X[n + r, ai] = 1.0                               # away rows: attack of away team
    X[n + r, T + hi] = -1.0
    y = np.concatenate([rows["FTHG"].to_numpy(float), rows["FTAG"].to_numpy(float)])
    sw = np.concatenate([w, w])
    glm = PoissonRegressor(alpha=1e-3, max_iter=300)
    glm.fit(X, y, sample_weight=sw)
    att, deff, home = glm.coef_[:T], glm.coef_[T:2 * T], glm.coef_[2 * T]
    c = glm.intercept_
    weakest = np.argsort(att + deff)[:5]             # promoted-team prior: 5 weakest fitted teams
    fill_att, fill_def = att[weakest].mean(), deff[weakest].mean()

    lam = np.exp(c + att[hi] - deff[ai] + home)
    mu = np.exp(c + att[ai] - deff[hi])
    hg, ag = rows["FTHG"].to_numpy(), rows["FTAG"].to_numpy()

    def neg_ll(rho):
        tau = np.ones(n)
        tau = np.where((hg == 0) & (ag == 0), 1 - lam * mu * rho, tau)
        tau = np.where((hg == 0) & (ag == 1), 1 + lam * rho, tau)
        tau = np.where((hg == 1) & (ag == 0), 1 + mu * rho, tau)
        tau = np.where((hg == 1) & (ag == 1), 1 - rho, tau)
        return -np.sum(w * np.log(np.clip(tau, 1e-9, None)))

    rho = minimize_scalar(neg_ll, bounds=(-0.15, 0.15), method="bounded").x

    def predict(home_teams, away_teams):
        """(n,3) 1X2 + (n,) OU2.5 over-prob + (n,) BTTS-prob from the score grid."""
        from scipy.stats import poisson
        a_h = np.array([att[ti[t]] if t in ti else fill_att for t in home_teams])
        d_h = np.array([deff[ti[t]] if t in ti else fill_def for t in home_teams])
        a_a = np.array([att[ti[t]] if t in ti else fill_att for t in away_teams])
        d_a = np.array([deff[ti[t]] if t in ti else fill_def for t in away_teams])
        lam_ = np.exp(c + a_h - d_a + home)
        mu_ = np.exp(c + a_a - d_h)
        k = np.arange(GOAL_GRID)
        px = poisson.pmf(k[None, :], lam_[:, None])
        py = poisson.pmf(k[None, :], mu_[:, None])
        grid = px[:, :, None] * py[:, None, :]
        grid[:, 0, 0] *= np.clip(1 - lam_ * mu_ * rho, 1e-9, None)
        grid[:, 0, 1] *= np.clip(1 + lam_ * rho, 1e-9, None)
        grid[:, 1, 0] *= np.clip(1 + mu_ * rho, 1e-9, None)
        grid[:, 1, 1] *= np.clip(1 - rho, 1e-9, None)
        grid /= grid.sum(axis=(1, 2), keepdims=True)
        x = np.arange(GOAL_GRID)[:, None]
        yg = np.arange(GOAL_GRID)[None, :]
        p1x2 = np.stack([(grid * (x > yg)).sum((1, 2)),
                         (grid * (x == yg)).sum((1, 2)),
                         (grid * (x < yg)).sum((1, 2))], axis=1)
        ou25 = (grid * ((x + yg) >= 3)).sum((1, 2))
        btts = (grid * ((x >= 1) & (yg >= 1))).sum((1, 2))
        return p1x2, ou25, btts

    return predict


def dc_predict_seasons(df, seasons, half_life):
    """Walk-forward DC predictions, refit per DC_REFIT_DAYS block (protocol v2)."""
    p1x2 = np.full((len(df), 3), np.nan)
    ou = np.full(len(df), np.nan)
    btts = np.full(len(df), np.nan)
    for s in seasons:
        start = season_cutoff(s)
        sdf = df[df["season"] == s]
        block = ((sdf["Date"] - start).dt.days // DC_REFIT_DAYS).clip(lower=0)
        for (div, b), sub in sdf.groupby([sdf["Div"], block]):
            cutoff = start + pd.Timedelta(days=DC_REFIT_DAYS * int(b))
            lo = cutoff - pd.DateOffset(years=DC_WINDOW_YEARS)
            train = df[(df["Div"] == div) & (df["Date"] < cutoff) & (df["Date"] >= lo)]
            predict = fit_dc_league(train, cutoff, half_life)
            p, o, bt = predict(sub["HomeTeam"].tolist(), sub["AwayTeam"].tolist())
            p1x2[sub.index] = p
            ou[sub.index] = o
            btts[sub.index] = bt
    return p1x2, ou, btts


def lgb_oof(df, feat_cols):
    """OOF LGB probs for CV seasons + holdout probs from the final model."""
    from lightgbm import LGBMClassifier
    probs = np.full((len(df), 3), np.nan)
    for s in CV_SEASONS:
        tr = df[df["season"] < s]
        va = df[df["season"] == s]
        model = LGBMClassifier(**LGB_PARAMS)
        model.fit(tr[feat_cols], tr["y"])
        probs[va.index] = model.predict_proba(va[feat_cols])
    for s in HOLDOUT_SEASONS:              # protocol v2: retrain at each holdout season boundary
        tr = df[df["Date"] < season_cutoff(s)]
        ho = df[df["season"] == s]
        model = LGBMClassifier(**LGB_PARAMS)
        model.fit(tr[feat_cols], tr["y"])
        probs[ho.index] = model.predict_proba(ho[feat_cols])
    return probs


def logit_combine(train_X_probs, train_y, apply_X_probs):
    """Multinomial LR on concatenated log-probs; NaN where any input is missing."""
    from sklearn.linear_model import LogisticRegression
    logs = [np.log(np.clip(p, 1e-9, None)) for p in train_X_probs]
    lr = LogisticRegression(C=10.0, max_iter=1000, random_state=SEED)
    lr.fit(np.hstack(logs), train_y)
    X = np.hstack([np.log(np.clip(p, 1e-9, None)) for p in apply_X_probs])
    out = np.full((len(X), len(lr.classes_)), np.nan)
    ok = np.isfinite(X).all(axis=1)
    out[ok] = lr.predict_proba(X[ok])
    return lr, out


def reliability(y, p, cls):
    """Markdown lines for one class's 10-bin calibration."""
    return [f"| {rng} | {n} | {pred:.3f} | {obs:.3f} |"
            for rng, n, pred, obs in demo.calibration_table(y, p, cls)]


def rps_per_match(y, p):
    """Per-match RPS vector (order H<D<A), for paired bootstraps."""
    outcome = np.zeros_like(p)
    outcome[np.arange(len(y)), y] = 1.0
    return np.sum((np.cumsum(p, 1) - np.cumsum(outcome, 1)) ** 2, 1) / 2


def ou_market(df):
    """No-vig P(over 2.5) per row from the first complete totals pair, + source."""
    p = np.full(len(df), np.nan)
    src = np.array([""] * len(df), dtype=object)
    for over_c, under_c in OU_CHAINS:
        if over_c not in df:
            continue
        odds = df[[over_c, under_c]].to_numpy(float)
        ok = (src == "") & np.isfinite(odds).all(1) & (odds > 1.0).all(1)
        inv = 1.0 / odds[ok]
        p[ok] = inv[:, 0] / inv.sum(1)
        src[ok] = over_c
    return p, src


def main():
    df = load()
    y = df["y"].to_numpy()
    rejected = {f"{p}xg{d}{w}" for p in ("h_", "a_") for d in ("f", "a")
                for w in (5, 10)}          # REJECTED 2026-08-21 (-0.00023 < gate), kept in CSV
    feat_cols = ["elo_diff"] + [c for c in df.columns
                                if c.startswith(("h_", "a_")) and c not in rejected]
    mkt, mkt_src = demo.market_probs(df)
    cv_mask = df["season"].isin(CV_SEASONS).to_numpy()
    ho_mask = df["season"].isin(HOLDOUT_SEASONS).to_numpy()

    print("tuning DC half-life on CV seasons...")
    best_hl, best_rps, dc_cv, hl_rps = None, np.inf, None, {}
    for hl in HALF_LIVES:
        p, _, _ = dc_predict_seasons(df, CV_SEASONS, hl)
        hl_rps[hl] = r = demo.rps(y[cv_mask], p[cv_mask])
        print(f"  half-life {hl}d: CV RPS {r:.4f}")
        if r < best_rps:
            best_hl, best_rps, dc_cv = hl, r, p
    print(f"chosen half-life: {best_hl}d")
    dc = dc_cv
    ho_p, ho_ou, ho_btts = dc_predict_seasons(df, HOLDOUT_SEASONS, best_hl)
    dc[ho_mask] = ho_p[ho_mask]

    print("LightGBM expanding-window OOF + holdout...")
    lgb = lgb_oof(df, feat_cols)

    stack_m = df["season"].isin(STACK_SEASONS).to_numpy()
    _, stack_all = logit_combine([dc[stack_m], lgb[stack_m]], y[stack_m],
                                 [dc, lgb])
    blend_m = df["season"].isin(BLEND_SEASONS).to_numpy() & np.isfinite(mkt).all(1)
    _, blend_all = logit_combine([stack_all[blend_m], mkt[blend_m]], y[blend_m],
                                 [stack_all, np.nan_to_num(mkt, nan=1 / 3)])

    ev = ho_mask & np.isfinite(mkt).all(1)
    yt = y[ev]
    models = {"Dixon-Coles": dc[ev], "LightGBM": lgb[ev], "Stack": stack_all[ev],
              "Market (no-vig close)": mkt[ev], "Blend (stack+market)": blend_all[ev]}
    for name, p in models.items():
        assert np.allclose(p.sum(1), 1.0, atol=1e-6), f"{name} probs do not sum to 1"

    # sanity: on the blend seasons (stack out-of-sample) the stack should not
    # be worse than the better single family
    bl_ev = blend_m
    cv_rps = {n: demo.rps(y[bl_ev], p[bl_ev]) for n, p in
              [("dc", dc), ("lgb", lgb), ("stack", stack_all)]}
    print(f"CV (2223+2324) RPS: {cv_rps}")
    assert cv_rps["stack"] <= min(cv_rps["dc"], cv_rps["lgb"]) + 0.002, \
        f"stack {cv_rps['stack']:.4f} worse than best single family"

    rng = np.random.default_rng(SEED)
    d = rps_per_match(yt, models["Blend (stack+market)"]) - \
        rps_per_match(yt, models["Market (no-vig close)"])
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10_000)])
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    frac_better = float((boots < 0).mean())

    over = ((df["FTHG"] + df["FTAG"]) >= 3).to_numpy(float)
    mkt_ou, ou_src = ou_market(df)
    ou_ev = ho_mask & np.isfinite(mkt_ou) & np.isfinite(ho_ou)
    btts_true = ((df["FTHG"] >= 1) & (df["FTAG"] >= 1)).to_numpy(float)
    btts_base = btts_true[df["Date"] < HOLDOUT_START].mean()
    bt_ev = ho_mask & np.isfinite(ho_btts)

    src_counts = pd.Series(mkt_src[ev]).value_counts().to_dict()
    ou_counts = pd.Series(ou_src[ou_ev]).value_counts().to_dict()

    lines = [
        "# EPL model report", "",
        f"Generated by `python epl/train_report.py` (seed {SEED}). "
        f"Data: {len(df)} matches, top-5 leagues, seasons 2011-12..2025-26 "
        "(football-data.co.uk).", "",
        f"- Train: everything before {HOLDOUT_START:%Y-%m-%d}; DC half-life "
        f"chosen on CV = {best_hl}d (CV RPS by candidate: "
        + ", ".join(f"{h}d={r:.4f}" for h, r in hl_rps.items())
        + " — evaluated inside train only).",
        f"- Protocol v2 (walk-forward): goal model refit every {DC_REFIT_DAYS} "
        "days; LGB retrained at each holdout season boundary. Applied "
        "identically to every model row; combiner fitting unchanged.",
        f"- Stack combiner fit on OOF predictions, seasons {STACK_SEASONS}.",
        f"- Blend weight fit on seasons {BLEND_SEASONS} (stack out-of-sample there).",
        f"- Holdout: seasons 2024-25 + 2025-26, {ev.sum()} matches with 1X2 closing "
        f"odds ({src_counts}).", "",
        "## Headline: 1X2 on the holdout", "",
        "| model | RPS | log loss |", "|---|---|---|",
    ]
    for name, p in models.items():
        lines.append(f"| {name} | {demo.rps(yt, p):.4f} | {demo.log_loss3(yt, p):.4f} |")
    lines += [
        "",
        f"**Blend minus market RPS: {d.mean():+.5f}** "
        f"(10,000-resample paired bootstrap 95% CI [{ci_lo:+.5f}, {ci_hi:+.5f}]; "
        f"blend better in {frac_better:.1%} of resamples).", "",
        "Honest framing (locked): the closing line is the benchmark and EPL 1X2 "
        "closes are brutally efficient. Matching their calibration with public "
        "data is the achievement; a model that 'beats the close' outright is a "
        "red flag for leakage. The blend test above is the real question.", "",
        "## Per-class reliability on the holdout (stack vs market)", "",
    ]
    for cls, cname in enumerate(["HOME", "DRAW", "AWAY"]):
        lines += [f"### {cname}", "", "stack:", "",
                  "| bin | n | mean pred | observed |", "|---|---|---|---|",
                  *reliability(yt, models["Stack"], cls), "", "market:", "",
                  "| bin | n | mean pred | observed |", "|---|---|---|---|",
                  *reliability(yt, models["Market (no-vig close)"], cls), ""]
    lines += [
        "## Goals model extras (holdout)", "",
        f"- O/U 2.5 Brier — DC model {np.mean((ho_ou[ou_ev] - over[ou_ev]) ** 2):.4f} "
        f"vs no-vig totals market {np.mean((mkt_ou[ou_ev] - over[ou_ev]) ** 2):.4f} "
        f"on {int(ou_ev.sum())} matches with totals odds ({ou_counts}).",
        f"- BTTS Brier — DC model {np.mean((ho_btts[bt_ev] - btts_true[bt_ev]) ** 2):.4f} "
        f"vs train-base-rate {np.mean((btts_base - btts_true[bt_ev]) ** 2):.4f} "
        f"({int(bt_ev.sum())} matches; football-data carries no BTTS odds, so no "
        "market comparison).", "",
        "## Locked rules for everything after this report", "",
        "1. Every experiment passes a PRE-REGISTERED adoption gate: metric + "
        "threshold stated before running (CV RPS improvement >= 0.0005 unless "
        "stated otherwise), documented adopt-or-REJECT here with numbers. "
        "Rejected features stay computed in features.csv, unused.",
        "2. Calibration layers and blend weights never see their own training "
        "data (OOF/holdout-clean first-stage predictions only).",
        "3. Counts get rate-x-exposure treatment; check missingness-as-signal "
        "before imputing.",
        "4. Draws are the calibration crux — per-class reliability, always.",
        "5. Betting features only after this report proves calibration, with "
        "honest EV framing: edges must clear the vig, and most won't.", "",
        "## Experiment log", "",
        "Gate (pre-registered before each run): expanding-window OOF CV RPS "
        "improvement >= 0.0005 over the six CV seasons. Rejected features stay "
        "computed in features.csv, unused.", "",
        *[f"- {e}" for e in EXPERIMENTS], "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT}")
    for name, p in models.items():
        print(f"  {name:24s} RPS {demo.rps(yt, p):.4f}  LL {demo.log_loss3(yt, p):.4f}")
    print(f"  blend-market {d.mean():+.5f}  CI [{ci_lo:+.5f}, {ci_hi:+.5f}]  "
          f"better {frac_better:.1%}")


if __name__ == "__main__":
    sys.exit(main())
