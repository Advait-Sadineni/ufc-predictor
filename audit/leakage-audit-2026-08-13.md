# Per-feature temporal-leakage audit — 2026-08-13

Per MEASUREMENT.md §7: for every feature the model trains on, name the timestamp at
which its value was knowable. If that timestamp can be at-or-after the fight being
predicted, it is leakage. Audited per feature, not per dataset.

**Scope:** the 110 winner-model columns (`train_report.py` `SNAP_FEATS` × diff/mean +
`red_corner, southpaw_vs_orthodox, rank_adv, ranked_diff` + 4 context), the blend
inputs (closing odds), and the prop-model paths in `props.py`. Backtest under audit:
train ≤ 2023-06-03, test after — the report.md protocol.

**Verification performed** (not just code reading):
- Replay pre-fight discipline: `snapshot()` runs before `update_state()`
  (build_features.py:573-574 vs :661-671); Elo/Glicko/opponent-quality scalars
  captured pre-mutation (:633-645). Differential check: `n_fights` monotone per
  fighter across all 8616 rows, 0 violations / 2724 fighters (same-date ties
  explain the 30 initial flags).
- Orientation: first-listed fighter is the red corner, not the winner — OUTCOME
  distribution W/L 5567 / L/W 3122 (64%/36%), so `red_corner` is legitimate
  pre-fight information, not a winner-listed-first leak.
- pre_ufc contamination proven by named cases (see table).
- ESPN scoreboard API returns CURRENT records even when queried with historical
  dates (Ngannou's computed pre-UFC includes his Oct 2025 PFL win despite his last
  UFC event being Jan 2023) — so no point-in-time re-crawl is available from
  that endpoint.

## Audit table (worst first)

| Feature | Source | Knowable when | Verdict | Evidence |
|---|---|---|---|---|
| `pre_ufc_wins/_losses/_fights/_winpct` (diff+mean, 8 cols) | ESPN pro record NOW − own UFC record NOW | **ESPN scrape date (2026)** — contains every non-UFC fight AFTER the row's date | **LEAK** | build_features.py:192-219, fetch_pro_records.py:35-52. Kevin Lee computed 9-1 vs true 8-1 (2022 Eagle FC win leaks into 2014-2023 rows); Ngannou 7-1 vs true 5-1 (2025 PFL win). `pre_ufc_winpct_diff` was #3 by permutation importance (+0.0036, report.md) |
| `southpaw_vs_orthodox` (+ stance in archetype traits) | ufc_fighter_tott.csv, current snapshot | scrape date; stance treated as career-constant | SUSPECT (minor) | build_features.py:145-156, :594-595. Stance changes are rare; no dated history available. Not fixable locally; magnitude ~0 |
| `odds_a/odds_b`, `mo_ko/sub/dec_*` (blend inputs, NOT tree features) | ufc-master closing odds | T-0 (bout start) — before the outcome, but AFTER any realistic bet placement | SUSPECT (deployment, not eval) | build_features.py:222-241; train_report.py:288-292, :327-337. Blend-vs-market comparison is timestamp-fair (both sides see T-0 info), so the 97.0% finding is internally valid; it does NOT imply a bettable pre-close edge. Already framed honestly in report.md |
| `rank_adv`, `ranked_diff` | ufc-master `*_match_weightclass_rank` | fight week (UFC rankings published pre-fight) | CLEAN (upstream trust) | Coverage starts 2013, exactly when UFC rankings began — consistent with point-in-time compilation. report.md's "2021 onward" caveat is wrong prose, not a leak |
| `red_corner` | ufcstats listing order | booking | CLEAN | Verified first-listed ≠ winner (36% L/W) |
| `height`, `reach`, `age` | tott (static) + DOB | time-invariant / birth | CLEAN | build_features.py:145-156 |
| All replay-state features: `elo, peak_elo, elo_decline, glicko*, n_fights, win_pct, streaks, slpm, sapm, str_acc/def, td_*, sub_att_p15, kd_*, ctrl_min_share, finish_rate, dec_win_rate, ko/sub_loss_rate, finished_rate, never_finished, head/leg/ground/clinch shares, absorbed_*, form_*, avg_opp_elo, form_opp_elo, sc_close/split/share, five_rd_fights, weight_change, layoff_days` | replay of prior fights | end of fighter's previous fight (< row date) | CLEAN | Snapshot-before-update verified in code + monotonicity differential check (0 violations) |
| `title_bout, women, sched_rounds, weight_lbs` | bout booking (WEIGHTCLASS / TIME FORMAT) | announcement | CLEAN | build_features.py:540-543, :596-599 |
| Prop labels `fight_secs, method_cls, total_td, td/sig/kd_landed_*` | post-fight | after the fight | CLEAN (labels only) | Never in `feats`; `fight_secs` enters prop TRAINING only as a Poisson exposure offset (props.py:180-188); predict-time duration comes from the model mixture (props.py:217-234) |

Columns computed in features.csv but NOT in the model (`vs_*` splits, `matchup_win`,
`chin_vs_power`, `grapple_threat`, `size_vs_div_*`, `glicko*`, cardio features,
`better_rank_adv`): out of scope, verified absent from `feats`.

**Counts: 1 LEAK (4 base features / 8 model columns) · 2 SUSPECT · everything else CLEAN.**

## The LEAK, precisely

`pre_ufc = ESPN pro record (current) − UFC record (current)`. The UFC portion cancels
exactly, so the residual is the fighter's ENTIRE non-UFC career — including regional/
Bellator/PFL fights fought AFTER the training row's date. Fighters cut from the UFC
disproportionately keep fighting elsewhere, so contaminated values correlate with
future career outcomes. Live picks are NOT affected (at prediction time "current" is
legitimately knowable); the inflated artifact is the validated backtest and the
learned weights.

**Fix applied** (branch `leakage-fixes`): remove the 4 features from `SNAP_FEATS`.
A true point-in-time recomputation needs dated non-UFC fight histories, which no local
or reachable source provides (ESPN's API returns current records for historical
dates). Removal is therefore the only guaranteed-clean value; it deletes legitimate
signal along with the leak, so the measured drop is an UPPER bound on the honest cost
and the pre-fix number is the inflated one.

## Pre-registered expectations (MEASUREMENT.md §11) — written before the rerun

Baseline (report.md @ 29cb40c, same features.csv, seed 42):
full test (n=1614) ensemble acc 0.644 / LL 0.637 / Brier 0.223;
odds subset (n=1182) ensemble 0.654 / 0.628 / 0.219, market 0.702 / 0.582 / 0.199,
blend 0.704 / 0.577 / 0.197, blend better in 97.0% of resamples, CI [-0.0002, +0.0101].

Noise context: project gate is 0.002 CV log loss; seed-to-seed spread ~±0.003.

Expectations after removing pre_ufc_* (perm importance of `pre_ufc_winpct_diff` was
+0.0036, #3; Phase 10 measured its adoption at +0.0014 overall CV / +0.0026 on the
low-experience slice):

1. Full-test ensemble log loss WORSENS by +0.001 to +0.004 (0.638-0.641); accuracy
   0.639-0.645. An honest drop is the success condition.
2. Odds-subset ensemble: acc 0.646-0.654, LL 0.629-0.632.
3. Blend: LL 0.577-0.580; blend-better resample share drops to 88-97% but the
   blend-beats-market finding SURVIVES (point estimate still below market 0.582) —
   the blend edge is mostly calibration + stats signal orthogonal to pre_ufc.
4. Tuned configs (best_lgb, best_xgb) probably unchanged.
5. Importance top-3 becomes age_diff, red_corner, form_sapm_diff/elo_diff.
6. If instead metrics are FLAT (within ±0.002) or improve, the Phase 10 adoption was
   noise + leak and removal is free — also a fine outcome, differently embarrassing.

## Results (after rerun)

_(appended after train_report.py completes on branch leakage-fixes)_
