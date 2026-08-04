"""UFC fight winner prediction — Phase 1 demo slice.

Downloads (once) the Ultimate UFC Dataset, trains a logistic regression on
pre-fight feature differentials with a strict temporal split, and prints
test-set accuracy + log loss against two baselines: better career record,
and the betting-market favorite (vig-removed implied probabilities).

Run: python demo.py
"""

import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

SEED = 42
DATA_PATH = Path(__file__).parent / "data" / "ufc-master.csv"
MIRRORS = [
    "https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/main/ufc-master.csv",
    "https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/master/ufc-master.csv",
]
REQUIRED_COLS = {"date", "Winner", "R_odds", "B_odds", "R_wins", "B_wins"}


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        print(f"Using cached dataset: {DATA_PATH}")
        return pd.read_csv(DATA_PATH, low_memory=False)
    DATA_PATH.parent.mkdir(exist_ok=True)
    for url in MIRRORS:
        try:
            print(f"Downloading {url} ...")
            raw = urllib.request.urlopen(url, timeout=60).read()
            df = pd.read_csv(io.BytesIO(raw), low_memory=False)
            if len(df) < 1000 or not REQUIRED_COLS.issubset(df.columns):
                raise ValueError("unexpected shape/columns")
            DATA_PATH.write_bytes(raw)
            print(f"Cached {len(df)} fights to {DATA_PATH}")
            return df
        except Exception as e:  # try next mirror
            print(f"  mirror failed: {e}")
    sys.exit("All dataset mirrors failed; cannot run demo.")


def american_to_prob(odds: pd.Series) -> pd.Series:
    """American odds -> implied win probability (includes vig)."""
    return np.where(odds < 0, -odds / (-odds + 100), 100 / (odds + 100))


def main() -> None:
    df = load_data()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["Winner"].isin(["Red", "Blue"])].sort_values("date").reset_index(drop=True)
    y = (df["Winner"] == "Red").astype(int)

    # Pre-fight feature differentials, Red minus Blue.
    X = pd.DataFrame({
        "reach_diff": df["R_Reach_cms"] - df["B_Reach_cms"],
        "height_diff": df["R_Height_cms"] - df["B_Height_cms"],
        "age_diff": df["R_age"] - df["B_age"],
        "win_streak_diff": df["R_current_win_streak"] - df["B_current_win_streak"],
        "sig_str_diff": df["R_avg_SIG_STR_landed"] - df["B_avg_SIG_STR_landed"],
        "td_avg_diff": df["R_avg_TD_landed"] - df["B_avg_TD_landed"],
        "southpaw_vs_orthodox": ((df["R_Stance"] == "Southpaw") & (df["B_Stance"] == "Orthodox")).astype(int)
                              - ((df["R_Stance"] == "Orthodox") & (df["B_Stance"] == "Southpaw")).astype(int),
    })

    # Strict temporal split: train on the first 80% of fight dates, test on the rest.
    cutoff = df["date"].quantile(0.8)
    train, test = df["date"] <= cutoff, df["date"] > cutoff
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(random_state=SEED, max_iter=1000),
    )
    model.fit(X[train], y[train])
    prob = model.predict_proba(X[test])[:, 1]
    yt = y[test]

    print(f"\nTrain: {train.sum()} fights through {cutoff.date()}")
    print(f"Test:  {test.sum()} fights, {df.loc[test, 'date'].min().date()} to {df.loc[test, 'date'].max().date()}")
    print(f"Red-corner base rate on test set: {yt.mean():.3f}\n")

    print(f"{'':32}{'accuracy':>10}{'log loss':>10}")
    print(f"{'Logistic regression':32}{accuracy_score(yt, prob > 0.5):>10.3f}{log_loss(yt, prob):>10.3f}")

    # Baseline (a): pick the fighter with the better career record (win pct, wins as tiebreak).
    rec_r = df["R_wins"] / (df["R_wins"] + df["R_losses"]).replace(0, np.nan)
    rec_b = df["B_wins"] / (df["B_wins"] + df["B_losses"]).replace(0, np.nan)
    pick_red = (rec_r.fillna(0.5) > rec_b.fillna(0.5)) | (
        (rec_r.fillna(0.5) == rec_b.fillna(0.5)) & (df["R_wins"] >= df["B_wins"]))
    acc_rec = accuracy_score(yt, pick_red[test].astype(int))
    print(f"{'Better career record':32}{acc_rec:>10.3f}{'n/a':>10}")

    # Baseline (b): market favorite, with vig-removed implied probabilities.
    has_odds = test & df["R_odds"].notna() & df["B_odds"].notna()
    p_r, p_b = american_to_prob(df["R_odds"]), american_to_prob(df["B_odds"])
    p_market = (p_r / (p_r + p_b))[has_odds]
    ym = y[has_odds]
    print(f"{'Market favorite (no-vig)':32}{accuracy_score(ym, p_market > 0.5):>10.3f}{log_loss(ym, p_market):>10.3f}"
          f"   ({has_odds.sum()}/{test.sum()} test fights have odds)")


if __name__ == "__main__":
    main()
