"""
feature_engineering.py:
computes rolling volatility, momentum, and pairwise correlations from
log-returns. fits a StandardScaler on training features and applies it
consistently to both splits.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import itertools

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

#window lengths in trading days
VOL_SHORT = 20   # ~1 month
VOL_LONG = 60   # ~1 quarter
MOM_SHORT = 20
MOM_LONG = 60
CORR_WINDOW = 60

#longest window determines how many rows are NaN at the start.
#all windows must be <= MAX_WINDOW or the warm-up logic below will silently under drop rows.
MAX_WINDOW = max(VOL_SHORT, VOL_LONG, MOM_SHORT, MOM_LONG, CORR_WINDOW)

# ---------------------------------------------------------------------------
# individual feature computations
# ---------------------------------------------------------------------------

def rolling_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    annualised realised volatility over a rolling window

    std of log-returns * sqrt(252) gives the annualised vol in the same
    units as typical quant reporting (e.g. "18% annualised vol")
    """
    vol = returns.rolling(window).std() * np.sqrt(252)
    vol.columns = [f"vol_{window}d_{col}" for col in returns.columns]
    return vol


def rolling_momentum(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    cumulative log-return over a rolling window

    simplest, most interpretable momentum signal. a positive
    value indicates the asset has trended up over the window while negative means down.
    """
    mom = returns.rolling(window).sum()
    mom.columns = [f"mom_{window}d_{col}" for col in returns.columns]
    return mom


def rolling_correlations(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    upper triangle of the rolling pairwise correlation matrix

    for N assets there are N*(N-1)/2 unique pairs. we discard the diagonal
    (always 1.0) and the lower triangle (redundant by symmetry).

    motivation: in bear regimes, cross asset equity correlations
    spike toward 1.0 as everything sells off together. this is one of the
    strongest features for the HMM regime detector.
    """
    assets = list(returns.columns)
    #upper triangle pairs: (asset_i, asset_j) where i < j
    pairs = list(itertools.combinations(assets, 2))

    #pandas computes the full N×N rolling correlation matrix efficiently
    corr_rolling = returns.rolling(window).corr()   # MultiIndex: (date, asset) × asset

    #unstack inner index -> (date) × (asset_i, asset_j) MultiIndex on columns
    corr_unstacked = corr_rolling.unstack(level=1)  # columns become (asset_i, asset_j)

    #select only upper triangle pairs
    corr_pairs = corr_unstacked[pairs].copy()
    corr_pairs.columns = [f"corr_{window}d_{a}_{b}" for a, b in pairs]

    return corr_pairs

# ---------------------------------------------------------------------------
# master feature builder
# ---------------------------------------------------------------------------

def compute_features(returns: pd.DataFrame) -> pd.DataFrame:
    """
    concatenate all feature families into 1 df

    the warm-up period (first MAX_WINDOW - 1 rows) is NaN across all features
    because every rolling computation needs a full window of history.
    we drop these rows here so the output has no NaNs.

    output shape: (T - MAX_WINDOW + 1, F)
    where F = 2*N (vol) + 2*N (mom) + N*(N-1)/2 (corr)
            = 2*5 + 2*5 + 10 = 30 for our 5 asset universe.
    """
    parts = [
        rolling_volatility(returns, VOL_SHORT),
        rolling_volatility(returns, VOL_LONG),
        rolling_momentum(returns, MOM_SHORT),
        rolling_momentum(returns, MOM_LONG),
        rolling_correlations(returns, CORR_WINDOW),
    ]

    features = pd.concat(parts, axis=1)

    #drop warm-up rows (NaN from rolling windows)
    n_before = len(features)
    features = features.dropna()
    n_dropped = n_before - len(features)
    print(f"[compute_features] Dropped {n_dropped} warm-up rows "
          f"(max window = {MAX_WINDOW}). "
          f"Feature matrix shape: {features.shape}")

    return features

# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------

def fit_scaler(features_train: pd.DataFrame) -> StandardScaler:
    """
    fit a StandardScaler on training features only

    important note: we keep the scaler object because the RL environment needs to apply
    the exact same transformation at inference time
    """
    scaler = StandardScaler()
    scaler.fit(features_train.values)
    return scaler


def apply_scaler(
    features: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    apply a prefitted scaler to a feature df

    returns a df with the same index and column names as the input
    so downstream code can still do date based slicing
    """
    scaled_values = scaler.transform(features.values)
    return pd.DataFrame(
        scaled_values,
        index=features.index,
        columns=features.columns,
    )

# ---------------------------------------------------------------------------
# alignment helper
# ---------------------------------------------------------------------------

def align_returns_to_features(
    returns: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    trim a returns df to match the feature matrix index

    after dropping warm-up rows, the feature matrix starts MAX_WINDOW - 1
    days later than the returns. the environment needs returns and features
    to share the same index so it can look up the return on day t when
    it knows the features on day t.
    """
    return returns.reindex(features.index)

# ---------------------------------------------------------------------------
# master entry point
# ---------------------------------------------------------------------------

def build_features(
    returns_train: pd.DataFrame,
    returns_val: pd.DataFrame,
    returns_test: pd.DataFrame,
) -> dict:
    """
    full feature pipeline: compute -> fit scaler -> scale -> align

    the scaler is fit on training data only. val and test features are
    transformed with the train fitted scaler so no future information leaks.

    returns a dict with:
        features_{train,val,test}: scaled feature DataFrames
        features_{train,val,test}_raw: unscaled (for inspection / HMM fitting)
        returns_{train,val,test}_aligned: returns trimmed to match feature index
        scaler: fitted StandardScaler
        feature_names: list of F column name strings
    """
    print("computing training features...")
    features_train_raw = compute_features(returns_train)

    print("computing val features...")
    features_val_raw = compute_features(returns_val)

    print("computing test features...")
    features_test_raw = compute_features(returns_test)

    print("fitting scaler on training data")
    scaler = fit_scaler(features_train_raw)

    features_train = apply_scaler(features_train_raw, scaler)
    features_val   = apply_scaler(features_val_raw,   scaler)
    features_test  = apply_scaler(features_test_raw,  scaler)

    #sanity checks
    assert not features_train.isnull().any().any(), "NaNs in training features"
    assert not features_val.isnull().any().any(),   "NaNs in val features"
    assert not features_test.isnull().any().any(),  "NaNs in test features"

    train_mean = features_train.mean().abs().max()
    train_std  = features_train.std().max()
    assert train_mean < 1e-6, f"scaled train mean not near zero: {train_mean:.6f}"
    assert abs(train_std - 1.0) < 0.01, f"scaled train std not near one: {train_std:.6f}"

    print(f"\nfeature summary:")
    print(f"Training features : {features_train.shape}  "
          f"({features_train.index[0].date()} → {features_train.index[-1].date()})")
    print(f"val features      : {features_val.shape}  "
          f"({features_val.index[0].date()} → {features_val.index[-1].date()})")
    print(f"test features     : {features_test.shape}  "
          f"({features_test.index[0].date()} → {features_test.index[-1].date()})")
    print(f"feature names     : {list(features_train.columns)}")

    return {
        "features_train":        features_train,
        "features_val":          features_val,
        "features_test":         features_test,
        "features_train_raw":    features_train_raw,
        "features_val_raw":      features_val_raw,
        "features_test_raw":     features_test_raw,
        "returns_train_aligned": align_returns_to_features(returns_train, features_train),
        "returns_val_aligned":   align_returns_to_features(returns_val,   features_val),
        "returns_test_aligned":  align_returns_to_features(returns_test,  features_test),
        "scaler":                scaler,
        "feature_names":         list(features_train.columns),
    }

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_pipeline import build_pipeline

    data, bench_returns = build_pipeline()
    feat = build_features(data["returns_train"], data["returns_val"], data["returns_test"])
