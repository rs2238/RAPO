"""
hmm_regime.py:
fits a K-state Gaussian HMM on daily log-returns to identify market regimes.
returns soft regime probability vectors appended to the RL agent's observation.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

N_STATES    = 3
N_ITER      = 200    # max Baum-Welch iterations per restart
N_RESTARTS  = 10     # number of EM restarts; keep the best log-likelihood
MIN_COVAR   = 1e-3   # covariance floor to prevent singular matrices
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def _fit_single(
    observations: np.ndarray,
    n_states: int,
    n_iter: int,
    seed: int,
) -> Tuple[hmm.GaussianHMM, float]:
    """single Baum-Welch run from one random initialisation"""
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        min_covar=MIN_COVAR,
        random_state=seed,
        verbose=False,
    )
    model.fit(observations)
    log_likelihood = model.score(observations)
    return model, log_likelihood

def fit_hmm(
    returns_train: pd.DataFrame,
    n_states: int = N_STATES,
    n_restarts: int = N_RESTARTS,
    n_iter: int = N_ITER,
) -> hmm.GaussianHMM:
    """
    fits a Gaussian HMM with multiple random restarts, returns the best model.
    Baum-Welch is EM and converges to a local optimum, restarts improve the fit.
    """
    observations = returns_train.values  # shape (T, N_assets)

    best_model, best_ll = None, -np.inf
    for i in range(n_restarts):
        try:
            model, ll = _fit_single(observations, n_states, n_iter,
                                    seed=RANDOM_SEED + i)
            if ll > best_ll:
                best_model, best_ll = model, ll
        except Exception as e:
            print(f"[fit_hmm] restart {i} failed ({e}), skipping.")

    if best_model is None:
        raise RuntimeError("all HMM fitting attempts failed.")

    converged = best_model.monitor_.converged
    print(f"[fit_hmm] best log-likelihood: {best_ll:.2f}")
    print(f"[fit_hmm] EM converged: {converged}")
    if not converged:
        print("[fit_hmm] warning: EM did not converge — consider increasing n_iter.")

    return best_model

# ---------------------------------------------------------------------------
# state labelling
# ---------------------------------------------------------------------------

def label_states(model: hmm.GaussianHMM) -> Dict[int, str]:
    """
    assigns bull/bear/high_vol labels by sorting states on mean log-return.
    labelling is heuristic — the HMM just learns clusters in return space.
    """
    #average mean return across assets for each state, shape (K,)
    mean_returns = model.means_.mean(axis=1)
    sorted_idx   = np.argsort(mean_returns)  # ascending

    bear_idx     = int(sorted_idx[0])
    high_vol_idx = int(sorted_idx[1])
    bull_idx     = int(sorted_idx[2])

    labels = {bull_idx: "bull", bear_idx: "bear", high_vol_idx: "high_vol"}

    print("\n[label_states] emission parameters per state:")
    print(f"  {'state':>6}  {'label':>8}  {'mean return':>12}  {'covariance trace':>16}")
    for k, label in sorted(labels.items()):
        mean_ret  = model.means_[k].mean()
        cov_trace = np.trace(model.covars_[k])
        print(f"  {k:>6}  {label:>8}  {mean_ret:>+12.5f}  {cov_trace:>16.6f}")

    #sanity check
    if np.trace(model.covars_[high_vol_idx]) < np.trace(model.covars_[bull_idx]):
        print("\n  warning: 'high_vol' state has less covariance than 'bull' — labels may be misleading.")

    return labels

# ---------------------------------------------------------------------------
# regime probabilities
# ---------------------------------------------------------------------------

def regime_probabilities(
    model: hmm.GaussianHMM,
    returns: pd.DataFrame,
    state_labels: Dict[int, str],
) -> pd.DataFrame:
    """
    computes posterior state probabilities using the forward-backward algorithm.
    output: (T, 3) DataFrame with columns [regime_bull, regime_bear, regime_high_vol].
    """
    posteriors = model.predict_proba(returns.values)  #(T, K)

    #always output columns in [bull, bear, high_vol] order regardless of internal hmmlearn index
    ordered_labels = ["bull", "bear", "high_vol"]
    label_to_idx   = {v: k for k, v in state_labels.items()}
    col_order      = [label_to_idx[lbl] for lbl in ordered_labels]

    probs = pd.DataFrame(
        posteriors[:, col_order],
        index=returns.index,
        columns=[f"regime_{lbl}" for lbl in ordered_labels],
    )
    return probs

# ---------------------------------------------------------------------------
# master pipeline
# ---------------------------------------------------------------------------

def build_regime_pipeline(
    returns_train: pd.DataFrame,
    returns_val: pd.DataFrame,
    returns_test: pd.DataFrame,
    features_train: pd.DataFrame,
    features_val: pd.DataFrame,
    features_test: pd.DataFrame,
) -> dict:
    """
    full HMM pipeline: fit -> label -> compute probabilities -> align.
    the HMM is fit on training returns only, then applied to val and test.
    """
    print("fitting HMM on training log-returns...")
    model = fit_hmm(returns_train)

    print("\nlabelling hidden states...")
    state_labels = label_states(model)

    print("\ncomputing regime probabilities...")
    probs_train_full = regime_probabilities(model, returns_train, state_labels)
    probs_val_full   = regime_probabilities(model, returns_val,   state_labels)
    probs_test_full  = regime_probabilities(model, returns_test,  state_labels)

    #trim to match the feature matrix (which dropped warm-up rows)
    regime_probs_train = probs_train_full.reindex(features_train.index)
    regime_probs_val   = probs_val_full.reindex(features_val.index)
    regime_probs_test  = probs_test_full.reindex(features_test.index)

    assert not regime_probs_train.isnull().any().any(), \
        "NaNs in train regime probs — check that features and returns share dates"
    assert not regime_probs_val.isnull().any().any(), \
        "NaNs in val regime probs — check that features and returns share dates"
    assert not regime_probs_test.isnull().any().any(), \
        "NaNs in test regime probs — check that features and returns share dates"

    dominant_train = regime_probs_train.idxmax(axis=1).value_counts(normalize=True)
    print("\nregime summary:")
    print(f"train: {regime_probs_train.shape}  "
          f"({regime_probs_train.index[0].date()} → {regime_probs_train.index[-1].date()})")
    print(f"val:   {regime_probs_val.shape}  "
          f"({regime_probs_val.index[0].date()} → {regime_probs_val.index[-1].date()})")
    print(f"test:  {regime_probs_test.shape}  "
          f"({regime_probs_test.index[0].date()} → {regime_probs_test.index[-1].date()})")
    print("\ntrain dominant-regime occupancy:")
    for regime, frac in dominant_train.items():
        print(f"  {regime}: {frac:.1%}")

    return {
        "model":              model,
        "state_labels":       state_labels,
        "regime_probs_train": regime_probs_train,
        "regime_probs_val":   regime_probs_val,
        "regime_probs_test":  regime_probs_test,
    }

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_pipeline import build_pipeline
    from feature_engineering import build_features

    data, bench_returns = build_pipeline()
    feat = build_features(data["returns_train"], data["returns_val"], data["returns_test"])

    regime = build_regime_pipeline(
        returns_train  = data["returns_train"],
        returns_val    = data["returns_val"],
        returns_test   = data["returns_test"],
        features_train = feat["features_train"],
        features_val   = feat["features_val"],
        features_test  = feat["features_test"],
    )
