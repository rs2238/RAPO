"""
baselines.py:
four baseline portfolio strategies — equal-weight, buy-and-hold, min-variance,
and momentum — simulated with consistent transaction costs and weight drift.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from typing import Callable

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

TC_RATE     = 0.001  # 10 bps per unit of L1 turnover — matches environment.py
MV_LOOKBACK = 252    # covariance estimation window: 1 trading year
MOM_LOOKBACK = 60    # momentum signal lookback: ~1 quarter
REBAL_FREQ  = 21     # monthly rebalancing for min-variance and momentum

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _solve_min_variance(cov: np.ndarray) -> np.ndarray:
    """
    long-only minimum-variance portfolio via SLSQP; falls back to equal weights on failure
    """
    N  = cov.shape[0]
    w0 = np.ones(N) / N  # equal-weight initialisation

    result = minimize(
        fun    = lambda w: float(w @ cov @ w),
        x0     = w0,
        method = "SLSQP",
        jac    = lambda w: 2.0 * cov @ w,
        bounds = [(0.0, 1.0)] * N,
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options = {"ftol": 1e-10, "maxiter": 500},
    )

    if result.success:
        w = np.clip(result.x, 0.0, None)
        return w / w.sum()  # renormalise for floating-point residuals
    return w0  #fallback


def _estimate_covariance(returns_window: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage estimator (more stable than raw sample covariance for small N)
    """
    lw = LedoitWolf()
    lw.fit(returns_window)
    return lw.covariance_

# ---------------------------------------------------------------------------
# simulation engine
# ---------------------------------------------------------------------------

def _simulate(
    returns: pd.DataFrame,
    target_weight_fn: Callable[[pd.DataFrame], np.ndarray],
    tc_rate: float,
    rebalance_freq: int,
    min_history: int = 1,
) -> pd.Series:
    """
    simulates a rebalancing strategy over a returns DataFrame.
    at each step: optionally rebalance, earn daily return, drift weights.
    returns (log_return_series, weights_dataframe).
    """
    T, N             = returns.shape
    log_ret_arr      = returns.values.astype(np.float64)
    current_weights  = np.ones(N) / N  # start equal-weighted
    port_log_returns = np.empty(T)
    weights_hist     = np.empty((T, N))

    for t in range(T):
        #1. rebalancing decision
        if t % rebalance_freq == 0:
            history = returns.iloc[:t]  #data before day t

            if len(history) >= min_history:
                target = target_weight_fn(history)
            else:
                target = np.ones(N) / N  #equal-weight fallback

            turnover        = np.abs(target - current_weights).sum()
            tc              = tc_rate * turnover
            current_weights = target
        else:
            tc = 0.0

        #2. portfolio return for day t
        arith_asset_rets     = np.expm1(log_ret_arr[t])         # e^r − 1
        port_arith_ret       = float(np.dot(current_weights, arith_asset_rets))
        log_port_ret         = float(np.log1p(port_arith_ret))  # log(1 + x)
        port_log_returns[t]  = log_port_ret - tc
        weights_hist[t]      = current_weights

        #3. weight drift
        new_values = current_weights * np.exp(log_ret_arr[t])
        total = new_values.sum()
        if total > 1e-12:
            current_weights = new_values / total

    log_ret_series = pd.Series(port_log_returns, index=returns.index, name="log_return")
    weights_df     = pd.DataFrame(weights_hist, index=returns.index,
                                  columns=returns.columns)
    return log_ret_series, weights_df

# ---------------------------------------------------------------------------
# weight functions
# ---------------------------------------------------------------------------

def _ew_weights(history: pd.DataFrame) -> np.ndarray:
    """equal weight: ignore history, return 1/N"""
    return np.ones(history.shape[1]) / history.shape[1]

def _mv_weights(history: pd.DataFrame, lookback: int) -> np.ndarray:
    """minimum variance weights using trailing lookback days of returns"""
    window = history.iloc[-lookback:].values
    cov    = _estimate_covariance(window)
    return _solve_min_variance(cov)

def _momentum_weights(history: pd.DataFrame, lookback: int) -> np.ndarray:
    """weights proportional to positive trailing cumulative log-return (zero for negative momentum)"""
    N = history.shape[1]
    if len(history) < lookback:
        return np.ones(N) / N

    momentum = history.iloc[-lookback:].sum().values  #cumulative log-returns
    pos_mom  = np.maximum(momentum, 0.0)
    total    = pos_mom.sum()

    if total < 1e-10:
        return np.ones(N) / N  #all assets negative: equal-weight fallback
    return pos_mom / total

# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------

def equal_weight(
    returns: pd.DataFrame,
    tc_rate: float = TC_RATE,
    rebalance_freq: int = 1,
) -> pd.Series:
    """equal-weight portfolio rebalanced at rebalance_freq"""
    return _simulate(returns, _ew_weights, tc_rate, rebalance_freq, min_history=1)[0]


def buy_and_hold(
    returns: pd.DataFrame,
) -> pd.Series:
    """equal-weight on day 0 then no rebalancing; weights drift freely with the market"""
    T, N             = returns.shape
    log_ret_arr      = returns.values.astype(np.float64)
    current_weights  = np.ones(N) / N
    port_log_returns = np.empty(T)
    weights_hist     = np.empty((T, N))

    for t in range(T):
        arith_asset_rets    = np.expm1(log_ret_arr[t])
        port_arith_ret      = float(np.dot(current_weights, arith_asset_rets))
        port_log_returns[t] = float(np.log1p(port_arith_ret))  # no TC
        weights_hist[t]     = current_weights

        new_values = current_weights * np.exp(log_ret_arr[t])
        total      = new_values.sum()
        if total > 1e-12:
            current_weights = new_values / total

    return pd.Series(port_log_returns, index=returns.index, name="log_return")


def min_variance(
    returns: pd.DataFrame,
    tc_rate: float = TC_RATE,
    lookback: int = MV_LOOKBACK,
    rebalance_freq: int = REBAL_FREQ,
) -> pd.Series:
    """rolling long only minimum-variance portfolio using Ledoit-Wolf covariance estimation"""
    fn = lambda history: _mv_weights(history, lookback)
    return _simulate(returns, fn, tc_rate, rebalance_freq, min_history=lookback)[0]


def momentum(
    returns: pd.DataFrame,
    tc_rate: float = TC_RATE,
    lookback: int = MOM_LOOKBACK,
    rebalance_freq: int = REBAL_FREQ,
) -> pd.Series:
    """cross sectional momentum portfolio: weights proportional to positive trailing returns"""
    fn = lambda history: _momentum_weights(history, lookback)
    return _simulate(returns, fn, tc_rate, rebalance_freq, min_history=lookback)[0]

# ---------------------------------------------------------------------------
# master
# ---------------------------------------------------------------------------

def run_all_baselines(
    returns_train: pd.DataFrame,
    returns_test: pd.DataFrame,
    tc_rate: float = TC_RATE,
) -> dict:
    """
    runs all four baselines over train and test periods.
    concatenates before simulating so min-variance and momentum have lookback history on the first test day.
    """
    returns_full = pd.concat([returns_train, returns_test])
    train_idx    = returns_train.index
    test_idx     = returns_test.index

    #each entry: (target_weight_fn, rebalance_freq, min_history)
    strategy_specs = {
        "equal_weight": (_ew_weights,                                   1,           1),
        "min_variance": (lambda h: _mv_weights(h, MV_LOOKBACK),         REBAL_FREQ,  MV_LOOKBACK),
        "momentum":     (lambda h: _momentum_weights(h, MOM_LOOKBACK),  REBAL_FREQ,  MOM_LOOKBACK),
    }

    results = {}

    #buy-and-hold handled separately (no rebalancing, no TC)
    print("  running buy_and_hold...", end=" ")
    bah_log_rets = buy_and_hold(returns_full)
    #reconstruct weight history using large rebalance_freq to approximate no rebalancing
    _, bah_weights_full = _simulate(
        returns_full, _ew_weights, tc_rate=0.0, rebalance_freq=10**9, min_history=1
    )
    results["buy_and_hold"] = {
        "log_returns_train": bah_log_rets.reindex(train_idx),
        "log_returns_test":  bah_log_rets.reindex(test_idx),
        "log_returns_full":  bah_log_rets,
        "weights_train":     bah_weights_full.reindex(train_idx),
        "weights_test":      bah_weights_full.reindex(test_idx),
    }
    print(f"train cumret = {bah_log_rets.reindex(train_idx).sum():+.3f}  |  "
          f"test cumret = {bah_log_rets.reindex(test_idx).sum():+.3f}")

    for name, (fn, freq, min_hist) in strategy_specs.items():
        print(f"  running {name}...", end=" ")
        full_log_rets, full_weights = _simulate(
            returns_full, fn, tc_rate, freq, min_history=min_hist
        )
        results[name] = {
            "log_returns_train": full_log_rets.reindex(train_idx),
            "log_returns_test":  full_log_rets.reindex(test_idx),
            "log_returns_full":  full_log_rets,
            "weights_train":     full_weights.reindex(train_idx),
            "weights_test":      full_weights.reindex(test_idx),
        }
        train_cumret = full_log_rets.reindex(train_idx).sum()
        test_cumret  = full_log_rets.reindex(test_idx).sum()
        print(f"train cumret = {train_cumret:+.3f}  |  test cumret = {test_cumret:+.3f}")

    return results

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_pipeline import build_pipeline
    from feature_engineering import build_features

    data, bench = build_pipeline()
    feat = build_features(data["returns_train"], data["returns_val"], data["returns_test"])

    print("\nrunning baselines...")
    baselines = run_all_baselines(
        feat["returns_train_aligned"],
        feat["returns_test_aligned"],
    )
