"""
evaluation.py:
computes performance metrics for the PPO agent and all baseline strategies
and builds a comparison table.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# display names
# ---------------------------------------------------------------------------

DISPLAY = {
    "ppo":          "PPO Agent",
    "equal_weight": "Equal-Weight",
    "buy_and_hold": "Buy & Hold",
    "min_variance": "Min-Variance",
    "momentum":     "Momentum",
}

# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def compute_metrics(log_returns: pd.Series, weights: pd.DataFrame = None) -> dict:
    """
    computes annualised return, vol, Sharpe, max drawdown, Calmar, and turnover.
    all inputs are net of transaction cost log-returns.
    """
    lr = log_returns.dropna()
    if len(lr) == 0:
        return {k: np.nan for k in [
            "Cumret", "Ann Return %", "Ann Vol %", "Sharpe",
            "Max Drawdown", "Calmar", "Avg Turnover",
        ]}

    ann_ret = lr.mean() * 252
    ann_vol = lr.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

    #drawdown: cumulative log-return minus its running maximum
    cum      = lr.cumsum()
    drawdown = cum - cum.cummax()
    max_dd   = float(drawdown.min())

    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else np.nan

    turnover = np.nan
    if weights is not None and len(weights) > 1:
        #L1 weight change at each step, averaged over the period
        turnover = float(weights.diff().abs().sum(axis=1).mean())

    return {
        "Cumret":       float(lr.sum()),
        "Ann Return %": float(ann_ret * 100),
        "Ann Vol %":    float(ann_vol * 100),
        "Sharpe":       float(sharpe),
        "Max Drawdown": float(max_dd),
        "Calmar":       float(calmar),
        "Avg Turnover": float(turnover),
    }

# ---------------------------------------------------------------------------
# comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    all_results: dict,
    period: str = "test",
) -> pd.DataFrame:
    """builds a DataFrame with one row per strategy and one column per metric"""
    rows = {}
    for name, res in all_results.items():
        lr  = res.get(f"log_returns_{period}")
        wts = res.get(f"weights_{period}")
        if lr is None:
            continue
        rows[DISPLAY.get(name, name)] = compute_metrics(lr, wts)

    table = pd.DataFrame(rows).T
    #round for display
    fmt = {
        "Cumret":       3, "Ann Return %": 2, "Ann Vol %": 2,
        "Sharpe":       3, "Max Drawdown": 3, "Calmar":    3,
        "Avg Turnover": 4,
    }
    return table.round(fmt)

# ---------------------------------------------------------------------------
# master evaluation
# ---------------------------------------------------------------------------

def run_full_evaluation(
    ppo_results: dict,
    baselines: dict,
    asset_names: list = None,
) -> dict:
    """computes and prints metrics for PPO and all baselines over train and test"""
    #unify all results under a common structure
    all_results = {}

    #PPO
    for period_key, res_key in [("train", "train_results"), ("test", "test_results")]:
        res = ppo_results[res_key]
        if "ppo" not in all_results:
            all_results["ppo"] = {}
        all_results["ppo"][f"log_returns_{period_key}"] = res["log_returns"]
        wts = res["weights"]
        if asset_names is not None and wts is not None:
            wts = pd.DataFrame(wts.values, index=wts.index, columns=asset_names)
        all_results["ppo"][f"weights_{period_key}"] = wts

    #baselines
    for name, res in baselines.items():
        all_results[name] = res

    #metrics tables
    print("\nperformance — training period:")
    table_train = build_comparison_table(all_results, period="train")
    print(table_train.to_string())

    print("\nperformance — test period (out-of-sample):")
    table_test = build_comparison_table(all_results, period="test")
    print(table_test.to_string())

    return {
        "all_results":   all_results,
        "metrics_train": table_train,
        "metrics_test":  table_test,
    }

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from data_pipeline import build_pipeline
    from feature_engineering import build_features
    from hmm_regime import build_regime_pipeline
    from baselines import run_all_baselines
    from ppo_agent import evaluate_policy, MODEL_PATH
    from stable_baselines3 import PPO
    from environment import make_eval_env, make_test_env

    data, bench = build_pipeline()
    feat = build_features(data["returns_train"], data["returns_val"], data["returns_test"])
    regime = build_regime_pipeline(
        data["returns_train"], data["returns_val"], data["returns_test"],
        feat["features_train"], feat["features_val"], feat["features_test"],
    )

    print("\nrunning baselines...")
    baselines = run_all_baselines(
        feat["returns_train_aligned"],
        feat["returns_test_aligned"],
    )

    #load saved PPO model and evaluate (avoids retraining)
    print("\nloading PPO model...")
    best_path  = MODEL_PATH + "_best/best_model.zip"
    model_path = best_path if os.path.exists(best_path) else MODEL_PATH + ".zip"
    model = PPO.load(model_path)

    eval_env = make_eval_env(feat, regime)
    test_env = make_test_env(feat, regime)

    print("evaluating PPO on train period...")
    train_res = evaluate_policy(model, eval_env)
    print("evaluating PPO on test period...")
    test_res  = evaluate_policy(model, test_env)

    ppo_results = {"train_results": train_res, "test_results": test_res}

    asset_names = list(data["returns_train"].columns)
    ev = run_full_evaluation(ppo_results, baselines, asset_names=asset_names)
