"""
ablation.py:
trains two PPO variants — with and without HMM regime features — to isolate
the causal effect of the regime signal on out-of-sample performance.
"""

import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from environment import make_train_env, make_val_env, make_eval_env, make_test_env
from ppo_agent import (
    make_model, train, evaluate_policy,
    TOTAL_TIMESTEPS, MODEL_PATH, SEED,
)
from evaluation import compute_metrics, DISPLAY


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "with_regime": {
        "use_regime": True,
        "model_path": "ablation_with_regime",
        "display":    "PPO + Regime",
    },
    "no_regime": {
        "use_regime": False,
        "model_path": "ablation_no_regime",
        "display":    "PPO (no regime)",
    },
}


# ---------------------------------------------------------------------------
# training helpers
# ---------------------------------------------------------------------------

def _make_envs(feat: dict, regime: dict, use_regime: bool, tc_rate: float):
    train_env = make_train_env(feat, regime, tc_rate=tc_rate, use_regime=use_regime)
    val_env   = make_val_env(feat,   regime, tc_rate=tc_rate, use_regime=use_regime)
    test_env  = make_test_env(feat,  regime, tc_rate=tc_rate, use_regime=use_regime)
    return train_env, val_env, test_env


def train_or_load(
    name: str,
    config: dict,
    feat: dict,
    regime: dict,
    tc_rate: float,
    force_retrain: bool = False,
) -> PPO:
    """loads a saved model if one exists; otherwise trains from scratch"""
    best_path  = config["model_path"] + "_best/best_model.zip"
    final_path = config["model_path"] + ".zip"

    if not force_retrain:
        #try the ablation-specific save path first
        for path in [best_path, final_path]:
            if os.path.exists(path):
                print(f"[{name}] loading saved model from {path}")
                train_env, eval_env, _ = _make_envs(
                    feat, regime, config["use_regime"], tc_rate
                )
                vec_env = DummyVecEnv([lambda: Monitor(train_env)])
                return PPO.load(path, env=vec_env)

        #fallback: if "with_regime" has no dedicated save, try the main PPO checkpoint
        if name == "with_regime":
            step6_path = MODEL_PATH + "_best/best_model.zip"
            if os.path.exists(step6_path):
                print(f"[{name}] loading checkpoint from {step6_path}")
                train_env, eval_env, _ = _make_envs(
                    feat, regime, config["use_regime"], tc_rate
                )
                vec_env = DummyVecEnv([lambda: Monitor(train_env)])
                return PPO.load(step6_path, env=vec_env)

    #train from scratch
    print(f"\n[{name}] training PPO (use_regime={config['use_regime']})...")
    train_env, eval_env, _ = _make_envs(feat, regime, config["use_regime"], tc_rate)
    model = make_model(train_env, seed=SEED)
    model = train(model, eval_env,
                  total_timesteps=TOTAL_TIMESTEPS,
                  save_path=config["model_path"])
    return model


# ---------------------------------------------------------------------------
# metrics helper
# ---------------------------------------------------------------------------

def _sharpe(lr: pd.Series) -> float:
    lr = lr.dropna()
    return float((lr.mean() / lr.std()) * np.sqrt(252)) if lr.std() > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# master ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    feat: dict,
    regime: dict,
    baselines: dict,
    tc_rate: float = 0.001,
    force_retrain: bool = False,
) -> dict:
    """train (or load) both PPO variants, evaluate both, and compare"""
    results = {}

    for name, config in ABLATION_CONFIGS.items():
        model = train_or_load(name, config, feat, regime, tc_rate, force_retrain)

        _, _, test_env = _make_envs(feat, regime, config["use_regime"], tc_rate)
        eval_env = make_eval_env(feat, regime, tc_rate=tc_rate, use_regime=config["use_regime"])

        print(f"\n[{name}] evaluating on train period...")
        train_res = evaluate_policy(model, eval_env)

        print(f"[{name}] evaluating on test period...")
        test_res = evaluate_policy(model, test_env)

        results[name] = {
            "config":        config,
            "model":         model,
            "train_results": train_res,
            "test_results":  test_res,
        }

    #print comparison table
    _print_comparison_table(results, baselines)

    return {"results": results}


# ---------------------------------------------------------------------------
# comparison table
# ---------------------------------------------------------------------------

def _print_comparison_table(ablation_results: dict, baselines: dict) -> None:
    """prints side-by-side metrics for both PPO variants and all baselines"""
    rows = []

    for name, res in ablation_results.items():
        display = ABLATION_CONFIGS[name]["display"]
        for period, key in [("Train", "train_results"), ("Test", "test_results")]:
            lr  = res[key]["log_returns"]
            wts = res[key]["weights"]
            m   = compute_metrics(lr, wts)
            rows.append({
                "Strategy": display,
                "Period":   period,
                **m,
            })

    for bname, bres in baselines.items():
        display = DISPLAY.get(bname, bname)
        for period, pkey in [("Train", "log_returns_train"), ("Test", "log_returns_test")]:
            lr  = bres.get(pkey)
            wts = bres.get(f"weights_{period.lower()}")
            if lr is None:
                continue
            m = compute_metrics(lr, wts)
            rows.append({
                "Strategy": display,
                "Period":   period,
                **m,
            })

    df = pd.DataFrame(rows).set_index(["Strategy", "Period"])
    fmt = {"Cumret": 3, "Ann Return %": 2, "Ann Vol %": 2,
           "Sharpe": 3, "Max Drawdown": 3, "Calmar": 3, "Avg Turnover": 4}
    df = df.round(fmt)

    for period in ["Train", "Test"]:
        print(f"\nablation — {period.lower()} period:")
        subset = df.xs(period, level="Period")
        print(subset.to_string())


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_pipeline import build_pipeline
    from feature_engineering import build_features
    from hmm_regime import build_regime_pipeline
    from baselines import run_all_baselines

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

    ablation = run_ablation(feat, regime, baselines, force_retrain=True)
