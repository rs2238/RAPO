"""
ppo_agent.py:
wraps stable-baselines3 PPO with hyperparameter config, a training loop,
and an evaluation function that returns log-returns for comparison with baselines.
"""

import os
import numpy as np
import pandas as pd
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback


# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS = 500_000  # total environment steps

#rollout and update
N_STEPS    = 1024   # steps per rollout before each update
BATCH_SIZE = 128    # mini-batch size; must divide N_STEPS
N_EPOCHS   = 5      # passes over rollout data per update (PPO range: 4-10)

#discount and advantage
GAMMA      = 0.999  # near-undiscounted; long-horizon rewards matter
GAE_LAMBDA = 0.95   # GAE bias-variance tradeoff; SB3 default

#PPO clipping and loss weights
CLIP_RANGE = 0.2    # standard PPO clipping range
ENT_COEF   = 0.05   # entropy bonus; discourages degenerate all-in allocations
VF_COEF    = 0.5    # value function loss weight; SB3 default

#optimiser
LEARNING_RATE = 3e-4  # Adam initial lr, decayed linearly to 0

NET_ARCH = dict(pi=[64, 32], vf=[64, 32])  # separate policy and value heads

SEED       = 42
LOG_FREQ   = 50_000  # console logging interval in steps
EVAL_FREQ  = 50_000  # EvalCallback evaluation frequency
MODEL_PATH = "ppo_portfolio"


# ---------------------------------------------------------------------------
# learning rate schedule
# ---------------------------------------------------------------------------

def linear_schedule(initial_lr: float) -> Callable[[float], float]:
    """decays lr linearly from initial_lr to 0 as progress_remaining goes from 1 to 0"""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_lr
    return schedule


# ---------------------------------------------------------------------------
# callback
# ---------------------------------------------------------------------------

class TrainingLogger(BaseCallback):
    """prints mean episodic reward and entropy every LOG_FREQ steps"""

    def __init__(self, log_freq: int = LOG_FREQ):
        super().__init__(verbose=0)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0 and self.n_calls > 0:
            stats = self.model.logger.name_to_value
            rew   = stats.get("rollout/ep_rew_mean", float("nan"))
            ent   = stats.get("train/entropy_loss",  float("nan"))
            print(f"  [{self.num_timesteps:>7,} steps]  "
                  f"ep_rew_mean = {rew:+.5f}  |  "
                  f"entropy = {ent:.5f}")
        return True  # returning False would stop training


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------

def make_model(train_env, seed: int = SEED) -> PPO:
    """wraps train_env in DummyVecEnv and Monitor, then constructs the PPO model"""
    vec_env = DummyVecEnv([lambda: Monitor(train_env)])

    model = PPO(
        policy        = "MlpPolicy",
        env           = vec_env,
        learning_rate = linear_schedule(LEARNING_RATE),
        n_steps       = N_STEPS,
        batch_size    = BATCH_SIZE,
        n_epochs      = N_EPOCHS,
        gamma         = GAMMA,
        gae_lambda    = GAE_LAMBDA,
        clip_range    = CLIP_RANGE,
        ent_coef      = ENT_COEF,
        vf_coef       = VF_COEF,
        policy_kwargs = {"net_arch": NET_ARCH},
        seed          = seed,
        verbose       = 0,
    )
    return model


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(
    model: PPO,
    eval_env,
    total_timesteps: int = TOTAL_TIMESTEPS,
    save_path: str = MODEL_PATH,
) -> PPO:
    """runs PPO training with EvalCallback saving the best checkpoint by val reward"""
    print(f"\ntraining PPO: {total_timesteps:,} total timesteps")
    print(f"  rollout: n_steps={N_STEPS}, batch={BATCH_SIZE}, epochs={N_EPOCHS}")
    print(f"  objective: gamma={GAMMA}, gae_lambda={GAE_LAMBDA}, "
          f"clip={CLIP_RANGE}, ent={ENT_COEF}")
    print(f"  network: pi={NET_ARCH['pi']}, vf={NET_ARCH['vf']}\n")

    eval_callback = EvalCallback(
        eval_env          = DummyVecEnv([lambda: Monitor(eval_env)]),
        best_model_save_path = save_path + "_best",
        log_path          = None,
        eval_freq         = EVAL_FREQ,
        n_eval_episodes   = 1,
        deterministic     = True,
        render            = False,
        verbose           = 0,
    )

    model.learn(
        total_timesteps     = total_timesteps,
        callback            = [eval_callback, TrainingLogger()],
        reset_num_timesteps = True,
    )

    model.save(save_path)
    print(f"\nfinal model saved → {save_path}.zip")
    print(f"best model saved  → {save_path}_best/best_model.zip")
    return model


# ---------------------------------------------------------------------------
# policy evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(model: PPO, env, deterministic: bool = True) -> dict:
    """
    rolls out one complete episode using the trained policy.
    returns log_returns, weights, and portfolio_value series.
    """
    obs, _ = env.reset()
    done   = False

    dates, log_rets, weights_hist, nav = [], [], [], []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        dates.append(info["date"])
        log_rets.append(info["reward"])
        weights_hist.append(info["weights"])
        nav.append(info["portfolio_value"])

    idx = pd.DatetimeIndex(dates)

    return {
        "log_returns":     pd.Series(log_rets,  index=idx, name="log_return"),
        "weights":         pd.DataFrame(weights_hist, index=idx),
        "portfolio_value": pd.Series(nav, index=idx),
    }


# ---------------------------------------------------------------------------
# master pipeline
# ---------------------------------------------------------------------------

def run_ppo_pipeline(
    feat:   dict,
    regime: dict,
    tc_rate: float = 0.001,
    total_timesteps: int = TOTAL_TIMESTEPS,
) -> dict:
    """wires up environments, trains PPO, and evaluates on train and test"""
    from environment import make_train_env, make_eval_env, make_val_env, make_test_env

    train_env = make_train_env(feat, regime, tc_rate=tc_rate)
    val_env   = make_val_env(feat,   regime, tc_rate=tc_rate)
    eval_env  = make_eval_env(feat,  regime, tc_rate=tc_rate)
    test_env  = make_test_env(feat,  regime, tc_rate=tc_rate)

    model = make_model(train_env)
    model = train(model, val_env, total_timesteps=total_timesteps)

    #load best checkpoint (may differ from final model)
    best_path = MODEL_PATH + "_best/best_model"
    if os.path.exists(best_path + ".zip"):
        print(f"\nloading best checkpoint from {best_path}.zip")
        model = PPO.load(best_path, env=DummyVecEnv([lambda: Monitor(eval_env)]))

    print("\nevaluating on training period (in-sample)...")
    train_results = evaluate_policy(model, eval_env)

    print("evaluating on test period (out-of-sample)...")
    test_results = evaluate_policy(model, test_env)

    #quick summary
    def _sharpe(lr: pd.Series) -> float:
        lr = lr.dropna()
        return float((lr.mean() / lr.std()) * np.sqrt(252)) if lr.std() > 0 else 0.0

    print("\nPPO results:")
    for label, res in [("train", train_results), ("test", test_results)]:
        lr = res["log_returns"]
        print(f"  {label}:  cumret = {lr.sum():+.3f}  |  "
              f"Sharpe = {_sharpe(lr):+.3f}  |  "
              f"final NAV = {res['portfolio_value'].iloc[-1]:.3f}")

    return {
        "model":         model,
        "train_results": train_results,
        "test_results":  test_results,
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_pipeline import build_pipeline
    from feature_engineering import build_features
    from hmm_regime import build_regime_pipeline

    data, bench = build_pipeline()
    feat = build_features(data["returns_train"], data["returns_val"], data["returns_test"])
    regime = build_regime_pipeline(
        data["returns_train"], data["returns_val"], data["returns_test"],
        feat["features_train"], feat["features_val"], feat["features_test"],
    )

    ppo = run_ppo_pipeline(feat, regime)
