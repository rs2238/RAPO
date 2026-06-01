"""
environment.py:
custom Gymnasium environment for daily portfolio rebalancing across N assets.
state: [features | current_weights | regime_probs], reward: log return minus TC.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

TC_RATE = 0.001  # 10 bps per unit of L1 turnover


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

class PortfolioEnv(gym.Env):
    """
    portfolio rebalancing environment.
    at each step the agent observes features and regime probs, outputs weights,
    and earns the next-day log portfolio return minus transaction costs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        returns: pd.DataFrame,
        regime_probs: pd.DataFrame,
        tc_rate: float = TC_RATE,
        random_start: bool = False,
        use_regime: bool = True,
    ):
        super().__init__()

        #validate alignment — all three DataFrames must share the same index
        assert features.index.equals(returns.index), \
            "features and returns index mismatch — run align_returns_to_features()"
        assert features.index.equals(regime_probs.index), \
            "features and regime_probs index mismatch"
        assert len(features) >= 2, \
            "need at least 2 timesteps (one to observe, one to realise return)"

        #store as numpy for fast indexing inside step()
        self.features_arr = features.values.astype(np.float32)
        self.returns_arr  = returns.values.astype(np.float32)
        self.regime_arr   = regime_probs.values.astype(np.float32)
        self.dates        = features.index

        self.T          = len(features)
        self.N          = returns.shape[1]
        self.F          = features.shape[1]
        self.tc_rate    = tc_rate
        self.random_start = random_start
        self.use_regime = use_regime

        #observation: features (F) + current weights (N) + regime probs (3 or 0)
        obs_dim = self.F + self.N + (3 if use_regime else 0)
        self.observation_space = spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape = (obs_dim,),
            dtype = np.float32,
        )

        #action: N raw values projected onto the simplex
        self.action_space = spaces.Box(
            low  = 0.0,
            high = 1.0,
            shape = (self.N,),
            dtype = np.float32,
        )

        #episode state (initialised in reset())
        self.current_step    = 0
        self.current_weights = np.ones(self.N, dtype=np.float32) / self.N
        self.portfolio_value = 1.0

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """concatenates features, current weights, and (optionally) regime probs"""
        parts = [
            self.features_arr[self.current_step],
            self.current_weights,
        ]
        if self.use_regime:
            parts.append(self.regime_arr[self.current_step])
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def _project_simplex(raw: np.ndarray) -> np.ndarray:
        """clips negatives to zero then normalises; falls back to equal weights if all-zero"""
        w = np.clip(raw, 0.0, None)
        total = w.sum()
        if total < 1e-8:
            return np.ones_like(raw) / len(raw)
        return w / total

    # ------------------------------------------------------------------
    # gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if self.random_start:
            #start anywhere that leaves room for at least one return step
            self.current_step = int(self.np_random.integers(0, self.T - 1))
        else:
            self.current_step = 0

        #start each episode with equal weights
        self.current_weights = np.ones(self.N, dtype=np.float32) / self.N
        self.portfolio_value = 1.0

        return self._get_obs(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """executes one rebalancing step: project weights, apply TC, earn next-day return"""
        assert self.current_step is not None, "call reset() before step()"

        #simplex projection
        new_weights = self._project_simplex(action)

        #transaction cost
        turnover = float(np.abs(new_weights - self.current_weights).sum())
        tc       = self.tc_rate * turnover

        #next-day returns
        next_step = self.current_step + 1
        if next_step >= self.T:
            return self._get_obs(), 0.0, True, False, {"warning": "stepped past end"}

        next_log_returns = self.returns_arr[next_step]  # shape (N,)

        #exact log portfolio return
        arith_asset_returns  = np.expm1(next_log_returns)   #eʳⁱ − 1
        portfolio_arith_ret  = float(np.dot(new_weights, arith_asset_returns))
        log_portfolio_return = float(np.log1p(portfolio_arith_ret))

        #reward
        reward = log_portfolio_return - tc

        #state update
        self.current_weights  = new_weights
        self.current_step     = next_step
        self.portfolio_value *= np.exp(log_portfolio_return - tc)

        terminated = (self.current_step >= self.T - 1)

        obs  = self._get_obs()
        info = {
            "date":                 self.dates[self.current_step],
            "log_portfolio_return": log_portfolio_return,
            "transaction_cost":     tc,
            "turnover":             turnover,
            "reward":               reward,
            "portfolio_value":      self.portfolio_value,
            "weights":              new_weights.copy(),
        }

        return obs, reward, terminated, False, info

    def render(self):
        pass


# ---------------------------------------------------------------------------
# factory helpers
# ---------------------------------------------------------------------------

def make_train_env(feat: dict, regime: dict, tc_rate: float = TC_RATE,
                   use_regime: bool = True) -> PortfolioEnv:
    """training env with random_start=True for episode variety"""
    return PortfolioEnv(
        features     = feat["features_train"],
        returns      = feat["returns_train_aligned"],
        regime_probs = regime["regime_probs_train"],
        tc_rate      = tc_rate,
        random_start = True,
        use_regime   = use_regime,
    )


def make_eval_env(feat: dict, regime: dict, tc_rate: float = TC_RATE,
                  use_regime: bool = True) -> PortfolioEnv:
    """training-period env with random_start=False for deterministic evaluation"""
    return PortfolioEnv(
        features     = feat["features_train"],
        returns      = feat["returns_train_aligned"],
        regime_probs = regime["regime_probs_train"],
        tc_rate      = tc_rate,
        random_start = False,
        use_regime   = use_regime,
    )


def make_val_env(feat: dict, regime: dict, tc_rate: float = TC_RATE,
                 use_regime: bool = True) -> PortfolioEnv:
    """validation env used by EvalCallback to save the best-generalising checkpoint"""
    return PortfolioEnv(
        features     = feat["features_val"],
        returns      = feat["returns_val_aligned"],
        regime_probs = regime["regime_probs_val"],
        tc_rate      = tc_rate,
        random_start = False,
        use_regime   = use_regime,
    )


def make_test_env(feat: dict, regime: dict, tc_rate: float = TC_RATE,
                  use_regime: bool = True) -> PortfolioEnv:
    """held-out test env; touch only once for final evaluation"""
    return PortfolioEnv(
        features     = feat["features_test"],
        returns      = feat["returns_test_aligned"],
        regime_probs = regime["regime_probs_test"],
        tc_rate      = tc_rate,
        random_start = False,
        use_regime   = use_regime,
    )


# ---------------------------------------------------------------------------
# sanity check
# ---------------------------------------------------------------------------

def run_env_checks(env: PortfolioEnv) -> None:
    """steps through with random actions to verify obs shape, reward finiteness, and termination"""
    from gymnasium.utils.env_checker import check_env
    print("[run_env_checks] running gymnasium check_env...")
    check_env(env, warn=True, skip_render_check=True)
    print("[run_env_checks] check_env passed.")

    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, "bad obs shape after reset"
    assert obs.dtype == np.float32, "obs dtype should be float32"

    total_steps, total_reward = 0, 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        assert obs.shape  == env.observation_space.shape, f"bad obs shape at step {total_steps}"
        assert np.isfinite(reward), f"non-finite reward at step {total_steps}: {reward}"
        assert abs(info["weights"].sum() - 1.0) < 1e-5, \
            f"weights don't sum to 1 at step {total_steps}: sum={info['weights'].sum()}"

        total_reward += reward
        total_steps  += 1

    print(f"[run_env_checks] episode complete: {total_steps} steps, "
          f"total reward = {total_reward:.4f}, "
          f"final portfolio value = {info['portfolio_value']:.4f}")
    assert total_steps == env.T - 1, \
        f"expected {env.T - 1} steps, got {total_steps}"
    print("[run_env_checks] all checks passed.")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_pipeline import build_pipeline
    from feature_engineering import build_features
    from hmm_regime import build_regime_pipeline

    data, bench = build_pipeline()
    feat   = build_features(data["returns_train"], data["returns_val"], data["returns_test"])
    regime = build_regime_pipeline(
        data["returns_train"], data["returns_val"], data["returns_test"],
        feat["features_train"], feat["features_val"], feat["features_test"],
    )

    print("\ntraining environment:")
    train_env = make_train_env(feat, regime)
    print(f"observation space : {train_env.observation_space}")
    print(f"action space      : {train_env.action_space}")
    print(f"timesteps (T)     : {train_env.T}")
    print(f"assets (N)        : {train_env.N}")
    print(f"features (F)      : {train_env.F}")
    print(f"obs dimension     : {train_env.observation_space.shape[0]}"
          f" = {train_env.F}F + {train_env.N}N + 3 regime")

    print("\nsanity checks (eval env, deterministic episode):")
    eval_env = make_eval_env(feat, regime)
    run_env_checks(eval_env)
