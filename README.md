# RAPO (Regime Aware Portfolio Optimization)

A PPO based portfolio allocation agent that conditions on real time market regime estimates from a Hidden Markov Model. The core claim is that feeding soft regime probabilities into the RL observation space, rather than treating all market conditions identically, produces a meaningfully lower volatility policy (isolated in a controlled ablation that holds all other hyperparameters constant).

## What the system does

At each trading day, the agent observes 30 technical features (rolling volatility, momentum, pairwise correlations across five sector ETFs) concatenated with three soft regime probabilities from a Gaussian HMM (bull / bear / high volatility). It outputs a long only portfolio weight vector. Reward is the log portfolio return minus a 10bps per unit transaction cost penalty. The HMM is the key design element: rather than forcing the agent to infer market conditions implicitly from price features alone, the regime probabilities give it an explicit, continuously updated signal about the current environment, letting it learn regime conditional policies like defensive rotation toward XLV/XLU when bear probability spikes.

**Asset universe:** XLK, XLF, XLE, XLV, XLU (S&P 500 sector ETFs)  
**Benchmark:** SPY (evaluation only, never traded)  
**Data range:** 2000-2024 | **Splits:** train 2000-2018, val 2019-2021, test 2022-2024

---

## Architecture

The pipeline runs in four stages. First, daily adjusted prices are fetched via yfinance and converted to log-returns. Second, two parallel processes run on the training data: a feature matrix is built from rolling volatility, momentum, and pairwise correlations (30 features across 5 assets), and a 3 state Gaussian HMM is fit to identify market regimes both of which are scaled and fit on train only to prevent leakage. Third, these are combined into a custom Gymnasium environment where the agent observes `[30 features | 5 current weights | 3 regime probs]`, outputs portfolio weights projected onto the simplex, and earns log portfolio return minus a 10bps per unit turnover penalty. Fourth, PPO trains for 500k steps with an EvalCallback that saves the best checkpoint by validation reward. The held out test set is touched once at the end.

---

## Results

Performance on the held out test period (2022-2024), net of transaction costs:

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max Drawdown |
|---|---|---|---|---|
| **PPO + Regime** | **7.72%** | **16.36%** | **0.472** | -17.6% |
| PPO (no regime) | 7.14% | 20.19% | 0.354 | -20.2% |
| Equal-Weight | 8.26% | 15.25% | 0.542 | -16.3% |
| Buy & Hold | 6.02% | 14.65% | 0.411 | -15.7% |
| Min-Variance | 3.67% | 13.85% | 0.265 | -17.0% |
| Momentum | -1.82% | 19.00% | -0.096 | -40.1% |

Equal-Weight has the best Sharpe on this test window (0.542). The PPO agent's edge is specifically in volatility reduction and drawdown control relative to the no regime baseline, not in dominating every metric against every strategy.

---

## Ablation study

The regime signal reduces annualized volatility by 3.8 percentage points out of sample (20.2% -> 16.4%), with all other hyperparameters held constant. This is the project's central empirical claim.

**Methodology:** Two PPO agents are trained identically (same architecture, same hyperparameters, same random seed, same train/val/test split), differing only in whether the 3 dimensional HMM regime probability vector is included in the observation. The `use_regime=False` variant observes `[30 features | 5 weights]` and the `use_regime=True` variant observes `[30 features | 5 weights | 3 regime probs]`. This isolates the marginal contribution of the regime signal from any benefit that comes from the RL training procedure itself.

**What the numbers show:** The regime aware agent produces meaningfully lower out of sample volatility without a proportional reduction in return, improving the Sharpe ratio. The no regime agent achieves competitive in sample performance but shows greater return dispersion on the test set, consistent with regime blind policies that happen to work in one environment failing to adapt when conditions shift.

---

## Design rationale

**Why PPO?** PPO's clipped surrogate objective prevents the catastrophic policy updates that can occur in portfolio environments where a single large position change sends returns to near zero. Its on policy nature also ensures the agent trains on fresh experience from the current policy rather than stale transitions, which matters when regime shifts can quickly make old experiences unrepresentative.

**Why a Gaussian HMM with 3 states?** Gaussian HMMs have tractable exact inference (forward-backward algorithm), interpretable parameters (emission means map cleanly to bull/bear/high vol), and a well understood fitting procedure. Three states is the minimal resolution that captures the empirically distinct equity regimes (trend up, trend down, high dispersion). More states risk overfitting to the training period's specific market history.

**Why soft regime probabilities rather than hard labels?** Hard Viterbi labels switch discretely and inject an artificial step function into the observation space. Soft posteriors let the agent act on regime uncertainty. A partially elevated bear probability of 0.4 is genuinely different information from 0.9, and the policy can reflect that. It also avoids the instability of policies that conditionally execute very different strategies triggered by a single bit flip.

**Why these five sector ETFs?** They cover technology, financials, energy, healthcare, and utilities, spanning the growth/value and cyclical/defensive axes that dominate sector rotation strategies. Pairwise correlations across these five are among the most informative features for the HMM: correlations spike toward 1.0 during broad selloffs, which is the primary signature of the bear regime.

**Why 10bps transaction cost?** 10bps per unit of L1 turnover is a conservative estimate for institutional ETF trading that includes both spread and market impact. It is deliberately consistent across PPO and all baselines so that turnover differences between strategies are reflected fairly in the comparison.

---

## Limitations

**Transaction cost model is simplified:** The flat 10bps linear TC model ignores size dependent market impact, bid ask spread variation across volatility regimes, and the distinction between market and limit orders. A real implementation would face higher costs during the high volatility regime, which is exactly when the agent is most likely to trade aggressively.

**HMM labels are post-hoc:** States are labeled bull/bear/high vol by sorting on mean emission return after fitting. The HMM learns clusters in return space; the labels are interpretive. In practice the "high vol" state sometimes has higher mean returns than expected, which the code flags as a warning.

**Backtest optimism:** The hyperparameter choices (gamma=0.999, entropy coefficient, network architecture) were selected with knowledge of the dataset. A proper walk forward evaluation would require holding a true out of sample period that was never used during development. The 2022-2024 test set here was not used for hyperparameter selection, but the overall approach was developed with awareness of the data's general properties.

**Single seed:** Results are reported for one random seed (42). RL training variance across seeds can be substantial, particularly for portfolio environments with sparse reward structure. A robust evaluation would average across multiple seeds.

**Static regime model:** The HMM is fit once on training data and applied forward. In practice, market microstructure and cross asset relationships evolve over decades; a production system would require periodic model refitting.

---

## Setup

```bash
pip install -r requirements.txt
python ppo_agent.py      # train from scratch + evaluate
python evaluation.py     # evaluate saved model
python ablation.py       # ablation study
```

Each module also runs standalone (`python data_pipeline.py`, `python hmm_regime.py`, etc.) and prints a diagnostic summary of its stage.
