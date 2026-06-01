"""
data_pipeline.py:
fetches daily OHLCV data, computes log-returns, handles missing data,
and produces clean dfs for feature engineering
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from typing import Tuple, Dict

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

ASSETS = ["XLK", "XLF", "XLE", "XLV", "XLU"]    # sector ETFs
BENCHMARK = "SPY"                               # kept separate and not traded, meant as a performance benchmark only
START_DATE   = "2000-01-01"
END_DATE     = "2024-12-31"
TRAIN_CUTOFF = "2018-12-31"
VAL_CUTOFF   = "2021-12-31"                     # train: 2000-2018, validation: 2019–2021, test: 2022–2024

#threshold: if >MAX_MISSING_FRAC of rows for an asset are NaN after forward filling, we raise rather than silently drop data
MAX_MISSING_FRAC = 0.01 

# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    fetches adjusted closing prices via yfinance

    auto_adjust=True:
      stock splits and dividends create discontinuities in raw
      price series. a 2 for 1 split looks like a 50% overnight return.
      auto_adjust folds both splits and dividends into the price series so
      that log-returns reflect actual investor P&L.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts for splits and dividends
        progress=False,
    )

    # yfinance returns a MultiIndex (field, ticker) when multiple tickers are requested. we only need Close prices.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "date"
    return prices

# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------

def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    handles missing data conservatively

    strategy:
      1. forward fill gaps of exactly 1 day (stale quotes, data vendor
         hiccups). we limit to ffill(limit=1). we never want to carry
         a price forward for more than one day because that fabricates
         a zero return day that could fool the HMM.
      2. after forward filling, check whether any asset still has an
         unacceptable fraction of NaNs. if so, raise (don't silently
         discard data without the user knowing).
      3. drop any remaining rows with NaN in any column. because ETFs
         all trade on the same calendar, post fill NaNs are almost always
         at the very start of the series before an ETF existed.
    """
    #forward fill single missing days
    prices_filled = prices.ffill(limit=1)

    #audit residual NaN fractions
    nan_fracs = prices_filled.isna().mean()
    bad_assets = nan_fracs[nan_fracs > MAX_MISSING_FRAC]
    if not bad_assets.empty:
        raise ValueError(
            f"Assets with excessive missing data after forward-fill:\n"
            f"{bad_assets}\n"
            f"Consider shortening the date range or replacing the asset."
        )

    #drop rows where anything is still NaN
    n_before = len(prices_filled)
    prices_clean = prices_filled.dropna()
    n_dropped = n_before - len(prices_clean)
    if n_dropped > 0:
        print(f"[clean_prices] Dropped {n_dropped} rows with residual NaNs "
              f"(likely prelisting dates at the start of the series).")

    return prices_clean

# ---------------------------------------------------------------------------
# log-returns
# ---------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """computes daily log-returns: r_t = ln(P_t / P_{t-1})"""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


# ---------------------------------------------------------------------------
# train/val/test split
# ---------------------------------------------------------------------------

def split_data(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    train_cutoff: str,
    val_cutoff: str,
) -> Dict[str, pd.DataFrame]:
    """
    partition prices and returns into train, val, and test sets

    boundaries:
      train: start -> train_cutoff (inclusive)
      val: train_cutoff+1 -> val_cutoff (inclusive)
      test: val_cutoff+1 -> end

    the .iloc[1:] drops the boundary date that loc includes on both sides
    to avoid the same day appearing in two splits
    """
    train_ts = pd.Timestamp(train_cutoff)
    val_ts   = pd.Timestamp(val_cutoff)

    return {
        "prices_train":  prices.loc[:train_ts],
        "prices_val":    prices.loc[train_ts:val_ts].iloc[1:],
        "prices_test":   prices.loc[val_ts:].iloc[1:],
        "returns_train": returns.loc[:train_ts],
        "returns_val":   returns.loc[train_ts:val_ts].iloc[1:],
        "returns_test":  returns.loc[val_ts:].iloc[1:],
    }

# ---------------------------------------------------------------------------
# sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
) -> None:
    """lightweight checks that catch common upstream data issues. call after cleaning, before splitting."""
    assert not prices.isnull().any().any(), "NaNs remain in prices"
    assert not returns.isnull().any().any(), "NaNs remain in returns"

    # log-returns should be approximately N(0, 0.01) for daily equity data.
    # a mean absolute return > 5% per day signals something is wrong.
    mean_abs = returns.abs().mean()
    suspicious = mean_abs[mean_abs > 0.05]
    if not suspicious.empty:
        print(f"warning: unusually large mean absolute returns:\n{suspicious}")

    # verify prices are positive (adjusted prices can be very small but should never be zero or negative)
    assert (prices > 0).all().all(), "non positive prices found"

    # check the index is monotonically increasing with no duplicates
    assert prices.index.is_monotonic_increasing, "price index not sorted"
    assert prices.index.is_unique, "duplicate dates in price index"

    print("[sanity_checks] all checks passed.")

# ---------------------------------------------------------------------------
# summary helper
# ---------------------------------------------------------------------------

def summarise(data: Dict[str, pd.DataFrame]) -> None:
    """prints a concise overview of the dataset"""
    tr = data["returns_train"]
    va = data["returns_val"]
    te = data["returns_test"]

    print("\ndataset summary:")
    print(f"assets         : {list(tr.columns)}")
    print(f"train period   : {tr.index[0].date()} → {tr.index[-1].date()}  "
          f"({len(tr)} trading days)")
    print(f"val period     : {va.index[0].date()} → {va.index[-1].date()}  "
          f"({len(va)} trading days)")
    print(f"test period    : {te.index[0].date()} → {te.index[-1].date()}  "
          f"({len(te)} trading days)")
    print(f"\ntrain log-return statistics (daily):")
    print(tr.describe().loc[["mean", "std", "min", "max"]].round(5))

# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def build_pipeline(
    assets: list[str] = ASSETS,
    benchmark: str = BENCHMARK,
    start: str = START_DATE,
    end: str = END_DATE,
    train_cutoff: str = TRAIN_CUTOFF,
    val_cutoff: str = VAL_CUTOFF,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    full pipeline: fetch -> clean -> log-returns -> split

    returns:
    data : dict with keys prices_train, prices_test, returns_train, returns_test
    benchmark_returns : log-returns for SPY (used only in evaluation, not training)
    """
    print(f"fetching {assets + [benchmark]} from {start} to {end}...")

    prices_raw = fetch_prices(assets, start, end)
    bench_raw  = fetch_prices([benchmark], start, end)

    #clean
    prices = clean_prices(prices_raw)
    bench  = clean_prices(bench_raw)

    #align benchmark to asset calendar (they should match for ETFs)
    bench = bench.reindex(prices.index).ffill(limit=1).dropna()

    #compute log-returns
    returns           = compute_log_returns(prices)
    benchmark_returns = compute_log_returns(bench)

    #sanity checks (on full series before split)
    run_sanity_checks(prices, returns)

    #train/val/test split
    data = split_data(prices, returns, train_cutoff, val_cutoff)

    #summary
    summarise(data)

    return data, benchmark_returns

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data, bench_returns = build_pipeline()
