from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd


def add_returns(
    df: pd.DataFrame,
    price_col: str = "Price",
    windows: Iterable[int] = (1, 3, 6, 12),
) -> pd.DataFrame:
    """
    Add simple and log returns at multiple horizons.

    Parameters
    ----------
    df : pandas.DataFrame
        Monthly price series (sorted).
    price_col : str, optional
        Price column name.
    windows : Iterable[int], optional
        Return horizons in months.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: ret_{k}m, log_ret_{k}m, optional log_price.
    """
    df = df.copy()
    p = df[price_col].astype(float)

    for k in windows:
        df[f"ret_{k}m"] = p.pct_change(k)  # simple return over k months
        df[f"log_ret_{k}m"] = np.log(p / p.shift(k))  # log return over k months

    return df


def add_volatility_and_zscore(
    df: pd.DataFrame,
    price_col: str = "Price",
    vol_windows: Iterable[int] = (3, 6, 12, 24),
    zscore_windows: Iterable[int] = (6, 12, 24),
) -> pd.DataFrame:
    """
    Add rolling volatility of monthly log returns and rolling z-scores of price.

    Parameters
    ----------
    df : pandas.DataFrame
        Monthly price series (sorted).
    price_col : str, optional
        Price column name.
    vol_windows : Iterable[int], optional
        Windows (months) for realized volatility of 1M log returns.
    zscore_windows : Iterable[int], optional
        Windows for rolling z-score of price.

    Returns
    -------
    pandas.DataFrame
        DataFrame with vol_{w}m and z_{w}m columns.
    """
    df = df.copy()
    log_ret_1m = np.log(
        df[price_col].astype(float) / df[price_col].astype(float).shift(1)
    )

    for w in vol_windows:
        df[f"vol_{w}m"] = log_ret_1m.rolling(w).std()

    for w in zscore_windows:
        roll = df[price_col].rolling(w)
        df[f"z_{w}m"] = (df[price_col] - roll.mean()) / roll.std()

    return df


def add_momentum_and_ma(
    df: pd.DataFrame,
    price_col: str = "Price",
    roc_windows: Iterable[int] = (3, 6, 12),
    sma_windows: Iterable[int] = (3, 6, 12, 24),
    cross_pairs: Iterable[tuple[int, int]] = ((3, 12), (6, 12), (6, 24)),
) -> pd.DataFrame:
    """
    Add momentum (rate of change), simple moving averages, and crossover signals.

    Parameters
    ----------
    df : pandas.DataFrame
        Monthly price series (sorted).
    price_col : str, optional
        Price column name.
    roc_windows : Iterable[int], optional
        Windows for rate-of-change (% change vs k months ago).
    sma_windows : Iterable[int], optional
        Windows for simple moving averages.
    cross_pairs : Iterable[Tuple[int, int]], optional
        (short, long) pairs to compute SMA crossovers.

    Returns
    -------
    pandas.DataFrame
        DataFrame with roc_{k}m, sma_{k}m, and cross_{s}v{l} columns (boolean).
    """
    df = df.copy()
    p = df[price_col].astype(float)

    for k in roc_windows:
        df[f"roc_{k}m"] = p.pct_change(k)

    for w in sma_windows:
        df[f"sma_{w}m"] = p.rolling(w).mean()

    for sht, lg in cross_pairs:
        short = f"sma_{sht}m"
        long_ = f"sma_{lg}m"
        col = f"cross_{sht}v{lg}"
        df[col] = (df[short] > df[long_]).astype(int)

    return df


def add_rsi(
    df: pd.DataFrame, price_col: str = "Price", window: int = 6
) -> pd.DataFrame:
    """
    Add a monthly RSI feature using an EMA-based implementation.

    Parameters
    ----------
    df : pandas.DataFrame
        Monthly price series (sorted).
    price_col : str, optional
        Price column name.
    window : int, optional
        RSI lookback in months (6–12 is typical for monthly data).

    Returns
    -------
    pandas.DataFrame
        DataFrame with rsi_{window} column (0-100).
    """
    df = df.copy()
    delta = df[price_col].astype(float).diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(gain, index=df.index).ewm(alpha=1 / window, adjust=False).mean()
    roll_down = (
        pd.Series(loss, index=df.index).ewm(alpha=1 / window, adjust=False).mean()
    )

    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    df[f"rsi_{window}"] = rsi

    return df


def add_macd(
    df: pd.DataFrame,
    price_col: str = "Price",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Add MACD features (EMA-based trend indicators) for monthly data.

    Parameters
    ----------
    df : pandas.DataFrame
        Monthly price series (sorted).
    price_col : str, optional
        Price column name.
    fast : int, optional
        Fast EMA window (months).
    slow : int, optional
        Slow EMA window (months).
    signal : int, optional
        Signal EMA window (months).

    Returns
    -------
    pandas.DataFrame
        DataFrame with macd, macd_signal, macd_hist columns.
    """
    df = df.copy()
    p = df[price_col].astype(float)
    ema_fast = p.ewm(span=fast, adjust=False).mean()
    ema_slow = p.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal
    return df


def add_drawdown_features(
    df: pd.DataFrame, price_col: str = "Price", lookback: int = 24
) -> pd.DataFrame:
    """
    Add rolling maximum drawdown and peak distance over a lookback window.

    Parameters
    ----------
    df : pandas.DataFrame
        Monthly price series (sorted).
    price_col : str, optional
        Price column name.
    lookback : int, optional
        Window (months) used to compute rolling drawdown metrics.

    Returns
    -------
    pandas.DataFrame
        DataFrame with mdd_{lookback}m and dist_from_peak_{lookback}m.
    """
    df = df.copy()
    p = df[price_col].astype(float)
    rolling_max = p.rolling(lookback).max()
    drawdown = p / rolling_max - 1.0
    df[f"mdd_{lookback}m"] = drawdown.rolling(lookback).min()
    df[f"dist_from_peak_{lookback}m"] = drawdown
    return df


def shift_features(
    df: pd.DataFrame,
    date_col: str = "Date",
    target_cols: Optional[list[str]] = None,
    shift_months: int = 1,
) -> pd.DataFrame:
    """
    Shift all feature columns down by `shift_months` to prevent look-ahead bias.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with features and targets.
    date_col : str, optional
        Name of the date column (kept unshifted).
    target_cols : List[str], optional
        Columns to exclude from shifting (targets remain aligned to t).
    shift_months : int, optional
        Number of months to shift features forward (default 1).

    Returns
    -------
    pandas.DataFrame
        DataFrame with shifted features and original targets.
    """
    df = df.copy()
    keep = {date_col}
    if target_cols:
        keep.update(target_cols)

    cols_to_shift = [c for c in df.columns if c not in keep]
    df[cols_to_shift] = df[cols_to_shift].shift(shift_months)
    return df


def build_features(
    df: pd.DataFrame,
    date_col: str = "Date",
    price_col: str = "Price",
    shift_months: int = 1,
) -> pd.DataFrame:
    """
    Full monthly feature pipeline for gold prices with anti-leakage shift.

    Steps
    -----
    2) Add returns (1/3/6/12m) and log price
    3) Add volatility (3/6/12/24m) and z-scores (6/12/24m)
    4) Add momentum and moving-average crossovers
    5) Add RSI and MACD
    6) Add drawdown metrics (24m)
    7) Shift features by 1 month to avoid look-ahead
    8) Drop rows with NaNs introduced by rolling windows

    Parameters
    ----------
    df : pandas.DataFrame
        Monthly price series with columns [Date, Price].
    date_col : str, optional
        Date column name.
    price_col : str, optional
        Price column name.
    target_cols : List[str], optional
        Target columns to keep unshifted (e.g., time_to_XXpct, event_XXpct, roi_XXpct_within_12).
    shift_months : int, optional
        Feature shift to prevent leakage (default 1).

    Returns
    -------
    pandas.DataFrame
        Feature-enriched DataFrame.
    """
    target_cols = [
        c for c in df.columns if c.startswith(("time_to_", "event_", "roi_"))
    ]
    x = add_returns(df, price_col=price_col)
    x = add_volatility_and_zscore(x, price_col=price_col)
    x = add_momentum_and_ma(x, price_col=price_col)
    x = add_rsi(x, price_col=price_col, window=6)
    x = add_macd(x, price_col=price_col)
    x = add_drawdown_features(x, price_col=price_col, lookback=24)

    x = shift_features(
        x, date_col=date_col, target_cols=target_cols, shift_months=shift_months
    )

    # Drop rows with missing values due to rolling/shift; keep targets intact
    if target_cols:
        cols = [date_col] + list(target_cols)
        feature_cols = [c for c in x.columns if c not in cols]
        x = x.dropna(subset=feature_cols).reset_index(drop=True)
    else:
        x = x.dropna().reset_index(drop=True)

    return x
