from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


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


def _add_log_price(
    df: pd.DataFrame, price_column: str = "Price", log_column: str = "log_price"
) -> pd.DataFrame:
    """
    Add a log-transformed version of a price column to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame that must contain a price column.
    price_column : str, optional
        Name of the price column to transform. Default is "Price".
    log_column : str, optional
        Name of the output column to store the log-transformed values.
        Default is "log_price".

    Returns
    -------
    pd.DataFrame
        A copy of the original DataFrame with an additional column containing
        the natural logarithm of the price values. Non-positive values are
        replaced with NaN.

    Raises
    ------
    KeyError
        If the specified price_column does not exist in the DataFrame.
    """
    if price_column not in df.columns:
        raise KeyError(f"Column '{price_column}' does not exist in the DataFrame.")

    df = df.copy()
    df[log_column] = np.where(df[price_column] > 0, np.log(df[price_column]), np.nan)
    return df


def _add_sarima_features(
    df: pd.DataFrame,
    price_col: str = "Price",
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[tuple[int, int, int, int]] = (1, 0, 1, 12),
) -> pd.DataFrame:
    """
    Fit a SARIMA model to a price time series and append model-derived features.

    This function fits a SARIMAX model to the column specified by `price_col`
    and adds the following columns to the returned DataFrame:

    - **sarima_fitted**: In-sample fitted values produced by the model.
    - **sarima_residual**: Difference between actual values and fitted values
      (`actual - sarima_fitted`), useful as a stationarized feature.
    - **sarima_forecast_1m**: One-step-ahead forecasts for each timestamp
      (i.e., the forecast for *t+1* based only on data available at *t*).
      This is obtained using `model.get_prediction(dynamic=False)`.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the time series.
    price_col : str, default="Price"
        Name of the column in `df` that holds the target price values.
    order : tuple[int, int, int], default=(1, 1, 1)
        The (p, d, q) order of the SARIMA non-seasonal component.
    seasonal_order : tuple[int, int, int, int] or None, default=(1, 0, 1, 12)
        The (P, D, Q, s) seasonal component. If `None`, no seasonal term is used.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with three extra SARIMA-based feature columns.
    """

    x = df.copy()
    y = x[price_col].astype(float)

    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order if seasonal_order else (0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    # In-sample fitted (same index)
    fitted = model.fittedvalues
    # One-step-ahead predictions (use only up-to-t info for t+1)
    # get_prediction with dynamic=False gives one-step-ahead in-sample
    pred_mean = model.get_prediction().predicted_mean

    x["sarima_fitted"] = fitted
    x["sarima_residual"] = y.values - fitted.values
    x["sarima_forecast_1m"] = pred_mean.values

    return x


def build_features(
    df: pd.DataFrame,
    date_col: str = "Date",
    price_col: str = "Price",
    shift_months: int = 1,
    include_event_targets: bool = True,
    include_sarima: bool = False,
    sarima_order: tuple[int, int, int] = (1, 1, 1),
    sarima_seasonal_order: Optional[tuple[int, int, int, int]] = (1, 0, 1, 12),
) -> pd.DataFrame:
    """
    Monthly feature pipeline for gold prices with anti-leakage shifting.
    Adds classic technical features + optional SARIMA features, then shifts all features.

    Steps
    -----
    1) Sort chronologically
    2) Log-price & returns (1/3/6/12m)
    3) Volatility (3/6/12/24m) & z-scores (6/12/24m)
    4) Momentum & MA crossovers
    5) RSI & MACD
    6) Drawdown (24m)
    7) (Optional) SARIMA features: fitted, residual, 1-step-ahead forecast
    8) Shift features by `shift_months` (targets stay unshifted)
    9) Drop rows with NaNs in features
    """
    if shift_months < 1:
        raise ValueError("shift_months must be >= 1")

    x = df.copy().sort_values(date_col).reset_index(drop=True)

    # Targets to keep untouched
    prefixes = ("time_to_", "roi_")
    if include_event_targets:
        prefixes += ("event_",)
    target_cols = [c for c in x.columns if c.startswith(prefixes)]

    # 2–6) Your existing feature blocks (assumed implemented elsewhere)
    x = _add_log_price(x)  # log_price
    x = add_returns(x, price_col=price_col)  # return_1m/3m/6m/12m
    x = add_volatility_and_zscore(x, price_col=price_col)  # vol_*, zscore_*
    x = add_momentum_and_ma(x, price_col=price_col)  # momentum, MA cross, price_to_ma*
    x = add_rsi(x, price_col=price_col, window=6)  # rsi_6
    x = add_macd(x, price_col=price_col)  # macd, macd_signal, macd_hist
    x = add_drawdown_features(x, price_col=price_col, lookback=24)  # dd_*, max_dd_24m

    # 7) SARIMA features (added BEFORE shifting so they get shifted too)
    if include_sarima:
        x = _add_sarima_features(
            x,
            price_col=price_col,
            order=sarima_order,
            seasonal_order=sarima_seasonal_order,
        )

    # 8) Shift all features to avoid look-ahead (date & targets preserved)
    x = shift_features(
        x,
        date_col=date_col,
        target_cols=target_cols,
        shift_months=shift_months,
    )

    # 9) Drop rows with missing values in features (keep date & targets)
    protect = [date_col] + target_cols
    feature_cols = [c for c in x.columns if c not in protect]
    x = x.dropna(subset=feature_cols).reset_index(drop=True)

    return x
