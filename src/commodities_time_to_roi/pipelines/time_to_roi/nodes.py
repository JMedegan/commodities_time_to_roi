import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_last_300_months(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Filter a monthly gold price dataset to retain only the most recent 300 months.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame that must contain at least a date column and
        corresponding price data.
    date_col : str, optional (default="Date")
        The column name containing the monthly date values.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing only the most recent 300 months of data,
        sorted in ascending date order.

    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    filtered = df.tail(300)

    logger.info(
        f"Filtered (monthly) range: {filtered[date_col].min().date()} → {filtered[date_col].max().date()}"
    )

    return filtered.reset_index(drop=True)


def add_multi_roi_time_targets(
    df: pd.DataFrame,
    project_params: dict,
    price_col: str = "Price",
    date_col: str = "Date",
    max_nan_frac: float = 0.05,  # keep targets where ≤ 5% NaN
) -> pd.DataFrame:
    """
    Add multiple ROI-based regression targets: number of months until each ROI
    threshold is reached, and **keep only targets where censored fraction ≤ max_nan_frac**.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly gold price dataset sorted chronologically.
    project_params : dict
        Contains:
        - "roi_targets": list of floats (e.g., [0.10, 0.20])
    price_col : str
        Name of the price column.
    date_col : str
        Name of the date column.
    max_nan_frac : float
        Max allowed fraction of NaN values (censored) to keep a target column.
        Default = 0.05 → keep only if at least 95% of rows have an observed time-to-ROI.

    Returns
    -------
    pd.DataFrame
        Copy of df including only *valid* time_to_* target columns.
    """
    df = df.copy()
    df = df.sort_values(date_col)

    prices = df[price_col].values
    n = len(df)

    kept_targets = []

    for roi in project_params["roi_targets"]:
        col = f"time_to_{int(roi * 100)}pct_months"
        df[col] = np.nan

        for i in range(n):
            current_price = prices[i]
            future_roi = (prices[i:] - current_price) / current_price

            idx = np.where(future_roi >= roi)[0]
            if len(idx) > 0:
                df.loc[i, col] = idx[0]

        # ---- Filter based on proportion of NaNs ----
        nan_frac = df[col].isna().mean()
        if nan_frac <= max_nan_frac:
            kept_targets.append(col)
        else:
            df.drop(columns=[col], inplace=True)

    return df


def add_multi_roi_classification_targets(
    df: pd.DataFrame,
    project_params: dict,
    horizon_months: int = 12,
    price_col: str = "Price",
    date_col: str = "Date",
    min_positive_frac: float = 0.10,
) -> pd.DataFrame:
    """
    Add multiple binary ROI targets indicating whether each ROI
    threshold is reached within a fixed horizon, and remove targets
    where the positive class ratio is below `min_positive_frac`.

    Parameters
    ----------
    df : pandas.DataFrame
        Monthly gold price dataset, one row per month.
    project_params : dict
        Must contain key:
        - "roi_targets": list of ROI thresholds (e.g., [0.10, 0.20]).
    horizon_months : int, optional
        Time horizon (in months) to check if the ROI is achieved.
    price_col : str, optional
        Name of the price column.
    date_col : str, optional
        Name of the datetime column.
    min_positive_frac : float, optional
        Minimum fraction of positive class for keeping the target.
        Defaults to 0.10 (10%).

    Returns
    -------
    pandas.DataFrame
        Original df plus selected binary ROI targets.
    """
    df = df.copy()
    df = df.sort_values(date_col)

    prices = df[price_col].to_numpy()
    n = len(df)
    created_cols = []

    # --- Create targets ---
    for roi in project_params["roi_targets"]:
        col = f"roi_{int(roi * 100)}pct_within_{horizon_months}"
        df[col] = False

        # Fill targets
        for i in range(n - horizon_months):
            current_price = prices[i]
            future_price = prices[i + horizon_months]
            roi_val = (future_price - current_price) / current_price
            df.loc[df.index[i], col] = roi_val >= roi

        created_cols.append(col)

    # --- Remove sparse targets ---
    to_drop = []
    for col in created_cols:
        pos_frac = df[col].mean()
        if pos_frac < min_positive_frac:
            to_drop.append(col)

    df = df.drop(columns=to_drop)

    return df


def impute_censored(
    df: pd.DataFrame,
    project_params: dict[any],
    date_col: str = "Date",
    window: int = 3,
    fallback: str = "max_plus_one",
) -> pd.DataFrame:
    """
    Impute censored time-to-ROI targets with the mean of the previous `window` observed values.
    For each NaN row we explicitly average rows [i-window : i-1].

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain time-to-ROI columns named like 'time_to_{pct}pct_months'.
    project_params : dict
        Contains 'roi_targets': list of floats, e.g. [0.10, 0.20, ...].
    date_col : str
        Column used to enforce chronological order.
    window : int
        Number of prior observations to average (default 3).
    fallback : str
        What to do if there are no prior observed values:
        - "max_plus_one": use (max observed) + 1 (conservative)
        - "leave_nan": keep NaN
        - "col_median": use column median (ignoring NaNs)

    Returns
    -------
    pandas.DataFrame
        Copy with NaNs imputed per rule above.
    """
    out = df.copy()
    out = out.sort_values(date_col)

    for roi in project_params["roi_targets"]:
        col = f"time_to_{int(roi * 100)}pct_months"
        if col in out.columns:
            # Precompute fallbacks
            col_max = out[col].max(skipna=True)
            col_median = out[col].median(skipna=True)

            # Loop only over NaN positions
            nan_idx = np.where(out[col].isna().to_numpy())[0]
            vals = out[col].to_numpy(dtype=float)

            for i in nan_idx:
                start = max(0, i - window)
                prev_vals = vals[start:i]  # strictly previous rows
                prev_vals = prev_vals[~np.isnan(prev_vals)]

                if prev_vals.size > 0:
                    vals[i] = float(prev_vals.mean())
                elif fallback == "max_plus_one" and pd.notna(col_max):
                    vals[i] = float(col_max + 1.0)
                elif fallback == "col_median" and pd.notna(col_median):
                    vals[i] = float(col_median)
                elif fallback == "leave_nan":
                    # keep NaN
                    continue
                else:
                    # default safety: leave NaN if nothing to use
                    continue

            out[col] = vals

    return out
